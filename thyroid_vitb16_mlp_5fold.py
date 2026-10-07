"""
Thyroid Nodule IMAGE Classification — ViT-B/16 + MLP  (5-Fold CV)
==================================================================
Single-image classifier (no video, no temporal modelling, no ROI branch).

Pipeline:
  image → ViT-B/16 (partially frozen) → 768-d feature → MLP → Benign/Malignant

Expected folder structure:
    DATA_ROOT/
        fold_0/
            train/benign/  train/malignant/
            val/benign/    val/malignant/
            test/benign/   test/malignant/
        fold_1/ ... fold_4/

Per-fold outputs:
  • best + last checkpoints
  • per-image CSV
  • training log

Aggregate report printed at the end (mean ± std across folds).
"""

import os
import csv
import sys
import sys as _sys

import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ─────────────────────────────────────────────────────────────────────
# Tee (stdout → console + file)
# ─────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, log_path: str, mode: str = "w"):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._file   = open(log_path, mode, encoding="utf-8")
        self._stdout = _sys.__stdout__

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self.flush()
        self._file.close()
        _sys.stdout = self._stdout


# ─────────────────────────────────────────────────────────────────────
# 0. Diagnostics
# ─────────────────────────────────────────────────────────────────────

def diagnose(fold_root: str):
    fold_root = Path(fold_root)
    print(f"\n{'='*60}")
    print(f"  {fold_root.name}")
    for split in ("train", "val", "test"):
        for cls in ("benign", "malignant"):
            cls_dir = fold_root / split / cls
            if not cls_dir.exists():
                print(f"  [{split}/{cls}]  MISSING FOLDER")
                continue
            imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            print(f"  [{split}/{cls}]  images={len(imgs)}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────

class ThyroidImageDataset(Dataset):
    LABEL_MAP  = {"benign": 0, "malignant": 1}
    CLASS_NAMES = ["Benign", "Malignant"]

    def __init__(self,
                 split_root: str,
                 img_size:   int  = 224,
                 augment:    bool = False):

        self.split_root = Path(split_root)
        self.samples: List[Tuple[Path, int]] = []

        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root not found: {self.split_root}")

        existing = {d.name.lower(): d
                    for d in self.split_root.iterdir() if d.is_dir()}

        for class_name, label in self.LABEL_MAP.items():
            cls_dir = existing.get(class_name.lower())
            if cls_dir is None:
                print(f"  [WARNING] Missing class folder: "
                      f"{self.split_root / class_name}")
                continue
            for img_path in sorted(p for p in cls_dir.iterdir()
                                   if p.suffix.lower() in IMAGE_EXTS):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.split_root}.")

        b = sum(1 for _, l in self.samples if l == 0)
        m = sum(1 for _, l in self.samples if l == 1)
        print(f"  [{self.split_root.name}] benign={b}, malignant={m}, "
              f"total={len(self.samples)}")

        norm = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2)]
                if augment else [])

        self.tf = transforms.Compose(
            aug + [transforms.Resize((img_size, img_size))] + norm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = self.tf(Image.open(img_path).convert("RGB"))
        return img, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. ViT-B/16 Encoder  (partial unfreeze: blocks 10-11 + norm)
# ─────────────────────────────────────────────────────────────────────

class ViTFrameEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights          = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit              = vit_b_16(weights=weights)
        self.patch_embed = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token
        self.hidden_dim  = vit.hidden_dim   # 768

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        # Partially unfreeze when freeze_backbone=False
        if not freeze_backbone:
            for name, p in self.named_parameters():
                if any(f"layers.{i}" in name for i in (10, 11)):
                    p.requires_grad = True
                if "ln" in name.lower() or "norm" in name.lower():
                    p.requires_grad = True

    def forward(self, x):                          # (B, 3, 224, 224)
        B = x.shape[0]
        x = self.patch_embed(x)                    # (B, 768, 14, 14)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, 196, 768)
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # (B, 197, 768)
        x = self.encoder(x)
        return x[:, 0]                             # (B, 768)


# ─────────────────────────────────────────────────────────────────────
# 3. MLP Classifier
# ─────────────────────────────────────────────────────────────────────

class ImageClassifier(nn.Module):
    def __init__(self,
                 num_classes:    int   = 2,
                 freeze_backbone: bool = False,
                 vit_embed_dim:  int   = 768,
                 hidden_dim:     int   = 256,
                 dropout:        float = 0.3):
        super().__init__()
        self.frame_encoder = ViTFrameEncoder(pretrained=True,
                                             freeze_backbone=freeze_backbone)
        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_embed_dim),
            nn.Linear(vit_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.frame_encoder(x))


# ─────────────────────────────────────────────────────────────────────
# 4. Train / Eval helpers
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, probs, all_labels = [], [], []
    total = 0.0
    for imgs, lbls in tqdm(loader, desc="Eval ", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        total += criterion(logits, lbls).item()
        p = F.softmax(logits, dim=-1)
        preds.extend(p.argmax(1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    auc = roc_auc_score(all_labels, probs) if len(set(all_labels)) > 1 else 0.0
    report = classification_report(
        all_labels, preds,
        target_names=["Benign", "Malignant"],
        output_dict=True, zero_division=0)
    return {"loss": total / len(loader),
            "accuracy": report["accuracy"],
            "auc": auc,
            "probs": probs,
            "labels": all_labels}


def _youden_threshold(labels, probs):
    """Optimal threshold via Youden-J, clamped to [0.2, 0.5]."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[idx], 0.2, 0.5))


# ─────────────────────────────────────────────────────────────────────
# 5. Per-fold test evaluation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_fold(
        fold_idx:   int,
        checkpoint: str,
        test_root:  str,
        val_probs:  list,
        val_labels: list,
        img_size:   int   = 224,
        batch_size: int   = 8,
        num_workers: int  = 4,
        dropout:    float = 0.3,
        freeze_backbone: bool = False,
        results_csv: str  = "",
        device_str: str   = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'─'*50}")
    print(f"  Test evaluation — fold_{fold_idx}")
    print(f"{'─'*50}")

    model = ImageClassifier(freeze_backbone=freeze_backbone,
                            dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    test_ds = ThyroidImageDataset(
        split_root=test_root, img_size=img_size, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    all_preds, all_probs, all_labels, all_names = [], [], [], []

    for batch_idx, (imgs, lbls) in enumerate(tqdm(test_loader, desc="Test")):
        imgs = imgs.to(device)
        p    = F.softmax(model(imgs), dim=-1)
        all_preds.extend(p.argmax(1).cpu().numpy().tolist())
        all_probs.extend(p[:, 1].cpu().numpy().tolist())
        all_labels.extend(lbls.numpy().tolist())
        start = batch_idx * batch_size
        end   = min(start + batch_size, len(test_ds.samples))
        for i in range(start, end):
            all_names.append(test_ds.samples[i][0].name)

    # ── Default threshold (0.5) ───────────────────────────────────────
    preds_05 = [int(p >= 0.5) for p in all_probs]
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm  = confusion_matrix(all_labels, preds_05)
    sens_05  = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0.0
    spec_05  = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0.0
    acc_05   = (cm[0,0]+cm[1,1]) / cm.sum()

    # ── Youden-J threshold (from val set) ────────────────────────────
    opt_thresh = _youden_threshold(val_labels, val_probs)
    preds_opt  = [int(p >= opt_thresh) for p in all_probs]
    cm_opt     = confusion_matrix(all_labels, preds_opt)
    sens_opt   = cm_opt[1,1]/(cm_opt[1,1]+cm_opt[1,0]) if (cm_opt[1,1]+cm_opt[1,0]) > 0 else 0.0
    spec_opt   = cm_opt[0,0]/(cm_opt[0,0]+cm_opt[0,1]) if (cm_opt[0,0]+cm_opt[0,1]) > 0 else 0.0
    acc_opt    = (cm_opt[0,0]+cm_opt[1,1]) / cm_opt.sum()

    print(f"\n  [threshold=0.50]")
    print(classification_report(all_labels, preds_05,
          target_names=["Benign","Malignant"], zero_division=0))
    print(f"  AUC={auc:.4f}  Sens={sens_05:.4f}  Spec={spec_05:.4f}")

    print(f"\n  [threshold={opt_thresh:.3f}  Youden-J from val]")
    print(classification_report(all_labels, preds_opt,
          target_names=["Benign","Malignant"], zero_division=0))
    print(f"  AUC={auc:.4f}  Sens={sens_opt:.4f}  Spec={spec_opt:.4f}")

    # ── CSV ──────────────────────────────────────────────────────────
    if results_csv:
        os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
        id2name = {0: "Benign", 1: "Malignant"}
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_name","gt_label","pred_05","pred_opt",
                               "benign_prob","malignant_prob","correct_05","correct_opt"])
            writer.writeheader()
            for nm, gt, p05, popt, mp in zip(
                    all_names, all_labels, preds_05, preds_opt, all_probs):
                writer.writerow({
                    "image_name":     nm,
                    "gt_label":       id2name[int(gt)],
                    "pred_05":        id2name[int(p05)],
                    "pred_opt":       id2name[int(popt)],
                    "benign_prob":    round(1.0 - float(mp), 4),
                    "malignant_prob": round(float(mp), 4),
                    "correct_05":     "yes" if int(gt)==int(p05) else "no",
                    "correct_opt":    "yes" if int(gt)==int(popt) else "no",
                })
        print(f"\n  Per-image CSV → {results_csv}")

    return {
        "auc": auc, "opt_thresh": opt_thresh,
        "sens_05": sens_05,  "spec_05": spec_05,  "acc_05": acc_05,
        "sens_opt": sens_opt, "spec_opt": spec_opt, "acc_opt": acc_opt,
    }


# ─────────────────────────────────────────────────────────────────────
# 6. Per-fold training
# ─────────────────────────────────────────────────────────────────────

def train_fold(
        fold_idx:        int,
        fold_root:       str,
        save_best_path:  str,
        save_last_path:  str,
        batch_size:      int   = 16,
        img_size:        int   = 224,
        epochs:          int   = 50,
        lr:              float = 3e-4,
        weight_decay:    float = 1e-4,
        freeze_backbone: bool  = False,
        dropout:         float = 0.3,
        hidden_dim:      int   = 256,
        warmup_epochs:   int   = 5,
        early_stop:      int   = 10,
        label_smoothing: float = 0.1,
        num_workers:     int   = 4,
        device_str:      str   = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    fold_root = Path(fold_root)

    diagnose(str(fold_root))

    train_ds = ThyroidImageDataset(
        str(fold_root / "train"), img_size=img_size, augment=True)
    val_ds   = ThyroidImageDataset(
        str(fold_root / "val"),   img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = ImageClassifier(
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    counts   = np.bincount([lbl for _, lbl in train_ds.samples])
    cw       = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs(os.path.dirname(os.path.abspath(save_best_path)), exist_ok=True)

    best_auc, no_improve = 0.0, 0
    best_val_probs, best_val_labels = [], []

    for epoch in range(1, epochs + 1):
        tr_loss  = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_met  = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Fold {fold_idx} | Ep {epoch:03d} | "
              f"LR {lr_now:.2e} | "
              f"TrLoss {tr_loss:.4f} | "
              f"ValLoss {val_met['loss']:.4f} | "
              f"Acc {val_met['accuracy']:.4f} | "
              f"AUC {val_met['auc']:.4f}")

        torch.save(model.state_dict(), save_last_path)

        if val_met["auc"] > best_auc:
            best_auc        = val_met["auc"]
            no_improve      = 0
            best_val_probs  = val_met["probs"]
            best_val_labels = val_met["labels"]
            torch.save(model.state_dict(), save_best_path)
            print(f"    ✓ Best model saved (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            print(f"    No improvement {no_improve}/{early_stop}")

        if no_improve >= early_stop:
            print(f"\n  Early stop at epoch {epoch}.")
            break

    print(f"\n  Fold {fold_idx} training done. Best Val AUC = {best_auc:.4f}")
    return best_val_probs, best_val_labels


# ─────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/best_frame_5fold"
    WEIGHTS_DIR    = "/root/autodl-tmp/suhel/thyroid_nodule/weights_images_5fold"
    RESULTS_DIR    = "/root/autodl-tmp/suhel/thyroid_nodule/results/5fold_images"

    # ── Hyper-parameters ─────────────────────────────────────────────
    BATCH_SIZE       = 16
    IMG_SIZE         = 224
    EPOCHS           = 50
    LR               = 3e-4
    WEIGHT_DECAY     = 1e-4
    FREEZE_BACKBONE  = False   # unfreeze ViT blocks 10-11 + norm
    DROPOUT          = 0.3
    HIDDEN_DIM       = 256
    WARMUP_EPOCHS    = 5
    EARLY_STOP       = 10
    LABEL_SMOOTHING  = 0.1
    NUM_WORKERS      = 4
    NUM_FOLDS        = 5
    # ─────────────────────────────────────────────────────────────────

    data_root = Path(DATA_ROOT)
    fold_dirs = sorted(d for d in data_root.iterdir()
                       if d.is_dir() and d.name.startswith("fold_"))
    if not fold_dirs:
        raise SystemExit(f"No fold_* folders found in {DATA_ROOT}")

    all_metrics = []

    for fold_dir in fold_dirs[:NUM_FOLDS]:
        fold_idx = int(fold_dir.name.split("_")[1])

        save_best = os.path.join(WEIGHTS_DIR,
                                 f"fold_{fold_idx}_ViTB16_MLP_best.pth")
        save_last = os.path.join(WEIGHTS_DIR,
                                 f"fold_{fold_idx}_ViTB16_MLP_last.pth")
        results_csv = os.path.join(RESULTS_DIR,
                                   f"fold_{fold_idx}_test_results.csv")

        print(f"\n{'#'*60}")
        print(f"  FOLD {fold_idx}")
        print(f"{'#'*60}")

        val_probs, val_labels = train_fold(
            fold_idx        = fold_idx,
            fold_root       = str(fold_dir),
            save_best_path  = save_best,
            save_last_path  = save_last,
            batch_size      = BATCH_SIZE,
            img_size        = IMG_SIZE,
            epochs          = EPOCHS,
            lr              = LR,
            weight_decay    = WEIGHT_DECAY,
            freeze_backbone = FREEZE_BACKBONE,
            dropout         = DROPOUT,
            hidden_dim      = HIDDEN_DIM,
            warmup_epochs   = WARMUP_EPOCHS,
            early_stop      = EARLY_STOP,
            label_smoothing = LABEL_SMOOTHING,
            num_workers     = NUM_WORKERS,
        )

        metrics = test_fold(
            fold_idx        = fold_idx,
            checkpoint      = save_best,
            test_root       = str(fold_dir / "test"),
            val_probs       = val_probs,
            val_labels      = val_labels,
            img_size        = IMG_SIZE,
            batch_size      = BATCH_SIZE,
            num_workers     = NUM_WORKERS,
            dropout         = DROPOUT,
            freeze_backbone = FREEZE_BACKBONE,
            results_csv     = results_csv,
        )
        all_metrics.append(metrics)

    # ── Aggregate summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  5-FOLD SUMMARY")
    print(f"{'='*60}")
    header = f"{'Fold':>5}  {'AUC':>6}  {'Thr':>5}  "  \
             f"{'Sens@0.5':>9}  {'Spec@0.5':>9}  {'Acc@0.5':>8}  " \
             f"{'Sens@opt':>9}  {'Spec@opt':>9}  {'Acc@opt':>8}"
    print(header)
    print("─" * len(header))

    for i, m in enumerate(all_metrics):
        print(f"  {i:>3}  "
              f"{m['auc']:>6.4f}  {m['opt_thresh']:>5.3f}  "
              f"{m['sens_05']:>9.4f}  {m['spec_05']:>9.4f}  {m['acc_05']:>8.4f}  "
              f"{m['sens_opt']:>9.4f}  {m['spec_opt']:>9.4f}  {m['acc_opt']:>8.4f}")

    def _ms(key):
        v = [m[key] for m in all_metrics]
        return np.mean(v), np.std(v)

    print("─" * len(header))
    for key, label in [
        ("auc",      "AUC"),
        ("sens_05",  "Sens@0.5"),
        ("spec_05",  "Spec@0.5"),
        ("acc_05",   "Acc@0.5"),
        ("sens_opt", "Sens@opt"),
        ("spec_opt", "Spec@opt"),
        ("acc_opt",  "Acc@opt"),
    ]:
        mu, sd = _ms(key)
        print(f"  {label:<12}  mean={mu:.4f}  std={sd:.4f}")

    print(f"\nAll weights  → {WEIGHTS_DIR}/")
    print(f"All CSVs     → {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────
# 8. Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LOG_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/Logs/images_5fold/pipeline_vitb16_mlp_5fold.txt"
    sys.stdout = Tee(LOG_PATH)
    try:
        main()
    finally:
        sys.stdout.close()
