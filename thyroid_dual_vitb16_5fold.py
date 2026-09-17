"""
Thyroid Nodule — Dual-Branch ViT-B/16 — 5-Fold Cross-Validation
================================================================
Input structure (from organize_5fold.py + preextract_rois_5fold.py):

    DATA_5FOLD/fold_k/{train|val|test}/{benign|malignant|indeterminate}/
        <video.mp4> or <cine_folder>/

    ROIS_5FOLD/fold_k/{train|val|test}/{benign|malignant|indeterminate}/
        <stem>/frame_0000.jpg ... manifest.json

For each fold k:
  - Train on fold_k/train, validate on fold_k/val, test on fold_k/test
  - Save best checkpoint per fold
  - Aggregate metrics across all 5 folds at the end

Architecture:
  Branch A (whole frame): frames → ViT-B/16 → TemporalTransformer → (B,768)
  Branch B (ROI crops)  : crops  → ViT-B/16 → TemporalTransformer → (B,768)
  Fusion                : concat → LayerNorm → MLP → N-class logits
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys as _sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from PIL import Image
import timm
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_5FOLD  = "/root/autodl-tmp/suhel/thyroid_nodule/data_5fold"
ROIS_5FOLD  = "/root/autodl-tmp/suhel/thyroid_nodule/rois_5fold"
WEIGHTS_DIR = "/root/autodl-tmp/suhel/thyroid_nodule/weights_5fold"
RESULTS_DIR = "/root/autodl-tmp/suhel/thyroid_nodule/results_5fold"
LOGS_DIR    = "/root/autodl-tmp/suhel/thyroid_nodule/logs_5fold"

VITB16_FINETUNED_CKPT = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_finetuned_last.pth"

# 2-class: benign vs malignant (indeterminate excluded from training)
# Set to None to include all found classes
LABEL_MAP = {"benign": 0, "malignant": 1}

BATCH_SIZE          = 2
MAX_FRAMES          = 32
IMG_SIZE            = 224
ROI_SIZE            = 224
EPOCHS              = 50
LR                  = 1e-4
BACKBONE_LR         = 1e-5
WEIGHT_DECAY        = 1e-4
DROPOUT             = 0.5
WARMUP_EPOCHS       = 5
EARLY_STOP_PATIENCE = 10
NUM_WORKERS         = 4
N_FOLDS             = 5
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Logger
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
# Dataset  (reads directly from fold_k/split/cls/ structure)
# ─────────────────────────────────────────────────────────────────────

class ThyroidDualDataset(Dataset):
    def __init__(self,
                 video_root: str,    # e.g. data_5fold/fold_0/train
                 roi_root:   str,    # e.g. rois_5fold/fold_0/train
                 label_map:  dict,   # {class_name: int}
                 max_frames: int  = 32,
                 img_size:   int  = 224,
                 roi_size:   int  = 224,
                 augment:    bool = False):

        self.video_root = Path(video_root)
        self.roi_root   = Path(roi_root)
        self.max_frames = max_frames
        self.roi_size   = roi_size
        self.label_map  = label_map
        self.class_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

        self.samples: List[Tuple[Path, Path, int, str]] = []

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        for class_name, label in label_map.items():
            cls_video_dir = self.video_root / class_name
            cls_roi_dir   = self.roi_root   / class_name
            if not cls_video_dir.exists():
                print(f"  [WARNING] Missing: {cls_video_dir}")
                continue

            # video files
            for vp in sorted(p for p in cls_video_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                roi_dir = cls_roi_dir / vp.stem
                if not (roi_dir / "manifest.json").exists():
                    print(f"  [WARNING] No ROI manifest for {vp.name} — skipping.")
                    continue
                self.samples.append((vp, roi_dir, label, "video"))

            # cine frame folders
            for cp in sorted(p for p in cls_video_dir.iterdir() if p.is_dir()):
                if not any(f.suffix.lower() in IMAGE_EXTS for f in cp.iterdir()):
                    continue
                roi_dir = cls_roi_dir / cp.name
                if not (roi_dir / "manifest.json").exists():
                    print(f"  [WARNING] No ROI manifest for {cp.name} — skipping.")
                    continue
                self.samples.append((cp, roi_dir, label, "cine"))

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.video_root}")

        counts = {cn: sum(1 for _, _, l, _ in self.samples
                          if l == label_map[cn])
                  for cn in self.class_names}
        print(f"  [{self.video_root.parent.name}/{self.video_root.name}]  "
              + "  ".join(f"{cn}={counts[cn]}" for cn in self.class_names)
              + f"  total={len(self.samples)}")

        norm = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2)]
                if augment else [])

        self.whole_tf = transforms.Compose(aug + [transforms.Resize((img_size, img_size))] + norm)
        self.roi_tf   = transforms.Compose(aug + [transforms.Resize((roi_size, roi_size))] + norm)

    def __len__(self):
        return len(self.samples)

    def _load_whole_video(self, vp: Path) -> torch.Tensor:
        cap     = cv2.VideoCapture(str(vp), cv2.CAP_FFMPEG)
        total   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, self.max_frames, dtype=int)
        frames, last = [], torch.zeros(3, 224, 224)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, bgr = cap.read()
            if ret:
                last = self.whole_tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            frames.append(last)
        cap.release()
        return torch.stack(frames)

    def _load_whole_cine(self, cp: Path) -> torch.Tensor:
        files   = sorted(p for p in cp.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        total   = max(len(files), 1)
        indices = np.linspace(0, total - 1, self.max_frames, dtype=int)
        frames, last = [], torch.zeros(3, 224, 224)
        for idx in indices:
            bgr = cv2.imread(str(files[int(idx)]))
            if bgr is not None:
                last = self.whole_tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            frames.append(last)
        return torch.stack(frames)

    def _load_roi_frames(self, roi_dir: Path) -> torch.Tensor:
        with open(roi_dir / "manifest.json") as f:
            fnames = json.load(f)["frames"]
        if not fnames:
            return torch.zeros(self.max_frames, 3, self.roi_size, self.roi_size)
        indices = np.linspace(0, len(fnames) - 1, self.max_frames, dtype=int)
        frames  = []
        for si in indices:
            p = roi_dir / fnames[int(si)]
            bgr = cv2.imread(str(p)) if p.exists() else None
            if bgr is not None:
                frames.append(self.roi_tf(
                    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
            else:
                frames.append(torch.zeros(3, self.roi_size, self.roi_size))
        return torch.stack(frames)

    def __getitem__(self, idx):
        src, roi_dir, label, kind = self.samples[idx]
        whole = self._load_whole_video(src) if kind == "video" else self._load_whole_cine(src)
        roi   = self._load_roi_frames(roi_dir)
        return whole, roi, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class ViTFrameEncoder(nn.Module):
    def __init__(self, finetuned_ckpt: str = VITB16_FINETUNED_CKPT,
                 freeze: bool = False):
        super().__init__()
        ckpt_exists = bool(finetuned_ckpt and Path(finetuned_ckpt).exists())
        self.backbone = timm.create_model("vit_base_patch16_224",
                                          pretrained=not ckpt_exists, num_classes=0)
        self.hidden_dim = self.backbone.embed_dim

        if ckpt_exists:
            sd = torch.load(finetuned_ckpt, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[ViTFrameEncoder] Loaded {finetuned_ckpt} "
                  f"| missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[ViTFrameEncoder] Using ImageNet pretrained weights.")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            TRAINABLE = {10, 11}
            for name, p in self.backbone.named_parameters():
                p.requires_grad = (any(f"blocks.{i}." in name for i in TRAINABLE)
                                   or name.startswith("norm."))
            n_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"[ViTFrameEncoder] Partial freeze — trainable: {n_train:,}")

    def forward(self, x):
        return self.backbone(x)


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, num_layers=1,
                 ff_dim=1024, dropout=0.1, max_frames=32):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames + 1, embed_dim) * 0.02)
        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embedding[:, :T + 1]
        return self.norm(self.transformer(x)[:, 0])


class DualBranchClassifier(nn.Module):
    def __init__(self, num_classes=2, max_frames=32,
                 freeze_backbone=True, dropout=0.3):
        super().__init__()
        self.frame_encoder   = ViTFrameEncoder(freeze=freeze_backbone)
        D = self.frame_encoder.hidden_dim
        self.whole_temporal  = TemporalTransformer(embed_dim=D, max_frames=max_frames, dropout=dropout)
        self.roi_temporal    = TemporalTransformer(embed_dim=D, max_frames=max_frames, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _encode(self, videos, temporal):
        B, T, C, H, W = videos.shape
        feats = self.frame_encoder(rearrange(videos, 'b t c h w -> (b t) c h w'))
        return temporal(rearrange(feats, '(b t) d -> b t d', b=B, t=T))

    def forward(self, whole, roi):
        return self.fusion(torch.cat([self._encode(whole, self.whole_temporal),
                                      self._encode(roi,   self.roi_temporal)], dim=-1))


# ─────────────────────────────────────────────────────────────────────
# Train / Eval helpers
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    for whole, roi, labels in tqdm(loader, desc="Train", leave=False):
        whole, roi, labels = whole.to(device), roi.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(whole, roi), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss = criterion(model(whole, roi), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds, scores, labels_all = [], [], []
    total = 0.0
    for whole, roi, lbls in tqdm(loader, desc="Eval ", leave=False):
        whole, roi, lbls = whole.to(device), roi.to(device), lbls.to(device)
        logits = model(whole, roi)
        total += criterion(logits, lbls).item()
        p = F.softmax(logits, dim=-1)
        preds.extend(p.argmax(1).cpu().numpy())
        scores.extend(p[:, 1].cpu().numpy())   # P(malignant)
        labels_all.extend(lbls.cpu().numpy())
    auc = roc_auc_score(labels_all, scores) if len(set(labels_all)) > 1 else 0.0
    acc = np.mean(np.array(preds) == np.array(labels_all))
    return {"loss": total / len(loader), "accuracy": acc, "auc": auc,
            "preds": preds, "scores": scores, "labels": labels_all}


# ─────────────────────────────────────────────────────────────────────
# Per-fold training
# ─────────────────────────────────────────────────────────────────────

def train_fold(fold_k: int, device: torch.device) -> dict:
    fold_dir = Path(DATA_5FOLD) / f"fold_{fold_k}"
    roi_dir  = Path(ROIS_5FOLD) / f"fold_{fold_k}"

    print(f"\n{'=' * 65}")
    print(f"  FOLD {fold_k}  |  video: {fold_dir}  |  roi: {roi_dir}")
    print(f"{'=' * 65}")

    num_classes = len(LABEL_MAP)

    train_ds = ThyroidDualDataset(str(fold_dir / "train"), str(roi_dir / "train"),
                                  LABEL_MAP, MAX_FRAMES, IMG_SIZE, ROI_SIZE, augment=True)
    val_ds   = ThyroidDualDataset(str(fold_dir / "val"),   str(roi_dir / "val"),
                                  LABEL_MAP, MAX_FRAMES, IMG_SIZE, ROI_SIZE, augment=False)
    test_ds  = ThyroidDualDataset(str(fold_dir / "test"),  str(roi_dir / "test"),
                                  LABEL_MAP, MAX_FRAMES, IMG_SIZE, ROI_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = DualBranchClassifier(num_classes=num_classes, max_frames=MAX_FRAMES,
                                 freeze_backbone=True, dropout=DROPOUT).to(device)

    counts  = np.bincount([lbl for _, _, lbl, _ in train_ds.samples], minlength=num_classes)
    weights = torch.tensor(1.0 / np.where(counts > 0, counts, 1).astype(float),
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    backbone_params = [p for p in model.frame_encoder.backbone.parameters() if p.requires_grad]
    head_params     = (list(model.whole_temporal.parameters()) +
                       list(model.roi_temporal.parameters()) +
                       list(model.fusion.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": LR,          "weight_decay": WEIGHT_DECAY},
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
    ])

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_path = os.path.join(WEIGHTS_DIR, f"dual_vitb16_fold{fold_k}_best.pth")
    last_path = os.path.join(WEIGHTS_DIR, f"dual_vitb16_fold{fold_k}_last.pth")

    best_auc, no_improve = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:03d} | lr={lr_now:.1e} | "
              f"TrLoss={train_loss:.4f} | "
              f"ValLoss={val_metrics['loss']:.4f} | "
              f"Acc={val_metrics['accuracy']:.4f} | "
              f"AUC={val_metrics['auc']:.4f}")

        torch.save(model.state_dict(), last_path)

        if val_metrics["auc"] > best_auc:
            best_auc, no_improve = val_metrics["auc"], 0
            torch.save(model.state_dict(), best_path)
            print(f"    → Best model saved (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # ── Test with best checkpoint ────────────────────────────────────
    print(f"\n  Testing fold {fold_k} with best checkpoint...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    class_names = [k for k, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
    print(f"\n  Fold {fold_k} Test Results:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"],
                                 target_names=class_names, zero_division=0))
    print(f"  Test AUC: {test_metrics['auc']:.4f}")

    cm          = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if cm.shape[0] > 1 and (cm[1,1]+cm[1,0]) > 0 else 0.0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0.0

    # Save per-fold CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"fold{fold_k}_test_results.csv")
    sample_names = [s[0].name for s in test_ds.samples]
    id2name = {v: k for k, v in LABEL_MAP.items()}
    rows = [{"sample": sn,
             "gt": id2name[int(gt)],
             "pred": id2name[int(pr)],
             "mal_score": round(float(sc), 4),
             "correct": "yes" if int(gt) == int(pr) else "no"}
            for sn, gt, pr, sc in zip(sample_names,
                                      test_metrics["labels"],
                                      test_metrics["preds"],
                                      test_metrics["scores"])]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"  Results saved → {csv_path}")

    return {
        "fold":        fold_k,
        "val_auc":     best_auc,
        "test_auc":    test_metrics["auc"],
        "test_acc":    test_metrics["accuracy"],
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ─────────────────────────────────────────────────────────────────────
# 5-fold runner
# ─────────────────────────────────────────────────────────────────────

def run_kfold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data root : {DATA_5FOLD}")
    print(f"ROI root  : {ROIS_5FOLD}")
    print(f"Classes   : {LABEL_MAP}\n")

    all_results = []
    for k in range(N_FOLDS):
        result = train_fold(k, device)
        all_results.append(result)

    # ── Aggregate summary ────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  5-FOLD SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Fold':>5}  {'ValAUC':>8}  {'TestAUC':>8}  "
          f"{'TestAcc':>8}  {'Sens':>8}  {'Spec':>8}")
    for r in all_results:
        print(f"  {r['fold']:>5}  {r['val_auc']:>8.4f}  {r['test_auc']:>8.4f}  "
              f"  {r['test_acc']:>8.4f}  {r['sensitivity']:>8.4f}  {r['specificity']:>8.4f}")

    metrics = ["val_auc", "test_auc", "test_acc", "sensitivity", "specificity"]
    print(f"  {'Mean':>5}  " + "  ".join(
        f"{np.mean([r[m] for r in all_results]):>8.4f}" for m in metrics))
    print(f"  {'Std':>5}  " + "  ".join(
        f"{np.std([r[m] for r in all_results]):>8.4f}" for m in metrics))

    # Save summary CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_csv = os.path.join(RESULTS_DIR, "5fold_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + metrics)
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerow({"fold": "mean", **{m: round(np.mean([r[m] for r in all_results]), 4) for m in metrics}})
        writer.writerow({"fold": "std",  **{m: round(np.std( [r[m] for r in all_results]), 4) for m in metrics}})
    print(f"\nSummary saved → {summary_csv}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "dual_vitb16_5fold.txt")
    sys.stdout = Tee(log_path)
    try:
        run_kfold()
    finally:
        sys.stdout.close()
