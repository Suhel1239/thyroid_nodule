"""
Thyroid Nodule Video Classification — Dual-Branch (ViT-B/16 LoRA finetuned)
=============================================================================
Uses ViT-B/16 finetuned with LoRA on thyroid images as the shared frame encoder.
The LoRA weights are already merged into the backbone before saving, so this
script loads a plain ViT-B/16 state dict — no peft dependency needed here.

Pipeline:
  Branch A (whole frame): video frames → ViT-B/16 → TemporalTransformer → (B, 768)
  Branch B (ROI crops)  : pre-saved crops → ViT-B/16 → TemporalTransformer → (B, 768)
  Fusion                : concat → LayerNorm → MLP → Benign/Malignant

Run finetune_vitb16_lora.py first to produce the backbone checkpoint.
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
from sklearn.metrics import classification_report, roc_auc_score
from PIL import Image
import timm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

EMBED_DIM = 768

# Backbone checkpoint produced by finetune_vitb16_lora.py (LoRA merged, plain state dict)
VITB16_FINETUNED_CKPT = os.environ.get(
    "VITB16_FINETUNED_CKPT",
    "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_lora_finetuned_best.pth",
)


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
# 0. Diagnostics
# ─────────────────────────────────────────────────────────────────────

def diagnose(data_root: str, roi_root: str, max_frames: int = 32):
    data_root = Path(data_root)
    print("\n" + "=" * 60)
    for split in ("train_pure", "val_pure", "test_pure"):
        for cls in ("benign", "malignant"):
            vid_dir = data_root / split / cls
            split_name = split.replace("_pure", "")
            roi_dir = (Path(roi_root)
                       / f"rois_{max_frames}_rfdetr_topn_withareafiltering"
                       / split_name / cls)
            if not vid_dir.exists():
                continue
            videos = [v for v in vid_dir.iterdir()
                      if v.is_file() and v.suffix.lower() in VIDEO_EXTS]
            rois   = list(roi_dir.glob("*/manifest.json")) if roi_dir.exists() else []
            total  = len(videos)
            status = ("OK" if len(rois) == total
                      else f"MISMATCH (videos={total}, roi_dirs={len(rois)})")
            print(f"  [{split}/{cls}]  videos={total}  roi_manifests={len(rois)}  {status}")
        print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────

class ThyroidDualDataset(Dataset):
    LABEL_MAP = {"benign": 0, "malignant": 1}

    def __init__(self,
                 video_root: str,
                 roi_root:   str,
                 max_frames: int  = 32,
                 img_size:   int  = 224,
                 roi_size:   int  = 224,
                 augment:    bool = False):

        self.video_root = Path(video_root)
        split_name      = Path(video_root).name.replace("_pure", "")
        self.roi_root   = (Path(roi_root)
                           / f"rois_{max_frames}_rfdetr_topn_withareafiltering"
                           / split_name)
        self.max_frames = max_frames
        self.roi_size   = roi_size
        self.samples: List[Tuple[Path, Path, int]] = []

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        existing = {d.name.lower(): d
                    for d in self.video_root.iterdir() if d.is_dir()}

        for class_name, label in self.LABEL_MAP.items():
            cls_vid_dir = existing.get(class_name.lower())
            if cls_vid_dir is None:
                print(f"  [WARNING] Missing class folder: {self.video_root / class_name}")
                continue
            cls_roi_dir = self.roi_root / class_name
            for vp in sorted(p for p in cls_vid_dir.iterdir()
                             if p.suffix.lower() in VIDEO_EXTS):
                roi_dir = cls_roi_dir / vp.stem
                if not (roi_dir / "manifest.json").exists():
                    print(f"  [WARNING] No ROI manifest for {vp.name} — skipping.")
                    continue
                self.samples.append((vp, roi_dir, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under {self.video_root}.\n"
                f"Run preextract_rois.py to generate ROI crops first.")

        b = sum(1 for _, _, l in self.samples if l == 0)
        m = sum(1 for _, _, l in self.samples if l == 1)
        print(f"  [{self.video_root.name}] benign={b}, malignant={m}, total={len(self.samples)}")

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

    def _load_whole_frames(self, video_path: Path) -> torch.Tensor:
        cap   = cv2.VideoCapture(str(video_path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
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

    def _load_roi_frames(self, roi_dir: Path) -> torch.Tensor:
        with open(roi_dir / "manifest.json") as f:
            manifest = json.load(f)
        fnames  = manifest["frames"]
        n_saved = len(fnames)
        if n_saved == 0:
            return torch.zeros(self.max_frames, 3, self.roi_size, self.roi_size)
        indices = np.linspace(0, n_saved - 1, self.max_frames, dtype=int)
        frames  = []
        for si in indices:
            img_path = roi_dir / fnames[int(si)]
            if img_path.exists():
                bgr = cv2.imread(str(img_path))
                if bgr is not None:
                    frames.append(self.roi_tf(
                        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
                    continue
            frames.append(torch.zeros(3, self.roi_size, self.roi_size))
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path, roi_dir, label = self.samples[idx]
        whole = self._load_whole_frames(video_path)
        roi   = self._load_roi_frames(roi_dir)
        return whole, roi, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. ViT-B/16 Frame Encoder (loads LoRA-merged backbone)
# ─────────────────────────────────────────────────────────────────────

class ViTFrameEncoder(nn.Module):
    """
    Shared frame encoder. Loads the LoRA-merged ViT-B/16 backbone.
    No peft dependency — checkpoint is a plain state dict.

    Input : (B, 3, 224, 224)
    Output: (B, 768)
    """

    def __init__(self, freeze: bool = True,
                 finetuned_ckpt: str = VITB16_FINETUNED_CKPT):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0)
        self.hidden_dim = self.backbone.embed_dim  # 768

        if finetuned_ckpt and Path(finetuned_ckpt).exists():
            sd = torch.load(finetuned_ckpt, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            n_total  = len(self.backbone.state_dict())
            n_loaded = n_total - len(missing)
            print(f"[ViTFrameEncoder] Loaded LoRA-merged weights: {finetuned_ckpt}")
            print(f"  {n_loaded}/{n_total} tensors | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"[ViTFrameEncoder] WARNING: checkpoint not found at {finetuned_ckpt}")
            print(f"  Run finetune_vitb16_lora.py first. Falling back to random init.")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[ViTFrameEncoder] Backbone frozen.")
        else:
            print(f"[ViTFrameEncoder] Backbone trainable.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 768)


# ─────────────────────────────────────────────────────────────────────
# 3. Temporal Transformer
# ─────────────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8,
                 num_layers: int = 2, ff_dim: int = 1024,
                 dropout: float = 0.1, max_frames: int = 32):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_frames + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x):           # (B, T, D)
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embedding[:, :T + 1]
        x   = self.transformer(x)
        return self.norm(x[:, 0])   # (B, D)


# ─────────────────────────────────────────────────────────────────────
# 4. Dual-Branch Classifier
# ─────────────────────────────────────────────────────────────────────

class DualBranchThyroidClassifier(nn.Module):
    """
    Branch A : whole video frames → ViT-B/16 (LoRA-finetuned) → TemporalTransformer → (B, D)
    Branch B : ROI crops          → ViT-B/16 (LoRA-finetuned) → TemporalTransformer → (B, D)
    Fusion   : concat(A,B) → LayerNorm → Linear(512) → GELU → Dropout → Linear(2)
    """
    def __init__(self,
                 num_classes:     int   = 2,
                 max_frames:      int   = 32,
                 freeze_backbone: bool  = True,
                 temporal_heads:  int   = 8,
                 temporal_layers: int   = 2,
                 dropout:         float = 0.3):
        super().__init__()

        self.frame_encoder = ViTFrameEncoder(freeze=freeze_backbone)
        embed_dim = self.frame_encoder.hidden_dim  # 768

        self.whole_temporal = TemporalTransformer(
            embed_dim=embed_dim, num_heads=temporal_heads,
            num_layers=temporal_layers, max_frames=max_frames, dropout=dropout)

        self.roi_temporal = TemporalTransformer(
            embed_dim=embed_dim, num_heads=temporal_heads,
            num_layers=temporal_layers, max_frames=max_frames, dropout=dropout)

        fused_dim = embed_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def _encode(self, videos: torch.Tensor,
                temporal: TemporalTransformer) -> torch.Tensor:
        B, T, C, H, W = videos.shape
        frames = rearrange(videos, 'b t c h w -> (b t) c h w')
        feats  = self.frame_encoder(frames)                     # (B*T, D)
        feats  = rearrange(feats, '(b t) d -> b t d', b=B, t=T)
        return temporal(feats)                                  # (B, D)

    def forward(self, whole: torch.Tensor,
                      roi:   torch.Tensor) -> torch.Tensor:
        whole_feat = self._encode(whole, self.whole_temporal)
        roi_feat   = self._encode(roi,   self.roi_temporal)
        fused      = torch.cat([whole_feat, roi_feat], dim=-1)
        return self.fusion(fused)


# ─────────────────────────────────────────────────────────────────────
# 5. Collate
# ─────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    whole  = torch.stack([b[0] for b in batch])
    roi    = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return whole, roi, labels


# ─────────────────────────────────────────────────────────────────────
# 6. Train / Eval
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
            scaler.step(optimizer)
            scaler.update()
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
    preds, probs, all_labels = [], [], []
    total = 0.0
    for whole, roi, lbls in tqdm(loader, desc="Eval ", leave=False):
        whole, roi, lbls = whole.to(device), roi.to(device), lbls.to(device)
        logits = model(whole, roi)
        total += criterion(logits, lbls).item()
        p = F.softmax(logits, dim=-1)
        preds.extend(p.argmax(1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    auc    = roc_auc_score(all_labels, probs) if len(set(all_labels)) > 1 else 0.0
    report = classification_report(all_labels, preds,
                 target_names=["Benign", "Malignant"],
                 output_dict=True, zero_division=0)
    return {"loss": total / len(loader), "accuracy": report["accuracy"], "auc": auc}


from sklearn.metrics import confusion_matrix
import csv

@torch.no_grad()
def evaluate_test_set(
        checkpoint: str,
        test_video_root: str,
        test_roi_root: str,
        max_frames: int = 32,
        img_size: int = 224,
        roi_size: int = 224,
        batch_size: int = 1,
        num_workers: int = 4,
        dropout: float = 0.3,
        results_csv: str = "/root/autodl-tmp/suhel/thyroid_nodule/results/test_results_vitb16_lora_ROI_best.csv",
        device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  Test-set evaluation")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Device     : {device}")
    print(f"{'=' * 60}\n")

    model = DualBranchThyroidClassifier(
        max_frames=max_frames, freeze_backbone=True, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print("Model loaded.\n")

    test_ds = ThyroidDualDataset(
        video_root=test_video_root, roi_root=test_roi_root,
        max_frames=max_frames, img_size=img_size, roi_size=roi_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_fn)

    all_preds, all_probs, all_labels = [], [], []
    all_video_names = [s[0].name for s in test_ds.samples]

    for whole, roi, lbls in tqdm(test_loader, desc="Testing"):
        whole, roi, lbls = whole.to(device), roi.to(device), lbls.to(device)
        p = F.softmax(model(whole, roi), dim=-1)
        all_preds.extend(p.argmax(1).cpu().numpy().tolist())
        all_probs.extend(p[:, 1].cpu().numpy().tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())

    print("\n" + "─" * 50)
    print("Classification Report (threshold 0.5):")
    print(classification_report(all_labels, all_preds,
          target_names=["Benign", "Malignant"], zero_division=0))

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    print(f"AUC-ROC : {auc:.4f}")

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    best_thr = float(thresholds[np.argmax(tpr - fpr)])
    print(f"Optimal threshold (Youden J): {best_thr:.4f}")
    preds_tuned = (np.array(all_probs) >= best_thr).astype(int).tolist()
    print("\nWith tuned threshold:")
    print(classification_report(all_labels, preds_tuned,
          target_names=["Benign", "Malignant"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0.0
    specificity = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0.0
    print(f"Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}")
    print("─" * 50)

    id2name = {0: "Benign", 1: "Malignant"}
    rows = [{"video_name": vn, "gt_label": id2name[int(gt)],
             "pred_label": id2name[int(pr)],
             "benign_prob": round(1.0 - float(mp), 4),
             "malignant_prob": round(float(mp), 4),
             "correct": "yes" if int(gt) == int(pr) else "no"}
            for vn, gt, pr, mp in zip(
                all_video_names, all_labels, all_preds, all_probs)]

    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {results_csv}")
    return {"auc": auc, "sensitivity": sensitivity, "specificity": specificity}


# ─────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    ROI_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    SAVE_BEST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_vitb16_lora_best_32_b2_rfdetr.pth"
    SAVE_LAST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_vitb16_lora_last_32_b2_rfdetr.pth"

    BATCH_SIZE          = 2
    MAX_FRAMES          = 32
    IMG_SIZE            = 224
    ROI_SIZE            = 224
    EPOCHS              = 50
    LR                  = 1e-4
    BACKBONE_LR         = 5e-6
    WEIGHT_DECAY        = 1e-4
    DROPOUT             = 0.3
    WARMUP_EPOCHS       = 3
    EARLY_STOP_PATIENCE = 10
    NUM_WORKERS         = 4

    diagnose(DATA_ROOT, ROI_ROOT, MAX_FRAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_ds = ThyroidDualDataset(
        video_root=os.path.join(DATA_ROOT, "train_pure"), roi_root=ROI_ROOT,
        max_frames=MAX_FRAMES, img_size=IMG_SIZE, roi_size=ROI_SIZE, augment=True)
    val_ds = ThyroidDualDataset(
        video_root=os.path.join(DATA_ROOT, "val_pure"), roi_root=ROI_ROOT,
        max_frames=MAX_FRAMES, img_size=IMG_SIZE, roi_size=ROI_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = DualBranchThyroidClassifier(
        max_frames=MAX_FRAMES, freeze_backbone=True, dropout=DROPOUT).to(device)

    counts    = np.bincount([lbl for _, _, lbl in train_ds.samples])
    weights   = torch.tensor(1.0 / counts.astype(float), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Unfreeze last 2 backbone blocks + norm for further adaptation on video data
    for blk in model.frame_encoder.backbone.blocks[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.frame_encoder.backbone.norm.parameters():
        p.requires_grad = True
    for name, p in model.named_parameters():
        if any(k in name for k in ("temporal", "fusion", "classifier")):
            p.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "frame_encoder.backbone" in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and "frame_encoder.backbone" not in n]

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
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_params:,}  (head @ lr={LR}, backbone_last2 @ lr={BACKBONE_LR})\n")

    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)

    best_auc, no_improve = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer,
                                      criterion, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | LR: {lr_now:.2e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f}")

        torch.save(model.state_dict(), SAVE_LAST_PATH)

        if val_metrics["auc"] > best_auc:
            best_auc, no_improve = val_metrics["auc"], 0
            torch.save(model.state_dict(), SAVE_BEST_PATH)
            print(f"  Saved best model (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{EARLY_STOP_PATIENCE} epoch(s)")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    print(f"Best  → {SAVE_BEST_PATH}")
    print(f"Last  → {SAVE_LAST_PATH}")


# ─────────────────────────────────────────────────────────────────────
# 8. Test-only
# ─────────────────────────────────────────────────────────────────────

def run_test_only():
    evaluate_test_set(
        checkpoint      = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_vitb16_lora_best_32_b2_rfdetr.pth",
        test_video_root = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all/test_pure",
        test_roi_root   = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        max_frames=32, img_size=224, roi_size=224, batch_size=2, num_workers=4, dropout=0.3,
        results_csv     = "/root/autodl-tmp/suhel/thyroid_nodule/results/test_results_vitb16_lora_ROI_best.csv",
    )


if __name__ == "__main__":
    log_path = ("/root/autodl-tmp/suhel/thyroid_nodule/Logs/with_areafiltering/"
                "dual_vitb16_lora_32frames_rfdetr.txt")
    sys.stdout = Tee(log_path)
    try:
        main()
        # run_test_only()
    finally:
        sys.stdout.close()
