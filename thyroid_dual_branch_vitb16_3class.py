"""
Thyroid Nodule Video Classification — Dual-Branch (ViT-B/16) — 3-Class
=======================================================================
Three output classes: Benign (0), Intermediate (1), Malignant (2).

Pipeline:
  Branch A (whole frame): video frames → ViT-B/16 → TemporalTransformer → (B, 768)
  Branch B (ROI crops)  : pre-saved crops → ViT-B/16 → TemporalTransformer → (B, 768)
  Fusion                : concat → LayerNorm → MLP → 3-class logits

Score thresholding (inference):
  malignant_prob >= mal_thr  → Malignant   (takes priority)
  benign_prob    >= ben_thr  → Benign
  else                       → Intermediate

Malignancy score (continuous 0→1 risk):
  score = 0.0 * P(Benign) + 0.5 * P(Intermediate) + 1.0 * P(Malignant)
  Risk bands: Low (<0.35), Medium (0.35–0.65), High (≥0.65)

Supports both video files (.mp4, .avi, …) and cine frame folders (directories
of images). ROI folder name suffix: _3class.
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
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from PIL import Image
import timm
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

VITB16_FINETUNED_CKPT = os.environ.get(
    "VITB16_FINETUNED_CKPT",
    "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_mae_pretrained.pth",
)

SCORE_WEIGHTS = np.array([0.0, 0.5, 1.0])  # Benign, Intermediate, Malignant


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
    roi_folder = f"rois_{max_frames}_rfdetr_topn_withareafiltering_withcine_3class"
    print("\n" + "=" * 60)
    for split in ("train_pure", "val_pure", "test_pure"):
        for cls in ("benign", "intermediate", "malignant"):
            cls_dir    = data_root / split / cls
            split_name = split.replace("_pure", "")
            roi_dir    = Path(roi_root) / roi_folder / split_name / cls
            if not cls_dir.exists():
                continue
            videos = [v for v in cls_dir.iterdir()
                      if v.is_file() and v.suffix.lower() in VIDEO_EXTS]
            cines  = [d for d in cls_dir.iterdir()
                      if d.is_dir() and any(
                          f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())]
            total = len(videos) + len(cines)
            rois  = list(roi_dir.glob("*/manifest.json")) if roi_dir.exists() else []
            status = ("OK" if len(rois) == total
                      else f"MISMATCH (sources={total}, roi_dirs={len(rois)})")
            print(f"  [{split}/{cls}]  videos={len(videos)}  cines={len(cines)}"
                  f"  roi_manifests={len(rois)}  {status}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────

class ThyroidDualDataset(Dataset):
    LABEL_MAP  = {"benign": 0, "intermediate": 1, "malignant": 2}
    CLASS_NAMES = ["Benign", "Intermediate", "Malignant"]

    def __init__(self,
                 video_root: str,
                 roi_root:   str,
                 max_frames: int  = 32,
                 img_size:   int  = 224,
                 roi_size:   int  = 224,
                 augment:    bool = False):

        self.video_root = Path(video_root)
        split_name      = Path(video_root).name.replace("_pure", "")
        roi_folder      = f"rois_{max_frames}_rfdetr_topn_withareafiltering_withcine_3class"
        self.roi_root   = Path(roi_root) / roi_folder / split_name
        self.max_frames = max_frames
        self.roi_size   = roi_size
        # samples: (source_path, roi_dir, label, kind)  kind="video"|"cine"
        self.samples: List[Tuple[Path, Path, int, str]] = []

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        existing = {d.name.lower(): d
                    for d in self.video_root.iterdir() if d.is_dir()}

        for class_name, label in self.LABEL_MAP.items():
            cls_dir = existing.get(class_name.lower())
            if cls_dir is None:
                print(f"  [WARNING] Missing class folder: {self.video_root / class_name}")
                continue
            cls_roi_dir = self.roi_root / class_name

            # video files
            for vp in sorted(p for p in cls_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                roi_dir = cls_roi_dir / vp.stem
                if not (roi_dir / "manifest.json").exists():
                    print(f"  [WARNING] No ROI manifest for video {vp.name} — skipping.")
                    continue
                self.samples.append((vp, roi_dir, label, "video"))

            # cine frame folders
            for cp in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
                if not any(f.suffix.lower() in IMAGE_EXTS for f in cp.iterdir()):
                    continue
                roi_dir = cls_roi_dir / cp.name
                if not (roi_dir / "manifest.json").exists():
                    print(f"  [WARNING] No ROI manifest for cine {cp.name} — skipping.")
                    continue
                self.samples.append((cp, roi_dir, label, "cine"))

        if not self.samples:
            raise RuntimeError(
                f"No samples found under {self.video_root}. "
                f"Run preextract_rois.py to generate ROI crops first.")

        counts = {cn: sum(1 for _, _, l, _ in self.samples if l == i)
                  for i, cn in enumerate(self.CLASS_NAMES)}
        print(f"  [{self.video_root.name}] "
              + "  ".join(f"{cn}={counts[cn]}" for cn in self.CLASS_NAMES)
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

    def _load_whole_frames_video(self, video_path: Path) -> torch.Tensor:
        cap     = cv2.VideoCapture(str(video_path))
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

    def _load_whole_frames_cine(self, cine_dir: Path) -> torch.Tensor:
        image_files = sorted(
            p for p in cine_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        total   = max(len(image_files), 1)
        indices = np.linspace(0, total - 1, self.max_frames, dtype=int)
        frames, last = [], torch.zeros(3, 224, 224)
        for idx in indices:
            bgr = cv2.imread(str(image_files[int(idx)]))
            if bgr is not None:
                last = self.whole_tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            frames.append(last)
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
        src_path, roi_dir, label, kind = self.samples[idx]
        if kind == "video":
            whole = self._load_whole_frames_video(src_path)
        else:
            whole = self._load_whole_frames_cine(src_path)
        roi = self._load_roi_frames(roi_dir)
        return whole, roi, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. ViT-B/16 Frame Encoder
# ─────────────────────────────────────────────────────────────────────

class ViTFrameEncoder(nn.Module):
    def __init__(self, finetuned_ckpt: str = VITB16_FINETUNED_CKPT,
                 freeze: bool = False):
        super().__init__()

        ckpt_exists = bool(finetuned_ckpt and Path(finetuned_ckpt).exists())

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=not ckpt_exists,
            num_classes=0,
        )
        self.hidden_dim = self.backbone.embed_dim  # 768

        if ckpt_exists:
            sd = torch.load(finetuned_ckpt, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            n_total  = len(self.backbone.state_dict())
            n_loaded = n_total - len(missing)
            print(f"[ViTFrameEncoder] Loaded custom weights : {finetuned_ckpt}")
            print(f"  {n_loaded}/{n_total} keys loaded "
                  f"| missing={len(missing)} unexpected={len(unexpected)}")
            w_norm = self.backbone.blocks[0].norm1.weight.norm().item()
            print(f"  block[0].norm1 weight norm = {w_norm:.4f}  "
                  f"({'OK' if 0.5 < w_norm < 5.0 else 'CHECK KEYS'})")
        else:
            print("[ViTFrameEncoder] Custom checkpoint not found — "
                  "using ImageNet pretrained weights.")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[ViTFrameEncoder] Backbone FROZEN.")
        else:
            n = sum(p.numel() for p in self.backbone.parameters())
            print(f"[ViTFrameEncoder] Backbone TRAINABLE ({n:,} params).")

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
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embedding[:, :T + 1]
        x   = self.transformer(x)
        return self.norm(x[:, 0])


# ─────────────────────────────────────────────────────────────────────
# 4. Dual-Branch Classifier
# ─────────────────────────────────────────────────────────────────────

class DualBranchThyroidClassifier(nn.Module):
    def __init__(self,
                 num_classes:     int   = 3,
                 max_frames:      int   = 32,
                 freeze_backbone: bool  = False,
                 temporal_heads:  int   = 8,
                 temporal_layers: int   = 2,
                 dropout:         float = 0.3):
        super().__init__()

        self.frame_encoder = ViTFrameEncoder(freeze=freeze_backbone)
        D = self.frame_encoder.hidden_dim  # 768

        self.whole_temporal = TemporalTransformer(
            embed_dim=D, num_heads=temporal_heads,
            num_layers=temporal_layers, max_frames=max_frames, dropout=dropout)

        self.roi_temporal = TemporalTransformer(
            embed_dim=D, num_heads=temporal_heads,
            num_layers=temporal_layers, max_frames=max_frames, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def _encode(self, videos: torch.Tensor,
                temporal: TemporalTransformer) -> torch.Tensor:
        B, T, C, H, W = videos.shape
        frames = rearrange(videos, 'b t c h w -> (b t) c h w')
        feats  = self.frame_encoder(frames)
        feats  = rearrange(feats, '(b t) d -> b t d', b=B, t=T)
        return temporal(feats)

    def forward(self, whole: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        whole_feat = self._encode(whole, self.whole_temporal)
        roi_feat   = self._encode(roi,   self.roi_temporal)
        return self.fusion(torch.cat([whole_feat, roi_feat], dim=-1))


# ─────────────────────────────────────────────────────────────────────
# 5. Collate
# ─────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    whole  = torch.stack([b[0] for b in batch])
    roi    = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return whole, roi, labels


# ─────────────────────────────────────────────────────────────────────
# 6. Score thresholding
# ─────────────────────────────────────────────────────────────────────

def threshold_predict(probs: np.ndarray,
                      mal_thr: float = 0.50,
                      ben_thr: float = 0.50) -> np.ndarray:
    """
    Priority: malignant > benign > intermediate (default).
    probs: (N, 3) — columns: Benign, Intermediate, Malignant
    """
    preds = np.full(len(probs), 1, dtype=int)          # default: Intermediate
    preds[probs[:, 0] >= ben_thr] = 0                  # Benign
    preds[probs[:, 2] >= mal_thr] = 2                  # Malignant (higher priority)
    return preds


def malignancy_score(probs: np.ndarray) -> np.ndarray:
    """Continuous 0→1 risk score. probs: (N, 3)."""
    return (probs * SCORE_WEIGHTS).sum(axis=1)


def risk_band(score: float) -> str:
    if score >= 0.65:
        return "High"
    if score >= 0.35:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────
# 7. Train / Eval
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
    preds, all_probs, all_labels = [], [], []
    total = 0.0
    CLASS_NAMES = ThyroidDualDataset.CLASS_NAMES

    for whole, roi, lbls in tqdm(loader, desc="Eval ", leave=False):
        whole, roi, lbls = whole.to(device), roi.to(device), lbls.to(device)
        logits = model(whole, roi)
        total += criterion(logits, lbls).item()
        p = F.softmax(logits, dim=-1)
        preds.extend(p.argmax(1).cpu().numpy())
        all_probs.extend(p.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, preds,
                 target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    auc = 0.0
    if len(set(all_labels)) > 1:
        y_bin = label_binarize(all_labels, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")

    return {"loss": total / len(loader),
            "accuracy": report["accuracy"],
            "auc": auc}


# ─────────────────────────────────────────────────────────────────────
# 8. Test-set evaluation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(
        checkpoint:      str,
        test_video_root: str,
        test_roi_root:   str,
        max_frames:      int   = 32,
        img_size:        int   = 224,
        roi_size:        int   = 224,
        batch_size:      int   = 1,
        num_workers:     int   = 4,
        dropout:         float = 0.3,
        mal_thr:         float = 0.50,
        ben_thr:         float = 0.50,
        results_csv:     str   = "/root/autodl-tmp/suhel/thyroid_nodule/results/"
                                 "test_results_3class.csv",
        device_str:      str   = "cuda",
):
    CLASS_NAMES = ThyroidDualDataset.CLASS_NAMES
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  Test-set evaluation (3-class)  |  checkpoint: {checkpoint}")
    print(f"{'=' * 60}\n")

    model = DualBranchThyroidClassifier(
        num_classes=3, max_frames=max_frames,
        freeze_backbone=False, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print("Model loaded.\n")

    test_ds = ThyroidDualDataset(
        video_root=test_video_root, roi_root=test_roi_root,
        max_frames=max_frames, img_size=img_size, roi_size=roi_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_fn)

    all_probs_list, all_labels = [], []
    sample_names = [s[0].name for s in test_ds.samples]

    for whole, roi, lbls in tqdm(test_loader, desc="Testing"):
        whole, roi, lbls = whole.to(device), roi.to(device), lbls.to(device)
        p = F.softmax(model(whole, roi), dim=-1)
        all_probs_list.extend(p.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    all_probs  = np.array(all_probs_list)
    all_labels = np.array(all_labels)

    preds_argmax   = all_probs.argmax(axis=1)
    preds_thr      = threshold_predict(all_probs, mal_thr=mal_thr, ben_thr=ben_thr)
    mal_scores     = malignancy_score(all_probs)

    print("\n" + "─" * 55)
    print("Classification Report — argmax:")
    print(classification_report(all_labels, preds_argmax,
          target_names=CLASS_NAMES, zero_division=0))

    print("Classification Report — score threshold "
          f"(mal≥{mal_thr}, ben≥{ben_thr}):")
    print(classification_report(all_labels, preds_thr,
          target_names=CLASS_NAMES, zero_division=0))

    auc = 0.0
    if len(set(all_labels)) > 1:
        y_bin = label_binarize(all_labels, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")
    print(f"Macro OvR AUC: {auc:.4f}")
    print(f"Mean malignancy score: {mal_scores.mean():.4f}")
    print(f"  Benign subset:       {mal_scores[all_labels == 0].mean():.4f}")
    print(f"  Intermediate subset: {mal_scores[all_labels == 1].mean():.4f}")
    print(f"  Malignant subset:    {mal_scores[all_labels == 2].mean():.4f}")
    print("─" * 55)

    id2name = {i: n for i, n in enumerate(CLASS_NAMES)}
    rows = []
    for i, (sname, gt, pa, pt, ms) in enumerate(zip(
            sample_names, all_labels, preds_argmax, preds_thr, mal_scores)):
        rows.append({
            "sample_name":       sname,
            "gt_label":          id2name[int(gt)],
            "malignancy_score":  round(float(ms), 4),
            "risk_band":         risk_band(float(ms)),
            "pred_argmax":       id2name[int(pa)],
            "pred_threshold":    id2name[int(pt)],
            "benign_prob":       round(float(all_probs[i, 0]), 4),
            "intermediate_prob": round(float(all_probs[i, 1]), 4),
            "malignant_prob":    round(float(all_probs[i, 2]), 4),
            "correct_argmax":    "yes" if int(gt) == int(pa) else "no",
            "correct_threshold": "yes" if int(gt) == int(pt) else "no",
        })

    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {results_csv}")
    return {"auc": auc}


# ─────────────────────────────────────────────────────────────────────
# 9. Main (training)
# ─────────────────────────────────────────────────────────────────────

def main():
    DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    ROI_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    SAVE_BEST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_vitb16_3class_best.pth"
    SAVE_LAST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_vitb16_3class_last.pth"

    BATCH_SIZE          = 2
    MAX_FRAMES          = 32
    IMG_SIZE            = 224
    ROI_SIZE            = 224
    EPOCHS              = 50
    LR                  = 1e-4
    BACKBONE_LR         = 1e-5
    WEIGHT_DECAY        = 1e-4
    DROPOUT             = 0.3
    WARMUP_EPOCHS       = 5
    EARLY_STOP_PATIENCE = 10
    NUM_WORKERS         = 4

    diagnose(DATA_ROOT, ROI_ROOT, MAX_FRAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_ds = ThyroidDualDataset(
        video_root=os.path.join(DATA_ROOT, "train_pure"), roi_root=ROI_ROOT,
        max_frames=MAX_FRAMES, img_size=IMG_SIZE, roi_size=ROI_SIZE, augment=True)
    val_ds = ThyroidDualDataset(
        video_root=os.path.join(DATA_ROOT, "val_pure"), roi_root=ROI_ROOT,
        max_frames=MAX_FRAMES, img_size=IMG_SIZE, roi_size=ROI_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)

    model = DualBranchThyroidClassifier(
        num_classes=3, max_frames=MAX_FRAMES,
        freeze_backbone=False, dropout=DROPOUT).to(device)

    # Class-balanced weights for 3 classes
    counts  = np.bincount([lbl for _, _, lbl, _ in train_ds.samples], minlength=3)
    weights = torch.tensor(1.0 / counts.astype(float), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    backbone_params = list(model.frame_encoder.backbone.parameters())
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

    n_backbone = sum(p.numel() for p in backbone_params)
    n_head     = sum(p.numel() for p in head_params)
    n_total    = sum(p.numel() for p in model.parameters())
    print(f"Params — backbone: {n_backbone:,} @ lr={BACKBONE_LR}  "
          f"head/temporal: {n_head:,} @ lr={LR}  total: {n_total:,}\n")

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
              f"AUC (macro OvR): {val_metrics['auc']:.4f}")

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
    print(f"Best → {SAVE_BEST_PATH}")
    print(f"Last → {SAVE_LAST_PATH}")


# ─────────────────────────────────────────────────────────────────────
# 10. Test-only entry point
# ─────────────────────────────────────────────────────────────────────

def run_test_only():
    evaluate_test_set(
        checkpoint      = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_vitb16_3class_best.pth",
        test_video_root = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all/test_pure",
        test_roi_root   = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        max_frames=32, img_size=224, roi_size=224,
        batch_size=2, num_workers=4, dropout=0.3,
        mal_thr=0.50, ben_thr=0.50,
        results_csv="/root/autodl-tmp/suhel/thyroid_nodule/results/test_results_3class.csv",
    )


if __name__ == "__main__":
    log_path = ("/root/autodl-tmp/suhel/thyroid_nodule/Logs/with_areafiltering/"
                "dual_vitb16_3class.txt")
    sys.stdout = Tee(log_path)
    try:
        main()
        # run_test_only()
    finally:
        sys.stdout.close()
