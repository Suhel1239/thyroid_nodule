############## With Early Stopping ##########################
"""
Thyroid Nodule Video Classification — Dual-Branch v5 (USFM)
============================================================
Requires ROI crops to be pre-extracted first:
    python preextract_rois.py

Pipeline:
  Branch A (whole frame): video frames → USFM → TemporalTransformer → (B, EMBED_DIM)
  Branch B (ROI crops)  : pre-saved crops → USFM → TemporalTransformer → (B, EMBED_DIM)
  Fusion                : concat → LayerNorm → MLP → Benign/Malignant

USFM (Ultrasound Foundation Model) replaces ViT-B/16 as the per-frame encoder.
USFM is a large vision-language model pre-trained on ultrasound images; its image
encoder produces patch embeddings compatible with the TemporalTransformer used here.

The USFM checkpoint is loaded from a local path (USFM_CKPT) or from HuggingFace
(set USFM_HF_REPO to the hub repo id).  The encoder exposes a 768-d CLS token by
default; if your checkpoint uses a different hidden dim set USFM_EMBED_DIM.

[CHANGE] Variable-length ROI manifests are supported:
  _load_roi_frames() reads however many .jpg files are in the manifest and
  resamples them uniformly to max_frames using np.linspace — the same strategy
  used for video frames in _load_whole_frames().
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from PIL import Image
import sys
import sys as _sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── USFM configuration ───────────────────────────────────────────────────────
# Set USFM_CKPT to a local .pth / .bin checkpoint, or leave as None to load
# from HuggingFace using USFM_HF_REPO.
USFM_CKPT      = os.environ.get("USFM_CKPT", None)          # e.g. "/path/to/usfm.pth"
USFM_HF_REPO   = os.environ.get("USFM_HF_REPO",
                                 "mkaichristensen/USFM")      # HuggingFace repo id
USFM_EMBED_DIM = int(os.environ.get("USFM_EMBED_DIM", 768))  # CLS token hidden dim


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
    roi_root  = Path(roi_root)
    print("\n" + "=" * 60)
    for split in ("train_pure", "val_pure", "test_pure"):
        for cls in ("benign", "malignant"):
            vid_dir = data_root / split / cls

            split_name = split.replace("_pure", "")
            roi_dir = (Path(roi_root)
                       / f"rois_{max_frames}_rfdetr_filtered_noblackframe"
                       / split_name / cls)

            if not vid_dir.exists():
                continue
            videos = [v for v in vid_dir.iterdir()
                      if v.is_file() and v.suffix.lower() in VIDEO_EXTS]
            folders = [d for d in vid_dir.iterdir()
                       if d.is_dir() and any(
                           f.suffix.lower() in IMAGE_EXTS
                           for f in d.iterdir())]
            rois = list(roi_dir.glob("*/manifest.json")) if roi_dir.exists() else []
            total  = len(videos) + len(folders)
            status = ("OK" if len(rois) == total
                      else f"MISMATCH (sources={total}, roi_dirs={len(rois)})")
            print(f"  [{split}/{cls}]  videos={len(videos)}  "
                  f"frame_folders={len(folders)}  "
                  f"roi_manifests={len(rois)}  {status}")

            if roi_dir.exists():
                saved_counts = []
                for mpath in roi_dir.glob("*/manifest.json"):
                    with open(mpath) as f:
                        m = json.load(f)
                    saved_counts.append(m.get("total_saved", len(m["frames"])))
                if saved_counts:
                    print(f"    ROI frames per video — "
                          f"min={min(saved_counts)}  "
                          f"max={max(saved_counts)}  "
                          f"mean={np.mean(saved_counts):.1f}  "
                          f"(will be resampled to {max_frames})")
        print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────

class ThyroidDualDataset(Dataset):
    """
    Returns (whole_frames, roi_frames, label)
      whole_frames : (T, 3, 224, 224)  — uniformly sampled from video
      roi_frames   : (T, 3, roi_size, roi_size) — resampled from however many
                     ROI crops were saved (variable per video → fixed T)
      label        : scalar long tensor
    """
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
                print(f"  [WARNING] Missing class folder: "
                      f"{self.video_root / class_name}")
                continue

            cls_roi_dir = self.roi_root / class_name

            for vp in sorted(p for p in cls_vid_dir.iterdir()
                             if p.suffix.lower() in VIDEO_EXTS):
                roi_dir = cls_roi_dir / vp.stem
                if not (roi_dir / "manifest.json").exists():
                    print(f"  [WARNING] No ROI manifest for {vp.name} — "
                          f"run preextract_rois.py first. Skipping.")
                    continue
                self.samples.append((vp, roi_dir, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under {self.video_root}.\n"
                f"Run preextract_rois.py to generate ROI crops first.")

        b = sum(1 for _, _, l in self.samples if l == 0)
        m = sum(1 for _, _, l in self.samples if l == 1)
        print(f"  [{self.video_root.name}] benign={b}, malignant={m}, "
              f"total={len(self.samples)}")

        norm = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2)]
                if augment else [])

        self.whole_tf = transforms.Compose(
            aug + [transforms.Resize((img_size, img_size))] + norm)
        self.roi_tf   = transforms.Compose(
            aug + [transforms.Resize((roi_size, roi_size))] + norm)

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
                rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last = self.whole_tf(Image.fromarray(rgb))
            frames.append(last)
        cap.release()
        return torch.stack(frames)   # (max_frames, 3, H, W)

    def _load_roi_frames(self, roi_dir: Path) -> torch.Tensor:
        with open(roi_dir / "manifest.json") as f:
            manifest = json.load(f)

        fnames  = manifest["frames"]
        n_saved = len(fnames)

        if n_saved == 0:
            print(f"  [WARNING] Empty manifest at {roi_dir} — "
                  f"returning zero tensor.")
            return torch.zeros(self.max_frames, 3,
                               self.roi_size, self.roi_size)

        sample_indices = np.linspace(0, n_saved - 1, self.max_frames, dtype=int)

        frames = []
        for si in sample_indices:
            fname    = fnames[int(si)]
            img_path = roi_dir / fname
            if img_path.exists():
                bgr = cv2.imread(str(img_path))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames.append(self.roi_tf(Image.fromarray(rgb)))
                    continue
            frames.append(torch.zeros(3, self.roi_size, self.roi_size))

        return torch.stack(frames)   # (max_frames, 3, roi_size, roi_size)

    def __getitem__(self, idx):
        video_path, roi_dir, label = self.samples[idx]
        whole = self._load_whole_frames(video_path)
        roi   = self._load_roi_frames(roi_dir)
        return whole, roi, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. USFM Frame Encoder  (shared across both branches)
# ─────────────────────────────────────────────────────────────────────

def _load_usfm_backbone() -> nn.Module:
    """
    Load the USFM image encoder.

    USFM (Ultrasound Foundation Model) is distributed as a Vision Transformer
    pretrained on large-scale ultrasound data.  The image encoder is a standard
    ViT whose forward() returns (B, num_patches+1, hidden_dim); we extract the
    CLS token at index 0.

    Loading order:
      1. Local checkpoint at USFM_CKPT (env var or module-level constant).
      2. HuggingFace hub repo USFM_HF_REPO using transformers AutoModel.

    The function returns the bare image-encoder nn.Module with attribute
    `hidden_dim` set to the CLS token dimension.
    """
    if USFM_CKPT and Path(USFM_CKPT).exists():
        # ── local checkpoint ─────────────────────────────────────────
        # Expects the checkpoint to contain either:
        #   { "image_encoder": state_dict }  or  a bare state_dict
        # and the architecture to be a timm ViT-Base (patch16, img224).
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required to load USFM from a local checkpoint. "
                "Install with:  pip install timm") from e

        backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,          # drop classification head → CLS output
        )
        state = torch.load(USFM_CKPT, map_location="cpu")
        if isinstance(state, dict) and "image_encoder" in state:
            state = state["image_encoder"]
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"  [USFM] missing keys in checkpoint ({len(missing)}): "
                  f"{missing[:5]} …")
        backbone.hidden_dim = backbone.embed_dim   # timm uses embed_dim
        print(f"  [USFM] Loaded local checkpoint: {USFM_CKPT}")
        return backbone

    # ── HuggingFace ──────────────────────────────────────────────────
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required to load USFM from HuggingFace. "
            "Install with:  pip install transformers") from e

    print(f"  [USFM] Loading from HuggingFace: {USFM_HF_REPO}")
    config   = AutoConfig.from_pretrained(USFM_HF_REPO)
    backbone = AutoModel.from_pretrained(USFM_HF_REPO)
    # HF vision models expose hidden_size in config
    backbone.hidden_dim = config.hidden_size
    return backbone


class USFMFrameEncoder(nn.Module):
    """
    Per-frame encoder backed by USFM.

    Input : (B, 3, 224, 224)
    Output: (B, hidden_dim)  — CLS token

    The backbone is shared between Branch A (whole frames) and Branch B
    (ROI crops); freeze=True keeps USFM weights fixed during training so
    only the TemporalTransformer and fusion head are updated.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.backbone   = _load_usfm_backbone()
        self.hidden_dim = getattr(self.backbone, "hidden_dim", USFM_EMBED_DIM)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, 224, 224)
        Returns (B, hidden_dim)
        """
        out = self.backbone(x)

        # timm ViT with num_classes=0 returns the CLS token directly as (B, D)
        if isinstance(out, torch.Tensor):
            if out.ndim == 2:
                return out                  # already (B, D)
            if out.ndim == 3:
                return out[:, 0]            # (B, T, D) → CLS

        # HuggingFace BaseModelOutput
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]

        # HuggingFace pooler_output (some vision models)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output

        raise ValueError(
            f"Unrecognised USFM backbone output type: {type(out)}. "
            "Override USFMFrameEncoder.forward() to handle your checkpoint.")


# ─────────────────────────────────────────────────────────────────────
# 3. Temporal Transformer  (one instance per branch)
# ─────────────────────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8,
                 num_layers: int = 2, ff_dim: int = 1024,
                 dropout: float = 0.1, max_frames: int = 32):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_frames + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x):                          # (B, T, D)
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)           # (B, T+1, D)
        x   = x + self.pos_embedding[:, :T + 1]
        x   = self.transformer(x)
        return self.norm(x[:, 0])                  # (B, D)


# ─────────────────────────────────────────────────────────────────────
# 4. Dual-Branch Classifier
# ─────────────────────────────────────────────────────────────────────

class DualBranchThyroidClassifier(nn.Module):
    """
    Branch A : whole video frames  → USFM → TemporalTransformer → (B, D)
    Branch B : ROI crops (resampled to max_frames)
                                   → USFM → TemporalTransformer → (B, D)
    Fusion   : concat(A,B) → LayerNorm → Linear(512) → GELU → Dropout → Linear(2)

    The USFM image encoder is shared between Branch A and Branch B.
    """
    def __init__(self,
                 num_classes:     int   = 2,
                 max_frames:      int   = 32,
                 freeze_backbone: bool  = True,
                 embed_dim:       int   = USFM_EMBED_DIM,
                 temporal_heads:  int   = 8,
                 temporal_layers: int   = 2,
                 dropout:         float = 0.3):
        super().__init__()

        self.frame_encoder  = USFMFrameEncoder(freeze=freeze_backbone)
        # use actual hidden dim from the loaded backbone
        embed_dim = self.frame_encoder.hidden_dim

        self.whole_temporal = TemporalTransformer(
            embed_dim=embed_dim, num_heads=temporal_heads,
            num_layers=temporal_layers, max_frames=max_frames, dropout=dropout)

        self.roi_temporal   = TemporalTransformer(
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
        feats  = self.frame_encoder(frames)               # (B*T, D)
        feats  = rearrange(feats, '(b t) d -> b t d', b=B, t=T)
        return temporal(feats)                            # (B, D)

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
    return {"loss": total / len(loader),
            "accuracy": report["accuracy"], "auc": auc}


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
        results_csv: str = "/root/autodl-tmp/suhel/thyroid_nodule/results/test_results_USFM_ROI_best_32frames_rfdetr.csv",
        device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  Test-set evaluation")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Test root  : {test_video_root}")
    print(f"  Device     : {device}")
    print(f"{'=' * 60}\n")

    model = DualBranchThyroidClassifier(
        max_frames=max_frames, freeze_backbone=True, dropout=dropout).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.\n")

    test_ds = ThyroidDualDataset(
        video_root=test_video_root,
        roi_root=test_roi_root,
        max_frames=max_frames,
        img_size=img_size,
        roi_size=roi_size,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn)

    all_preds, all_probs, all_labels = [], [], []
    all_video_names = [s[0].name for s in test_ds.samples]

    for whole, roi, lbls in tqdm(test_loader, desc="Testing"):
        whole = whole.to(device)
        roi   = roi.to(device)
        lbls  = lbls.to(device)

        logits = model(whole, roi)
        p      = F.softmax(logits, dim=-1)

        all_preds.extend(p.argmax(1).cpu().numpy().tolist())
        all_probs.extend(p[:, 1].cpu().numpy().tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())

    print("\n" + "─" * 50)
    print("Classification Report (default threshold 0.5):")
    print("─" * 50)
    print(classification_report(all_labels, all_preds,
          target_names=["Benign", "Malignant"], zero_division=0))

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    print(f"AUC-ROC : {auc:.4f}")

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    J        = tpr - fpr
    best_thr = float(thresholds[np.argmax(J)])
    print(f"\nOptimal threshold (Youden J): {best_thr:.4f}")

    preds_tuned = (np.array(all_probs) >= best_thr).astype(int).tolist()
    print("\nWith tuned threshold:")
    print(classification_report(all_labels, preds_tuned,
          target_names=["Benign", "Malignant"], zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    print(f"              Pred Benign  Pred Malignant")
    print(f"  GT Benign       {cm[0, 0]:>5}          {cm[0, 1]:>5}")
    print(f"  GT Malignant    {cm[1, 0]:>5}          {cm[1, 1]:>5}")

    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0.0
    specificity = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0.0
    print(f"\nSensitivity (malignant recall) : {sensitivity:.4f}")
    print(f"Specificity (benign recall)    : {specificity:.4f}")
    print("─" * 50)

    id2name = {0: "Benign", 1: "Malignant"}
    rows = []
    for vname, gt, pred, mal_prob in zip(
            all_video_names, all_labels, all_preds, all_probs):
        rows.append({
            "video_name":     vname,
            "gt_label":       id2name.get(int(gt), str(gt)),
            "pred_label":     id2name.get(int(pred), str(pred)),
            "benign_prob":    round(1.0 - float(mal_prob), 4),
            "malignant_prob": round(float(mal_prob), 4),
            "correct":        "yes" if int(gt) == int(pred) else "no",
        })

    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_name", "gt_label", "pred_label",
                           "benign_prob", "malignant_prob", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-video results saved → {results_csv}")
    return {"auc": auc, "sensitivity": sensitivity,
            "specificity": specificity, "predictions": rows}


# ─────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    ROI_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    SAVE_BEST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_USFM_best_64_b2_rfdetr.pth"
    SAVE_LAST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_USFM_last_64_b2_rfdetr.pth"

    BATCH_SIZE          = 2
    MAX_FRAMES          = 32
    IMG_SIZE            = 224
    ROI_SIZE            = 224
    EPOCHS              = 50
    LR                  = 3e-4
    WEIGHT_DECAY        = 1e-4
    FREEZE_BACKBONE     = True
    DROPOUT             = 0.3
    WARMUP_EPOCHS       = 5
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
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_fn)

    model = DualBranchThyroidClassifier(
        max_frames=MAX_FRAMES, freeze_backbone=FREEZE_BACKBONE, dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    counts    = np.bincount([lbl for _, _, lbl in train_ds.samples])
    weights   = torch.tensor(1.0 / counts.astype(float),
                             dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_LAST_PATH), exist_ok=True)

    best_auc   = 0.0
    no_improve = 0

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
            best_auc   = val_metrics["auc"]
            no_improve = 0
            torch.save(model.state_dict(), SAVE_BEST_PATH)
            print(f"  Saved best model (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement in val AUC for {no_improve}/"
                  f"{EARLY_STOP_PATIENCE} epoch(s)")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    print(f"Best checkpoint → {SAVE_BEST_PATH}")
    print(f"Last checkpoint → {SAVE_LAST_PATH} (epoch {epoch})")


# ─────────────────────────────────────────────────────────────────────
# 8. Test-only
# ─────────────────────────────────────────────────────────────────────

def run_test_only():
    evaluate_test_set(
        checkpoint      = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/dual_USFM_best_64_b2_rfdetr.pth",
        test_video_root = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all/test_pure",
        test_roi_root   = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        max_frames      = 32,
        img_size        = 224,
        roi_size        = 224,
        batch_size      = 2,
        num_workers     = 4,
        dropout         = 0.3,
        results_csv     = "/root/autodl-tmp/suhel/thyroid_nodule/results/test_results_USFM_ROI_best_1_with_repetition_32frames_rfdetr_withareafiltering_newtestset.csv",
        device_str      = "cuda",
    )


if __name__ == "__main__":
    log_path = ("/root/autodl-tmp/suhel/thyroid_nodule/Logs/with_areafiltering/"
                "test_results_USFM_ROI_best_1_32frames_rfdetr_withareafiltering_and_frame_repitition_withnewtestset.txt")
    sys.stdout = Tee(log_path)
    try:
        # main()
        run_test_only()
    finally:
        sys.stdout.close()
