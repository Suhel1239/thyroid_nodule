"""
Thyroid Nodule Video Classification — Temporal + RF-DETR Frame-Voting Fusion
=============================================================================
Two-branch architecture:

  Branch A (Temporal) : frames → shared ViT-B/16 → TemporalTransformer → logits_A
  Branch B (Voting)   : frames → RF-DETR per-frame classification
                        → average class probs across frames → log-probs as logits_B

  Fusion: concat(logits_A, logits_B) → MLP → final logits

RF-DETR is fully frozen (pretrained classifier). Branch B acts as a frame-level
consensus vote from the detection model. Branch A captures temporal dynamics.
The fusion MLP learns to trust each branch.

Backbone : ViT-B/16 finetuned on thyroid images (finetune_vitb16_images.py).
           Falls back to ImageNet pretrained if checkpoint not found.
Detector : RFDETRMedium — expected to output class_id in {0=benign, 1=malignant}.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys as _sys
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from PIL import Image
import timm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

VITB16_FINETUNED_CKPT = os.environ.get(
    "VITB16_FINETUNED_CKPT",
    "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_finetuned_best.pth",
)
RFDETR_CKPT = os.environ.get(
    "RFDETR_CKPT",
    "/home/suhel.khan/thyroid_nodule_p/RFDETR_runs/single/checkpoint_best_regular.pth",
)
RFDETR_THRESHOLD  = 0.25
RFDETR_NUM_CLASSES = 2   # 0=benign, 1=malignant


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
# RF-DETR per-frame classifier (frozen, no gradient)
# ─────────────────────────────────────────────────────────────────────

class RFDETRFrameVoter:
    """
    Wraps a frozen RF-DETR model.  For each RGB frame it returns a
    probability vector over num_classes.

    If the detector finds no detection above threshold, it returns a
    uniform distribution so that silent frames don't bias the vote.

    Initialized lazily so it is safe when forked into DataLoader workers.
    """

    def __init__(self, ckpt_path: str, threshold: float = RFDETR_THRESHOLD,
                 num_classes: int = RFDETR_NUM_CLASSES):
        self.ckpt_path   = ckpt_path
        self.threshold   = threshold
        self.num_classes = num_classes
        self._model      = None

    def _init(self):
        if self._model is None:
            from rfdetr import RFDETRMedium
            self._model = RFDETRMedium.from_checkpoint(self.ckpt_path)

    def frame_probs(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Returns (num_classes,) float32 probability array for one frame.
        Uses the highest-confidence detection; falls back to uniform if none.
        """
        self._init()
        dets = self._model.predict(frame_rgb, threshold=self.threshold)
        probs = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)
        if len(dets) > 0:
            best     = int(np.argmax(dets.confidence))
            conf     = float(dets.confidence[best])
            class_id = int(dets.class_id[best])
            if 0 <= class_id < self.num_classes:
                probs[:] = (1.0 - conf) / max(self.num_classes - 1, 1)
                probs[class_id] = conf
        return probs

    def video_logits(self, raw_frames: np.ndarray) -> np.ndarray:
        """
        raw_frames : (T, H, W, 3) uint8 RGB
        Returns (num_classes,) log-prob array — mean vote across T frames.
        """
        per_frame = np.stack([self.frame_probs(raw_frames[t])
                              for t in range(len(raw_frames))])   # (T, C)
        mean_probs = per_frame.mean(axis=0)                        # (C,)
        return np.log(mean_probs + 1e-8)                           # log-probs


# ─────────────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────

class ThyroidVideoDataset(Dataset):
    """
    Returns (whole_frames, raw_frames, label).

    whole_frames : (T, 3, 224, 224) normalised float tensor — for ViT Branch A
    raw_frames   : (T, H, W, 3)     uint8 RGB tensor        — for RF-DETR Branch B
    label        : scalar long tensor
    """

    LABEL_MAP = {"benign": 0, "malignant": 1}

    def __init__(self,
                 video_root: str,
                 max_frames: int  = 32,
                 img_size:   int  = 224,
                 augment:    bool = False):

        self.video_root = Path(video_root)
        self.max_frames = max_frames
        self.img_size   = img_size
        self.samples: List[Tuple[Path, int]] = []

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        existing = {d.name.lower(): d
                    for d in self.video_root.iterdir() if d.is_dir()}

        for class_name, label in self.LABEL_MAP.items():
            cls_dir = existing.get(class_name.lower())
            if cls_dir is None:
                print(f"  [WARNING] Missing class folder: {self.video_root / class_name}")
                continue
            for vp in sorted(p for p in cls_dir.iterdir()
                             if p.suffix.lower() in VIDEO_EXTS):
                self.samples.append((vp, label))

        if not self.samples:
            raise RuntimeError(f"No video samples found under {self.video_root}.")

        b = sum(1 for _, l in self.samples if l == 0)
        m = sum(1 for _, l in self.samples if l == 1)
        print(f"  [{self.video_root.name}] benign={b}, malignant={m}, total={len(self.samples)}")

        norm = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2)]
                if augment else [])
        self.transform = transforms.Compose(
            aug + [transforms.Resize((img_size, img_size))] + norm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap   = cv2.VideoCapture(str(video_path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, self.max_frames, dtype=int)

        whole_tensors, raw_list = [], []
        last_bgr   = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        last_whole = self.transform(Image.fromarray(cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)))

        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, bgr = cap.read()
            if ret:
                last_bgr   = bgr
                last_whole = self.transform(
                    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            whole_tensors.append(last_whole)
            raw_list.append(cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB))   # (H, W, 3) uint8

        cap.release()

        whole      = torch.stack(whole_tensors)              # (T, 3, H, W)
        raw_frames = torch.from_numpy(
            np.stack(raw_list).astype(np.uint8))             # (T, H, W, 3)
        return whole, raw_frames, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. Shared ViT-B/16 Frame Encoder
# ─────────────────────────────────────────────────────────────────────

class ViTFrameEncoder(nn.Module):
    """Input: (B, 3, 224, 224) → Output: (B, 768)"""

    def __init__(self, finetuned_ckpt: str = VITB16_FINETUNED_CKPT,
                 freeze: bool = True):
        super().__init__()
        self.backbone   = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0)
        self.hidden_dim = self.backbone.embed_dim  # 768

        if finetuned_ckpt and Path(finetuned_ckpt).exists():
            sd = torch.load(finetuned_ckpt, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            n_total  = len(self.backbone.state_dict())
            n_loaded = n_total - len(missing)
            print(f"[ViTFrameEncoder] Loaded finetuned weights: {finetuned_ckpt}")
            print(f"  {n_loaded}/{n_total} tensors | missing={len(missing)} "
                  f"unexpected={len(unexpected)}")
        else:
            print("[ViTFrameEncoder] Using ImageNet pretrained weights.")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[ViTFrameEncoder] Backbone frozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, 768)


# ─────────────────────────────────────────────────────────────────────
# 3. Branch A — Temporal Transformer
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        B, T, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embedding[:, :T + 1]
        x   = self.transformer(x)
        return self.norm(x[:, 0])   # (B, D)


# ─────────────────────────────────────────────────────────────────────
# 4. Full Model
# ─────────────────────────────────────────────────────────────────────

class ThyroidRFDETRVotingClassifier(nn.Module):
    """
    Branch A: whole frames → ViT → TemporalTransformer → Linear(768,2) → logits_A
    Branch B: raw frames  → RFDETRFrameVoter (frozen, no grad)
              per-frame classification → mean probs → log-probs → logits_B

    Fusion: concat(logits_A, logits_B) → LayerNorm → Linear(64) → GELU → Dropout → Linear(2)
    """

    def __init__(self,
                 num_classes:      int   = 2,
                 max_frames:       int   = 32,
                 freeze_backbone:  bool  = True,
                 finetuned_ckpt:   str   = VITB16_FINETUNED_CKPT,
                 rfdetr_ckpt:      str   = RFDETR_CKPT,
                 rfdetr_threshold: float = RFDETR_THRESHOLD,
                 temporal_heads:   int   = 8,
                 temporal_layers:  int   = 2,
                 dropout:          float = 0.3):
        super().__init__()

        self.frame_encoder = ViTFrameEncoder(
            finetuned_ckpt=finetuned_ckpt, freeze=freeze_backbone)
        D = self.frame_encoder.hidden_dim  # 768

        # Branch A
        self.temporal   = TemporalTransformer(
            embed_dim=D, num_heads=temporal_heads,
            num_layers=temporal_layers, max_frames=max_frames, dropout=dropout)
        self.temporal_head = nn.Linear(D, num_classes)

        # Branch B — RF-DETR voter (not an nn.Module; no learnable params)
        self.voter = RFDETRFrameVoter(
            ckpt_path=rfdetr_ckpt, threshold=rfdetr_threshold,
            num_classes=num_classes)

        # Fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(num_classes * 2),
            nn.Linear(num_classes * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _rfdetr_vote(self, raw_batch: torch.Tensor) -> torch.Tensor:
        """
        raw_batch : (B, T, H, W, 3) uint8 CPU/GPU tensor
        Returns   : (B, num_classes) float32 tensor on same device as model params
        """
        device = next(self.parameters()).device
        raw_np  = raw_batch.cpu().numpy()    # (B, T, H, W, 3)
        logits  = np.stack([self.voter.video_logits(raw_np[b])
                            for b in range(raw_np.shape[0])])   # (B, C)
        return torch.tensor(logits, dtype=torch.float32, device=device)

    def _encode_video(self, whole: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, D)"""
        B, T, C, H, W = whole.shape
        frames = rearrange(whole, 'b t c h w -> (b t) c h w')
        feats  = self.frame_encoder(frames)
        return rearrange(feats, '(b t) d -> b t d', b=B, t=T)

    def forward(self, whole: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # Branch A
        feats    = self._encode_video(whole)                         # (B, T, D)
        logits_a = self.temporal_head(self.temporal(feats))          # (B, 2)
        # Branch B  (no gradient flows through RF-DETR)
        with torch.no_grad():
            logits_b = self._rfdetr_vote(raw)                        # (B, 2)
        fused = torch.cat([logits_a, logits_b], dim=-1)              # (B, 4)
        return self.fusion(fused)                                     # (B, 2)

    def forward_with_branch_logits(self, whole: torch.Tensor, raw: torch.Tensor):
        feats    = self._encode_video(whole)
        logits_a = self.temporal_head(self.temporal(feats))
        with torch.no_grad():
            logits_b = self._rfdetr_vote(raw)
        fused = torch.cat([logits_a, logits_b], dim=-1)
        final = self.fusion(fused)
        return final, logits_a, logits_b


# ─────────────────────────────────────────────────────────────────────
# 5. Collate
# ─────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    whole  = torch.stack([b[0] for b in batch])   # (B, T, 3, H, W)
    raw    = torch.stack([b[1] for b in batch])   # (B, T, H, W, 3)
    labels = torch.stack([b[2] for b in batch])
    return whole, raw, labels


# ─────────────────────────────────────────────────────────────────────
# 6. Train / Eval
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    for whole, raw, labels in tqdm(loader, desc="Train", leave=False):
        whole, labels = whole.to(device), labels.to(device)
        # raw stays on CPU — RF-DETR runs on CPU via numpy
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(whole, raw), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(whole, raw), labels)
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
    for whole, raw, lbls in tqdm(loader, desc="Eval ", leave=False):
        whole, lbls = whole.to(device), lbls.to(device)
        logits = model(whole, raw)
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


# ─────────────────────────────────────────────────────────────────────
# 7. Test-set evaluation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test_set(
        checkpoint:      str,
        test_video_root: str,
        max_frames:      int   = 32,
        batch_size:      int   = 1,
        num_workers:     int   = 0,
        dropout:         float = 0.3,
        results_csv:     str   = "/root/autodl-tmp/suhel/thyroid_nodule/results/"
                                 "test_results_rfdetr_voting.csv",
        device_str:      str   = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  Test-set evaluation  |  checkpoint: {checkpoint}")
    print(f"{'=' * 60}\n")

    model = ThyroidRFDETRVotingClassifier(
        max_frames=max_frames, freeze_backbone=True, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    test_ds = ThyroidVideoDataset(
        video_root=test_video_root, max_frames=max_frames, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)

    all_preds, all_probs, all_labels = [], [], []
    branch_probs = {"A": [], "B": []}
    all_video_names = [s[0].name for s in test_ds.samples]

    for whole, raw, lbls in tqdm(test_loader, desc="Testing"):
        whole, lbls = whole.to(device), lbls.to(device)
        final, la, lb = model.forward_with_branch_logits(whole, raw)
        p = F.softmax(final, dim=-1)
        all_preds.extend(p.argmax(1).cpu().numpy().tolist())
        all_probs.extend(p[:, 1].cpu().numpy().tolist())
        all_labels.extend(lbls.cpu().numpy().tolist())
        branch_probs["A"].extend(F.softmax(la, -1)[:, 1].cpu().numpy().tolist())
        # lb is already log-probs from RF-DETR voter; softmax to get probs
        branch_probs["B"].extend(F.softmax(lb, -1)[:, 1].cpu().numpy().tolist())

    print("\n" + "─" * 50)
    print("Final Fusion — Classification Report:")
    print(classification_report(all_labels, all_preds,
          target_names=["Benign", "Malignant"], zero_division=0))

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    print(f"AUC-ROC (fusion)    : {auc:.4f}")
    for name, bp in branch_probs.items():
        b_auc = roc_auc_score(all_labels, bp) if len(set(all_labels)) > 1 else 0.0
        print(f"AUC-ROC (branch {name})  : {b_auc:.4f}")

    cm          = confusion_matrix(all_labels, all_preds)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0.0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    print(f"Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}")
    print("─" * 50)

    id2name = {0: "Benign", 1: "Malignant"}
    rows = [
        {"video_name": vn, "gt": id2name[int(gt)], "pred": id2name[int(pr)],
         "prob_malignant": round(float(mp), 4),
         "branch_A_prob":  round(float(ba), 4),
         "branch_B_rfdetr_prob": round(float(bb), 4),
         "correct": "yes" if int(gt) == int(pr) else "no"}
        for vn, gt, pr, mp, ba, bb in zip(
            all_video_names, all_labels, all_preds, all_probs,
            branch_probs["A"], branch_probs["B"])
    ]

    os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {results_csv}")
    return {"auc": auc, "sensitivity": sensitivity, "specificity": specificity}


# ─────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
    SAVE_BEST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/rfdetr_voting_best.pth"
    SAVE_LAST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/rfdetr_voting_last.pth"

    BATCH_SIZE          = 2
    MAX_FRAMES          = 32
    EPOCHS              = 50
    LR                  = 1e-4
    BACKBONE_LR         = 5e-6
    WEIGHT_DECAY        = 1e-4
    DROPOUT             = 0.3
    WARMUP_EPOCHS       = 3
    EARLY_STOP_PATIENCE = 10
    NUM_WORKERS         = 0   # RF-DETR runs inside worker; 0 is safest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_ds = ThyroidVideoDataset(
        video_root=os.path.join(DATA_ROOT, "train_pure"),
        max_frames=MAX_FRAMES, augment=True)
    val_ds = ThyroidVideoDataset(
        video_root=os.path.join(DATA_ROOT, "val_pure"),
        max_frames=MAX_FRAMES, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = ThyroidRFDETRVotingClassifier(
        max_frames=MAX_FRAMES, freeze_backbone=True,
        finetuned_ckpt=VITB16_FINETUNED_CKPT,
        rfdetr_ckpt=RFDETR_CKPT, dropout=DROPOUT).to(device)

    counts    = np.bincount([lbl for _, lbl in train_ds.samples])
    weights   = torch.tensor(1.0 / counts.astype(float), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Unfreeze last 2 ViT blocks + all branch/fusion heads
    for p in model.parameters():
        p.requires_grad = False
    for blk in model.frame_encoder.backbone.blocks[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.frame_encoder.backbone.norm.parameters():
        p.requires_grad = True
    for name, p in model.named_parameters():
        if any(k in name for k in ("temporal", "temporal_head", "fusion")):
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
    print(f"Trainable: {n_params:,}  (heads @ lr={LR}, backbone_last2 @ lr={BACKBONE_LR})\n")

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
# 9. Test-only entry point
# ─────────────────────────────────────────────────────────────────────

def run_test_only():
    evaluate_test_set(
        checkpoint      = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/rfdetr_voting_last.pth",
        test_video_root = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all/test_pure",
        max_frames      = 32, batch_size=1, num_workers=0, dropout=0.3,
        results_csv     = "/root/autodl-tmp/suhel/thyroid_nodule/results/"
                          "test_results_rfdetr_voting.csv",
    )


if __name__ == "__main__":
    log_path = ("/root/autodl-tmp/suhel/thyroid_nodule/Logs/with_areafiltering/"
                "rfdetr_voting_32frames.txt")
    sys.stdout = Tee(log_path)
    try:
        main()
        # run_test_only()
    finally:
        sys.stdout.close()
