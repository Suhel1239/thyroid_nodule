"""
ViT-B/16 Image-Level Finetuning for Thyroid Nodule Classification
==================================================================
Step 1 of 2: Finetune ViT-B/16 on individual frames extracted from videos.
             Each frame is treated as an independent image sample.

Step 2: Use the saved checkpoint in thyroid_dual_branch_vitb16.py by setting
        VITB16_FINETUNED_CKPT to the path produced here.

Data expected at:
    {DATA_ROOT}/{split}_pure/{class}/*.mp4   (videos — frames extracted on-the-fly)
  OR
    {DATA_ROOT}/{split}_pure/{class}/{video_stem}/*.jpg  (pre-extracted frame folders)

Both layouts are auto-detected per video.

Output checkpoint: SAVE_BEST_PATH (plain state dict of the ViT backbone, no head)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys as _sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
from PIL import Image
import timm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
SAVE_BEST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_finetuned_best.pth"
SAVE_LAST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_finetuned_last.pth"
LOG_PATH       = "/root/autodl-tmp/suhel/thyroid_nodule/Logs/finetune_vitb16_images.txt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
IMG_SIZE            = 224
EPOCHS              = 30
LR                  = 1e-4       # head LR
BACKBONE_LR         = 5e-6      # last-4-block backbone LR
WEIGHT_DECAY        = 1e-4
DROPOUT             = 0.1
WARMUP_EPOCHS       = 2
EARLY_STOP_PATIENCE = 8
NUM_WORKERS         = 4
FRAMES_PER_VIDEO    = 16         # how many frames to sample per video for training

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


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
# 1. Dataset — frame-level
# ─────────────────────────────────────────────────────────────────────

class ThyroidFrameDataset(Dataset):
    """
    Each __getitem__ returns a single frame (image) + label.
    Samples frames from videos or pre-extracted frame folders.
    """
    LABEL_MAP = {"benign": 0, "malignant": 1}

    def __init__(self,
                 video_root:      str,
                 frames_per_video: int  = 16,
                 img_size:         int  = 224,
                 augment:          bool = False):

        self.video_root      = Path(video_root)
        self.frames_per_video = frames_per_video
        # (source_path, frame_index_or_None, label)
        # source_path is a video file or an image file
        self.samples: List[Tuple[Path, int, int]] = []

        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        existing = {d.name.lower(): d
                    for d in self.video_root.iterdir() if d.is_dir()}

        for class_name, label in self.LABEL_MAP.items():
            cls_dir = existing.get(class_name.lower())
            if cls_dir is None:
                print(f"  [WARNING] Missing class folder: {self.video_root / class_name}")
                continue

            for entry in sorted(cls_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                    # video file — will sample frames on-the-fly
                    cap   = cv2.VideoCapture(str(entry))
                    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
                    cap.release()
                    indices = np.linspace(0, total - 1, frames_per_video, dtype=int)
                    for idx in indices:
                        self.samples.append((entry, int(idx), label))

                elif entry.is_dir():
                    # pre-extracted frame folder
                    imgs = sorted(p for p in entry.iterdir()
                                  if p.suffix.lower() in IMAGE_EXTS)
                    if not imgs:
                        continue
                    chosen = [imgs[i] for i in
                              np.linspace(0, len(imgs) - 1, frames_per_video, dtype=int)]
                    for img_path in chosen:
                        self.samples.append((img_path, -1, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {self.video_root}.")

        b = sum(1 for _, _, l in self.samples if l == 0)
        m = sum(1 for _, _, l in self.samples if l == 1)
        print(f"  [{self.video_root.name}] benign_frames={b}, "
              f"malignant_frames={m}, total={len(self.samples)}")

        norm = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomRotation(10),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)]
                if augment else [])

        self.transform = transforms.Compose(
            aug + [transforms.Resize((img_size, img_size))] + norm)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, frame_idx, label = self.samples[idx]

        if frame_idx == -1:
            # pre-extracted image file
            bgr = cv2.imread(str(src))
            if bgr is None:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            # video file — seek to frame
            cap = cv2.VideoCapture(str(src))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            cap.release()
            if ret:
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. Model — ViT-B/16 with classification head
# ─────────────────────────────────────────────────────────────────────

class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,   # returns CLS token (B, 768)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────
# 3. Train / Eval
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

    auc    = roc_auc_score(all_labels, probs) if len(set(all_labels)) > 1 else 0.0
    report = classification_report(all_labels, preds,
                 target_names=["Benign", "Malignant"],
                 output_dict=True, zero_division=0)
    return {"loss": total / len(loader),
            "accuracy": report["accuracy"], "auc": auc}


# ─────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_ds = ThyroidFrameDataset(
        video_root=os.path.join(DATA_ROOT, "train_pure"),
        frames_per_video=FRAMES_PER_VIDEO,
        img_size=IMG_SIZE, augment=True)

    val_ds = ThyroidFrameDataset(
        video_root=os.path.join(DATA_ROOT, "val_pure"),
        frames_per_video=FRAMES_PER_VIDEO,
        img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = ViTClassifier(num_classes=2, dropout=DROPOUT).to(device)

    counts    = np.bincount([lbl for _, _, lbl in train_ds.samples])
    weights   = torch.tensor(1.0 / counts.astype(float),
                             dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Freeze all backbone, unfreeze last 4 blocks + norm
    for p in model.backbone.parameters():
        p.requires_grad = False
    for blk in model.backbone.blocks[-4:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.backbone.norm.parameters():
        p.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "backbone" in n]
    head_params     = list(model.head.parameters())

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
    print(f"Trainable parameters: {n_params:,}  "
          f"(head @ lr={LR}, backbone_last4 @ lr={BACKBONE_LR})\n")

    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)

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

        # Save backbone only (no classification head) for transfer to dual-branch
        torch.save(model.backbone.state_dict(), SAVE_LAST_PATH)

        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            no_improve = 0
            torch.save(model.backbone.state_dict(), SAVE_BEST_PATH)
            print(f"  Saved best backbone (AUC={best_auc:.4f}) → {SAVE_BEST_PATH}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{EARLY_STOP_PATIENCE} epoch(s)")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    print(f"Best backbone checkpoint → {SAVE_BEST_PATH}")
    print(f"\nTo use in dual-branch training, set in thyroid_dual_branch_vitb16.py:")
    print(f"  VITB16_FINETUNED_CKPT = \"{SAVE_BEST_PATH}\"")


if __name__ == "__main__":
    sys.stdout = Tee(LOG_PATH)
    try:
        main()
    finally:
        sys.stdout.close()
