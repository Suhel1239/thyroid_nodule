"""
ViT-B/16 LoRA Finetuning on Thyroid Nodule Images
===================================================
Finetunes ViT-B/16 using LoRA on the attention Q/V projections.
Trains only ~0.5% of parameters — strong regularization for medical datasets.

After training, LoRA weights are merged into the backbone and saved as a
plain state dict — drop-in compatible with any script that loads ViT-B/16.

Data layout:
    {DATA_ROOT}/train/benign/*.jpg
    {DATA_ROOT}/train/malignant/*.jpg
    {DATA_ROOT}/val/benign/*.jpg
    {DATA_ROOT}/val/malignant/*.jpg

Requirements:
    pip install peft
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
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
from peft import LoraConfig, get_peft_model, TaskType

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = "/root/autodl-tmp/suhel/thyroid_nodule/Image_dataset/TN5000_forReview/categorized_dataset_balanced"
SAVE_BEST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_lora_finetuned_best.pth"
SAVE_LAST_PATH = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_lora_finetuned_last.pth"
LOG_PATH       = "/root/autodl-tmp/suhel/thyroid_nodule/Logs/finetune_vitb16_lora.txt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
IMG_SIZE            = 224
EPOCHS              = 100
LR                  = 1e-4      # LoRA params + head LR
WEIGHT_DECAY        = 1e-4
DROPOUT             = 0.1
WARMUP_EPOCHS       = 3
EARLY_STOP_PATIENCE = 10
NUM_WORKERS         = 4

# ── LoRA config ────────────────────────────────────────────────────────────────
LORA_R          = 16    # rank
LORA_ALPHA      = 32    # scaling = alpha / r
LORA_DROPOUT    = 0.1
# Target the Q and V projection matrices in every attention block
LORA_TARGET_MODULES = ["qkv"]   # timm ViT fuses Q,K,V into one matrix named "qkv"

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
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────

class ThyroidImageDataset(Dataset):
    LABEL_MAP = {"benign": 0, "malignant": 1}

    def __init__(self, root: str, img_size: int = 224, augment: bool = False):
        self.root    = Path(root)
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        for class_name, label in self.LABEL_MAP.items():
            cls_dir = self.root / class_name
            if not cls_dir.exists():
                print(f"  [WARNING] Missing class folder: {cls_dir}")
                continue
            imgs = sorted(p for p in cls_dir.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            for img_path in imgs:
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}")

        b = sum(1 for _, l in self.samples if l == 0)
        m = sum(1 for _, l in self.samples if l == 1)
        print(f"  [{self.root.name}] benign={b}  malignant={m}  total={len(self.samples)}")

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
        img_path, label = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.transform(img), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────
# 2. Model — ViT-B/16 + LoRA + classification head
# ─────────────────────────────────────────────────────────────────────

class ViTLoRAClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()

        # Load ViT-B/16 backbone
        backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0)

        # Freeze all backbone parameters — LoRA adapters will be trainable
        for p in backbone.parameters():
            p.requires_grad = False

        # Apply LoRA to Q/V attention projections
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.print_trainable_parameters()

        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_merged_backbone_state_dict(self):
        """Merge LoRA weights into backbone and return plain state dict."""
        merged = self.backbone.merge_and_unload()
        return merged.state_dict()


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
    return {"loss": total / len(loader), "accuracy": report["accuracy"], "auc": auc}


# ─────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data root: {DATA_ROOT}\n")
    print(f"LoRA rank={LORA_R}  alpha={LORA_ALPHA}  target={LORA_TARGET_MODULES}\n")

    train_ds = ThyroidImageDataset(
        root=os.path.join(DATA_ROOT, "train"), img_size=IMG_SIZE, augment=True)
    val_ds   = ThyroidImageDataset(
        root=os.path.join(DATA_ROOT, "val"),   img_size=IMG_SIZE, augment=False)

    counts    = np.bincount([l for _, l in train_ds.samples])
    weights   = torch.tensor(1.0 / counts.astype(float), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = ViTLoRAClassifier(num_classes=2, dropout=DROPOUT).to(device)

    # Only LoRA adapters + head are trainable
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)

    best_auc   = 0.0
    no_improve = 0
    best_state = None   # keep best merged state dict in memory

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

        # Merge LoRA → plain backbone and save
        merged_sd = model.get_merged_backbone_state_dict()
        torch.save(merged_sd, SAVE_LAST_PATH)

        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in merged_sd.items()}
            torch.save(best_state, SAVE_BEST_PATH)
            print(f"  Saved best backbone (AUC={best_auc:.4f}) → {SAVE_BEST_PATH}")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{EARLY_STOP_PATIENCE} epoch(s)")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nDone. Best Val AUC: {best_auc:.4f}")
    print(f"Merged backbone saved → {SAVE_BEST_PATH}")
    print(f"\nSet in dual-branch video script:")
    print(f'  VITB16_FINETUNED_CKPT = "{SAVE_BEST_PATH}"')


if __name__ == "__main__":
    sys.stdout = Tee(LOG_PATH)
    try:
        main()
    finally:
        sys.stdout.close()
