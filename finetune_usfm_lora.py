"""
Fine-tune USFM on ultrasound image classification with LoRA
============================================================
Expects data organised as:
    data_root/
        train/
            benign/   *.jpg / *.png ...
            malignant/
        val/
            benign/
            malignant/
        test/          (optional)
            benign/
            malignant/

Download the USFM pretrained checkpoint from the official repo:
    https://github.com/openmedlab/USFM
and set USFM_CKPT below (or pass --usfm_ckpt on the command line).

Dependencies:
    pip install torch torchvision timm peft scikit-learn tqdm
"""

import argparse
import os
import csv
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve)

# ── LoRA via PEFT ────────────────────────────────────────────────────────────
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARNING] peft not installed — running without LoRA. "
          "Install with:  pip install peft")

# ── timm for ViT backbone ─────────────────────────────────────────────────────
try:
    import timm
except ImportError as e:
    raise ImportError("timm is required.  Install with:  pip install timm") from e


# ─────────────────────────────────────────────────────────────────────────────
# 1.  USFM backbone loader
# ─────────────────────────────────────────────────────────────────────────────

def load_usfm_vit(ckpt_path: str, img_size: int = 224) -> nn.Module:
    """
    Build a timm ViT-Base/16 and load USFM pretrained weights.

    USFM was pre-trained with MAE-style MIM on ultrasound images.
    The checkpoint contains only the encoder weights (no decoder).
    Key mapping handles both bare state-dicts and nested dicts such as
    {"model": ..., "image_encoder": ..., "state_dict": ...}.
    """
    # Build ViT-Base/16 without classification head
    vit = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        img_size=img_size,
        num_classes=0,        # no head — we attach our own below
    )

    if ckpt_path and Path(ckpt_path).exists():
        raw = torch.load(ckpt_path, map_location="cpu")

        # unwrap common wrapper keys
        for key in ("model", "image_encoder", "state_dict", "encoder"):
            if isinstance(raw, dict) and key in raw:
                raw = raw[key]
                break

        # strip "backbone." / "encoder." / "module." prefixes if present
        def _strip(sd, prefix):
            return {k[len(prefix):]: v for k, v in sd.items()
                    if k.startswith(prefix)}

        for prefix in ("backbone.", "encoder.", "module."):
            stripped = _strip(raw, prefix)
            if stripped:
                raw = stripped
                break

        missing, unexpected = vit.load_state_dict(raw, strict=False)
        print(f"[USFM] Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"  missing  ({len(missing)}): {missing[:4]} ...")
        if unexpected:
            print(f"  unexpected ({len(unexpected)}): {unexpected[:4]} ...")
    else:
        print("[USFM] No checkpoint found — using random ViT-Base/16 weights.")

    return vit   # output shape: (B, embed_dim=768)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LoRA wrapper
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(vit: nn.Module,
               r: int = 16,
               lora_alpha: int = 32,
               lora_dropout: float = 0.05,
               target_modules: list = None) -> nn.Module:
    """
    Apply LoRA to the query and value projection matrices of every
    transformer block in the ViT encoder.

    timm ViT-Base/16 attention projection names:
        blocks.{i}.attn.qkv   (fused Q,K,V — PEFT targets individual names)
    We target the inner linear layers via their module names.
    """
    if not PEFT_AVAILABLE:
        print("[LoRA] peft not available — fine-tuning full model instead.")
        for p in vit.parameters():
            p.requires_grad = True
        return vit

    if target_modules is None:
        # timm ViT-Base uses a single fused qkv Linear; PEFT can target it
        # by matching the substring.  We also target the output projection.
        target_modules = ["qkv", "proj"]

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        # PEFT TaskType.FEATURE_EXTRACTION wraps the model as-is
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    vit = get_peft_model(vit, lora_cfg)
    vit.print_trainable_parameters()
    return vit


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Full classifier
# ─────────────────────────────────────────────────────────────────────────────

class USFMClassifier(nn.Module):
    """
    USFM (ViT-Base/16) encoder  +  lightweight classification head.

    Architecture:
        image (B,3,224,224)
          → USFM ViT encoder (with LoRA on qkv/proj)
          → CLS token (B, 768)
          → LayerNorm → Linear(512) → GELU → Dropout → Linear(num_classes)
    """

    def __init__(self,
                 usfm_ckpt: str,
                 num_classes: int = 2,
                 img_size: int = 224,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 dropout: float = 0.3,
                 use_lora: bool = True):
        super().__init__()

        # ── encoder ───────────────────────────────────────────────────
        vit = load_usfm_vit(usfm_ckpt, img_size=img_size)
        self.embed_dim = vit.embed_dim   # 768 for ViT-Base

        if use_lora and PEFT_AVAILABLE:
            self.encoder = apply_lora(vit,
                                      r=lora_r,
                                      lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout)
        else:
            # freeze the whole backbone; only head trains
            for p in vit.parameters():
                p.requires_grad = False
            self.encoder = vit

        # ── head ──────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT with num_classes=0 returns CLS token (B, 768)
        feat = self.encoder(x)
        if feat.ndim == 3:          # (B, T, D) — take CLS
            feat = feat[:, 0]
        return self.head(feat)      # (B, num_classes)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Data
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(data_root: str,
                  img_size: int = 224,
                  batch_size: int = 16,
                  num_workers: int = 4):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    loaders = {}
    for split, tf in [("train", train_tf), ("val", val_tf), ("test", val_tf)]:
        split_dir = Path(data_root) / split
        if not split_dir.exists():
            print(f"  [INFO] {split_dir} not found — skipping {split} split.")
            continue
        ds = datasets.ImageFolder(str(split_dir), transform=tf)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)
        cls_counts = np.bincount([y for _, y in ds.samples])
        print(f"  [{split}] classes={ds.classes}  "
              + "  ".join(f"{c}={n}" for c, n in zip(ds.classes, cls_counts)))

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train / Eval
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []
    for imgs, labels in tqdm(loader, desc="Eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        p = F.softmax(logits, dim=-1)
        all_preds.extend(p.argmax(1).cpu().tolist())
        all_probs.extend(p[:, 1].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n   = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    auc = (roc_auc_score(all_labels, all_probs)
           if len(set(all_labels)) > 1 else 0.0)
    return {"loss": total_loss / n, "acc": acc, "auc": auc,
            "preds": all_preds, "probs": all_probs, "labels": all_labels}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Test-set report
# ─────────────────────────────────────────────────────────────────────────────

def test_report(metrics: dict, class_names: list, results_csv: str = None):
    labels, preds, probs = (metrics["labels"],
                            metrics["preds"],
                            metrics["probs"])
    print("\n" + "=" * 55)
    print("Test-set results")
    print("=" * 55)
    print(classification_report(labels, preds,
                                target_names=class_names, zero_division=0))
    print(f"AUC-ROC: {metrics['auc']:.4f}")

    # Youden optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_thr = float(thresholds[np.argmax(tpr - fpr)])
    print(f"Optimal threshold (Youden J): {best_thr:.4f}")
    tuned = (np.array(probs) >= best_thr).astype(int).tolist()
    print("\nWith tuned threshold:")
    print(classification_report(labels, tuned,
                                target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels, preds)
    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1].sum() > 0 else 0.0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0].sum() > 0 else 0.0
    print(f"Sensitivity: {sens:.4f}   Specificity: {spec:.4f}")

    if results_csv:
        os.makedirs(os.path.dirname(os.path.abspath(results_csv)), exist_ok=True)
        with open(results_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["gt", "pred", "prob_malignant"])
            w.writeheader()
            for gt, pr, pb in zip(labels, preds, probs):
                w.writerow({"gt": class_names[gt],
                            "pred": class_names[pr],
                            "prob_malignant": round(pb, 4)})
        print(f"Saved per-sample CSV -> {results_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune USFM on ultrasound classification with LoRA")
    p.add_argument("--data_root",   required=True,
                   help="Root folder with train/val/test subfolders")
    p.add_argument("--usfm_ckpt",   default="",
                   help="Path to USFM_latest.pth pretrained checkpoint")
    p.add_argument("--save_dir",    default="./checkpoints",
                   help="Directory to save best/last model")
    p.add_argument("--results_csv", default="./results/test_results.csv")

    # model
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--use_lora",    action="store_true", default=True)
    p.add_argument("--no_lora",     dest="use_lora", action="store_false")
    p.add_argument("--lora_r",      type=int,   default=16)
    p.add_argument("--lora_alpha",  type=int,   default=32)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--dropout",     type=float, default=0.3)

    # training
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--head_lr",     type=float, default=1e-3,
                   help="Separate (higher) LR for the classification head")
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--warmup_epochs",type=int,  default=3)
    p.add_argument("--patience",    type=int,   default=8,
                   help="Early-stopping patience (epochs without AUC gain)")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"\nDevice : {device}")
    print(f"LoRA   : {args.use_lora}  (r={args.lora_r}, alpha={args.lora_alpha})\n")

    # ── data ──────────────────────────────────────────────────────────
    loaders = build_loaders(args.data_root, args.img_size,
                            args.batch_size, args.num_workers)
    if "train" not in loaders:
        raise RuntimeError("No train split found under data_root.")

    train_loader = loaders["train"]
    class_names  = train_loader.dataset.classes   # e.g. ["benign","malignant"]

    # ── model ─────────────────────────────────────────────────────────
    model = USFMClassifier(
        usfm_ckpt   = args.usfm_ckpt,
        num_classes = args.num_classes,
        img_size    = args.img_size,
        lora_r      = args.lora_r,
        lora_alpha  = args.lora_alpha,
        lora_dropout= args.lora_dropout,
        dropout     = args.dropout,
        use_lora    = args.use_lora,
    ).to(device)

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total:,}  trainable={trainable:,} "
          f"({100*trainable/total:.2f}%)\n")

    # ── loss with class-balance weighting ────────────────────────────
    counts  = np.bincount([y for _, y in train_loader.dataset.samples])
    weights = torch.tensor(1.0 / counts.astype(float),
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── optimiser: separate LRs for LoRA params and head ─────────────
    head_params = list(model.head.parameters())
    head_ids    = {id(p) for p in head_params}
    lora_params = [p for p in model.trainable_params()
                   if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    # ── cosine LR schedule with linear warmup ────────────────────────
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = (torch.cuda.amp.GradScaler()
                 if device.type == "cuda" else None)

    # ── training loop ─────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt = os.path.join(args.save_dir, "usfm_lora_best.pth")
    last_ckpt = os.path.join(args.save_dir, "usfm_lora_last.pth")

    best_auc, no_improve = 0.0, 0
    val_loader = loaders.get("val")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler)

        log = (f"Epoch {epoch:03d} | "
               f"LR(lora)={optimizer.param_groups[0]['lr']:.2e} | "
               f"Train loss={tr_loss:.4f}  acc={tr_acc:.4f}")

        if val_loader:
            vm = evaluate(model, val_loader, criterion, device)
            log += (f" | Val loss={vm['loss']:.4f}  "
                    f"acc={vm['acc']:.4f}  AUC={vm['auc']:.4f}")
            auc = vm["auc"]
        else:
            auc = tr_acc   # fallback: use train acc as proxy

        scheduler.step()
        print(log)

        # save last
        _save(model, last_ckpt, args)

        # save best
        if auc > best_auc:
            best_auc   = auc
            no_improve = 0
            _save(model, best_ckpt, args)
            # also export merged encoder weights for reuse in other tasks
            encoder_ckpt = os.path.join(args.save_dir, "usfm_encoder_merged_best.pth")
            _save_encoder(model, encoder_ckpt)
            print(f"  => Best model saved  (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nBest val AUC : {best_auc:.4f}")
    print(f"Best ckpt    : {best_ckpt}")

    # ── test-set evaluation ───────────────────────────────────────────
    if "test" in loaders:
        print("\nLoading best checkpoint for test evaluation...")
        _load(model, best_ckpt, device)
        tm = evaluate(model, loaders["test"], criterion, device)
        test_report(tm, class_names, args.results_csv)

    # ── final encoder export ──────────────────────────────────────────
    # Reload best weights and export a clean merged encoder checkpoint
    # that can be dropped into any other pipeline (e.g. dual-branch video
    # classifier) without needing peft installed.
    _load(model, best_ckpt, device)
    final_encoder_ckpt = os.path.join(args.save_dir, "usfm_encoder_merged_final.pth")
    _save_encoder(model, final_encoder_ckpt)
    print(f"\nMerged encoder saved -> {final_encoder_ckpt}")
    print("Load it in another script with:")
    print("  vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)")
    print(f"  vit.load_state_dict(torch.load('{final_encoder_ckpt}')['encoder'])")


def _save(model: nn.Module, path: str, args):
    """Full checkpoint: state_dict + training args."""
    payload = {"state_dict": model.state_dict(), "args": vars(args)}
    torch.save(payload, path)


def _save_encoder(model: nn.Module, path: str):
    """
    Export only the ViT encoder weights with LoRA merged in.

    After merging, the saved weights are plain ViT-Base/16 weights —
    no peft dependency needed to load them.  The file contains:
        {
          "encoder":   <merged ViT state_dict>,   # drop-in backbone
          "head":      <classifier head state_dict>,
        }
    """
    encoder = model.encoder

    # merge LoRA deltas into the base weights (peft helper)
    if PEFT_AVAILABLE and hasattr(encoder, "merge_adapter"):
        encoder = encoder.merge_and_unload()   # returns a plain nn.Module

    payload = {
        "encoder": encoder.state_dict(),
        "head":    model.head.state_dict(),
    }
    torch.save(payload, path)
    print(f"  [export] Merged encoder weights -> {path}")


def _load(model: nn.Module, path: str, device):
    payload = torch.load(path, map_location=device)
    sd = payload.get("state_dict", payload)
    model.load_state_dict(sd, strict=True)


if __name__ == "__main__":
    main()
