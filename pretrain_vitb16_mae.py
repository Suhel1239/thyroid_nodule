"""
ViT-B/16 MAE Pretraining on Thyroid Nodule Images (no labels)
==============================================================
Masked Autoencoder (MAE) self-supervised pretraining.
Random patches are masked; the model learns to reconstruct them.
No category labels required — any folder of thyroid images will do.

After pretraining, load the encoder backbone in your downstream scripts:
    VITB16_FINETUNED_CKPT = "/path/to/vitb16_mae_pretrained.pth"

Data layout (flat or nested — any images found recursively):
    {DATA_ROOT}/**/*.jpg

References: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import sys as _sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import timm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ──────────────────────────────────────────────────────────────────────
# Point at ANY folder containing thyroid images — no benign/malignant split needed.
DATA_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/Image_dataset/TN5000_forReview"
SAVE_BEST_PATH  = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_mae_pretrained.pth"
SAVE_LAST_PATH  = "/root/autodl-tmp/suhel/thyroid_nodule/weights_videos/vitb16_mae_last.pth"
LOG_PATH        = "/root/autodl-tmp/suhel/thyroid_nodule/Logs/pretrain_vitb16_mae.txt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
IMG_SIZE            = 224
PATCH_SIZE          = 16          # must match ViT-B/16
NUM_PATCHES         = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
MASK_RATIO          = 0.75        # fraction of patches to mask (MAE default)
EPOCHS              = 200
LR                  = 1.5e-4      # base lr; scaled by batch size below
MIN_LR              = 1e-6
WEIGHT_DECAY        = 0.05
WARMUP_EPOCHS       = 40
NUM_WORKERS         = 4

# Decoder config (lightweight — only encoder weights are saved)
DECODER_EMBED_DIM   = 512
DECODER_DEPTH       = 8
DECODER_NUM_HEADS   = 16

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
# 1. Dataset (no labels)
# ─────────────────────────────────────────────────────────────────────

class ThyroidUnlabeledDataset(Dataset):
    """Loads all images recursively from data_root — labels not used."""

    def __init__(self, data_root: str, img_size: int = 224, augment: bool = True):
        self.paths = sorted(
            p for p in Path(data_root).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

        if not self.paths:
            raise RuntimeError(f"No images found under {data_root}")

        print(f"  Found {len(self.paths)} images under {data_root}")

        norm = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])]
        aug  = ([transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomRotation(15),
                 transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1)]
                if augment else [])
        self.transform = transforms.Compose(
            aug + [transforms.Resize((img_size, img_size))] + norm)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bgr = cv2.imread(str(self.paths[idx]))
        if bgr is None:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.transform(img)   # (3, H, W) — no label


# ─────────────────────────────────────────────────────────────────────
# 2. Patch utilities
# ─────────────────────────────────────────────────────────────────────

def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    imgs : (B, 3, H, W)
    return: (B, N, patch_size**2 * 3)  where N = (H/p)*(W/p)
    """
    B, C, H, W = imgs.shape
    p = patch_size
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)   # (B, h, w, p, p, C)
    x = x.reshape(B, h * w, p * p * C)
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, img_size: int) -> torch.Tensor:
    """
    patches : (B, N, patch_size**2 * 3)
    return  : (B, 3, H, W)
    """
    B, N, _ = patches.shape
    p = patch_size
    h = w = img_size // p
    x = patches.reshape(B, h, w, p, p, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)   # (B, 3, h, p, w, p)
    x = x.reshape(B, 3, h * p, w * p)
    return x


def random_masking(x: torch.Tensor, mask_ratio: float):
    """
    x          : (B, N, D)  — patch tokens (no CLS)
    mask_ratio : fraction to mask

    Returns:
        x_visible  : (B, N_vis, D)
        mask       : (B, N)   bool — True where masked
        ids_restore: (B, N)   long — to restore original order
    """
    B, N, D = x.shape
    n_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)        # ascending → keep first n_keep
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep   = ids_shuffle[:, :n_keep]
    x_visible  = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
    mask.scatter_(1, ids_keep, False)   # False = visible, True = masked

    return x_visible, mask, ids_restore


# ─────────────────────────────────────────────────────────────────────
# 3. MAE Model
# ─────────────────────────────────────────────────────────────────────

class MAEEncoder(nn.Module):
    """ViT-B/16 encoder — processes only visible (unmasked) patches."""

    def __init__(self):
        super().__init__()
        vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        # Reuse patch embedding, positional embedding, CLS token, blocks, norm
        self.patch_embed = vit.patch_embed    # (B,3,H,W) → (B,N,768)
        self.cls_token   = vit.cls_token      # (1,1,768)
        self.pos_embed   = vit.pos_embed      # (1,N+1,768)
        self.blocks      = vit.blocks
        self.norm        = vit.norm
        self.embed_dim   = 768

    def forward(self, x: torch.Tensor, ids_keep: torch.Tensor):
        """
        x        : (B, 3, H, W)
        ids_keep : (B, N_vis) indices of visible patches
        Returns  : (B, N_vis+1, 768)  — CLS + visible tokens
        """
        # Patch embed
        tokens = self.patch_embed(x)    # (B, N, 768)
        B, N, D = tokens.shape

        # Add positional embedding (skip CLS position 0)
        tokens = tokens + self.pos_embed[:, 1:, :]

        # Keep only visible patches
        tokens = torch.gather(
            tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        cls = cls + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens   # (B, N_vis+1, D)


class MAEDecoder(nn.Module):
    """
    Lightweight decoder: takes encoder output + mask tokens → reconstructs all patches.
    Only used during pretraining — discarded afterwards.
    """

    def __init__(self, encoder_dim: int = 768,
                 decoder_dim: int = DECODER_EMBED_DIM,
                 decoder_depth: int = DECODER_DEPTH,
                 decoder_num_heads: int = DECODER_NUM_HEADS,
                 num_patches: int = NUM_PATCHES,
                 patch_size: int = PATCH_SIZE):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed     = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim))  # +1 for CLS

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=decoder_num_heads,
            dim_feedforward=decoder_dim * 4,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=decoder_depth)
        self.norm        = nn.LayerNorm(decoder_dim)
        self.pred        = nn.Linear(decoder_dim, patch_size ** 2 * 3)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed,  std=0.02)

    def forward(self, enc_tokens: torch.Tensor, ids_restore: torch.Tensor):
        """
        enc_tokens  : (B, N_vis+1, encoder_dim)  — CLS + visible encoder output
        ids_restore : (B, N)                      — to restore original patch order
        Returns     : (B, N, patch_size**2 * 3)  — reconstructed patches (all N)
        """
        B = enc_tokens.shape[0]
        N = ids_restore.shape[1]

        # Project encoder dim → decoder dim
        tokens = self.decoder_embed(enc_tokens)   # (B, N_vis+1, D_dec)

        # Separate CLS from patch tokens
        cls, vis_tokens = tokens[:, :1], tokens[:, 1:]   # (B,1,D), (B,N_vis,D)

        # Build full sequence: visible tokens + mask tokens, then restore order
        n_vis   = vis_tokens.shape[1]
        n_mask  = N - n_vis
        mask_tokens = self.mask_token.expand(B, n_mask, -1)

        full = torch.cat([vis_tokens, mask_tokens], dim=1)   # (B, N, D)
        full = torch.gather(
            full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1]))

        # Re-attach CLS and add positional embedding
        full = torch.cat([cls, full], dim=1)             # (B, N+1, D)
        full = full + self.pos_embed[:, :N + 1]

        full = self.transformer(full)
        full = self.norm(full)

        pred = self.pred(full[:, 1:])   # drop CLS → (B, N, p*p*3)
        return pred


class MAEModel(nn.Module):
    def __init__(self, mask_ratio: float = MASK_RATIO, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.patch_size  = patch_size
        self.encoder     = MAEEncoder()
        self.decoder     = MAEDecoder(encoder_dim=self.encoder.embed_dim)

    def forward(self, imgs: torch.Tensor):
        """imgs: (B, 3, H, W) — returns (loss, pred_patches, mask)"""
        # Tokenise (for masking only; encoder re-does patch embed internally)
        B, C, H, W = imgs.shape
        N = (H // self.patch_size) ** 2

        # Random masking
        noise    = torch.rand(B, N, device=imgs.device)
        ids_shuf = torch.argsort(noise, dim=1)
        ids_rest = torch.argsort(ids_shuf, dim=1)
        n_keep   = int(N * (1 - self.mask_ratio))
        ids_keep = ids_shuf[:, :n_keep]

        mask = torch.ones(B, N, device=imgs.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)   # True = masked

        # Encode visible patches
        enc_out = self.encoder(imgs, ids_keep)         # (B, N_vis+1, 768)

        # Decode → reconstruct all patches
        pred = self.decoder(enc_out, ids_rest)         # (B, N, p*p*3)

        # Target: normalised pixel values per patch
        target = patchify(imgs, self.patch_size)       # (B, N, p*p*3)
        # Normalise target per patch (MAE default)
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        # Loss only on masked patches
        loss = F.mse_loss(pred[mask], target[mask])
        return loss, pred, mask

    def get_encoder_state_dict(self):
        """Returns the encoder backbone state dict for downstream loading."""
        # Reconstruct a timm ViT state dict from our encoder pieces
        sd = {}
        for k, v in self.encoder.patch_embed.state_dict().items():
            sd[f"patch_embed.{k}"] = v
        sd["cls_token"]  = self.encoder.cls_token
        sd["pos_embed"]  = self.encoder.pos_embed
        for i, blk in enumerate(self.encoder.blocks):
            for k, v in blk.state_dict().items():
                sd[f"blocks.{i}.{k}"] = v
        for k, v in self.encoder.norm.state_dict().items():
            sd[f"norm.{k}"] = v
        return sd


# ─────────────────────────────────────────────────────────────────────
# 4. LR schedule (cosine with linear warmup, scaled by batch)
# ─────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, epochs: int,
                    min_lr: float, base_lr: float):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale    = min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine
        return scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Mask ratio: {MASK_RATIO}  |  Epochs: {EPOCHS}\n")

    ds = ThyroidUnlabeledDataset(DATA_ROOT, img_size=IMG_SIZE, augment=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model = MAEModel(mask_ratio=MASK_RATIO).to(device)

    # Scale LR by batch size (MAE convention: base_lr at batch=256)
    effective_lr = LR * BATCH_SIZE / 256
    print(f"Effective LR: {effective_lr:.2e}  (base={LR} × batch={BATCH_SIZE}/256)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=effective_lr, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, effective_lr)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    n_params = sum(p.numel() for p in model.parameters())
    n_enc    = sum(p.numel() for p in model.encoder.parameters())
    print(f"Total params: {n_params:,}  |  Encoder: {n_enc:,}\n")

    os.makedirs(os.path.dirname(SAVE_BEST_PATH), exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for imgs in tqdm(loader, desc=f"Epoch {epoch:03d}", leave=False):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.autocast(device_type="cuda"):
                    loss, _, _ = model(imgs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _ = model(imgs)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | LR: {lr_now:.2e} | Recon Loss: {avg_loss:.4f}")

        # Save encoder backbone (timm-compatible state dict)
        enc_sd = model.get_encoder_state_dict()
        torch.save(enc_sd, SAVE_LAST_PATH)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(enc_sd, SAVE_BEST_PATH)
            print(f"  Saved best encoder (loss={best_loss:.4f}) → {SAVE_BEST_PATH}")

    print(f"\nDone. Best reconstruction loss: {best_loss:.4f}")
    print(f"Encoder backbone saved → {SAVE_BEST_PATH}")
    print(f"\nLoad in downstream scripts:")
    print(f'  VITB16_FINETUNED_CKPT = "{SAVE_BEST_PATH}"')


if __name__ == "__main__":
    sys.stdout = Tee(LOG_PATH)
    try:
        main()
    finally:
        sys.stdout.close()
