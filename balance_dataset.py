"""
Balance dataset by undersampling the majority class (benign) to match minority (malignant).
Creates a new folder with the same train/val/test / benign/malignant structure.
Original files are copied (not moved).
"""

import os
import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SRC_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/categorized_dataset"
DST_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/categorized_dataset_balanced"
SEED     = 42

random.seed(SEED)

SPLITS  = ["train", "val", "test"]
CLASSES = ["benign", "malignant"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def get_images(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def balance_split(split: str):
    src_split = Path(SRC_ROOT) / split
    dst_split = Path(DST_ROOT) / split

    # Collect images per class
    class_images = {}
    for cls in CLASSES:
        src_cls = src_split / cls
        if not src_cls.exists():
            print(f"  [WARNING] {src_cls} not found — skipping.")
            class_images[cls] = []
        else:
            class_images[cls] = get_images(src_cls)

    counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    print(f"\n[{split}] original counts: {counts}")

    # Undersample majority to match minority
    min_count = min(counts.values())
    print(f"[{split}] balancing to {min_count} per class")

    for cls in CLASSES:
        imgs    = class_images[cls]
        chosen  = random.sample(imgs, min_count) if len(imgs) > min_count else imgs
        dst_cls = dst_split / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        for src_path in chosen:
            shutil.copy2(src_path, dst_cls / src_path.name)

        print(f"  {cls}: {len(chosen)} images → {dst_cls}")


def main():
    print(f"Source : {SRC_ROOT}")
    print(f"Output : {DST_ROOT}")

    for split in SPLITS:
        if not (Path(SRC_ROOT) / split).exists():
            print(f"\n[{split}] not found — skipping.")
            continue
        balance_split(split)

    print("\nDone. Balanced dataset ready at:")
    print(f"  {DST_ROOT}")


if __name__ == "__main__":
    main()
