"""
Create 5-Fold Image Dataset from Video Folds
=============================================
Matches images to their corresponding video fold by comparing the patient ID
— the part of the filename BEFORE the first underscore.

Example
-------
Video in fold_1/train/benign : 10006399578_anon_vid_001.mp4
  → matching key              : "10006399578"
Images in your image dataset  : 10006399578_frame_001.jpg
                                 10006399578_frame_002.jpg
  → matching key              : "10006399578"  ✓
  → copied to                 : image_folds/fold_1/train/benign/

Folder layout expected
----------------------
VIDEO_FOLD_ROOT/
    fold_1/
        train/
            benign/     *.mp4
            malignant/  *.mp4
        val/
            benign/
            malignant/
    fold_2/ ...
    ...

IMAGE_SRC_ROOT/         (your current single train/val/test split)
    train/
        benign/     *.jpg  (or any image extension)
        malignant/
    val/
        benign/
        malignant/
    test/               (optional)
        benign/
        malignant/

Output
------
IMAGE_FOLD_OUT/
    fold_1/
        train/benign/   ← copies of matched images
        train/malignant/
        val/benign/
        val/malignant/
    fold_2/ ...
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_FOLD_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/folds"
IMAGE_SRC_ROOT  = "/root/autodl-tmp/suhel/thyroid_nodule/Image_dataset/TN5000_forReview/categorized_dataset_balanced"
IMAGE_FOLD_OUT  = "/root/autodl-tmp/suhel/thyroid_nodule/image_folds"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

CLASSES = ["benign", "malignant"]


# ─────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────

def patient_key(filename: str) -> str:
    """
    Extract the matching key = everything before the first underscore.

    Examples
      "10006399578_anon_vid_001.mp4" → "10006399578"
      "10006399578_frame_001.jpg"    → "10006399578"
    """
    stem = Path(filename).stem          # strip extension
    return stem.split("_")[0]


# ─────────────────────────────────────────────────────────────────────
# 1. Index all images by (class, patient_key)
# ─────────────────────────────────────────────────────────────────────

def build_image_index(image_src_root: str):
    """
    Returns dict: { (class_name, patient_key) : [Path, ...] }
    Scans all splits (train/val/test) so every image is findable.
    """
    index = defaultdict(list)
    src   = Path(image_src_root)

    for split_dir in src.iterdir():
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            for img in cls_dir.iterdir():
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    key = patient_key(img.name)
                    index[(cls_name, key)].append(img)

    total = sum(len(v) for v in index.values())
    print(f"[Index] {total} images indexed across "
          f"{len({k[0] for k in index})} classes, "
          f"{len({k[1] for k in index})} unique patient keys")
    return index


# ─────────────────────────────────────────────────────────────────────
# 2. Build one fold
# ─────────────────────────────────────────────────────────────────────

def build_fold(fold_dir: Path, image_index: dict, out_dir: Path):
    """
    fold_dir  : e.g. folds/fold_1/
    out_dir   : e.g. image_folds/fold_1/
    """
    copied_total  = 0
    missing_total = 0

    for split_dir in sorted(fold_dir.iterdir()):   # train, val, (test)
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()

            # Collect all video patient keys in this split/class
            video_keys = set()
            for vf in cls_dir.iterdir():
                if vf.is_file() and vf.suffix.lower() in VIDEO_EXTS:
                    video_keys.add(patient_key(vf.name))

            if not video_keys:
                print(f"  [WARN] No videos found in {cls_dir} — skipping")
                continue

            # Destination folder
            dst = out_dir / split_name / cls_name
            dst.mkdir(parents=True, exist_ok=True)

            copied  = 0
            missing = 0

            for vk in sorted(video_keys):
                imgs = image_index.get((cls_name, vk), [])
                if not imgs:
                    print(f"    [MISSING] No images for patient key '{vk}' "
                          f"in class '{cls_name}'")
                    missing += 1
                    continue
                for img_path in imgs:
                    shutil.copy2(img_path, dst / img_path.name)
                    copied += 1

            print(f"  [{split_name}/{cls_name}] "
                  f"{len(video_keys)} video-keys → "
                  f"{copied} images copied, {missing} keys missing")
            copied_total  += copied
            missing_total += missing

    return copied_total, missing_total


# ─────────────────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────────────────

def main():
    fold_root = Path(VIDEO_FOLD_ROOT)
    out_root  = Path(IMAGE_FOLD_OUT)

    if not fold_root.exists():
        raise FileNotFoundError(f"Video fold root not found: {fold_root}")

    fold_dirs = sorted(d for d in fold_root.iterdir()
                       if d.is_dir() and d.name.startswith("fold"))
    if not fold_dirs:
        raise RuntimeError(f"No fold_* directories found under {fold_root}")

    print(f"Found {len(fold_dirs)} fold(s): {[d.name for d in fold_dirs]}\n")

    # Build image index once — reuse for every fold
    image_index = build_image_index(IMAGE_SRC_ROOT)
    print()

    grand_copied  = 0
    grand_missing = 0

    for fold_dir in fold_dirs:
        print(f"── {fold_dir.name} ──────────────────────────────")
        out_dir = out_root / fold_dir.name
        c, m    = build_fold(fold_dir, image_index, out_dir)
        grand_copied  += c
        grand_missing += m
        print()

    print("=" * 50)
    print(f"Done.")
    print(f"  Total images copied  : {grand_copied}")
    print(f"  Patient keys missing : {grand_missing}")
    print(f"  Output               : {IMAGE_FOLD_OUT}")

    # Print a summary of what was created
    print("\nOutput structure:")
    for fold_dir in sorted(out_root.iterdir()):
        if not fold_dir.is_dir():
            continue
        for split_dir in sorted(fold_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                n = sum(1 for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
                print(f"  {fold_dir.name}/{split_dir.name}/{cls_dir.name}: {n} images")


if __name__ == "__main__":
    main()
