"""
Organize raw videos / cine folders into a 5-fold cross-validation structure.

Input:
    DATA_ROOT/
        b/   ← benign  (videos or cine folders)
        n/   ← indeterminate
        m/   ← malignant

Output:
    OUTPUT_ROOT/
        fold_0/
            train/benign/<video or cine folder copied here>
            val/benign/...
            test/benign/...
            train/indeterminate/...
            ...
        fold_1/
            ...
        5fold_splits.json   ← full manifest of which files go where

Files are COPIED (not moved / symlinked) so the originals are untouched.
Cine folders are copied recursively.

Fold assignment for fold k:
    test  = chunk k
    val   = chunk (k+1) % n_folds
    train = remaining chunks
"""

import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT   = "/root/autodl-tmp/suhel/thyroid_nodule/data"
OUTPUT_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/data_5fold"

CLASS_MAP = {          # subfolder in DATA_ROOT  →  label in output
    "b": "benign",
    "n": "indeterminate",
    "m": "malignant",
}

N_FOLDS     = 5
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_sources(data_root: Path, class_map: dict) -> dict[str, list]:
    """Returns {label: [(path, kind, stem), ...]}."""
    per_class = {label: [] for label in class_map.values()}
    for folder, label in class_map.items():
        cls_dir = data_root / folder
        if not cls_dir.exists():
            print(f"  [WARNING] not found: {cls_dir}")
            continue
        for entry in sorted(cls_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                per_class[label].append((entry, "video", entry.name))
            elif entry.is_dir():
                has_imgs = any(f.suffix.lower() in IMAGE_EXTS
                               for f in entry.iterdir() if f.is_file())
                if has_imgs:
                    per_class[label].append((entry, "cine", entry.name))
    return per_class


def make_folds(per_class: dict, n_folds: int, seed: int) -> list[dict]:
    """
    Returns list of n_folds dicts:
      { label: { "train": [...], "val": [...], "test": [...] } }
    """
    rng = random.Random(seed)
    class_chunks = {}
    for label, sources in per_class.items():
        shuffled = list(sources)
        rng.shuffle(shuffled)
        class_chunks[label] = [shuffled[i::n_folds] for i in range(n_folds)]

    splits = []
    for k in range(n_folds):
        fold = {}
        for label, chunks in class_chunks.items():
            test_idx  = k
            val_idx   = (k + 1) % n_folds
            train_idx = [i for i in range(n_folds)
                         if i != test_idx and i != val_idx]
            fold[label] = {
                "test":  chunks[test_idx],
                "val":   chunks[val_idx],
                "train": [s for i in train_idx for s in chunks[i]],
            }
        splits.append(fold)
    return splits


def copy_source(src: Path, dst_dir: Path, kind: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        return  # already copied
    if kind == "video":
        shutil.copy2(src, dst)
    else:  # cine folder
        shutil.copytree(src, dst)


def main():
    data_root   = Path(DATA_ROOT).resolve()
    output_root = Path(OUTPUT_ROOT).resolve()

    print(f"Source root : {data_root}")
    for folder, label in CLASS_MAP.items():
        print(f"  {folder}/ → '{label}'")

    per_class = collect_sources(data_root, CLASS_MAP)
    for label, srcs in per_class.items():
        n_vid  = sum(1 for _, k, _ in srcs if k == "video")
        n_cine = sum(1 for _, k, _ in srcs if k == "cine")
        print(f"  {label}: {len(srcs)} sources  (videos={n_vid}, cine={n_cine})")

    fold_splits = make_folds(per_class, N_FOLDS, RANDOM_SEED)
    labels = list(CLASS_MAP.values())

    print(f"\nFold sizes  (train / val / test):")
    for k, fold in enumerate(fold_splits):
        parts = [f"{cls}: {len(fold[cls]['train'])}/{len(fold[cls]['val'])}/{len(fold[cls]['test'])}"
                 for cls in labels]
        print(f"  fold_{k}: " + "   ".join(parts))

    # ── Save split manifest ──────────────────────────────────────────
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for k, fold in enumerate(fold_splits):
        manifest[f"fold_{k}"] = {
            label: {
                split: [str(p) for p, _, _ in srcs]
                for split, srcs in splits.items()
            }
            for label, splits in fold.items()
        }
    manifest_path = output_root / "5fold_splits.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")

    # ── Copy files into fold structure ───────────────────────────────
    print(f"\nCopying files to: {output_root}\n")
    for k, fold in enumerate(fold_splits):
        print(f"── fold_{k} ──────────────────────────────")
        for split_name in ("train", "val", "test"):
            for label in labels:
                sources = fold[label][split_name]
                if not sources:
                    continue
                dst_cls = output_root / f"fold_{k}" / split_name / label
                for src, kind, name in tqdm(
                        sources, desc=f"fold_{k}/{split_name}/{label}", leave=False):
                    copy_source(src, dst_cls, kind)
                print(f"  fold_{k}/{split_name}/{label}: {len(sources)} sources")

    print("\nDone.")
    print(f"Output structure:")
    print(f"  {output_root}/")
    print(f"    fold_0/train/benign/   fold_0/val/benign/   fold_0/test/benign/")
    print(f"    fold_0/train/indeterminate/  ...")
    print(f"    fold_1/  fold_2/  fold_3/  fold_4/")


if __name__ == "__main__":
    main()
