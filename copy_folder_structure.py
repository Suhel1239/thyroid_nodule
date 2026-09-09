"""
Copy files from SOURCE_ROOT matching names found in REFERENCE_ROOT,
preserving the same train/val/test / benign/malignant folder structure.

Reference folder  → defines WHICH filenames to copy (names only, not content)
Source folder     → WHERE to find the actual files to copy
Output folder     → destination, mirroring the reference structure

Works with both video files and cine image sub-folders.
For cine folders: if the folder name exists in both reference and source, the
whole folder is copied recursively.

CONFIG:
    REFERENCE_ROOT  — folder whose filenames are used as the pick-list
    SOURCE_ROOT     — folder from which matching files/folders are copied
    OUTPUT_ROOT     — destination (created if absent)
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
REFERENCE_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/reference_folder"
SOURCE_ROOT    = "/root/autodl-tmp/suhel/thyroid_nodule/source_folder"
OUTPUT_ROOT    = "/root/autodl-tmp/suhel/thyroid_nodule/output_folder"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def collect_names(folder: Path) -> set:
    """Return the set of entry names (files + dirs) directly inside folder."""
    if not folder.exists():
        return set()
    return {e.name for e in folder.iterdir()}


def copy_entry(src: Path, dst_dir: Path):
    """Copy a single file or cine folder into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        return
    if src.is_file():
        shutil.copy2(src, dst)
    elif src.is_dir():
        shutil.copytree(src, dst)


def process_leaf(ref_leaf: Path, src_leaf: Path, out_leaf: Path):
    """
    ref_leaf, src_leaf, out_leaf are all cls-level dirs
    (e.g. .../train/benign/).
    Copy every entry from src_leaf whose name appears in ref_leaf.
    """
    ref_names = collect_names(ref_leaf)
    if not ref_names:
        return

    # Build a name→path map for the source leaf
    src_map = {}
    if src_leaf.exists():
        for e in src_leaf.iterdir():
            src_map[e.name] = e

    found = 0
    missing = []
    for name in sorted(ref_names):
        if name in src_map:
            copy_entry(src_map[name], out_leaf)
            found += 1
        else:
            missing.append(name)

    rel = out_leaf.relative_to(Path(OUTPUT_ROOT))
    print(f"  {rel}: copied {found}/{len(ref_names)}", end="")
    if missing:
        print(f"  [missing {len(missing)}: {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else "") + "]")
    else:
        print()


def main():
    ref_root = Path(REFERENCE_ROOT).resolve()
    src_root = Path(SOURCE_ROOT).resolve()
    out_root = Path(OUTPUT_ROOT).resolve()

    if not ref_root.exists():
        raise SystemExit(f"[ERROR] REFERENCE_ROOT not found: {ref_root}")
    if not src_root.exists():
        raise SystemExit(f"[ERROR] SOURCE_ROOT not found: {src_root}")

    print(f"Reference : {ref_root}")
    print(f"Source    : {src_root}")
    print(f"Output    : {out_root}\n")

    # Walk reference structure: split / cls
    splits = sorted(d for d in ref_root.iterdir() if d.is_dir())
    if not splits:
        raise SystemExit("[ERROR] No subdirectories found in REFERENCE_ROOT")

    for split_dir in splits:          # train / val / test
        cls_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for cls_dir in cls_dirs:      # benign / malignant / ...
            ref_leaf = cls_dir
            src_leaf = src_root / split_dir.name / cls_dir.name
            out_leaf = out_root / split_dir.name / cls_dir.name
            process_leaf(ref_leaf, src_leaf, out_leaf)

    print("\nDone.")
    print(f"Output: {out_root}/")


if __name__ == "__main__":
    main()
