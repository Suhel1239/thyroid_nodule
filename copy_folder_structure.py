"""
Copy files from SOURCE_ROOT matching names found in REFERENCE_ROOT,
preserving the same train/val/test / benign/malignant folder structure.

SOURCE_ROOT does NOT need to mirror the reference structure — all files and
cine folders are discovered recursively and looked up purely by name.

Reference folder  → defines WHICH filenames/folder-names to copy
Source folder     → flat or arbitrary structure; searched recursively for matches
Output folder     → destination, mirroring the reference structure

CONFIG:
    REFERENCE_ROOT  — folder whose filenames are used as the pick-list
    SOURCE_ROOT     — folder searched recursively for matching names
    OUTPUT_ROOT     — destination (created if absent)
"""

import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
REFERENCE_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all/train_3_class"
SOURCE_ROOT    = "/root/autodl-tmp/suhel/thyroid_nodule/enhanced_malignant"
OUTPUT_ROOT    = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all/train_3_cl_with_enh_mal"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_source_map(src_root: Path) -> dict[str, Path]:
    """
    Recursively scan src_root and build {name: path} for every video file
    and every cine folder (a directory that directly contains image files).
    If a name appears more than once the first occurrence wins and a warning
    is printed.
    """
    src_map: dict[str, Path] = {}

    def _warn_dup(name, existing, new):
        print(f"  [WARNING] duplicate name '{name}' in source — "
              f"keeping {existing}, ignoring {new}")

    for entry in sorted(src_root.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
            if entry.name in src_map:
                _warn_dup(entry.name, src_map[entry.name], entry)
            else:
                src_map[entry.name] = entry
        elif entry.is_dir():
            has_imgs = any(f.suffix.lower() in IMAGE_EXTS
                           for f in entry.iterdir() if f.is_file())
            if has_imgs:
                if entry.name in src_map:
                    _warn_dup(entry.name, src_map[entry.name], entry)
                else:
                    src_map[entry.name] = entry

    return src_map


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


def process_leaf(ref_leaf: Path, src_map: dict[str, Path], out_leaf: Path, out_root: Path):
    """
    For every entry in ref_leaf, look it up in the flat src_map by name
    and copy into out_leaf.
    """
    if not ref_leaf.exists():
        return
    ref_names = {e.name for e in ref_leaf.iterdir()}
    if not ref_names:
        return

    found = 0
    missing = []
    for name in sorted(ref_names):
        if name in src_map:
            copy_entry(src_map[name], out_leaf)
            found += 1
        else:
            missing.append(name)

    rel = out_leaf.relative_to(out_root)
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

    print("Building flat source map (scanning SOURCE_ROOT recursively)...")
    src_map = build_source_map(src_root)
    print(f"  Found {len(src_map)} unique entries in source.\n")

    # Walk reference structure: cls / cine_folder
    # (reference is cls-level dirs containing cine folders, no train/val/test split)
    cls_dirs = sorted(d for d in ref_root.iterdir() if d.is_dir())
    if not cls_dirs:
        raise SystemExit("[ERROR] No subdirectories found in REFERENCE_ROOT")

    for cls_dir in cls_dirs:          # benign / malignant / ...
        out_leaf = out_root / cls_dir.name
        process_leaf(cls_dir, src_map, out_leaf, out_root)

    print("\nDone.")
    print(f"Output: {out_root}/")


if __name__ == "__main__":
    main()
