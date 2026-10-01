"""
Build a CSV with columns [video, label, type] from three category folders.

Each category folder may contain:
  • .mp4 / video files directly            → type = "video"
  • sub-folders whose name has "video"     → type = "stanford"
  • sub-folders whose name has "transverse"→ type = "pocus"
  • other sub-folders (frames/images)      → type = "cine"

Sub-folders can themselves contain frames (images) or video files.
The `video` column holds just the file/folder name, not the full path.

Edit FOLDERS below to point to your three directories.
"""

import csv
import os
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ── CONFIG ────────────────────────────────────────────────────────────
FOLDERS = {
    "benign":        "/root/autodl-tmp/suhel/thyroid_nodule/folder1",
    "malignant":     "/root/autodl-tmp/suhel/thyroid_nodule/folder2",
    "indeterminate": "/root/autodl-tmp/suhel/thyroid_nodule/folder3",
}

OUTPUT_CSV = "/root/autodl-tmp/suhel/thyroid_nodule/video_labels.csv"
# ─────────────────────────────────────────────────────────────────────


def _get_type(entry: Path) -> str:
    """Determine type from entry kind and name."""
    if entry.is_file():
        return "video"
    name_lower = entry.name.lower()
    if "video" in name_lower:
        return "stanford"
    if "transverse" in name_lower:
        return "pocus"
    return "cine"


def _has_content(folder: Path) -> bool:
    """True if folder contains at least one image or video file."""
    return any(f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
               for f in folder.iterdir() if f.is_file())


def _collect_sources(folder: Path):
    """Yield (entry, type) for each source inside the category folder."""
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
            yield entry, _get_type(entry)
        elif entry.is_dir() and _has_content(entry):
            yield entry, _get_type(entry)


def main():
    rows = []
    for label, folder in FOLDERS.items():
        folder = Path(folder)
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}")
            continue
        sources = list(_collect_sources(folder))
        counts = {}
        for _, t in sources:
            counts[t] = counts.get(t, 0) + 1
        print(f"  {label}: {len(sources)} sources  {counts}  ({folder})")
        for src, src_type in sources:
            rows.append({"video": src.name, "label": label, "type": src_type})

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label", "type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
