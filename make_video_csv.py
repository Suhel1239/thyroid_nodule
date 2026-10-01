"""
Build a CSV with columns [video, label] from three category folders.

Each category folder may contain:
  • video files  (.mp4, .avi, …)  → one row per file
  • sub-folders (cine / DICOM sequences) → one row per sub-folder

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


def _collect_sources(folder: Path):
    """
    Yield paths of all sources inside folder:
      - video files directly in folder
      - sub-folders that contain at least one image file (cine sequences)
    """
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
            yield entry
        elif entry.is_dir():
            has_images = any(f.suffix.lower() in IMAGE_EXTS
                             for f in entry.iterdir() if f.is_file())
            if has_images:
                yield entry


def main():
    rows = []
    for label, folder in FOLDERS.items():
        folder = Path(folder)
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}")
            continue
        sources = list(_collect_sources(folder))
        n_vid  = sum(1 for s in sources if s.is_file())
        n_cine = sum(1 for s in sources if s.is_dir())
        print(f"  {label}: {len(sources)} sources "
              f"(videos={n_vid}, cine_folders={n_cine})  ({folder})")
        for src in sources:
            rows.append({"video": str(src), "label": label})

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
