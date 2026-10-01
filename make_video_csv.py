"""
Build a CSV with columns [video, label] from three category folders.
Each folder contains video files; the folder name becomes the label.

Edit FOLDERS below to point to your three directories.
"""

import csv
import os
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── CONFIG ────────────────────────────────────────────────────────────
FOLDERS = {
    "benign":        "/root/autodl-tmp/suhel/thyroid_nodule/folder1",
    "malignant":     "/root/autodl-tmp/suhel/thyroid_nodule/folder2",
    "indeterminate": "/root/autodl-tmp/suhel/thyroid_nodule/folder3",
}

OUTPUT_CSV = "/root/autodl-tmp/suhel/thyroid_nodule/video_labels.csv"
# ─────────────────────────────────────────────────────────────────────


def main():
    rows = []
    for label, folder in FOLDERS.items():
        folder = Path(folder)
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}")
            continue
        videos = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        print(f"  {label}: {len(videos)} videos  ({folder})")
        for v in videos:
            rows.append({"video": str(v), "label": label})

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
