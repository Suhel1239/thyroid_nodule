"""
Organize image folders into class subfolders using an Excel label file.

Excel file must have:
    FOLDER_COL  — folder name (matches a subfolder under SOURCE_ROOT)
    LABEL_COL   — class label (e.g. benign / malignant)

Output:
    OUTPUT_ROOT/
        benign/
            folder_a/   ← all images from SOURCE_ROOT/folder_a/
        malignant/
            folder_b/
        ...
"""

import shutil
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
EXCEL_FILE  = "/root/autodl-tmp/suhel/thyroid_nodule/labels.xlsx"
SOURCE_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/source_folders"
OUTPUT_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/organized_output"

FOLDER_COL  = "folder_name"   # column name for folder names in the Excel file
LABEL_COL   = "label"         # column name for class labels

SHEET_NAME  = 0               # sheet index or name; 0 = first sheet
# ─────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def main():
    src_root = Path(SOURCE_ROOT).resolve()
    out_root = Path(OUTPUT_ROOT).resolve()

    if not src_root.exists():
        raise SystemExit(f"[ERROR] SOURCE_ROOT not found: {src_root}")

    # ── Read Excel ───────────────────────────────────────────────────
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, dtype=str)
    df.columns = df.columns.str.strip()

    if FOLDER_COL not in df.columns:
        raise SystemExit(f"[ERROR] Column '{FOLDER_COL}' not found in Excel.\n"
                         f"  Available columns: {list(df.columns)}")
    if LABEL_COL not in df.columns:
        raise SystemExit(f"[ERROR] Column '{LABEL_COL}' not found in Excel.\n"
                         f"  Available columns: {list(df.columns)}")

    df = df[[FOLDER_COL, LABEL_COL]].dropna()
    df[FOLDER_COL] = df[FOLDER_COL].str.strip()
    df[LABEL_COL]  = df[LABEL_COL].str.strip()

    print(f"Excel   : {EXCEL_FILE}  ({len(df)} rows)")
    print(f"Source  : {src_root}")
    print(f"Output  : {out_root}")
    print(f"Labels  : {sorted(df[LABEL_COL].unique())}\n")

    copied = 0; skipped = 0; missing = 0

    for _, row in df.iterrows():
        folder_name = row[FOLDER_COL]
        label       = row[LABEL_COL]
        src_folder  = src_root / folder_name

        if not src_folder.exists():
            print(f"  [MISSING] {folder_name}")
            missing += 1
            continue

        imgs = sorted(f for f in src_folder.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        if not imgs:
            print(f"  [NO IMAGES] {folder_name}")
            skipped += 1
            continue

        dst_folder = out_root / label / folder_name
        if dst_folder.exists():
            print(f"  [SKIP already exists] {label}/{folder_name}")
            skipped += 1
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dst_folder / img.name)

        print(f"  {label}/{folder_name}  ({len(imgs)} images)")
        copied += 1

    print(f"\nDone.  copied={copied}  skipped={skipped}  missing={missing}")
    print(f"Output: {out_root}/")


if __name__ == "__main__":
    main()
