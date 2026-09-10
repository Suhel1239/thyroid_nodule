"""
Organize image folders into class subfolders using an Excel label file.

Excel file must have:
    FOLDER_COL  — folder name (matches a subfolder under SOURCE_ROOT)
    LABEL_COL   — class label (e.g. benign / malignant)

Output:
    OUTPUT_ROOT/
        benign/
            folder_a__img001.jpg   ← images flat, prefixed with folder name
            folder_a__img002.jpg
        malignant/
            folder_b__img001.jpg
        ...
"""

import shutil
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
EXCEL_FILE  = "/Users/suhelkhan/PyCharmMiscProject/thyroid_nodule/TTSH_nodule_images/koios AI final analysis for medical board.xlsx"
SOURCE_ROOT = "/Users/suhelkhan/PyCharmMiscProject/thyroid_nodule/TTSH_nodule_images/nodule_images_TTSH"
OUTPUT_ROOT = "/Users/suhelkhan/PyCharmMiscProject/thyroid_nodule/TTSH_nodule_images/TTSH_images_classified"

FOLDER_COL  = "nodulecode"            # column name for folder names in the Excel file
LABEL_COL   = "benignniftporca"       # column name for class labels

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

        dst_dir = out_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        n_copied = 0
        for img in imgs:
            # prefix image filename with folder name: foldername__original.jpg
            new_name = f"{folder_name}__{img.name}"
            dst = dst_dir / new_name
            if dst.exists():
                continue
            shutil.copy2(img, dst)
            n_copied += 1

        if n_copied == 0:
            print(f"  [SKIP all exist] {label}/{folder_name}")
            skipped += 1
        else:
            print(f"  {label}/  ← {folder_name}  ({n_copied} images)")
            copied += 1

    print(f"\nDone.  copied={copied}  skipped={skipped}  missing={missing}")
    print(f"Output: {out_root}/")


if __name__ == "__main__":
    main()
