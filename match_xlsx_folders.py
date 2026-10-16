"""
Match XLSX column entries against files/subfolders in multiple folders.

Matching rule
-------------
File or subfolder name before the first '_' must equal the column value.
  column value : "10006399578"
  file name    : "10006399578_anon_vid_001.mp4"  → match ✓
  folder name  : "10006399578_scans"             → match ✓

Output
------
Prints per-folder: matched count + list of matched names.
Also saves a summary CSV to the same directory as the xlsx.
"""

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
XLSX_FILE   = "/path/to/your_file.xlsx"
SHEET_NAME  = 0                          # 0 = first sheet, or use sheet name
COLUMN_NAME = "name"                     # column header that contains the names

# Folders to search (add as many as needed)
SEARCH_FOLDERS = [
    "/path/to/folder1",
    "/path/to/folder2",
]
# ──────────────────────────────────────────────────────────────────────────────


def key_from_name(name: str) -> str:
    """Everything before the first underscore."""
    return name.split("_")[0]


def load_column_values(xlsx_path: str, sheet, col: str) -> set:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    values = set(str(v).strip() for v in df[col].dropna())
    print(f"[XLSX] {len(values)} unique values loaded from column '{col}'")
    return values


def search_folder(folder_path: str, target_values: set):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[WARN] Folder not found: {folder_path}")
        return []

    matched = []
    for entry in sorted(folder.iterdir()):
        key = key_from_name(entry.name)
        if key in target_values:
            matched.append(entry.name)

    return matched


def main():
    values = load_column_values(XLSX_FILE, SHEET_NAME, COLUMN_NAME)

    results = {}   # folder_path → [matched names]
    for folder in SEARCH_FOLDERS:
        matched = search_folder(folder, values)
        results[folder] = matched

    # ── Print results ──────────────────────────────────────────────────────────
    print()
    total = 0
    for folder, matched in results.items():
        print(f"Folder : {folder}")
        print(f"  Matched : {len(matched)}")
        for name in matched:
            print(f"    {name}")
        print()
        total += len(matched)

    print(f"Total matched across all folders: {total}")

    # ── Save summary CSV ───────────────────────────────────────────────────────
    rows = []
    for folder, matched in results.items():
        for name in matched:
            rows.append({"folder": folder, "matched_file": name,
                         "key": key_from_name(name)})

    out_csv = Path(XLSX_FILE).with_suffix(".match_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Summary saved to: {out_csv}")


if __name__ == "__main__":
    main()
