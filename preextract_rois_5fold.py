"""
5-Fold cross-validation ROI extraction using RF-DETR.

Step 1 — Pool all sources (videos + cine folders) across every existing
         split for each class, then divide into 5 stratified folds.

Step 2 — For each fold k (0-4):
           test  = fold k
           val   = fold (k+1) % 5
           train = remaining 3 folds

Step 3 — Run RF-DETR ROI extraction on every source in every split/fold
         and save crops to:
           output_root/5fold/fold_{k}/{train|val|test}/{cls}/{stem}/
               frame_0000.jpg ... frame_NNNN.jpg
               manifest.json

Tiered confidence fallback (same as single-run script):
  Tier 0 : conf >= conf_thr  AND  area >= area_min_threshold  →  ROI crop
  Tier 1 : conf in [MODEL_DETECT_THR, conf_thr)  OR  area too small  →  loose crop
  Tier 2 : no detection  →  full-frame resize (last resort)

Always saves exactly n_save frames per source.
"""

import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from rfdetr import RFDETRMedium
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
# DATA_ROOT must contain one subfolder per class, each holding videos/cine folders.
# Example:
#   DATA_ROOT/b/   ← benign
#   DATA_ROOT/n/   ← indeterminate
#   DATA_ROOT/m/   ← malignant
#
# CLASS_MAP maps subfolder name → output label used in the saved directory tree.
DATA_ROOT         = "/root/autodl-tmp/suhel/thyroid_nodule/data"
RFDETR_CHECKPOINT = "/root/autodl-tmp/suhel/thyroid_nodule/RFDETR_for_ROI/single/checkpoint_best_regular.pth"
OUTPUT_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"

CLASS_MAP = {               # subfolder_name : label used in output paths
    "benign":        "benign",
    "malignant":     "malignant",
    "intermediate":  "intermediate",
}

MAX_FRAMES        = 16      # frames sampled per source
N_SAVE            = 16      # frames to keep per source
ROI_SIZE          = 224
CONF_THR          = 0.70    # tier-0 confidence threshold
PAD_FRAC          = 0.10
AREA_FILTER_MODE  = "iqr"   # "iqr" or "mean"
N_FOLDS           = 5
RANDOM_SEED       = 42

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS       = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MODEL_DETECT_THR = 0.25   # low threshold to capture tier-1 fallback detections


# ── Source discovery ──────────────────────────────────────────────────

def collect_all_sources(data_root: Path, class_map: dict) -> dict[str, list]:
    """
    Read directly from data_root/<folder>/ for each entry in class_map.
    Returns {label: [(path, kind, stem), ...]}.
    """
    per_class = {label: [] for label in class_map.values()}

    for folder, label in class_map.items():
        cls_dir = data_root / folder
        if not cls_dir.exists():
            print(f"  [WARNING] folder not found: {cls_dir}")
            continue
        for entry in sorted(cls_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                per_class[label].append((entry, "video", entry.stem))
            elif entry.is_dir():
                has_imgs = any(f.suffix.lower() in IMAGE_EXTS
                               for f in entry.iterdir() if f.is_file())
                if has_imgs:
                    per_class[label].append((entry, "cine", entry.name))

    return per_class


def make_5fold_splits(
    per_class: dict[str, list],
    n_folds:   int  = 5,
    seed:      int  = 42,
) -> list[dict[str, dict[str, list]]]:
    """
    Returns a list of n_folds dicts, each shaped:
      { cls: { "train": [...], "val": [...], "test": [...] } }

    For fold k:
      test  = chunk k
      val   = chunk (k+1) % n_folds
      train = remaining chunks
    """
    rng = random.Random(seed)
    splits = []

    # shuffle and chunk each class independently (stratified)
    class_chunks = {}
    for cls, sources in per_class.items():
        shuffled = list(sources)
        rng.shuffle(shuffled)
        chunks = [shuffled[i::n_folds] for i in range(n_folds)]
        class_chunks[cls] = chunks

    for k in range(n_folds):
        fold = {}
        for cls, chunks in class_chunks.items():
            test_idx  = k
            val_idx   = (k + 1) % n_folds
            train_idx = [i for i in range(n_folds)
                         if i != test_idx and i != val_idx]
            fold[cls] = {
                "test":  chunks[test_idx],
                "val":   chunks[val_idx],
                "train": [s for i in train_idx for s in chunks[i]],
            }
        splits.append(fold)

    return splits


# ── Frame sampling ────────────────────────────────────────────────────

def _sample_video(video_path: Path, max_frames: int):
    cap   = cv2.VideoCapture(str(video_path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, max_frames, dtype=int)
    frames = []
    for seq_idx, vid_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vid_idx))
        ret, bgr = cap.read()
        frames.append((seq_idx, bgr if ret else None))
    cap.release()
    return frames


def _sample_cine(cine_dir: Path, max_frames: int):
    files = sorted(f for f in cine_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    n = len(files)
    if n == 0:
        return []
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    return [(seq_idx, cv2.imread(str(files[int(i)])))
            for seq_idx, i in enumerate(indices)]


# ── Crop helper ───────────────────────────────────────────────────────

def _crop_and_resize(bgr, xyxy, pad_frac, roi_size):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw = int((x2 - x1) * pad_frac);  ph = int((y2 - y1) * pad_frac)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
    crop = bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (roi_size, roi_size)) if crop.size > 0 else None


# ── Per-source processing ─────────────────────────────────────────────

def _process_source(
    source_path, kind, save_dir, model,
    max_frames, n_save, roi_size, conf_thr, pad_frac, area_filter_mode,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists():
        tqdm.write(f"    [skip] {source_path.name}")
        return

    raw_frames = (_sample_video(source_path, max_frames)
                  if kind == "video"
                  else _sample_cine(source_path, max_frames))
    if not raw_frames:
        tqdm.write(f"    {source_path.name}: no frames — skipping")
        return

    # ── Inference ────────────────────────────────────────────────────
    records = []
    for frame_idx, bgr in raw_frames:
        rec = {"frame_idx": frame_idx, "bgr": bgr,
               "conf": None, "area": None, "xyxy": None, "tier": 2}
        if bgr is not None:
            rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb, threshold=MODEL_DETECT_THR)
            if len(detections) > 0:
                best_idx        = int(np.argmax(detections.confidence))
                best_conf       = float(detections.confidence[best_idx])
                x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)
                area            = int((x2 - x1) * (y2 - y1))
                rec.update(conf=best_conf, area=area,
                           xyxy=(x1, y1, x2, y2),
                           tier=(0 if best_conf >= conf_thr else 1))
        records.append(rec)

    # ── Area statistics (tier-0 only) ────────────────────────────────
    areas = np.array([r["area"] for r in records
                      if r["tier"] == 0 and r["area"] is not None], float)
    if len(areas) == 0:
        area_min = mean_area = std_area = Q1 = Q3 = IQR = 0.0
    else:
        mean_area = float(areas.mean());  std_area = float(areas.std())
        Q1 = float(np.percentile(areas, 25));  Q3 = float(np.percentile(areas, 75))
        IQR = Q3 - Q1
        area_min = max(0.0, (Q1 - 1.5 * IQR) if area_filter_mode == "iqr"
                       else (mean_area - std_area))

    for r in records:
        if r["tier"] == 0 and r["area"] is not None and r["area"] < area_min:
            r["tier"] = 1

    # ── Priority list: tier0 → tier1 → tier2, each by conf desc ─────
    tier0 = sorted([r for r in records if r["tier"] == 0],
                   key=lambda r: r["conf"], reverse=True)
    tier1 = sorted([r for r in records if r["tier"] == 1],
                   key=lambda r: r["conf"], reverse=True)
    tier2 = [r for r in records if r["tier"] == 2]
    selected = sorted((tier0 + tier1 + tier2)[:n_save],
                      key=lambda r: r["frame_idx"])

    # ── Save ─────────────────────────────────────────────────────────
    saved_files = [];  saved_confs = [];  saved_tiers = []
    for rec in selected:
        bgr = rec["bgr"]
        if bgr is None:
            continue
        if rec["xyxy"] is not None:
            img = _crop_and_resize(bgr, rec["xyxy"], pad_frac, roi_size)
            if img is None:
                img = cv2.resize(bgr, (roi_size, roi_size))
        else:
            img = cv2.resize(bgr, (roi_size, roi_size))
        fname = f"frame_{rec['frame_idx']:04d}.jpg"
        cv2.imwrite(str(save_dir / fname), img)
        saved_files.append(fname);  saved_confs.append(rec["conf"])
        saved_tiers.append(rec["tier"])

    confs_valid = [c for c in saved_confs if c is not None]
    with open(manifest_path, "w") as f:
        json.dump({
            "source_kind":        kind,
            "frames":             saved_files,
            "saved_tiers":        saved_tiers,
            "total_saved":        len(saved_files),
            "total_sampled":      max_frames,
            "n_save_target":      n_save,
            "n_tier0":            sum(1 for t in saved_tiers if t == 0),
            "n_tier1_conf":       sum(1 for t in saved_tiers if t == 1),
            "n_tier2_nodet":      sum(1 for t in saved_tiers if t == 2),
            "saved_confs":        [round(c, 4) if c else None for c in saved_confs],
            "min_saved_conf":     round(min(confs_valid), 4) if confs_valid else None,
            "max_saved_conf":     round(max(confs_valid), 4) if confs_valid else None,
            "mean_saved_conf":    round(float(np.mean(confs_valid)), 4) if confs_valid else None,
            "area_threshold_px2": round(area_min, 1),
            "mean_area_px2":      round(mean_area, 1),
            "Q1_px2":             round(Q1, 1),
            "Q3_px2":             round(Q3, 1),
            "IQR_px2":            round(IQR, 1),
            "conf_threshold":     conf_thr,
            "area_filter_mode":   area_filter_mode,
        }, f, indent=2)

    tier_str = (f"t0={sum(1 for t in saved_tiers if t==0)} "
                f"t1={sum(1 for t in saved_tiers if t==1)} "
                f"t2={sum(1 for t in saved_tiers if t==2)}")
    tqdm.write(
        f"    {source_path.name} [{kind}]: saved={len(saved_files)}/{n_save}  {tier_str}"
        + (f"  conf=[{min(confs_valid):.3f}-{max(confs_valid):.3f}]"
           if confs_valid else "  no detections")
    )


# ── Main ──────────────────────────────────────────────────────────────

def run(
    data_root:         str  = DATA_ROOT,
    rfdetr_checkpoint: str  = RFDETR_CHECKPOINT,
    output_root:       str  = OUTPUT_ROOT,
    max_frames:        int  = MAX_FRAMES,
    n_save:            int  = N_SAVE,
    roi_size:          int  = ROI_SIZE,
    conf_thr:          float = CONF_THR,
    pad_frac:          float = PAD_FRAC,
    area_filter_mode:  str  = AREA_FILTER_MODE,
    n_folds:           int  = N_FOLDS,
    seed:              int  = RANDOM_SEED,
    class_map:         dict = None,
):
    assert n_save <= max_frames

    if class_map is None:
        class_map = CLASS_MAP

    data_root   = Path(data_root)
    output_root = Path(output_root)

    # ── Pool all sources ─────────────────────────────────────────────
    print(f"Collecting sources from: {data_root}")
    if not data_root.exists():
        raise SystemExit(f"\n[ERROR] DATA_ROOT does not exist: {data_root}\nEdit DATA_ROOT in the CONFIG block.")
    for folder, label in class_map.items():
        cls_dir = data_root / folder
        exists  = "OK" if cls_dir.exists() else "MISSING"
        print(f"  {folder}/ → label '{label}'  [{exists}]")
    per_class = collect_all_sources(data_root, class_map)
    total_sources = sum(len(v) for v in per_class.values())
    if total_sources == 0:
        raise SystemExit(
            f"\n[ERROR] No sources found under {data_root}\n"
            f"Check that DATA_ROOT contains subfolders: {list(class_map.keys())}\n"
            f"Each subfolder should hold video files or cine image folders."
        )
    for cls, srcs in per_class.items():
        n_vid  = sum(1 for _, k, _ in srcs if k == "video")
        n_cine = sum(1 for _, k, _ in srcs if k == "cine")
        print(f"  {cls}: {len(srcs)} total  (videos={n_vid}, cine={n_cine})")

    # ── Build 5-fold splits ──────────────────────────────────────────
    fold_splits = make_5fold_splits(per_class, n_folds=n_folds, seed=seed)
    labels = list(class_map.values())

    print(f"\nFold sizes (train / val / test) per fold:")
    for k, fold in enumerate(fold_splits):
        parts = []
        for cls in labels:
            tr = len(fold[cls]["train"])
            va = len(fold[cls]["val"])
            te = len(fold[cls]["test"])
            parts.append(f"{cls}: {tr}/{va}/{te}")
        print(f"  fold {k}: " + "  |  ".join(parts))

    # ── Save fold manifests (CSV-style JSON for reference) ───────────
    folds_meta_path = output_root / "5fold_splits.json"
    folds_meta = {}
    for k, fold in enumerate(fold_splits):
        folds_meta[f"fold_{k}"] = {
            cls: {
                split: [str(p) for p, _, _ in srcs]
                for split, srcs in splits.items()
            }
            for cls, splits in fold.items()
        }
    folds_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_meta_path, "w") as f:
        json.dump(folds_meta, f, indent=2)
    print(f"\nFold split manifest saved → {folds_meta_path}")

    # ── Load RF-DETR ─────────────────────────────────────────────────
    print(f"\n[RF-DETR] Loading checkpoint: {rfdetr_checkpoint}")
    model = RFDETRMedium.from_checkpoint(rfdetr_checkpoint)
    print(f"[RF-DETR] Model loaded.")
    print(f"  max_frames       : {max_frames}")
    print(f"  n_save           : {n_save}")
    print(f"  conf_thr         : >= {conf_thr:.0%}  (tier 0)")
    print(f"  fallback tier 1  : [{MODEL_DETECT_THR:.0%}, {conf_thr:.0%})")
    print(f"  fallback tier 2  : no detection (full-frame resize)")
    print(f"  area_filter_mode : {area_filter_mode.upper()}\n")


    # ── Extract ROIs for every fold ──────────────────────────────────
    for k, fold in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"  FOLD {k}")
        print(f"{'='*60}")

        for split_name in ("train", "val", "test"):
            for cls in labels:
                sources = fold[cls][split_name]
                if not sources:
                    continue

                n_vid  = sum(1 for _, kind, _ in sources if kind == "video")
                n_cine = sum(1 for _, kind, _ in sources if kind == "cine")
                print(f"\n  fold{k}/{split_name}/{cls}: "
                      f"{len(sources)} sources (vid={n_vid} cine={n_cine})")

                for source_path, kind, stem in tqdm(
                        sources, desc=f"fold{k}/{split_name}/{cls}"):
                    save_dir = output_root / f"fold_{k}" / split_name / cls / stem
                    _process_source(
                        source_path      = source_path,
                        kind             = kind,
                        save_dir         = save_dir,
                        model            = model,
                        max_frames       = max_frames,
                        n_save           = n_save,
                        roi_size         = roi_size,
                        conf_thr         = conf_thr,
                        pad_frac         = pad_frac,
                        area_filter_mode = area_filter_mode,
                    )

    print("\n✅ 5-fold pre-extraction complete.")
    print(f"   Output: {output_root}/")
    print(f"   Structure: fold_{{0-4}}/{{train|val|test}}/{{cls}}/{{stem}}/")


if __name__ == "__main__":
    run()
