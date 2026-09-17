"""
RF-DETR ROI extraction over an existing data_5fold directory.
Frame selection uses sharpness (variance of Laplacian on the ROI crop)
as the primary ranking criterion, inspired by Dr Shu Wen's nodule-tracking
pipeline.  Within each tier, frames are ranked:
    1. sharpness (desc)   ← variance of Laplacian on the crop / full frame
    2. confidence (desc)
    3. earlier frame index (asc)
Tier priority is still  tier0 → tier1 → tier2  as a fallback pool.

Input  (DATA_ROOT):
    data_5fold/
        fold_0/train/benign/<video or cine folder>
        fold_0/train/indeterminate/...
        fold_0/val/...  fold_0/test/...
        fold_1/ ... fold_4/

Output (OUTPUT_ROOT) — identical structure, ROI crops instead of raw files:
    rois_5fold_sharp/
        fold_0/train/benign/<stem>/frame_XXXX.jpg  manifest.json
        ...

Tiered confidence fallback:
  Tier 0 : conf >= conf_thr  AND  area >= area_min_threshold  →  ROI crop
  Tier 1 : conf in [MODEL_DETECT_THR, conf_thr)  OR  area too small  →  loose crop
  Tier 2 : no detection  →  full-frame resize (last resort)

Always saves exactly n_save frames per source, ranked by sharpness.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from rfdetr import RFDETRMedium
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT         = "/root/autodl-tmp/suhel/thyroid_nodule/data_5fold"
OUTPUT_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/rois_5fold_sharp"
RFDETR_CHECKPOINT = "/root/autodl-tmp/suhel/thyroid_nodule/RFDETR_for_ROI/single/checkpoint_best_regular.pth"

MAX_FRAMES        = 32   # uniformly sampled frames to run inference on
N_SAVE            = 32   # top-N sharpest frames to keep
ROI_SIZE          = 224
CONF_THR          = 0.70
PAD_FRAC          = 0.10
AREA_FILTER_MODE  = "iqr"   # "iqr" or "mean"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS       = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MODEL_DETECT_THR = 0.25


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
    if not files:
        return []
    indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
    return [(seq_idx, cv2.imread(str(files[int(i)])))
            for seq_idx, i in enumerate(indices)]


# ── Crop helper ───────────────────────────────────────────────────────

def _crop(bgr, xyxy, pad_frac, roi_size):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw = int((x2 - x1) * pad_frac);  ph = int((y2 - y1) * pad_frac)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
    crop = bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (roi_size, roi_size)) if crop.size > 0 else None


# ── Sharpness (variance of Laplacian on the ROI region) ──────────────

def _sharpness(bgr: np.ndarray, xyxy=None, pad_frac: float = PAD_FRAC) -> float:
    """
    Compute variance-of-Laplacian sharpness on the ROI crop.
    If xyxy is None, uses the full frame.
    Higher = sharper / more in-focus.
    """
    if xyxy is not None:
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw = int((x2 - x1) * pad_frac);  ph = int((y2 - y1) * pad_frac)
        x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
        x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
        region = bgr[y1:y2, x1:x2]
        if region.size == 0:
            region = bgr
    else:
        region = bgr
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Per-source processing ─────────────────────────────────────────────

def _process_source(source_path, kind, save_dir, model):
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists():
        tqdm.write(f"    [skip] {source_path.name}")
        return

    raw_frames = (_sample_video(source_path, MAX_FRAMES)
                  if kind == "video"
                  else _sample_cine(source_path, MAX_FRAMES))
    if not raw_frames:
        tqdm.write(f"    {source_path.name}: no frames — skipping")
        return

    # ── Inference + sharpness scoring ────────────────────────────────
    records = []
    for frame_idx, bgr in raw_frames:
        rec = {"frame_idx": frame_idx, "bgr": bgr,
               "conf": None, "area": None, "xyxy": None,
               "tier": 2, "sharpness": 0.0}
        if bgr is not None:
            rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb, threshold=MODEL_DETECT_THR)
            if len(detections) > 0:
                best_idx        = int(np.argmax(detections.confidence))
                best_conf       = float(detections.confidence[best_idx])
                x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)
                xyxy = (x1, y1, x2, y2)
                rec.update(conf=best_conf,
                           area=int((x2 - x1) * (y2 - y1)),
                           xyxy=xyxy,
                           tier=(0 if best_conf >= CONF_THR else 1))
            # sharpness: on ROI crop if detected, else full frame
            rec["sharpness"] = round(_sharpness(bgr, rec["xyxy"]), 4)
        records.append(rec)

    # ── Area stats (tier-0 only) ──────────────────────────────────────
    areas = np.array([r["area"] for r in records
                      if r["tier"] == 0 and r["area"] is not None], float)
    if len(areas) == 0:
        area_min = mean_area = 0.0
    else:
        mean_area = float(areas.mean())
        Q1  = float(np.percentile(areas, 25))
        Q3  = float(np.percentile(areas, 75))
        IQR = Q3 - Q1
        area_min = max(0.0, (Q1 - 1.5 * IQR) if AREA_FILTER_MODE == "iqr"
                       else (mean_area - float(areas.std())))

    for r in records:
        if r["tier"] == 0 and r["area"] is not None and r["area"] < area_min:
            r["tier"] = 1

    # ── Rank by sharpness (primary), conf (secondary), frame idx (tiebreak)
    # Pool: tier0 first, then tier1, then tier2 as last resort
    def _sort_key(r):
        return (-r["sharpness"], -(r["conf"] or 0.0), r["frame_idx"])

    tier0 = sorted([r for r in records if r["tier"] == 0], key=_sort_key)
    tier1 = sorted([r for r in records if r["tier"] == 1], key=_sort_key)
    tier2 = sorted([r for r in records if r["tier"] == 2], key=_sort_key)

    selected = sorted((tier0 + tier1 + tier2)[:N_SAVE],
                      key=lambda r: r["frame_idx"])

    # ── Save ──────────────────────────────────────────────────────────
    saved_files = []; saved_confs = []; saved_tiers = []; saved_sharp = []
    for rec in selected:
        bgr = rec["bgr"]
        if bgr is None:
            continue
        if rec["xyxy"] is not None:
            img = _crop(bgr, rec["xyxy"], PAD_FRAC, ROI_SIZE)
            if img is None:
                img = cv2.resize(bgr, (ROI_SIZE, ROI_SIZE))
        else:
            img = cv2.resize(bgr, (ROI_SIZE, ROI_SIZE))
        fname = f"frame_{rec['frame_idx']:04d}.jpg"
        cv2.imwrite(str(save_dir / fname), img)
        saved_files.append(fname)
        saved_confs.append(rec["conf"])
        saved_tiers.append(rec["tier"])
        saved_sharp.append(rec["sharpness"])

    confs_valid = [c for c in saved_confs if c is not None]
    sharp_valid = [s for s in saved_sharp if s is not None]
    with open(manifest_path, "w") as f:
        json.dump({
            "source_kind":           kind,
            "frames":                saved_files,
            "saved_tiers":           saved_tiers,
            "saved_sharpness":       saved_sharp,
            "total_saved":           len(saved_files),
            "total_sampled":         MAX_FRAMES,
            "n_save_target":         N_SAVE,
            "n_tier0":               sum(1 for t in saved_tiers if t == 0),
            "n_tier1_conf":          sum(1 for t in saved_tiers if t == 1),
            "n_tier2_nodet":         sum(1 for t in saved_tiers if t == 2),
            "saved_confs":           [round(c, 4) if c else None for c in saved_confs],
            "min_saved_conf":        round(min(confs_valid), 4) if confs_valid else None,
            "max_saved_conf":        round(max(confs_valid), 4) if confs_valid else None,
            "mean_saved_conf":       round(float(np.mean(confs_valid)), 4) if confs_valid else None,
            "min_sharpness":         round(min(sharp_valid), 4) if sharp_valid else None,
            "max_sharpness":         round(max(sharp_valid), 4) if sharp_valid else None,
            "mean_sharpness":        round(float(np.mean(sharp_valid)), 4) if sharp_valid else None,
            "area_threshold_px2":    round(area_min, 1),
            "mean_area_px2":         round(mean_area, 1),
            "conf_threshold":        CONF_THR,
            "area_filter_mode":      AREA_FILTER_MODE,
            "selection_rule":        "tier0 > tier1 > tier2; within each tier: sharpness desc, conf desc, frame_idx asc",
        }, f, indent=2)

    tier_str = (f"t0={sum(1 for t in saved_tiers if t==0)} "
                f"t1={sum(1 for t in saved_tiers if t==1)} "
                f"t2={sum(1 for t in saved_tiers if t==2)}")
    sharp_str = (f"  sharp=[{min(sharp_valid):.1f}-{max(sharp_valid):.1f}]"
                 if sharp_valid else "")
    tqdm.write(
        f"    {source_path.name} [{kind}]: saved={len(saved_files)}/{N_SAVE}  {tier_str}"
        + (f"  conf=[{min(confs_valid):.3f}-{max(confs_valid):.3f}]"
           if confs_valid else "  no detections")
        + sharp_str
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data_root   = Path(DATA_ROOT).resolve()
    output_root = Path(OUTPUT_ROOT).resolve()

    if not data_root.exists():
        raise SystemExit(f"[ERROR] DATA_ROOT not found: {data_root}")

    print(f"Input  : {data_root}")
    print(f"Output : {output_root}")
    print(f"\n[RF-DETR] Loading checkpoint: {RFDETR_CHECKPOINT}")
    model = RFDETRMedium.from_checkpoint(RFDETR_CHECKPOINT)
    print(f"[RF-DETR] Model loaded.")
    print(f"  max_frames : {MAX_FRAMES}  |  n_save : {N_SAVE}")
    print(f"  conf_thr   : >= {CONF_THR:.0%} (tier0)  "
          f"fallback: [{MODEL_DETECT_THR:.0%},{CONF_THR:.0%}) (tier1)  no-det (tier2)")
    print(f"  selection  : sharpness (VoL on ROI) > confidence > frame_idx\n")

    fold_dirs = sorted(d for d in data_root.iterdir()
                       if d.is_dir() and d.name.startswith("fold_"))
    if not fold_dirs:
        raise SystemExit(f"[ERROR] No fold_* folders found in {data_root}")

    for fold_dir in fold_dirs:
        print(f"\n{'='*60}")
        print(f"  {fold_dir.name}")
        print(f"{'='*60}")

        for split_dir in sorted(fold_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue

                sources = []
                for entry in sorted(cls_dir.iterdir()):
                    if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                        sources.append((entry, "video", entry.stem))
                    elif entry.is_dir():
                        has_imgs = any(f.suffix.lower() in IMAGE_EXTS
                                       for f in entry.iterdir() if f.is_file())
                        if has_imgs:
                            sources.append((entry, "cine", entry.name))

                if not sources:
                    continue

                rel = f"{fold_dir.name}/{split_dir.name}/{cls_dir.name}"
                print(f"\n  {rel}: {len(sources)} sources")

                for src_path, kind, stem in tqdm(sources, desc=rel):
                    save_dir = (output_root / fold_dir.name
                                / split_dir.name / cls_dir.name / stem)
                    _process_source(src_path, kind, save_dir, model)

    print("\n✅ Done.")
    print(f"   Output: {output_root}/")
    print(f"   Structure mirrors: {data_root}/")


if __name__ == "__main__":
    main()
