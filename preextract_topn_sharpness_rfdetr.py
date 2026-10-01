"""
Extract top-N ROI crops per video/cine using RF-DETR detection ranked by
Laplacian sharpness (primary) then confidence (secondary).

Frame selection priority:
  Tier 0 — conf >= conf_thr AND passes area filter  → crop + sharpness ranked
  Tier 1 — conf in [0.25, conf_thr) OR area fails   → crop + sharpness ranked
  Tier 2 — no detection at all                       → full-frame resize (last resort)

Always saves exactly n_save frames per source, named frame_XXXX.jpg.
Output: output_root/rois_{n_save}_rfdetr_sharpness/{split}/{cls}/{stem}/
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from rfdetr import RFDETRMedium
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_DETECT_THR = 0.25


# ── Frame sampling ────────────────────────────────────────────────────

def _sample_frames_from_video(video_path: Path, max_frames: int):
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


def _sample_frames_from_cine(cine_dir: Path, max_frames: int):
    image_files = sorted(
        f for f in cine_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    n = len(image_files)
    if n == 0:
        return []
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    frames = []
    for seq_idx, file_idx in enumerate(indices):
        bgr = cv2.imread(str(image_files[int(file_idx)]))
        frames.append((seq_idx, bgr))
    return frames


# ── Sharpness ─────────────────────────────────────────────────────────

def _laplacian_sharpness(bgr: np.ndarray, xyxy: tuple, pad_frac: float) -> float:
    """Variance of Laplacian on the padded ROI crop — higher = sharper."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _crop_roi(bgr: np.ndarray, xyxy: tuple, pad_frac: float, roi_size: int):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (roi_size, roi_size))


# ── Per-source processing ─────────────────────────────────────────────

def _process_source(
    source_path:       Path,
    kind:              str,
    save_dir:          Path,
    model,
    max_frames:        int,
    n_save:            int,
    roi_size:          int,
    conf_thr:          float,
    pad_frac:          float,
    area_filter_mode:  str,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists():
        tqdm.write(f"  [skip] {source_path.name} already done")
        return

    # ── sample frames ─────────────────────────────────────────────────
    if kind == "video":
        raw_frames = _sample_frames_from_video(source_path, max_frames)
    else:
        raw_frames = _sample_frames_from_cine(source_path, max_frames)

    if not raw_frames:
        tqdm.write(f"  {source_path.name}: no frames — skipping")
        return

    # ── inference: best detection + sharpness per frame ───────────────
    records = []

    for frame_idx, bgr in raw_frames:
        rec = {
            "frame_idx": frame_idx, "bgr": bgr,
            "conf": None, "area": None, "xyxy": None,
            "sharpness": None, "tier": 2,
        }

        if bgr is not None:
            rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb, threshold=MODEL_DETECT_THR)

            if len(detections) > 0:
                best_idx  = int(np.argmax(detections.confidence))
                best_conf = float(detections.confidence[best_idx])
                x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)
                area = int((x2 - x1) * (y2 - y1))

                sharpness = _laplacian_sharpness(bgr, (x1, y1, x2, y2), pad_frac)

                rec["conf"]      = best_conf
                rec["area"]      = area
                rec["xyxy"]      = (x1, y1, x2, y2)
                rec["sharpness"] = sharpness
                rec["tier"]      = 1 if best_conf < conf_thr else 0

        records.append(rec)

    # ── area threshold from tier-0 detections ─────────────────────────
    areas = np.array(
        [r["area"] for r in records if r["tier"] == 0], dtype=float)

    if len(areas) == 0:
        area_min_threshold = 0.0
        mean_area = std_area = Q1 = Q3 = IQR = 0.0
    else:
        mean_area = float(areas.mean())
        std_area  = float(areas.std())
        Q1  = float(np.percentile(areas, 25))
        Q3  = float(np.percentile(areas, 75))
        IQR = Q3 - Q1
        if area_filter_mode == "iqr":
            area_min_threshold = max(0.0, Q1 - 1.5 * IQR)
        else:
            area_min_threshold = max(0.0, mean_area - std_area)

    for r in records:
        if r["tier"] == 0 and r["area"] < area_min_threshold:
            r["tier"] = 1   # demote to fallback

    # ── rank within each tier: sharpness DESC, then confidence DESC ────
    tier0 = sorted(
        [r for r in records if r["tier"] == 0],
        key=lambda r: (r["sharpness"], r["conf"]), reverse=True)
    tier1 = sorted(
        [r for r in records if r["tier"] == 1],
        key=lambda r: (r["sharpness"], r["conf"]), reverse=True)
    tier2 = [r for r in records if r["tier"] == 2]

    priority = tier0 + tier1 + tier2

    # ── select exactly n_save and re-sort temporally ───────────────────
    selected = sorted(priority[:n_save], key=lambda r: r["frame_idx"])

    # ── save crops ────────────────────────────────────────────────────
    saved_files  = []
    saved_confs  = []
    saved_sharp  = []
    saved_tiers  = []

    for rec in selected:
        bgr = rec["bgr"]
        if bgr is None:
            continue

        if rec["xyxy"] is not None:
            img = _crop_roi(bgr, rec["xyxy"], pad_frac, roi_size)
            if img is None:
                img = cv2.resize(bgr, (roi_size, roi_size))
        else:
            img = cv2.resize(bgr, (roi_size, roi_size))

        fname = f"frame_{rec['frame_idx']:04d}.jpg"
        cv2.imwrite(str(save_dir / fname), img)
        saved_files.append(fname)
        saved_confs.append(rec["conf"])
        saved_sharp.append(rec["sharpness"])
        saved_tiers.append(rec["tier"])

    n_kept = len(saved_files)
    confs_v = [c for c in saved_confs if c is not None]
    sharp_v = [s for s in saved_sharp if s is not None]

    with open(manifest_path, "w") as f:
        json.dump({
            "source_kind"        : kind,
            "frames"             : saved_files,
            "saved_tiers"        : saved_tiers,
            "total_saved"        : n_kept,
            "total_sampled"      : max_frames,
            "n_save_target"      : n_save,
            "n_tier0"            : sum(1 for t in saved_tiers if t == 0),
            "n_tier1_fallback"   : sum(1 for t in saved_tiers if t == 1),
            "n_tier2_nodet"      : sum(1 for t in saved_tiers if t == 2),
            "saved_confs"        : [round(c, 4) if c is not None else None
                                    for c in saved_confs],
            "saved_sharpness"    : [round(s, 4) if s is not None else None
                                    for s in saved_sharp],
            "min_sharpness"      : round(min(sharp_v), 4) if sharp_v else None,
            "max_sharpness"      : round(max(sharp_v), 4) if sharp_v else None,
            "mean_sharpness"     : round(float(np.mean(sharp_v)), 4) if sharp_v else None,
            "min_conf"           : round(min(confs_v), 4) if confs_v else None,
            "max_conf"           : round(max(confs_v), 4) if confs_v else None,
            "area_threshold_px2" : round(area_min_threshold, 1),
            "mean_area_px2"      : round(mean_area, 1),
            "Q1_px2"             : round(Q1, 1),
            "Q3_px2"             : round(Q3, 1),
            "conf_threshold"     : conf_thr,
            "area_filter_mode"   : area_filter_mode,
        }, f, indent=2)

    tier_str = (f"tier0={sum(1 for t in saved_tiers if t==0)} "
                f"tier1={sum(1 for t in saved_tiers if t==1)} "
                f"tier2={sum(1 for t in saved_tiers if t==2)}")
    tqdm.write(
        f"  {source_path.name} [{kind}]: saved={n_kept}/{n_save}  {tier_str}"
        + (f"  sharpness=[{min(sharp_v):.1f}-{max(sharp_v):.1f}]"
           if sharp_v else "")
    )


# ── Main entry point ──────────────────────────────────────────────────

def preextract_rois(
    data_root:         str,
    rfdetr_checkpoint: str,
    output_root:       str,
    max_frames:        int   = 32,
    n_save:            int   = 32,
    roi_size:          int   = 224,
    conf_thr:          float = 0.60,
    pad_frac:          float = 0.10,
    area_filter_mode:  str   = "iqr",
):
    assert n_save <= max_frames, \
        f"n_save ({n_save}) must be <= max_frames ({max_frames})"

    print(f"[RF-DETR] Loading checkpoint: {rfdetr_checkpoint}")
    model = RFDETRMedium.from_checkpoint(rfdetr_checkpoint)
    print(f"[RF-DETR] Model loaded.")
    print(f"  max_frames       : {max_frames}")
    print(f"  n_save           : {n_save}")
    print(f"  conf_thr         : >= {conf_thr:.0%}  (tier 0)")
    print(f"  ranking          : Laplacian sharpness DESC, then confidence DESC")
    print(f"  area_filter_mode : {area_filter_mode.upper()}\n")

    data_root   = Path(data_root)
    output_root = Path(output_root)

    for split in ("train", "val", "test"):
        for cls in ("benign", "malignant", "indeterminate"):
            cls_dir = data_root / split / cls
            if not cls_dir.exists():
                continue

            sources = []
            for entry in sorted(cls_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                    sources.append((entry, "video", entry.stem))
                elif entry.is_dir():
                    has_images = any(
                        f.suffix.lower() in IMAGE_EXTS
                        for f in entry.iterdir() if f.is_file()
                    )
                    if has_images:
                        sources.append((entry, "cine", entry.name))

            if not sources:
                continue

            n_vid  = sum(1 for _, k, _ in sources if k == "video")
            n_cine = sum(1 for _, k, _ in sources if k == "cine")
            print(f"{split}/{cls}: {len(sources)} sources "
                  f"(videos={n_vid}, cine_folders={n_cine})")

            for source_path, kind, stem in tqdm(sources, desc=f"{split}/{cls}"):
                save_dir = (output_root
                            / f"rois_{n_save}_rfdetr_sharpness"
                            / split / cls / stem)
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

    print("\n✅ Pre-extraction complete.")


if __name__ == "__main__":
    preextract_rois(
        data_root         = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        rfdetr_checkpoint = "/root/autodl-tmp/suhel/thyroid_nodule/RFDETR_for_ROI/single/checkpoint_best_regular.pth",
        output_root       = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        max_frames        = 100,
        n_save            = 32,
        roi_size          = 224,
        conf_thr          = 0.70,
        pad_frac          = 0.10,
        area_filter_mode  = "iqr",
    )
