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

MODEL_DETECT_THR = 0.25   # low threshold so we capture fallback detections too


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


def _crop_and_resize(bgr, xyxy, pad_frac, roi_size):
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

    # ── SAMPLE FRAMES ─────────────────────────────────────────────────
    if kind == "video":
        raw_frames = _sample_frames_from_video(source_path, max_frames)
    else:
        raw_frames = _sample_frames_from_cine(source_path, max_frames)

    if not raw_frames:
        tqdm.write(f"  {source_path.name}: no frames found — skipping")
        return

    # ── INFERENCE: collect best detection per frame ────────────────────
    # tier 0 = conf >= conf_thr (preferred)
    # tier 1 = conf in [MODEL_DETECT_THR, conf_thr) (fallback)
    # tier 2 = no detection (last resort, full-frame crop)
    records = []   # all frames, with metadata

    for frame_idx, bgr in raw_frames:
        rec = {"frame_idx": frame_idx, "bgr": bgr,
               "conf": None, "area": None, "xyxy": None, "tier": 2}

        if bgr is not None:
            rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(rgb, threshold=MODEL_DETECT_THR)

            if len(detections) > 0:
                best_idx  = int(np.argmax(detections.confidence))
                best_conf = float(detections.confidence[best_idx])
                x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)
                area = int((x2 - x1) * (y2 - y1))

                rec["conf"] = best_conf
                rec["area"] = area
                rec["xyxy"] = (x1, y1, x2, y2)
                rec["tier"] = 1 if best_conf < conf_thr else 0

        records.append(rec)

    # ── AREA STATISTICS (on tier-0 detections only) ────────────────────
    areas = np.array(
        [r["area"] for r in records if r["tier"] == 0],
        dtype=float,
    )

    if len(areas) == 0:
        area_min_threshold = 0.0
        mean_area = std_area = Q1 = Q3 = IQR = 0.0
    else:
        mean_area = float(areas.mean())
        std_area  = float(areas.std())
        Q1        = float(np.percentile(areas, 25))
        Q3        = float(np.percentile(areas, 75))
        IQR       = Q3 - Q1
        if area_filter_mode == "iqr":
            area_min_threshold = max(0.0, Q1 - 1.5 * IQR)
        else:
            area_min_threshold = max(0.0, mean_area - std_area)

    # Demote tier-0 frames that fail the area filter → tier 1
    for r in records:
        if r["tier"] == 0 and r["area"] < area_min_threshold:
            r["tier"] = 1

    # ── BUILD PRIORITY LIST ────────────────────────────────────────────
    # Sort each tier by conf descending; tier 2 has no conf so just append.
    tier0 = sorted([r for r in records if r["tier"] == 0],
                   key=lambda r: r["conf"], reverse=True)
    tier1 = sorted([r for r in records if r["tier"] == 1],
                   key=lambda r: r["conf"], reverse=True)
    tier2 = [r for r in records if r["tier"] == 2]

    priority = tier0 + tier1 + tier2   # best → worst

    # ── SELECT EXACTLY n_save FRAMES ──────────────────────────────────
    selected = priority[:n_save]

    # Re-sort temporally for consistent filenames
    selected_ordered = sorted(selected, key=lambda r: r["frame_idx"])

    # ── SAVE CROPS ────────────────────────────────────────────────────
    saved_files  = []
    saved_confs  = []
    saved_tiers  = []
    n_skipped    = 0

    for rec in selected_ordered:
        bgr = rec["bgr"]
        if bgr is None:
            n_skipped += 1
            continue

        if rec["xyxy"] is not None:
            img = _crop_and_resize(bgr, rec["xyxy"], pad_frac, roi_size)
            if img is None:
                img = cv2.resize(bgr, (roi_size, roi_size))
        else:
            # tier 2: no detection — resize full frame
            img = cv2.resize(bgr, (roi_size, roi_size))

        fname = f"frame_{rec['frame_idx']:04d}.jpg"
        cv2.imwrite(str(save_dir / fname), img)
        saved_files.append(fname)
        saved_confs.append(rec["conf"])
        saved_tiers.append(rec["tier"])

    n_kept = len(saved_files)

    # ── MANIFEST ──────────────────────────────────────────────────────
    confs_valid = [c for c in saved_confs if c is not None]
    with open(manifest_path, "w") as f:
        json.dump({
            "source_kind"        : kind,
            "frames"             : saved_files,
            "saved_tiers"        : saved_tiers,
            "total_saved"        : n_kept,
            "total_sampled"      : max_frames,
            "n_save_target"      : n_save,
            "n_tier0"            : sum(1 for t in saved_tiers if t == 0),
            "n_tier1_conf"       : sum(1 for t in saved_tiers if t == 1),
            "n_tier2_nodet"      : sum(1 for t in saved_tiers if t == 2),
            "n_skipped_bad_bgr"  : n_skipped,
            "saved_confs"        : [round(c, 4) if c is not None else None
                                    for c in saved_confs],
            "min_saved_conf"     : round(min(confs_valid), 4) if confs_valid else None,
            "max_saved_conf"     : round(max(confs_valid), 4) if confs_valid else None,
            "mean_saved_conf"    : round(float(np.mean(confs_valid)), 4) if confs_valid else None,
            "area_threshold_px2" : round(area_min_threshold, 1),
            "mean_area_px2"      : round(mean_area, 1),
            "Q1_px2"             : round(Q1, 1),
            "Q3_px2"             : round(Q3, 1),
            "IQR_px2"            : round(IQR, 1),
            "conf_threshold"     : conf_thr,
            "area_filter_mode"   : area_filter_mode,
        }, f, indent=2)

    tier_str = (f"tier0={sum(1 for t in saved_tiers if t==0)} "
                f"tier1={sum(1 for t in saved_tiers if t==1)} "
                f"tier2={sum(1 for t in saved_tiers if t==2)}")
    tqdm.write(
        f"  {source_path.name} [{kind}]: saved={n_kept}/{n_save}  {tier_str}"
        + (f"  conf=[{min(confs_valid):.3f}-{max(confs_valid):.3f}]"
           if confs_valid else "")
    )


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
    print(f"  fallback tier 1  : [{MODEL_DETECT_THR:.0%}, {conf_thr:.0%})")
    print(f"  fallback tier 2  : no detection (full-frame resize)")
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
                            / f"rois_{n_save}_rfdetr_topn_withareafiltering_3class"
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
        max_frames        = 32,
        n_save            = 32,
        roi_size          = 224,
        conf_thr          = 0.70,
        pad_frac          = 0.10,
        area_filter_mode  = "iqr",
    )
