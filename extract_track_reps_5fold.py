"""
Nodule tracking + sharpest-middle-frame extraction for regular video files.

Walks:
    DATA_ROOT/train/benign/<video.mp4>
    DATA_ROOT/train/malignant/<video.mp4>
    DATA_ROOT/val/...
    DATA_ROOT/test/...

For each video:
  1. Run RF-DETR detector on every frame (sampled at FRAME_STEP).
  2. Track detections across frames with ByteTrack.
  3. Score every detection's ROI sharpness (variance of Laplacian).
  4. For eligible tracks (min size + min observations), pick the sharpest
     frame from the middle 20% of the track.
  5. Save the cropped ROI + a manifest JSON.

Output mirrors the same split/class structure:
    OUTPUT_ROOT/train/benign/<video_stem>/
        track_0001_best_frame_000042.jpg
        manifest.json
    OUTPUT_ROOT/train/malignant/...
"""

import cv2
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT   = "/root/autodl-tmp/suhel/thyroid_nodule/data_5fold/fold_0"
OUTPUT_ROOT = "/root/autodl-tmp/suhel/thyroid_nodule/track_reps/fold_0"
RFDETR_CHECKPOINT = "/root/autodl-tmp/suhel/thyroid_nodule/RFDETR_for_ROI/single/checkpoint_best_regular.pth"

FRAME_STEP              = 1      # process every Nth frame (1 = all frames)
DETECTOR_THRESHOLD      = 0.10   # RF-DETR detection confidence threshold
TRACKER_THRESHOLD       = 0.25   # ByteTrack activation threshold
MIN_TRACK_LONG_AXIS_PX  = 80.0   # reject tracks whose median bbox longest side < this
MIN_TRACK_OBSERVATIONS  = 10     # reject tracks seen in fewer frames than this
ROI_SIZE                = 224    # output crop size
PAD_FRAC                = 0.10   # padding around bbox before cropping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
SPLITS     = ("train", "val", "test")
CLASSES    = ("benign", "malignant")


# ── Helpers ───────────────────────────────────────────────────────────

def _sharpness(bgr: np.ndarray, xyxy=None, pad_frac: float = PAD_FRAC) -> float:
    if xyxy is not None:
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw = int((x2 - x1) * pad_frac); ph = int((y2 - y1) * pad_frac)
        x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
        x2 = min(W, x2 + pw); y2 = min(H, y2 + ph)
        region = bgr[y1:y2, x1:x2]
        if region.size == 0:
            region = bgr
    else:
        region = bgr
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _crop_roi(bgr: np.ndarray, xyxy, pad_frac: float, size: int) -> np.ndarray:
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw = int((x2 - x1) * pad_frac); ph = int((y2 - y1) * pad_frac)
    x1 = max(0, x1 - pw); y1 = max(0, y1 - ph)
    x2 = min(W, x2 + pw); y2 = min(H, y2 + ph)
    crop = bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size)) if crop.size > 0 else cv2.resize(bgr, (size, size))


def _middle_observations(ordered: list, frac_lo=0.4, frac_hi=0.6) -> list:
    n = len(ordered)
    start = math.floor(n * frac_lo)
    stop  = min(n, max(start + 1, math.ceil(n * frac_hi)))
    return ordered[start:stop]


def _track_long_axis(xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return max(x2 - x1, y2 - y1)


# ── Per-video pipeline ────────────────────────────────────────────────

def process_video(video_path: Path, save_dir: Path, model, tracker_cls):
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "manifest.json"
    if manifest_path.exists():
        tqdm.write(f"    [skip] {video_path.name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        tqdm.write(f"    [ERROR] cannot open {video_path.name}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    import supervision as sv
    tracker = tracker_cls(
        track_activation_threshold=TRACKER_THRESHOLD,
        frame_rate=fps / FRAME_STEP,
    )

    # ── Frame-by-frame detection + tracking ──────────────────────────
    track_obs: dict[int, list] = defaultdict(list)   # track_id → list of obs
    frame_index = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = model.predict(rgb, threshold=DETECTOR_THRESHOLD,
                             include_source_image=False)
        if isinstance(dets, list):
            dets = dets[0]

        # build supervision Detections
        if len(dets) > 0:
            xyxy_arr = dets.xyxy.astype(np.float32)
            conf_arr = dets.confidence.astype(np.float32)
            cls_arr  = (dets.class_id if hasattr(dets, "class_id")
                        else np.zeros(len(conf_arr), np.int32))
        else:
            xyxy_arr = np.zeros((0, 4), np.float32)
            conf_arr = np.zeros(0, np.float32)
            cls_arr  = np.zeros(0, np.int32)

        sv_dets = sv.Detections(xyxy=xyxy_arr, confidence=conf_arr,
                                class_id=cls_arr)
        tracked = tracker.update_with_detections(sv_dets)

        # store each tracked detection with its frame + sharpness
        for i in range(len(tracked)):
            tid  = int(tracked.tracker_id[i])
            xyxy = tuple(tracked.xyxy[i].astype(int))
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            sharp = _sharpness(bgr, xyxy)
            track_obs[tid].append({
                "frame_index": frame_index,
                "xyxy":        xyxy,
                "confidence":  conf,
                "sharpness":   sharp,
                "bgr":         bgr.copy(),    # keep frame for saving
            })

        frame_index += 1

    cap.release()

    if not track_obs:
        tqdm.write(f"    {video_path.name}: no tracked detections")
        with open(manifest_path, "w") as f:
            json.dump({"status": "no_tracked_detection", "saved": []}, f, indent=2)
        return

    # ── Select best representative per eligible track ─────────────────
    saved_files  = []
    track_summaries = []

    for tid, obs in sorted(track_obs.items()):
        ordered = sorted(obs, key=lambda x: x["frame_index"])
        median_long = float(np.median([_track_long_axis(o["xyxy"]) for o in ordered]))

        eligible = (median_long >= MIN_TRACK_LONG_AXIS_PX and
                    len(ordered) >= MIN_TRACK_OBSERVATIONS)

        track_summaries.append({
            "track_id":            tid,
            "observation_count":   len(ordered),
            "median_long_axis_px": round(median_long, 2),
            "eligible":            eligible,
        })

        if not eligible:
            continue

        midpoint = (ordered[0]["frame_index"] + ordered[-1]["frame_index"]) / 2
        middle   = _middle_observations(ordered)

        chosen = max(middle, key=lambda x: (
            x["sharpness"],
            x["confidence"],
            -abs(x["frame_index"] - midpoint),
            -x["frame_index"],
        ))

        roi = _crop_roi(chosen["bgr"], chosen["xyxy"], PAD_FRAC, ROI_SIZE)
        fname = f"track_{tid:04d}_best_frame_{chosen['frame_index']:06d}.jpg"
        cv2.imwrite(str(save_dir / fname), roi)

        saved_files.append({
            "file":        fname,
            "track_id":    tid,
            "frame_index": chosen["frame_index"],
            "sharpness":   round(chosen["sharpness"], 4),
            "confidence":  round(chosen["confidence"], 4),
            "xyxy":        list(chosen["xyxy"]),
            "track_observations":       len(ordered),
            "middle_observations_used": len(middle),
            "median_long_axis_px":      round(median_long, 2),
        })

    status = "selected" if saved_files else "no_eligible_track"
    with open(manifest_path, "w") as f:
        json.dump({
            "status":         status,
            "video":          video_path.name,
            "total_frames":   frame_index,
            "frame_step":     FRAME_STEP,
            "track_summaries": track_summaries,
            "saved":          saved_files,
            "config": {
                "detector_threshold":     DETECTOR_THRESHOLD,
                "tracker_threshold":      TRACKER_THRESHOLD,
                "min_track_long_axis_px": MIN_TRACK_LONG_AXIS_PX,
                "min_track_observations": MIN_TRACK_OBSERVATIONS,
                "roi_size":               ROI_SIZE,
                "selection_rule": (
                    "middle 20% of track observations, "
                    "ranked: sharpness > confidence > proximity to midpoint"
                ),
            },
        }, f, indent=2)

    tqdm.write(
        f"    {video_path.name}: "
        f"tracks={len(track_obs)}  eligible={len(saved_files)}  "
        f"status={status}"
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    data_root   = Path(DATA_ROOT).resolve()
    output_root = Path(OUTPUT_ROOT).resolve()

    if not data_root.exists():
        raise SystemExit(f"[ERROR] DATA_ROOT not found: {data_root}")

    print(f"Input  : {data_root}")
    print(f"Output : {output_root}")
    print(f"\n[RF-DETR] Loading: {RFDETR_CHECKPOINT}")

    from rfdetr import RFDETRMedium
    import supervision as sv
    model = RFDETRMedium.from_checkpoint(RFDETR_CHECKPOINT)
    print("[RF-DETR] Loaded.\n")

    for split in SPLITS:
        for cls in CLASSES:
            src_cls = data_root / split / cls
            if not src_cls.exists():
                continue

            videos = sorted(v for v in src_cls.iterdir()
                            if v.is_file() and v.suffix.lower() in VIDEO_EXTS)
            if not videos:
                continue

            rel = f"{split}/{cls}"
            print(f"\n  {rel}: {len(videos)} videos")

            for video_path in tqdm(videos, desc=rel):
                save_dir = output_root / split / cls / video_path.stem
                process_video(video_path, save_dir, model, sv.ByteTrack)

    print("\n✅ Done.")
    print(f"   Output: {output_root}/")


if __name__ == "__main__":
    main()
