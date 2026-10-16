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
MIN_SAVE_CONF           = 0.50   # skip source entirely if best frame conf < this
MIN_TRACK_LONG_AXIS_PX  = 80.0   # reject tracks whose median bbox longest side < this
MIN_TRACK_OBSERVATIONS  = 10     # reject tracks seen in fewer frames than this
ROI_SIZE                = 224    # output crop size
PAD_FRAC                = 0.10   # padding around bbox before cropping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
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


# ── Frame streaming (video file or cine folder) ───────────────────────

def _stream_frames(source: Path, kind: str):
    """
    Yields (frame_index, bgr) for every FRAME_STEP-th frame.
    kind: "video" | "cine"
    Also returns estimated fps.
    """
    if kind == "video":
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return 25.0, iter([])
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        def _gen():
            idx = 0
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                if idx % FRAME_STEP == 0:
                    yield idx, bgr
                idx += 1
            cap.release()

        return fps, _gen()

    else:  # cine folder
        files = sorted(f for f in source.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS)

        def _gen():
            for idx, fpath in enumerate(files):
                if idx % FRAME_STEP != 0:
                    continue
                bgr = cv2.imread(str(fpath))
                if bgr is not None:
                    yield idx, bgr

        return 25.0, _gen()   # cine folders have no real fps; 25 is fine for tracker


# ── Per-source pipeline ───────────────────────────────────────────────

def process_source(source: Path, kind: str, save_dir: Path, model, tracker_cls):
    # Output is a single .jpg named after the source; skip if it already exists.
    out_jpg = save_dir.parent / f"{save_dir.name}.jpg"
    if out_jpg.exists():
        tqdm.write(f"    [skip] {source.name}")
        return

    fps, frame_gen = _stream_frames(source, kind)
    if frame_gen is None:
        tqdm.write(f"    [ERROR] cannot open {source.name}")
        return

    import supervision as sv
    tracker = tracker_cls(
        track_activation_threshold=TRACKER_THRESHOLD,
        frame_rate=fps / FRAME_STEP,
    )

    # ── Frame-by-frame detection + tracking ──────────────────────────
    track_obs: dict[int, list] = defaultdict(list)

    for frame_index, bgr in frame_gen:
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = model.predict(rgb, threshold=DETECTOR_THRESHOLD,
                             include_source_image=False)
        if isinstance(dets, list):
            dets = dets[0]

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

        for i in range(len(tracked)):
            tid   = int(tracked.tracker_id[i])
            xyxy  = tuple(int(v) for v in tracked.xyxy[i])
            conf  = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            sharp = _sharpness(bgr, xyxy)
            track_obs[tid].append({
                "frame_index": frame_index,
                "xyxy":        xyxy,
                "confidence":  conf,
                "sharpness":   sharp,
                "bgr":         bgr.copy(),
            })

    # ── Flatten all observations for a global fallback pool ─────────
    all_obs = [o for obs in track_obs.values() for o in obs]

    # ── Pick the single best frame across all eligible tracks ────────
    best_chosen = None
    best_score  = None

    for tid, obs in sorted(track_obs.items()):
        ordered = sorted(obs, key=lambda x: x["frame_index"])
        median_long = float(np.median([_track_long_axis(o["xyxy"]) for o in ordered]))

        if (median_long < MIN_TRACK_LONG_AXIS_PX or
                len(ordered) < MIN_TRACK_OBSERVATIONS):
            continue

        midpoint = (ordered[0]["frame_index"] + ordered[-1]["frame_index"]) / 2
        middle   = _middle_observations(ordered)

        chosen = max(middle, key=lambda x: (
            x["sharpness"],
            x["confidence"],
            -abs(x["frame_index"] - midpoint),
        ))

        score = (chosen["sharpness"], chosen["confidence"])
        if best_score is None or score > best_score:
            best_score  = score
            best_chosen = chosen

    fallback = False
    if best_chosen is None or best_chosen["confidence"] < MIN_SAVE_CONF:
        # Fall back to highest-confidence frame across all tracked observations
        if all_obs:
            best_chosen = max(all_obs, key=lambda x: (x["confidence"], x["sharpness"]))
            fallback = True
        else:
            tqdm.write(f"    {source.name}: no tracked detections — skipped")
            return

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), best_chosen["bgr"])

    tqdm.write(
        f"    {source.name} [{kind}]: "
        f"tracks={len(track_obs)}  "
        f"frame={best_chosen['frame_index']}  "
        f"sharp={best_chosen['sharpness']:.1f}  "
        f"conf={best_chosen['confidence']:.3f}"
        + ("  [fallback: highest conf]" if fallback else "")
        + f"  → {out_jpg.name}"
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

    # Collect all video files and cine folders anywhere under data_root
    sources = []
    for entry in sorted(data_root.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
            rel_parent = entry.parent.relative_to(data_root)
            # save_dir.parent / save_dir.name + ".jpg" is the output file
            save_dir   = output_root / rel_parent / entry.stem
            sources.append((entry, "video", save_dir))
        elif entry.is_dir() and entry != data_root:
            has_imgs = any(f.suffix.lower() in IMAGE_EXTS
                           for f in entry.iterdir() if f.is_file())
            if has_imgs:
                # skip directories that are themselves inside a cine folder
                parent_has_imgs = any(f.suffix.lower() in IMAGE_EXTS
                                      for f in entry.parent.iterdir() if f.is_file())
                if parent_has_imgs:
                    continue
                rel_parent = entry.parent.relative_to(data_root)
                save_dir   = output_root / rel_parent / entry.name
                sources.append((entry, "cine", save_dir))

    if not sources:
        raise SystemExit(f"[ERROR] No video files or cine folders found under {data_root}")

    n_vid  = sum(1 for _, k, _ in sources if k == "video")
    n_cine = sum(1 for _, k, _ in sources if k == "cine")
    print(f"\nFound {len(sources)} sources  (videos={n_vid}, cine={n_cine})\n")

    for src_path, kind, save_dir in tqdm(sources, desc="sources"):
        process_source(src_path, kind, save_dir, model, sv.ByteTrack)

    print("\n✅ Done.")
    print(f"   Output: {output_root}/")


if __name__ == "__main__":
    main()
