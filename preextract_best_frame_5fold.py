"""
RF-DETR best-frame extraction for a 5-fold data directory.
Saves exactly ONE whole frame per video file or cine image folder.

Frame selection rule
--------------------
1. Sample up to MAX_FRAMES uniformly from the source.
2. Run RF-DETR on each sampled frame.
3. Among frames whose best-detection confidence >= CONF_SAVE_THR,
   pick the one with the highest sharpness (variance of Laplacian on the
   detection ROI; full frame if no detection).
4. If no frame clears CONF_SAVE_THR, fall back to the frame with the
   highest detection confidence (any detections >= CONF_THR threshold).
5. If zero detections across all sampled frames, save the sharpest
   whole frame (fallback-of-last-resort, prints a WARNING).

Input structure (DATA_ROOT):
    data_5fold/
        fold_0/train/benign/<video.mp4 or cine_folder/>
        fold_0/train/malignant/...
        fold_0/val/...  fold_0/test/...
        fold_1/...  ...  fold_4/...

Output structure (OUTPUT_ROOT):
    best_frame_5fold/
        fold_0/train/benign/<stem>.jpg   ← one .jpg per source
        fold_0/train/malignant/...
        ...
"""

import os
import cv2
import numpy as np
from pathlib import Path
from rfdetr import RFDETRMedium
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT         = "/root/autodl-tmp/suhel/thyroid_nodule/data_5fold"
OUTPUT_ROOT       = "/root/autodl-tmp/suhel/thyroid_nodule/best_frame_5fold"
RFDETR_CHECKPOINT = "/root/autodl-tmp/suhel/thyroid_nodule/RFDETR_for_ROI/single/checkpoint_best_regular.pth"

MAX_FRAMES    = 100    # uniformly sampled frames per source
CONF_THR      = 0.25   # RF-DETR inference threshold (keep detections above this)
CONF_SAVE_THR = 0.60   # prefer frames whose best detection conf >= this
PAD_FRAC      = 0.10   # padding around bbox when computing sharpness

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ── Helpers ───────────────────────────────────────────────────────────

def _sharpness(bgr: np.ndarray, xyxy=None) -> float:
    """Variance of Laplacian on the detection ROI (or full frame if no box)."""
    if xyxy is not None:
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = xyxy
        pw = int((x2 - x1) * PAD_FRAC)
        ph = int((y2 - y1) * PAD_FRAC)
        x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
        x2 = min(W, x2 + pw);  y2 = min(H, y2 + ph)
        region = bgr[y1:y2, x1:x2]
        if region.size == 0:
            region = bgr
    else:
        region = bgr
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _iter_video_frames(video_path: Path, max_frames: int):
    cap   = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, max_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, bgr = cap.read()
        if ret:
            yield bgr
    cap.release()


def _iter_cine_frames(cine_dir: Path, max_frames: int):
    files = sorted(p for p in cine_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        return
    indices = np.linspace(0, len(files) - 1, max_frames, dtype=int)
    for idx in indices:
        bgr = cv2.imread(str(files[int(idx)]))
        if bgr is not None:
            yield bgr


# ── Per-source processing ─────────────────────────────────────────────

def _pick_best_frame(frames, model):
    """
    Returns (bgr, conf, mode_str) for the best frame, or (None, 0, 'no_frames').

    mode:
      'high_conf'   – best sharpness among frames with conf >= CONF_SAVE_THR
      'fallback'    – no frame cleared CONF_SAVE_THR; best conf frame used
      'no_det'      – zero detections; sharpest whole frame used
      'no_frames'   – input was empty
    """
    if not frames:
        return None, 0.0, "no_frames"

    high_pool = []   # (sharpness, conf, bgr) for frames >= CONF_SAVE_THR
    any_pool  = []   # (conf, sharpness, bgr)  for all frames with any detection

    for bgr in frames:
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = model.predict(rgb, threshold=CONF_THR)

        if len(dets) == 0:
            continue

        best_idx        = int(np.argmax(dets.confidence))
        best_conf       = float(dets.confidence[best_idx])
        x1, y1, x2, y2 = dets.xyxy[best_idx].astype(int)
        xyxy            = (x1, y1, x2, y2)
        sharp           = _sharpness(bgr, xyxy)

        any_pool.append((best_conf, sharp, bgr))

        if best_conf >= CONF_SAVE_THR:
            high_pool.append((sharp, best_conf, bgr))

    if high_pool:
        sharp, conf, bgr = max(high_pool, key=lambda t: (t[0], t[1]))
        return bgr, conf, "high_conf"

    if any_pool:
        conf, sharp, bgr = max(any_pool, key=lambda t: (t[0], t[1]))
        return bgr, conf, "fallback"

    # No detections at all — save sharpest whole frame
    best_bgr   = max(frames, key=lambda b: _sharpness(b))
    return best_bgr, 0.0, "no_det"


def _process_source(src_path: Path, kind: str, out_path: Path, model):
    if out_path.exists():
        tqdm.write(f"    [skip] {src_path.name}")
        return

    frames = list(_iter_video_frames(src_path, MAX_FRAMES)
                  if kind == "video"
                  else _iter_cine_frames(src_path, MAX_FRAMES))

    if not frames:
        tqdm.write(f"    [WARN] {src_path.name}: no readable frames — skipped")
        return

    bgr, conf, mode = _pick_best_frame(frames, model)

    if bgr is None:
        tqdm.write(f"    [WARN] {src_path.name}: _pick_best_frame returned None — skipped")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)

    tag = {"high_conf": "", "fallback": "  [fallback: best conf]",
           "no_det":    "  [WARNING: no detections — sharpest frame saved]"}.get(mode, "")
    tqdm.write(
        f"    {src_path.name} [{kind}]:  conf={conf:.3f}  mode={mode}"
        f"  frames_sampled={len(frames)}{tag}"
        f"  → {out_path.name}"
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
    model = RFDETRMedium.from_checkpoint(RFDETR_CHECKPOINT)
    print("[RF-DETR] Loaded.")
    print(f"  max_frames    : {MAX_FRAMES}")
    print(f"  conf_thr      : {CONF_THR}  (inference threshold)")
    print(f"  conf_save_thr : {CONF_SAVE_THR}  (prefer frames above this)")
    print(f"  frame saved   : ONE whole frame per source (no ROI crop)\n")

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
                out_cls_dir = output_root / fold_dir.name / split_dir.name / cls_dir.name
                print(f"\n  {rel}: {len(sources)} sources")

                for src_path, kind, stem in tqdm(sources, desc=rel):
                    out_path = out_cls_dir / f"{stem}.jpg"
                    _process_source(src_path, kind, out_path, model)

    print("\n✅ Done.")
    print(f"   Output: {output_root}/")
    print(f"   One .jpg per source, whole frame (no ROI crop).")


if __name__ == "__main__":
    main()
