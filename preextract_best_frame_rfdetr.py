import os
import cv2
import numpy as np
from pathlib import Path
from rfdetr import RFDETRMedium
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _run_rfdetr_on_frames(frames_bgr, model, conf_thr, conf_save_thr, roi_size, out_path):
    """
    Common logic: iterate BGR frames, apply RF-DETR, save best frame.
    Returns True if saved, False if zero detections.
    """
    best_fallback_conf  = -1.0
    best_fallback_frame = None
    saved = False

    for bgr in frames_bgr:
        rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(rgb, threshold=conf_thr)

        if len(detections) == 0:
            continue

        best_idx  = int(np.argmax(detections.confidence))
        best_conf = float(detections.confidence[best_idx])

        if best_conf > best_fallback_conf:
            best_fallback_conf  = best_conf
            best_fallback_frame = bgr.copy()

        if best_conf >= conf_save_thr:
            cv2.imwrite(str(out_path), cv2.resize(bgr, (roi_size, roi_size)))
            saved = True
            break

    if not saved:
        if best_fallback_frame is not None:
            cv2.imwrite(str(out_path),
                        cv2.resize(best_fallback_frame, (roi_size, roi_size)))
            return True, best_fallback_conf  # saved via fallback
        return False, 0.0

    return True, conf_save_thr  # saved via threshold


def _iter_video_frames(video_path: Path, max_frames: int):
    """Yield up to max_frames uniformly sampled BGR frames from a video file."""
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
    """Yield up to max_frames uniformly sampled BGR frames from a cine image folder."""
    image_files = sorted(
        p for p in cine_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    total = max(len(image_files), 1)
    indices = np.linspace(0, total - 1, max_frames, dtype=int)
    for idx in indices:
        bgr = cv2.imread(str(image_files[int(idx)]))
        if bgr is not None:
            yield bgr


def preextract_rois(
    data_root:         str,
    rfdetr_checkpoint: str,
    output_root:       str,
    max_frames:        int   = 32,
    roi_size:          int   = 224,
    conf_thr:          float = 0.25,
    conf_save_thr:     float = 0.80,
    pad_frac:          float = 0.10,
):
    """
    For every video file AND cine frame folder in
        data_root/{train_3_class,val_3_class,test_3_class}/{benign,malignant}/:

      - Samples up to max_frames uniformly
      - Runs RF-DETR frame by frame
      - FIRST PRIORITY: if best detection >= conf_save_thr, save that frame and stop
      - FALLBACK: save the frame with the highest confidence seen (even below threshold)
      - WARNING: if zero detections across all frames, nothing is saved

    Output structure:
        output_root/best_frame_rfdetr/{split}/{cls}/{stem}.jpg
        (stem = video filename without extension, or cine folder name)
    """
    print(f"[RF-DETR] Loading checkpoint: {rfdetr_checkpoint}")
    model = RFDETRMedium.from_checkpoint(rfdetr_checkpoint)
    print("[RF-DETR] Model loaded.\n")

    data_root   = Path(data_root)
    output_root = Path(output_root)

    for split in ("train_3_class", "val_3_class", "test_3_class"):
        for cls in ("benign", "malignant"):

            cls_dir = data_root / split / cls
            if not cls_dir.exists():
                continue

            save_dir = output_root / "best_frame_rfdetr" / split / cls
            save_dir.mkdir(parents=True, exist_ok=True)

            # Collect all sources: video files + cine folders
            sources = []
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    sources.append(("video", p, p.stem))
                elif p.is_dir() and any(
                        f.suffix.lower() in IMAGE_EXTS for f in p.iterdir()):
                    sources.append(("cine", p, p.name))

            print(f"\n{split}/{cls}: {len(sources)} sources "
                  f"({sum(1 for k,_,_ in sources if k=='video')} videos, "
                  f"{sum(1 for k,_,_ in sources if k=='cine')} cines)")

            for kind, src_path, stem in tqdm(sources, desc=f"{split}/{cls}"):

                out_path = save_dir / f"{stem}.jpg"
                if out_path.exists():
                    continue

                if kind == "video":
                    frames = list(_iter_video_frames(src_path, max_frames))
                else:
                    frames = list(_iter_cine_frames(src_path, max_frames))

                if not frames:
                    print(f"  [WARNING] '{stem}' — could not read any frames.")
                    continue

                saved, conf = _run_rfdetr_on_frames(
                    frames, model, conf_thr, conf_save_thr, roi_size, out_path)

                if not saved:
                    print(f"  [WARNING] '{stem}' — "
                          f"zero detections across all {len(frames)} sampled frames. "
                          f"No frame saved.")
                elif conf < conf_save_thr:
                    print(f"  [fallback] '{stem}' ({kind}) — "
                          f"no frame >= {conf_save_thr:.2f}, "
                          f"saved best frame at conf={conf:.4f}")

    print("\n✅ Pre-extraction complete.")


# ── RUN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preextract_rois(
        data_root         = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        rfdetr_checkpoint = "/root/autodl-tmp/suhel/thyroid_nodule/RFDETR_for_ROI/single/checkpoint_best_regular.pth",
        output_root       = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all",
        max_frames        = 100,
        roi_size          = 224,
        conf_thr          = 0.50,
        conf_save_thr     = 0.90,
        pad_frac          = 0.10,
    )
