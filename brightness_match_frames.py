"""
Match brightness of frame folders to a reference video.

Steps:
  1. Sample frames from the reference video → compute median luminance (target).
  2. For each subfolder in root that contains images:
       a. Compute median luminance of the folder's frames (source).
       b. Apply a global gain  gain = target / source  to every frame.
       c. Save to output_root/<same relative path>/<filename>.
  3. Video files in root are ignored entirely.

Hardcoded paths — edit the CONFIG block below before running.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
ROOT_DIR        = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
OUTPUT_ROOT_DIR = "/root/autodl-tmp/suhel/thyroid_nodule/brightness_matched_frames"
REFERENCE_VIDEO = "/root/autodl-tmp/suhel/thyroid_nodule/reference.mp4"

# How many frames to sample from the reference video for the target luminance
REF_SAMPLE_FRAMES = 64
# How many frames to sample from each folder for its source luminance estimate
SRC_SAMPLE_FRAMES = 32
# ─────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _median_luminance_video(video_path: Path, n_samples: int) -> float:
    """Sample n_samples frames uniformly from a video, return median V-channel mean."""
    cap   = cv2.VideoCapture(str(video_path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    lums = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, bgr = cap.read()
        if ret:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lums.append(float(hsv[:, :, 2].mean()))
    cap.release()
    return float(np.median(lums)) if lums else 128.0


def _median_luminance_folder(image_files: list, n_samples: int) -> float:
    """Sample n_samples images uniformly from list, return median V-channel mean."""
    total   = max(len(image_files), 1)
    indices = np.linspace(0, total - 1, n_samples, dtype=int)
    lums = []
    for idx in indices:
        bgr = cv2.imread(str(image_files[int(idx)]))
        if bgr is not None:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lums.append(float(hsv[:, :, 2].mean()))
    return float(np.median(lums)) if lums else 128.0


def _build_gamma_lut(gamma: float) -> np.ndarray:
    """Pre-compute uint8 lookup table for gamma correction."""
    table = np.array([
        min(255, int(255.0 * (i / 255.0) ** gamma + 0.5))
        for i in range(256)
    ], dtype=np.uint8)
    return table


def _apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma to V channel via LUT.
    gamma < 1 → brightens; blacks stay black, contrast preserved."""
    lut = _build_gamma_lut(gamma)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = lut[hsv[:, :, 2]]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process(root: Path, output_root: Path, ref_video: Path):
    print(f"Reference video : {ref_video}")
    target_lum = _median_luminance_video(ref_video, REF_SAMPLE_FRAMES)
    print(f"Target luminance: {target_lum:.2f}\n")

    # Collect immediate and nested subfolders that contain images (no video processing)
    frame_folders = []
    for d in sorted(root.rglob("*")):
        if d.is_dir():
            imgs = sorted(f for f in d.iterdir()
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if imgs:
                frame_folders.append((d, imgs))

    if not frame_folders:
        print("No image folders found under root.")
        return

    print(f"Found {len(frame_folders)} frame folder(s).\n")

    for folder, image_files in frame_folders:
        rel = folder.relative_to(root)
        out_dir = output_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        src_lum = _median_luminance_folder(image_files, SRC_SAMPLE_FRAMES)
        # gamma = log(target/255) / log(src/255); gamma<1 brightens without lifting blacks
        eps   = 1e-6
        gamma = (np.log(target_lum / 255.0 + eps) /
                 np.log(src_lum    / 255.0 + eps))
        gamma = float(np.clip(gamma, 0.2, 5.0))

        print(f"{rel}  src_lum={src_lum:.2f}  gamma={gamma:.3f}  ({len(image_files)} frames)")

        for img_path in tqdm(image_files, desc=str(rel), leave=False):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            out_bgr = _apply_gamma(bgr, gamma)
            cv2.imwrite(str(out_dir / img_path.name), out_bgr)

    print("\nDone.")


if __name__ == "__main__":
    root      = Path(ROOT_DIR).resolve()
    out_root  = Path(OUTPUT_ROOT_DIR).resolve()
    ref_video = Path(REFERENCE_VIDEO).resolve()

    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    if not ref_video.exists():
        raise SystemExit(f"Reference video not found: {ref_video}")

    process(root, out_root, ref_video)
