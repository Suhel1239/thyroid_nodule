"""
Interactive frame browser — play videos / cine folders and press S to save.

Controls:
  S        — save current frame as {source_name}.jpg
  SPACE    — pause / resume
  D / →    — step forward one frame (while paused)
  A / ←    — step backward one frame (while paused)
  N        — next source
  P        — previous source
  Q / ESC  — quit

Usage:
  python frame_browser.py --root /path/to/root --output /path/to/saved_frames
"""

import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

DELAY_MS   = 60    # ~16 fps default; increase to slow down

# ── Configure these paths ─────────────────────────────────────────────
ROOT_DIR   = "/root/autodl-tmp/suhel/thyroid_nodule/extracted_videos_all"
OUTPUT_DIR = "/root/autodl-tmp/suhel/thyroid_nodule/saved_best_frames"
# ─────────────────────────────────────────────────────────────────────


# ── Source discovery ──────────────────────────────────────────────────

def discover_sources(root: Path) -> list[tuple[str, Path, str]]:
    """
    Walk root (one level of subfolders allowed).
    Returns list of (kind, path, stem):
      kind = "video"  → path is a video file,  stem = filename without ext
      kind = "cine"   → path is an image dir,   stem = folder name
    """
    sources = []

    def _scan_dir(d: Path):
        entries = sorted(d.iterdir())
        for e in entries:
            if e.is_file() and e.suffix.lower() in VIDEO_EXTS:
                sources.append(("video", e, e.stem))
            elif e.is_dir():
                imgs = [f for f in e.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                if imgs:
                    sources.append(("cine", e, e.name))

    _scan_dir(root)
    # also check one level of subfolders
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            _scan_dir(sub)

    # deduplicate while preserving order
    seen = set()
    unique = []
    for item in sources:
        key = str(item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# ── Frame loading ─────────────────────────────────────────────────────

def load_video_frames(video_path: Path) -> list[np.ndarray]:
    cap    = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(bgr)
    cap.release()
    return frames


def load_cine_frames(cine_dir: Path) -> list[np.ndarray]:
    image_files = sorted(
        f for f in cine_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    frames = []
    for p in image_files:
        bgr = cv2.imread(str(p))
        if bgr is not None:
            frames.append(bgr)
    return frames


# ── OSD overlay ───────────────────────────────────────────────────────

def _put_text(img, text, pos, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_osd(frame: np.ndarray, source_name: str, frame_idx: int,
             total: int, paused: bool, saved: bool) -> np.ndarray:
    img = frame.copy()
    H, W = img.shape[:2]

    status = "PAUSED" if paused else "PLAYING"
    color  = (0, 255, 255) if paused else (0, 255, 0)
    saved_tag = "  [SAVED]" if saved else ""

    _put_text(img, f"{source_name}{saved_tag}", (10, 24), color=(200, 200, 255))
    _put_text(img, f"Frame {frame_idx + 1}/{total}  {status}", (10, 48),
              color=color)
    _put_text(img, "S=save  SPACE=pause  A/D=step  N/P=source  Q=quit",
              (10, H - 10), scale=0.45, color=(180, 180, 180))
    return img


# ── Main browser loop ─────────────────────────────────────────────────

def browse(sources: list, output_dir: Path, delay_ms: int):
    if not sources:
        print("No sources found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    win = "Frame Browser"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    src_idx   = 0
    frames    = []
    frame_idx = 0
    paused    = False
    saved     = False
    needs_load = True

    while True:
        # ── load source ───────────────────────────────────────────────
        if needs_load:
            kind, path, stem = sources[src_idx]
            print(f"\n[{src_idx + 1}/{len(sources)}] Loading {kind}: {path.name}")
            if kind == "video":
                frames = load_video_frames(path)
            else:
                frames = load_cine_frames(path)

            if not frames:
                print(f"  Warning: no frames loaded for {path.name}, skipping.")
                src_idx = (src_idx + 1) % len(sources)
                continue

            frame_idx  = 0
            saved      = False
            needs_load = False
            print(f"  {len(frames)} frames  |  S=save  SPACE=pause  N=next  P=prev  Q=quit")

        # ── display ───────────────────────────────────────────────────
        display = draw_osd(frames[frame_idx], stem,
                           frame_idx, len(frames), paused, saved)
        cv2.imshow(win, display)

        # ── key handling ──────────────────────────────────────────────
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

        if key in (ord('q'), 27):           # Q / ESC → quit
            break

        elif key == ord('s'):               # S → save current frame
            out_path = output_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_path), frames[frame_idx])
            saved = True
            print(f"  Saved → {out_path}")

        elif key == ord(' '):               # SPACE → toggle pause
            paused = not paused

        elif key in (ord('d'), 83):         # D / → → step forward
            frame_idx = min(frame_idx + 1, len(frames) - 1)

        elif key in (ord('a'), 81):         # A / ← → step backward
            frame_idx = max(frame_idx - 1, 0)

        elif key == ord('n'):               # N → next source
            src_idx    = (src_idx + 1) % len(sources)
            needs_load = True

        elif key == ord('p'):              # P → previous source
            src_idx    = (src_idx - 1) % len(sources)
            needs_load = True

        else:
            # advance frame automatically when playing
            if not paused:
                frame_idx += 1
                if frame_idx >= len(frames):
                    frame_idx = 0   # loop

    cv2.destroyAllWindows()
    print("\nDone.")


def main():
    root   = Path(ROOT_DIR).resolve()
    output = Path(OUTPUT_DIR).resolve()

    if not root.exists():
        sys.exit(f"Root does not exist: {root}")

    print(f"Scanning: {root}")
    sources = discover_sources(root)
    print(f"Found {len(sources)} sources "
          f"({sum(1 for k,_,_ in sources if k=='video')} videos, "
          f"{sum(1 for k,_,_ in sources if k=='cine')} cine folders)")
    print(f"Saving frames to: {output}\n")

    browse(sources, output, DELAY_MS)


if __name__ == "__main__":
    main()
