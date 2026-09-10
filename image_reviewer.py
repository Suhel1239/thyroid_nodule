"""
Interactive image reviewer.
  SPACE / RIGHT  → next image (keep)
  D              → delete current image
  LEFT           → previous image
  Q / ESC        → quit

Usage:
    python image_reviewer.py /path/to/folder
"""

import sys
import os
from pathlib import Path
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

WINDOW = "Image Reviewer  [SPACE=next  D=delete  LEFT=prev  Q=quit]"


def load(path: Path, win_w: int = 1280, win_h: int = 800):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(win_w / w, win_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def main():
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".")
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    images = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {folder}")

    print(f"Folder : {folder}")
    print(f"Images : {len(images)}")
    print(f"Keys   : SPACE/→ next  |  D delete  |  ← prev  |  Q/ESC quit\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    idx = 0
    deleted = 0

    while 0 <= idx < len(images):
        path = images[idx]

        img = load(path)
        if img is None:
            print(f"  [skip unreadable] {path.name}")
            images.pop(idx)
            continue

        # overlay filename + position
        label = f"{idx + 1}/{len(images)}  {path.name}"
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q'), 27):          # Q / ESC → quit
            break
        elif key in (ord(' '), 83, 0):               # SPACE / → / numpad 0
            idx += 1
        elif key in (ord('d'), ord('D')):             # D → delete
            os.remove(path)
            print(f"  [deleted] {path.name}")
            images.pop(idx)
            deleted += 1
            # stay at same idx (now points to next image)
        elif key in (81, 2):                          # ← → prev
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    print(f"\nDone.  deleted={deleted}  remaining={len(images)}")


if __name__ == "__main__":
    main()
