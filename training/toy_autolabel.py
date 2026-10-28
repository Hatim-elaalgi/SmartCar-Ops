"""Automatic labelling of the toy-track video with SAM 2 video tracking.

For each obstacle sequence we give ONE click on the obstacle in a reference frame; SAM 2 follows the obstacle
forward and backward through the frames and gives its exact outline -> bounding box -> YOLO label.
Frames outside the sequences contain no obstacle (empty-track negatives, empty label files).

    docker compose --profile train exec -w /workspace trainer python training/toy_autolabel.py

Writes datasets/toy/auto/labels/*.txt and review sheets (boxes drawn) in datasets/toy/raw/review/.
"""
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.sam import SAM2VideoPredictor

ROOT = Path("/workspace/datasets/toy")
FRAMES = sorted(glob.glob(str(ROOT / "raw/frames/*.jpg")))
CLASSES = ["column", "box"]
# (class, first frame, last frame, reference frame, click point (x, y) on the obstacle in the reference frame,
#  drop dark pixels: the black tape touching the column was merged into its outline in the close-up sequence)
SEGMENTS = [
    ("column", 1, 42, 20, (290, 200), True),
    ("box", 87, 157, 120, (260, 250), False),
    ("box", 158, 170, 163, (420, 340), False),
    ("column", 223, 238, 231, (585, 220), False),
]
EXCLUDE_FRAMES = [170]    # only a sliver of the box is visible and SAM lost it: neither "box" nor "empty"
DARK = 70                 # grey level under which a pixel is tape, not obstacle
MIN_AREA = 0.002          # ignore masks smaller than 0.2 % of the image (object left the frame)
TMP = Path("/tmp/toy_seg")


def frame_path(i):
    return FRAMES[i - 1]          # files are numbered from 001


def track(order, point, drop_dark=False):
    """Run SAM 2 on the frames in `order`, prompt = point on the first one.
    Returns ({frame: box or None}, {frame: mask}, (w, h))."""
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    h, w = cv2.imread(frame_path(order[0])).shape[:2]
    video = TMP / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in order:
        writer.write(cv2.imread(frame_path(i)))
    writer.release()
    predictor = SAM2VideoPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                                                  model="sam2.1_b.pt", verbose=False))
    boxes, masks = {}, {}
    for i, r in zip(order, predictor(source=str(video), points=[list(point)], labels=[1], stream=True)):
        box = None
        if r.masks is not None and len(r.masks.data):
            m = r.masks.data[0].cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if drop_dark:
                gray = cv2.cvtColor(cv2.imread(frame_path(i)), cv2.COLOR_BGR2GRAY)
                m = (m & (gray > DARK)).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if n > 1:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                x, y, bw, bh, area = stats[k]
                if area >= MIN_AREA * w * h:
                    box = (int(x), int(y), int(x + bw), int(y + bh))
                    masks[i] = ((lab == k) * 255).astype(np.uint8)
        boxes[i] = box
    return boxes, masks, (w, h)


def main():
    labels = {i: [] for i in range(1, len(FRAMES) + 1)}
    all_masks = {}
    size = None
    for cls, first, last, ref, point, drop_dark in SEGMENTS:
        fwd, fwd_m, size = track(list(range(ref, last + 1)), point, drop_dark)
        bwd, bwd_m, _ = track(list(range(ref, first - 1, -1)), point, drop_dark)
        found = 0
        for i in range(first, last + 1):
            box = fwd.get(i) or bwd.get(i)
            if box:
                labels[i].append((CLASSES.index(cls), box))
                all_masks[i] = fwd_m.get(i) if fwd.get(i) else bwd_m.get(i)
                found += 1
        print(f"{cls:<7} frames {first:03d}-{last:03d}: boxes on {found}/{last - first + 1}", flush=True)

    w, h = size
    out, mask_dir = ROOT / "auto/labels", ROOT / "auto/masks"     # masks = obstacle cut-outs for copy-paste
    for d in (out, mask_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for i, objs in labels.items():
        if i in EXCLUDE_FRAMES:
            continue
        lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                 for c, (x1, y1, x2, y2) in objs]
        (out / (Path(frame_path(i)).stem + ".txt")).write_text("\n".join(lines))
        if objs and all_masks.get(i) is not None:
            cv2.imwrite(str(mask_dir / f"{Path(frame_path(i)).stem}_{objs[0][0]}.png"), all_masks[i])

    # review sheets: every frame of the sequences with its box
    colors = [(0, 165, 255), (255, 80, 0)]
    for k, (cls, first, last, *_) in enumerate(SEGMENTS):
        tiles = []
        for i in range(first, last + 1):
            im = cv2.imread(frame_path(i))
            for c, (x1, y1, x2, y2) in labels[i]:
                cv2.rectangle(im, (x1, y1), (x2, y2), colors[c], 4)
            im = cv2.resize(im, (208, 116))
            cv2.putText(im, f"{i:03d}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(im)
        cols = 8
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[j:j + cols]) for j in range(0, len(tiles), cols)])
        cv2.imwrite(str(ROOT / f"raw/review/auto_{k + 1}_{cls}_{first:03d}-{last:03d}.jpg"), sheet)
    print("labels:", out, "| review sheets in datasets/toy/raw/review/")


if __name__ == "__main__":
    main()
