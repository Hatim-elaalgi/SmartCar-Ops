"""Builds the toy-track dataset from the video frames labelled by toy_autolabel.py.

  1. split   — the video is one continuous drive (neighbouring frames are almost identical), so each sequence is cut
               in TIME blocks: a middle block for validation, another for test, the rest for training, with gaps.
  2. degrade — "ESP32-CAM look" copies of training frames (lower resolution, strong JPEG, blur, colour cast),
               so the model also works on the real car camera (the video was filmed with a phone).
  3. paste   — copy-paste: obstacles cut out with their SAM outline, pasted on empty-track training frames at new
               places / distances (bigger when closer to the camera), with exact boxes. Training split only.
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/workspace/datasets/toy")
FRAMES = ROOT / "raw" / "frames"
LABELS = ROOT / "auto" / "labels"
MASKS = ROOT / "auto" / "masks"
CLASSES = ["column", "box"]
# Continuous parts of the video: obstacle sequences and empty-track parts (frame numbers, inclusive).
SEQUENCES = [(1, 42), (43, 86), (87, 157), (158, 169), (171, 222), (223, 238), (239, 249)]


def frame_name(i):
    return f"ezgif-frame-{i:03d}"


def split_frames():
    """{frame number: 'train' | 'val' | 'test'} — time blocks inside every sequence, with gaps."""
    split = {}
    for first, last in SEQUENCES:
        frames = [i for i in range(first, last + 1) if (LABELS / f"{frame_name(i)}.txt").exists()]
        n = len(frames)
        for k, i in enumerate(frames):
            t = k / n
            if 0.38 <= t < 0.48:
                split[i] = "val"
            elif 0.52 <= t < 0.62:
                split[i] = "test"
            elif t < 0.34 or t >= 0.66:
                split[i] = "train"
            # frames in the gaps are left out
    return split


def read_labels(i):
    text = (LABELS / f"{frame_name(i)}.txt").read_text().strip()
    return [line.split() for line in text.splitlines() if line.strip()]


ESP32_SIZE = (320, 240)      # the ESP32-CAM streams 320 x 240 (QVGA, 4:3)


def esp32_frame(img, labels, rng):
    """Turn a phone frame (16:9) into an ESP32-CAM-like frame: random 4:3 crop -> 320 x 240, boxes adjusted
    (an obstacle cut by the crop is kept only if at least 40 % of it is still visible), then the camera look."""
    h, w = img.shape[:2]
    cw = min(w, int(round(h * 4 / 3)))
    x0 = rng.randint(0, w - cw)
    out = []
    for c, cx, cy, bw, bh in (map(float, l) for l in labels):
        x1, x2 = (cx - bw / 2) * w - x0, (cx + bw / 2) * w - x0
        vx1, vx2 = max(0.0, x1), min(float(cw), x2)
        if vx2 - vx1 < 0.4 * (x2 - x1):
            if vx2 > vx1:
                return None, None              # half-visible obstacle: ambiguous, skip this crop
            continue
        out.append([str(int(c)), f"{(vx1 + vx2) / 2 / cw:.6f}", f"{cy:.6f}", f"{(vx2 - vx1) / cw:.6f}", f"{bh:.6f}"])
    small = cv2.resize(img[:, x0:x0 + cw], ESP32_SIZE, interpolation=cv2.INTER_AREA)
    return esp32_look(small, rng), out


def esp32_look(img, rng):
    """Mimic the ESP32-CAM sensor: softer image, colour cast, noise, strong JPEG."""
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.4, 1.2))
    gain = np.array([rng.uniform(0.85, 1.1), rng.uniform(0.9, 1.1), rng.uniform(0.85, 1.15)])
    img = np.clip(img.astype(np.float32) * gain * rng.uniform(0.75, 1.15), 0, 255).astype(np.uint8)
    noise = np.random.default_rng(rng.randint(0, 10 ** 6)).normal(0, rng.uniform(2, 6), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(18, 45)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cutouts(train_frames):
    """Obstacle images + alpha masks from the training frames (skip ones cut by the image border)."""
    out = []
    for path in sorted(MASKS.glob("*.png")):
        stem, cls = path.stem.rsplit("_", 1)
        i = int(stem.split("-")[-1])
        if i not in train_frames:
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(FRAMES / f"{stem}.jpg"))
        ys, xs = np.nonzero(mask)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        h, w = mask.shape
        if x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1:     # partial obstacle: not a clean cut-out
            continue
        out.append((int(cls), img[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()))
    return out


def paste(bg, pieces, rng):
    """Paste 1-2 obstacles on an empty-track frame. Returns (image, labels)."""
    h, w = bg.shape[:2]
    img, labels, taken = bg.copy(), [], []
    for cls, piece, mask in rng.sample(pieces, k=1 if rng.random() < 0.7 else 2):
        y_bottom = rng.uniform(0.45, 1.0) * h                      # where the obstacle stands on the floor
        depth = (y_bottom - 0.45 * h) / (0.55 * h)                 # 0 = far (horizon), 1 = at the camera
        target_h = h * (0.12 + 0.75 * depth) * rng.uniform(0.8, 1.2)
        s = target_h / piece.shape[0]
        pw, ph = max(4, int(piece.shape[1] * s)), max(4, int(piece.shape[0] * s))
        if ph > h or pw > w * 0.6:
            continue
        p = cv2.resize(piece, (pw, ph), interpolation=cv2.INTER_AREA)
        a = cv2.GaussianBlur(cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_LINEAR), (5, 5), 0) / 255.0
        x0 = rng.randint(0, w - pw)
        y0 = int(min(h, y_bottom) - ph)
        if y0 < 0 or any(not (x0 + pw < tx1 or x0 > tx2) for tx1, tx2 in taken):
            continue
        region = img[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
        ratio = np.clip(region.mean() / max(1.0, p.mean()), 0.85, 1.15)          # match the local brightness
        p = np.clip(p.astype(np.float32) * ratio, 0, 255)
        img[y0:y0 + ph, x0:x0 + pw] = (a[..., None] * p + (1 - a[..., None]) * region).astype(np.uint8)
        taken.append((x0, x0 + pw))
        labels.append(f"{cls} {(x0 + pw / 2) / w:.6f} {(y0 + ph / 2) / h:.6f} {pw / w:.6f} {ph / h:.6f}")
    return img, labels


def save(ds, split, name, img, labels):
    cv2.imwrite(str(ds / "images" / split / f"{name}.jpg"), img)
    (ds / "labels" / split / f"{name}.txt").write_text("\n".join(" ".join(l) for l in labels))


def build(ds, degrade_p=0.6, synth_n=400, seed=0):
    rng = random.Random(seed)
    shutil.rmtree(ds, ignore_errors=True)
    splits = ("train", "val", "test", "test_esp32")
    for s in splits:
        (ds / "images" / s).mkdir(parents=True)
        (ds / "labels" / s).mkdir(parents=True)
    split = split_frames()
    counts = {s: {"real": 0, "obstacles": 0, "empty": 0, "esp32_look": 0, "pasted": 0} for s in splits}
    test_rng = random.Random(seed + 1000)        # fixed: the ESP32-style test set is the same on every run
    for i, s in split.items():
        img = cv2.imread(str(FRAMES / f"{frame_name(i)}.jpg"))
        labels = read_labels(i)
        save(ds, s, f"toy_{i:03d}", img, labels)
        counts[s]["real"] += 1
        counts[s]["obstacles" if labels else "empty"] += 1
        if s == "train" and rng.random() < degrade_p:
            small, small_labels = esp32_frame(img, labels, rng)
            if small is not None:
                save(ds, s, f"toy_{i:03d}_esp", small, small_labels)
                counts[s]["esp32_look"] += 1
        if s == "test":                           # same test frames, converted to 320 x 240 ESP32-CAM style
            for _ in range(5):
                small, small_labels = esp32_frame(img, labels, test_rng)
                if small is not None:
                    save(ds, "test_esp32", f"toy_{i:03d}_esp", small, small_labels)
                    counts["test_esp32"]["real"] += 1
                    counts["test_esp32"]["obstacles" if small_labels else "empty"] += 1
                    break
    train_frames = {i for i, s in split.items() if s == "train"}
    pieces = cutouts(train_frames)
    empty_train = [i for i in train_frames if not read_labels(i)]
    made = 0
    while made < synth_n and pieces and empty_train:
        bg = cv2.imread(str(FRAMES / f"{frame_name(rng.choice(empty_train))}.jpg"))
        img, labels = paste(bg, pieces, rng)
        if not labels:
            continue
        labels = [l.split() for l in labels]
        if rng.random() < degrade_p:
            img, labels = esp32_frame(img, labels, rng)
            if img is None:
                continue
        save(ds, "train", f"paste_{made:04d}", img, labels)
        made += 1
    counts["train"]["pasted"] = made
    (ds / "data.yaml").write_text(f"path: {ds}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                  f"test_esp32: {ds / 'images' / 'test_esp32'}\nnames:\n"   # custom split: absolute
                                  + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
    return counts, len(pieces)


def preview(ds, split="train", n=12, seed=1, cols=4):
    """Grid of random images of a split with their boxes (BGR array)."""
    rng = random.Random(seed)
    files = sorted((ds / "images" / split).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no images in {ds / 'images' / split} — run td.build() again "
                                "(if this notebook was open before an update: Restart the kernel, then Run All)")
    colors = [(0, 165, 255), (255, 80, 0)]
    tiles = []
    for f in rng.sample(files, min(n, len(files))):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        for line in (ds / "labels" / split / f"{f.stem}.txt").read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), colors[int(c)], 3)
        img = cv2.resize(img, (416, 232))
        cv2.putText(img, f.stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
