"""Draw detections and checkpoint markers on camera frames (colours are BGR)."""
import cv2
import numpy as np

BOX = (229, 135, 57)       # series blue  #3987e5
MARKER = (112, 158, 25)    # series aqua  #199e70
INK = (236, 239, 240)
SHADE = (25, 26, 26)


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA)


def draw_detections(img, detections):
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX, 2)
        _label(img, f"{d.cls} {d.conf:.0%}", x1, y1, BOX)
    return img


def draw_annotations(img, annotations):
    """annotations: ("box", (x1,y1,x2,y2), label[, bgr]) or ("marker", corners Nx2, label)."""
    for kind, geom, text, *color in annotations:
        if kind == "box":
            c = color[0] if color else BOX
            x1, y1, x2, y2 = geom
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
            _label(img, text, x1, y1, c)
        elif kind == "marker":
            pts = np.asarray(geom, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, MARKER, 2)
            x, y = pts[:, 0, :].min(axis=0)
            _label(img, text, int(x), int(y), MARKER)
    return img


def placeholder(text, sub="", size=(640, 480)):
    w, h = size
    img = np.full((h, w, 3), SHADE, np.uint8)
    for gx in range(0, w, 40):
        cv2.line(img, (gx, 0), (gx, h), (34, 35, 35), 1)
    for gy in range(0, h, 40):
        cv2.line(img, (0, gy), (w, gy), (34, 35, 35), 1)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(img, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, sub, ((w - sw) // 2, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (183, 194, 195), 1,
                    cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()
