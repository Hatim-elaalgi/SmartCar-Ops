"""Optional: make the Real-world simulation use YOUR real-road model.

Set in .env:
    DETECTOR_REAL=yolo:models/real_road.pt
    REAL_IMAGES_DIR=datasets/road_samples      (a folder of real road photos inside the project)

When the simulator meets an obstacle it then picks a random photo, runs your model on it, and stores the
model's class, confidence and the annotated photo as the obstacle snapshot.
"""
import logging
import random
import threading
import uuid

import cv2

import config
from sources import detector as detectors
from sources.drawing import draw_detections

log = logging.getLogger(__name__)


class RealImageModel:
    def __init__(self):
        self.enabled = False
        self.status = "Disabled — simulated classes"
        self.detector = None
        self.images = []
        if config.DETECTOR_REAL == "none" or not config.REAL_IMAGES_DIR:
            return
        folder = config.BASE_DIR / config.REAL_IMAGES_DIR
        self.images = [p for p in folder.glob("**/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        self.detector = detectors.load(config.DETECTOR_REAL)
        if not self.images:
            self.status = f"No images found in {config.REAL_IMAGES_DIR}"
        elif self.detector.error:
            self.status = self.detector.name
        else:
            self.enabled = True
            self.status = f"{self.detector.name} on {len(self.images)} real images"

    def sample(self):
        """Run the model on a random real image. Returns a dict or None."""
        try:
            path = random.choice(self.images)
            frame = cv2.imread(str(path))
            if frame is None:
                return None
            dets = self.detector.detect(frame)
            if not dets:
                return None
            best = max(dets, key=lambda d: d.conf)
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = best.box
            fname = f"real_{uuid.uuid4().hex[:10]}.jpg"
            cv2.imwrite(str(config.SNAPSHOT_DIR / fname), draw_detections(frame.copy(), dets))
            return {"cls": best.cls, "conf": best.conf, "proximity": min(1.0, (y2 - y1) / h),
                    "in_path": 0.3 * w <= (x1 + x2) / 2 <= 0.7 * w, "snapshot": fname}
        except Exception:
            log.exception("Real model failed")
            return None


_instance = None
_lock = threading.Lock()


def get():
    global _instance
    with _lock:
        if _instance is None:
            _instance = RealImageModel()
        return _instance
