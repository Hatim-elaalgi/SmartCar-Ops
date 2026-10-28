"""Pluggable object detectors. Choose one with DETECTOR_PROTO / DETECTOR_REAL in .env:

  none                       no detection (camera stream only)
  auto                       models/real_road.json if the training notebook produced it, else none
  profile:models/real_road.json   several ONNX models combined (written by training/train_road_obstacles.ipynb)
  onnx:models/model.onnx     one ONNX model exported by Ultralytics (runs on CPU with onnxruntime)
  yolo:models/best.pt        Ultralytics YOLO weights (build the image with WITH_YOLO=1)
  python:my_module:MyClass   your own class in models/my_module.py with
                             .detect(frame_bgr) -> [{"cls": str, "conf": float, "box": [x1, y1, x2, y2]}, ...]
  esp32                      inference runs ON the camera: GET http://<camera-ip>/detections returns the same JSON list
"""
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

import config

log = logging.getLogger(__name__)


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in pixels


def _coerce(item):
    if isinstance(item, Detection):
        return item
    return Detection(str(item["cls"]), float(item["conf"]), tuple(int(v) for v in item["box"]))


class NoDetector:
    name = "None (stream only)"
    min_interval = 0.0
    error = None

    def detect(self, frame):
        return []


class YoloDetector:
    min_interval = 0.0
    error = None

    def __init__(self, weights, conf=0.35):
        from ultralytics import YOLO  # only available when the image is built with WITH_YOLO=1
        path = Path(weights) if Path(weights).is_absolute() else config.BASE_DIR / weights
        self.model = YOLO(str(path))
        self.conf = conf
        self.name = f"YOLO · {path.name}"

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        out = []
        for b in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Detection(result.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
        return out


class PythonDetector:
    min_interval = 0.0
    error = None

    def __init__(self, target):
        module_name, class_name = target.split(":")
        if str(config.MODELS_DIR) not in sys.path:
            sys.path.insert(0, str(config.MODELS_DIR))
        self.impl = getattr(importlib.import_module(module_name), class_name)()
        self.name = f"Custom · {class_name}"
        self.min_interval = getattr(self.impl, "min_interval", 0.0)

    def detect(self, frame):
        return [_coerce(d) for d in self.impl.detect(frame)]


class Esp32Detector:
    """The model runs on the ESP32-CAM; we only fetch its latest results (throttled to spare the camera)."""
    name = "On-camera (ESP32)"
    min_interval = 0.5
    error = None

    def __init__(self, ip_getter):
        self.ip_getter = ip_getter

    def detect(self, frame):
        ip = self.ip_getter()
        if not ip:
            return []
        r = requests.get(f"http://{ip}/detections", timeout=0.6)
        r.raise_for_status()
        return [_coerce(d) for d in r.json()]


class OnnxDetector:
    min_interval = 0.0
    error = None

    def __init__(self, path):
        from sources.onnx_yolo import OnnxYolo
        self.model = OnnxYolo(config.BASE_DIR / path)
        self.name = f"ONNX · {Path(path).name}"

    def detect(self, frame):
        return self.model.detect(frame)


DEFAULT_PROFILE = "models/real_road.json"          # real-world dashcam (training/train_road_obstacles.ipynb)
PROTO_PROFILE = "models/proto_toy.json"            # toy track / ESP32-CAM (training/train_toy_obstacles.ipynb)


def load(spec, ip_getter=None, default_profile=DEFAULT_PROFILE):
    spec = (spec or "none").strip()
    if spec == "auto":
        spec = f"profile:{default_profile}" if (config.BASE_DIR / default_profile).exists() else "none"
    kind, _, arg = spec.partition(":")
    try:
        if kind == "profile":
            from sources.onnx_yolo import EnsembleDetector
            return EnsembleDetector(config.BASE_DIR / arg)
        if kind == "onnx":
            return OnnxDetector(arg)
        if kind == "yolo":
            return YoloDetector(arg)
        if kind == "python":
            return PythonDetector(arg)
        if kind == "esp32":
            return Esp32Detector(ip_getter or (lambda: None))
        return NoDetector()
    except Exception as exc:  # a broken model must never stop the dashboard
        log.exception("Could not load detector %s", spec)
        d = NoDetector()
        d.name = f"None — failed to load '{spec}'"
        d.error = str(exc)
        return d
