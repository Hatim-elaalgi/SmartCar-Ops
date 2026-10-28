"""Run Ultralytics YOLO models exported to ONNX with onnxruntime (CPU, no PyTorch needed in the dashboard).

Handles both export layouts:
  * end-to-end (YOLO26 default): output (1, N, 6) = x1, y1, x2, y2, score, class   — no NMS needed
  * classic (YOLOv8 / YOLO11):   output (1, 4 + classes, anchors) = cx, cy, w, h, class scores — NMS here

A *profile* (models/real_road.json, written by the training notebook) combines several models and maps each
model's class names to the dashboard's obstacle classes (e.g. COCO "car" -> "vehicle").
"""
import ast
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import config
from sources.detector import Detection


class OnnxYolo:
    def __init__(self, path, imgsz=640, conf=0.35, iou=0.5, class_map=None, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(1, (os.cpu_count() or 4) // 3)
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else imgsz
        self.conf, self.iou = conf, iou
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names = {int(k): v for k, v in names.items()}
        self.class_map = class_map          # None = keep every class with its own name
        self.path = Path(path)
        self.last_ms = 0.0

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = img
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), r, left, top

    def detect(self, frame):
        t0 = time.perf_counter()
        blob, r, dx, dy = self._letterbox(frame)
        out = self.session.run(None, {self.input.name: blob})[0][0]
        h, w = frame.shape[:2]
        boxes, scores, classes = [], [], []
        if out.ndim == 2 and out.shape[-1] == 6 and out.shape[0] < out.shape[1] * 100:   # end-to-end (N, 6)
            for x1, y1, x2, y2, s, c in out:
                if s >= self.conf:
                    boxes.append([x1, y1, x2, y2]); scores.append(float(s)); classes.append(int(c))
        else:                                                                           # classic (4+nc, anchors)
            preds = out.T if out.shape[0] < out.shape[1] else out
            cls_scores = preds[:, 4:]
            cls = cls_scores.argmax(1)
            sc = cls_scores[np.arange(len(cls)), cls]
            keep = sc >= self.conf
            xywh, sc, cls = preds[keep, :4], sc[keep], cls[keep]
            xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            idx = cv2.dnn.NMSBoxesBatched(
                [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
                sc.tolist(), cls.tolist(), self.conf, self.iou) if len(sc) else []
            for i in np.array(idx).flatten():
                boxes.append(xyxy[i].tolist()); scores.append(float(sc[i])); classes.append(int(cls[i]))
        found = []
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
            name = self.names.get(c, str(c))
            if self.class_map is not None:
                name = self.class_map.get(name)
                if name is None:
                    continue
            x1, x2 = (np.clip([x1, x2], dx, dx + w * r) - dx) / r
            y1, y2 = (np.clip([y1, y2], dy, dy + h * r) - dy) / r
            found.append(Detection(name, s, (int(x1), int(y1), int(x2), int(y2))))
        self.last_ms = 0.8 * self.last_ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.last_ms else \
            (time.perf_counter() - t0) * 1000
        return found


class EnsembleDetector:
    """Several ONNX models on the same frame; results are merged (each already mapped to dashboard classes)."""
    min_interval = 0.0
    error = None

    def __init__(self, profile_path):
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        self.name = profile.get("name", "Ensemble")
        self.models, self.missing = [], []
        for m in profile["models"]:
            path = config.BASE_DIR / m["path"]
            if not path.exists():                 # e.g. the hazards model before the training notebook has run
                self.missing.append(m.get("id", path.stem))
                continue
            model = OnnxYolo(path, imgsz=m.get("imgsz", 640), conf=m.get("conf", 0.35), class_map=m.get("classes"))
            model.id = m.get("id", path.stem)
            self.models.append(model)
        if self.missing:
            self.name += f" — not trained yet: {', '.join(self.missing)}"
        if not self.models:
            raise FileNotFoundError(f"no model file found for {profile_path}")
        # The models run at the same time (onnxruntime releases the GIL): frame time = slowest model, not the sum.
        self._pool = ThreadPoolExecutor(max_workers=len(self.models), thread_name_prefix="onnx")

    def detect(self, frame):
        found = []
        for result in self._pool.map(lambda m: m.detect(frame), self.models):
            found += result
        return found

    def info(self):
        return [{"id": m.id, "file": m.path.name, "classes": sorted(set((m.class_map or m.names).values()) if
                                                                     m.class_map else m.names.values()),
                 "conf": m.conf, "ms": round(m.last_ms, 1)} for m in self.models]
