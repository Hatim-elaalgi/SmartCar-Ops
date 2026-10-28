"""Real-world dashcam: a real road video -> the trained models -> real obstacles on the map.

The video is the camera of the simulated car CAR-01: each detection is placed on the map ahead of CAR-01
(at the estimated distance), stored as an obstacle with a snapshot, and makes CAR-01 slow down or stop.

Video source (REAL_VIDEO_SOURCE in .env, or the Camera page in Real-world mode):
  demo/videos           folder of clips played in a loop (real dashcam clips cut by the training notebook)
  path/to/clip.mp4      one video file inside the project
  http://.../video      a live stream, e.g. a phone running the "IP Webcam" app, fixed on a real car's dashboard
  esp32                 the prototype's ESP32-CAM (point it at the road, or at a screen playing a road video)
"""
import logging
import math
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

import config
from core import events, obstacles, settings, trips
from core.hub import hub
from core.severity import LEVEL_META, level_for, score
from sources import detector as detectors
from sources.drawing import draw_annotations, placeholder

log = logging.getLogger(__name__)
MODE = config.REAL
VIDEO_EXT = (".mp4", ".webm", ".mkv", ".avi", ".mov")
# Box colours by severity level (BGR)
LEVEL_BGR = {"critical": (59, 59, 208), "high": (90, 131, 236), "medium": (25, 178, 250), "low": (12, 163, 12)}
# Road users move: they are stored on the map only when they force the car to act (in path and close).
# Static hazards (pothole, cone, barrier, sign) stay where they are, so every sighting is mapped.
ROAD_USERS = {"pedestrian", "cyclist", "animal", "vehicle"}
STORE_ROAD_USER_WITHIN_M = 15
# Reaction when the obstacle is in the car's path: (brake below m, slow down below m)
REACTION_M = {"pedestrian": (15, 30), "cyclist": (12, 25), "animal": (15, 30), "vehicle": (8, 15),
              "pothole": (0, 25), "cone": (0, 25), "barrier": (6, 25), "debris": (0, 20), "roadwork": (0, 0)}
# Typical real heights (m) used for the distance estimate; potholes are flat -> road geometry instead.
KNOWN_HEIGHT_M = {"pedestrian": 1.7, "cyclist": 1.7, "vehicle": 1.5, "animal": 0.8, "cone": 0.7, "barrier": 1.0,
                  "roadwork": 1.2}
LIST_EVERY_S = 0.5     # live detection list: at most one entry per class every 0.5 s (the models run ~15x/s)
# Looping video files replay the same hazards at ever new places on the map: their unreviewed obstacles expire.
LOOP_OBSTACLE_TTL_S = 2 * 3600


def _credits(folder):
    """CREDITS.md lines like: - `01_city.mp4` — title · author · licence"""
    try:
        text = (folder / "CREDITS.md").read_text(encoding="utf-8")
    except OSError:
        return {}
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"`([^`]+)`\s+[—-]\s+(.+)", text)}


class DashcamService(threading.Thread):
    def __init__(self, camera, fleet):
        super().__init__(daemon=True, name="dashcam")
        self.camera, self.fleet = camera, fleet
        self._lock = threading.Lock()
        self._switch = threading.Event()
        self._new_frame = threading.Event()
        self.source = settings.get("real_video_source") or config.REAL_VIDEO_SOURCE
        self.status, self.error = "starting", None
        self.clip, self.credit, self.resolution = None, None, None
        self.video_fps = self.read_fps = self.det_fps = 0.0
        self.frames = 0
        self.last_frame_ts = 0.0
        self._frame, self._frame_id = None, 0
        self._jpeg_cache = {}
        self.annotations, self._annot_frame = [], -1
        self.detector = None
        self._last_cls, self._last_listed = {}, {}

    # --- public -----------------------------------------------------------------------
    @property
    def active(self):
        return self.status == "playing" and time.time() - self.last_frame_ts < 3

    def info(self):
        models = self.detector.info() if hasattr(self.detector, "info") else []
        return {"status": "playing" if self.active else self.status, "source": self.source, "clip": self.clip,
                "credit": self.credit, "resolution": self.resolution, "video_fps": round(self.video_fps, 1),
                "read_fps": round(self.read_fps, 1), "det_fps": round(self.det_fps, 1), "frames": self.frames,
                "error": self.error, "detector": self.detector.name if self.detector else "—",
                "detector_error": getattr(self.detector, "error", None), "models": models,
                "car_id": config.DASHCAM_CAR_ID}

    def set_source(self, source):
        source = (source or "").strip() or config.REAL_VIDEO_SOURCE
        settings.set("real_video_source", source)
        self.source = source
        self._switch.set()

    def latest_jpeg(self, annotated=True):
        with self._lock:
            frame, fid = self._frame, self._frame_id
        if frame is None or not self.active:
            text = {"no source": "No dashcam video", "starting": "Loading the models…"}.get(self.status, "Dashcam offline")
            sub = self.error or "Run the training notebook to get the demo clips, or set a stream URL"
            key = f"ph-{text}-{sub}"
            if key not in self._jpeg_cache:
                self._jpeg_cache = {key: placeholder(text, sub[:90])}
            return self._jpeg_cache[key], key
        key = (fid, annotated)
        if key not in self._jpeg_cache:
            img = draw_annotations(frame.copy(), self.annotations) if annotated and self.annotations else frame
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._jpeg_cache = {key: buf.tobytes()}
        return self._jpeg_cache[key], key

    # --- reading ----------------------------------------------------------------------
    def run(self):
        self.detector = detectors.load(config.DETECTOR_REAL)
        if isinstance(self.detector, detectors.NoDetector) and not self.detector.error:
            self.error = "No model yet — run training/train_road_obstacles.ipynb"
        threading.Thread(target=self._perception_loop, daemon=True, name="dashcam-perception").start()
        while True:
            self._switch.clear()
            try:
                src = self.source
                if src == "esp32":
                    self._play_esp32()
                elif re.match(r"^(https?|rtsp)://", src):
                    self._play_capture(src, live=True)
                else:
                    self._play_files(config.BASE_DIR / src)
            except Exception as exc:
                log.exception("Dashcam source failed")
                self.status, self.error = "error", str(exc)[:160]
                self._switch.wait(5)

    def _play_files(self, path):
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT) if path.is_dir() else \
            ([path] if path.exists() else [])
        if not files:
            self.status = "no source"
            self.error = f"No video in {path.relative_to(config.BASE_DIR) if path.is_relative_to(config.BASE_DIR) else path}"
            self._switch.wait(10)
            return
        credits = _credits(path if path.is_dir() else path.parent)
        for f in files:
            if self._switch.is_set():
                return
            obstacles.expire(MODE, "camera", LOOP_OBSTACLE_TTL_S)
            self.clip, self.credit = f.name, credits.get(f.name)
            self._play_capture(str(f), live=False)

    def _play_capture(self, src, live):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status, self.error = "error", f"Cannot open {src}"
            self._switch.wait(5)
            return
        if live:
            self.clip, self.credit = src, "live stream"
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        self.video_fps = fps if 1 <= fps <= 120 else 15
        period, next_t, fails = 1.0 / self.video_fps, time.time(), 0
        try:
            while not self._switch.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not live:
                        return              # end of the clip -> next one
                    fails += 1
                    if fails > 50:
                        raise RuntimeError("live stream stopped")
                    time.sleep(0.1)
                    continue
                fails = 0
                self._set_frame(frame)
                if not live:                # play files at their real speed
                    next_t += period
                    delay = next_t - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -1:
                        next_t = time.time()
        finally:
            cap.release()

    def _play_esp32(self):
        self.clip, self.credit, last = "ESP32-CAM", "prototype camera", None
        while not self._switch.is_set():
            with self.camera._lock:
                frame, fid = self.camera._frame, self.camera._frame_id
            if frame is not None and fid != last and self.camera.connected:
                self._set_frame(frame.copy())
                last = fid
            elif not self.camera.connected:
                self.status, self.error = "error", "ESP32-CAM not connected"
            time.sleep(0.03)

    def _set_frame(self, frame):
        now = time.time()
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])))
        with self._lock:
            if self.last_frame_ts:
                inst = 1.0 / max(1e-3, now - self.last_frame_ts)
                self.read_fps = 0.9 * self.read_fps + 0.1 * inst if self.read_fps else inst
            self._frame, self._frame_id = frame, self._frame_id + 1
            self.last_frame_ts, self.frames = now, self.frames + 1
            self.resolution = f"{frame.shape[1]}×{frame.shape[0]}"
        if self.status != "playing":
            self.status, self.error = "playing", None
            events.log(MODE, "info", "camera", f"Dashcam of {config.DASHCAM_CAR_ID} started ({self.clip})")
        self._new_frame.set()

    # --- perception -------------------------------------------------------------------
    def _perception_loop(self):
        last_id, t_prev = None, None
        while True:
            self._new_frame.wait(1.0)
            self._new_frame.clear()
            with self._lock:
                frame, fid = self._frame, self._frame_id
            if frame is None or fid == last_id or not self.active or self.detector is None:
                continue
            last_id = fid
            try:
                dets = self.detector.detect(frame)
            except Exception as exc:
                log.exception("Detector failed")
                self.error = f"Model error: {exc}"[:160]
                time.sleep(1)
                continue
            now = time.time()
            if t_prev:
                inst = 1.0 / max(1e-3, now - t_prev)
                self.det_fps = 0.85 * self.det_fps + 0.15 * inst if self.det_fps else inst
            t_prev = now
            self.annotations = self._handle(dets, frame, now)

    def _distance(self, cls, box, w, h):
        """Monocular distance estimate (pinhole camera, focal from the lens angle):
        objects with a typical height -> d = real height x focal / box height in pixels;
        flat objects (potholes) or boxes cut by the image edge -> flat-road geometry from the box bottom."""
        x1, y1, x2, y2 = box
        focal = (w / 2) / math.tan(math.radians(config.DASHCAM_HFOV_DEG) / 2)
        real_h = KNOWN_HEIGHT_M.get(cls)
        if real_h and y1 > 2 and y2 < h - 2 and y2 - y1 > 6:
            d = real_h * focal / (y2 - y1)
        else:
            below = y2 - config.DASHCAM_HORIZON * h
            d = 80.0 if below <= 2 else config.DASHCAM_HEIGHT_M * focal / below
        return max(1.5, min(80.0, d))

    @staticmethod
    def _in_path(box, w, h):
        """In the car's own lane: box centre inside a lane-shaped corridor, or the box covers the centre line."""
        x1, _, x2, y2 = box
        depth = max(0.0, (y2 / h - config.DASHCAM_HORIZON) / (1 - config.DASHCAM_HORIZON))
        centre_in_lane = abs((x1 + x2) / 2 - w / 2) <= w * (0.04 + 0.20 * depth)
        return centre_in_lane or (x1 <= w / 2 <= x2 and depth > 0.05)

    def _handle(self, dets, frame, now):
        h, w = frame.shape[:2]
        car = hub.car(MODE, config.DASHCAM_CAR_ID) or {}
        annotations, reaction = [], None
        for d in dets:
            dist = self._distance(d.cls, d.box, w, h)
            in_path = self._in_path(d.box, w, h)
            proximity = max(0.0, min(1.0, 1 - dist / 50))
            sev = score(d.cls, d.conf, proximity, in_path)
            lvl = level_for(sev)
            annotations.append(("box", d.box, f"{d.cls.replace('_', ' ')} {d.conf:.0%} · {dist:.0f} m", LEVEL_BGR[lvl]))
            if now - self._last_listed.get(d.cls, 0) >= LIST_EVERY_S:
                self._last_listed[d.cls] = now
                hub.add_detection({"cls": d.cls, "conf": d.conf, "proximity": proximity, "in_path": in_path,
                                   "distance": dist, "level": lvl, "ts": now, "source": "camera"}, mode=MODE)
            if in_path:
                brake_m, slow_m = REACTION_M.get(d.cls, (10, 20))
                action = "STOPPED" if dist < brake_m else "AVOIDING" if dist < slow_m else None
                rank = {"STOPPED": 2, "AVOIDING": 1}
                # strongest reaction wins (braking beats slowing down), then the closest obstacle
                if action and (reaction is None or (rank[action], -dist) > (rank[reaction[0]], -reaction[1])):
                    reaction = (action, dist)
            if d.cls in ROAD_USERS and not (in_path and dist < STORE_ROAD_USER_WITHIN_M):
                continue
            if now - self._last_cls.get(d.cls, 0) < 1.5:          # one database write per class every 1.5 s
                continue
            self._last_cls[d.cls] = now
            ahead = self.fleet.point_ahead(config.DASHCAM_CAR_ID, dist)
            if not ahead:
                continue
            x, y, edge, trip_id = ahead
            oid, is_new = obstacles.record(MODE, d.cls, d.conf, proximity, in_path, x, y,
                                           car_id=config.DASHCAM_CAR_ID, trip_id=trip_id, edge=edge,
                                           distance=round(dist, 1), source="camera", ts=now)
            if is_new:
                snap = draw_annotations(frame.copy(), [annotations[-1]])
                snap = cv2.resize(snap, (960, round(h * 960 / w))) if w > 960 else snap
                name = f"dash_{oid}_{uuid.uuid4().hex[:6]}.jpg"
                cv2.imwrite(str(config.SNAPSHOT_DIR / name), snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                obstacles.set_snapshot(oid, name)
                trips.bump(trip_id, obstacles=1, avoid=int(in_path))
        if reaction and car.get("state") in ("CRUISING", "AVOIDING", "STOPPED"):
            action, dist = reaction
            self.fleet.react(config.DASHCAM_CAR_ID, action, 2.0 if action == "STOPPED" else 3.0, dist)
        return annotations
