"""Turns ESP32-CAM frames into prototype telemetry. The Arduino has no wireless link, so everything the
dashboard knows about the live toy car comes from the picture:

  position  ArUco markers printed at the track checkpoints (download them on the Route planner page).
            Between two markers the position is estimated from the measured speed.
  motion    pixel change between frames -> moving / stopped
  obstacles your trained model (sources/detector.py) -> class, confidence, proximity (box height),
            in path (box centre in the middle of the image)
"""
import logging
import time

import cv2
import numpy as np

import config
from core import events, geo, network, obstacles, toy_car, trips
from core.hub import hub
from sources.drawing import draw_annotations

log = logging.getLogger(__name__)
MODE = config.PROTO
LIST_EVERY_S = 0.5     # live detection list: at most one entry per class every 0.5 s (the model runs ~15x/s)


class TrackLocalizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.route, self.idx, self.last_node = None, None, None
        self.marker_ts, self.traveled, self.done = None, 0.0, 0.0
        self.speed = config.PROTO_SPEED_CM_S
        self._last_update = None

    def on_marker(self, node, ts):
        """Returns True when the car reached a new checkpoint."""
        net = network.proto_network()
        traj = network.active_trajectory()
        route = traj["nodes"] if traj else [node]
        if route != self.route:
            self.route, self.idx, self.done = route, None, 0.0
        if self.last_node == node:
            return False
        start = self.idx + 1 if self.idx is not None else 0
        j = next((k for k in range(start, len(route)) if route[k] == node), None)
        if j is None and node in route:
            j = route.index(node)
        if self.idx is not None and j == self.idx + 1 and self.marker_ts:
            seg = net.edges.get(network.edge_key(route[self.idx], node))
            dt = ts - self.marker_ts
            if seg and dt > 0.5:
                self.speed = max(3.0, min(80.0, 0.6 * self.speed + 0.4 * seg["length"] / dt))
                self.done += seg["length"]
        self.idx, self.last_node, self.marker_ts, self.traveled = j, node, ts, 0.0
        return True

    def update(self, ts, moving):
        if self._last_update is not None and moving:
            self.traveled += self.speed * (ts - self._last_update)
        self._last_update = ts

    def _current_edge(self, net):
        if self.idx is None or not self.route or self.idx >= len(self.route) - 1:
            return None
        a, b = self.route[self.idx], self.route[self.idx + 1]
        return (a, b) if net.G.has_edge(a, b) else None

    def position(self):
        net = network.proto_network()
        if self.last_node is None or self.last_node not in net.nodes:
            return None
        edge = self._current_edge(net)
        if not edge:
            x, y = net.xy(self.last_node)
            return x, y, 0.0
        pts = net.edge_geometry(*edge)
        cum = geo.cumulative(MODE, pts)
        return geo.point_along(MODE, pts, cum, min(self.traveled, cum[-1] * 0.97))

    def distance(self):
        net = network.proto_network()
        edge = self._current_edge(net)
        extra = min(self.traveled, net.edges[network.edge_key(*edge)]["length"]) if edge else 0.0
        return self.done + extra

    def progress(self):
        if not self.route or self.idx is None:
            return None
        total = network.proto_network().path_length(self.route) if len(self.route) > 1 else 0
        return min(1.0, self.distance() / total) if total else None


class Perception:
    def __init__(self, detector):
        self.detector = detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
        self.aruco = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        self.loc = TrackLocalizer()
        self.prev_small, self.motion, self.moving = None, 0.0, False
        self.trip_id = None
        self.annotations = []
        self.markers_seen = 0
        self._last_detect = self._last_pub = self._last_point = self._last_inpath = 0.0
        self._last_cls, self._last_listed, self._last_avoid = {}, {}, {}
        self._avoid_path, self._avoid_ts = None, 0.0     # estimated detour drawn on the maps
        self._errors_logged = False

    # --- per frame ------------------------------------------------------------------
    def process(self, frame, ts, fps):
        h, w = frame.shape[:2]
        small = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        if self.prev_small is not None:
            diff = float(np.mean(cv2.absdiff(small, self.prev_small)))
            self.motion = 0.7 * self.motion + 0.3 * diff
        self.prev_small = small
        self.moving = self.motion > config.MOTION_THRESHOLD

        annotations = []
        net = network.proto_network()
        corners, ids, _ = self.aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            for c, marker_id in zip(corners, ids.ravel()):
                node = net.markers.get(int(marker_id))
                annotations.append(("marker", c.reshape(-1, 2), f"CP {node or '?'} · #{int(marker_id)}"))
                if node and self.loc.on_marker(node, ts):
                    self._on_checkpoint(node, ts)
        self.loc.update(ts, self.moving)

        if ts - self._last_detect >= self.detector.min_interval:
            self._last_detect = ts
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                detections = []
                if not self._errors_logged:
                    events.log("system", "warning", "camera", f"Detector error: {exc}")
                    self._errors_logged = True
            for d in detections:
                annotations.append(("box", d.box, f"{d.cls} {d.conf:.0%}"))
                self._on_detection(d, frame, w, h, ts)
        self.annotations = annotations
        if ts - self._last_pub >= 0.25:
            self._publish(ts, fps)
            self._last_pub = ts
        return annotations

    def state(self, ts):
        if not self.moving:
            return "STOPPED"
        if ts - self._last_inpath < toy_car.AVOID_S:      # the sketch's detour around the obstacle
            return "AVOIDING"
        return "CRUISING"

    def _on_checkpoint(self, node, ts):
        self.markers_seen += 1
        traj = network.active_trajectory()
        route = traj["nodes"] if traj else []
        events.log(MODE, "info", "checkpoint", f"Checkpoint {node} reached")
        if self.trip_id is None and route and node == route[0]:
            self.trip_id = trips.start(MODE, config.PROTO_CAR_ID, route_name=traj["name"], ts=ts, source="camera")
            events.log(MODE, "info", "trip", f"Run started on '{traj['name']}'")
        elif self.trip_id and route and node == route[-1] and self.loc.idx == len(route) - 1:
            dist = self.loc.distance()
            trips.finish(self.trip_id, round(dist, 1), ts)
            events.log(MODE, "info", "trip", f"Run on '{traj['name']}' finished — {dist:.0f} cm")
            self.trip_id = None

    def _on_detection(self, d, frame, w, h, ts):
        x1, y1, x2, y2 = d.box
        proximity = max(0.0, min(1.0, (y2 - y1) / h))
        in_path = 0.3 * w <= (x1 + x2) / 2 <= 0.7 * w
        if in_path and proximity > 0.25:
            self._last_inpath = ts
        if ts - self._last_listed.get(d.cls, 0) >= LIST_EVERY_S:
            self._last_listed[d.cls] = ts
            hub.add_detection({"cls": d.cls, "conf": d.conf, "proximity": proximity, "in_path": in_path, "ts": ts,
                               "source": "camera"})
        # The track's obstacles are known: a detection updates the matching one (no new obstacle is created).
        avoiding = in_path and proximity > 0.25 and ts - self._last_avoid.get(d.cls, 0) >= toy_car.AVOID_S
        if not avoiding and ts - self._last_cls.get(d.cls, 0) < 2.0:      # don't hammer the database at 15 fps
            return
        self._last_cls[d.cls] = ts
        snapshot = None
        if avoiding:                                    # the car goes around it: keep a picture of it
            self._last_avoid[d.cls] = ts
            self._avoid_path, self._avoid_ts = self._detour_path(), ts
            snapshot = f"cam_{d.cls}_{int(ts)}.jpg"
            cv2.imwrite(str(config.SNAPSHOT_DIR / snapshot), draw_annotations(frame.copy(), [("box", d.box, d.cls)]))
        oid = obstacles.track_sighting(d.cls, d.conf, proximity, trip_id=self.trip_id, source="camera", ts=ts,
                                       encounter=avoiding, snapshot=snapshot)
        if oid is None and snapshot:                    # not one of the track's obstacles: ignored
            (config.SNAPSHOT_DIR / snapshot).unlink(missing_ok=True)
        elif oid is not None and avoiding:
            trips.bump(self.trip_id, obstacles=1, avoid=1)
            events.log(MODE, "info", "obstacle", f"Car went around the {d.cls} ({d.conf:.0%})", ts)

    def _detour_path(self):
        """Estimated detour from where the car is now (needs its position on the route: ArUco checkpoints)."""
        loc, net = self.loc, network.proto_network()
        if not loc.route or loc.idx is None or len(loc.route) < 2 or not net.is_valid_path(loc.route):
            return None
        pts = net.path_points(loc.route)
        cum = geo.cumulative(MODE, pts)
        edges = net.path_edges(loc.route)
        s = sum(net.edges[e]["length"] for e in edges[:loc.idx])
        if loc.idx < len(edges):
            s += min(loc.traveled, net.edges[edges[loc.idx]]["length"])
        x, y, h = toy_car.track_point(pts, cum, s)
        return toy_car.avoid_track(pts, cum, s, net.inside_side(x, y, h))

    def _publish(self, ts, fps):
        pos = self.loc.position()
        state = self.state(ts)
        traj = network.active_trajectory()
        t = {"car_id": config.PROTO_CAR_ID, "ts": ts, "state": state,
             "speed": self.loc.speed if self.moving else 0.0, "fps": round(fps, 1), "motion": round(self.motion, 1),
             "last_marker": self.loc.last_node, "route": traj["name"] if traj else "", "trip_id": self.trip_id,
             "progress": self.loc.progress(), "trip_dist": self.loc.distance() if self.trip_id else 0,
             "position_known": pos is not None, "source": "camera",
             "avoid_path": self._avoid_path if ts - self._avoid_ts < toy_car.AVOID_S else None}
        t.update({"x": pos[0], "y": pos[1], "heading": pos[2]} if pos else {"x": None, "y": None, "heading": 0.0})
        hub.publish(MODE, t)
        if self.trip_id and pos and ts - self._last_point >= 1:
            trips.add_point(self.trip_id, ts, pos[0], pos[1], round(t["speed"], 1), state)
            self._last_point = ts

    def on_disconnect(self):
        if self.trip_id:
            trips.finish(self.trip_id, round(self.loc.distance(), 1), time.time(), status="interrupted")
            self.trip_id = None
        self.prev_small, self.motion, self.moving = None, 0.0, False
