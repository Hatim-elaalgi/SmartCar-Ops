"""In-memory live state shared by the data sources (simulators, camera) and the pages.

Every source publishes the same telemetry dict, so pages never care where data comes from:
    {"car_id", "ts", "x", "y", "heading", "speed", "state", "trip_id", ...optional fields}
"""
import threading
import time
from collections import defaultdict, deque

import config

HISTORY_SECONDS = 900


class Hub:
    def __init__(self):
        self._lock = threading.RLock()
        self._latest = {config.REAL: {}, config.PROTO: {}}
        self._history = {config.REAL: defaultdict(deque), config.PROTO: defaultdict(deque)}
        self._source = {config.REAL: "simulated", config.PROTO: "offline"}
        self._detections = {config.REAL: deque(maxlen=60), config.PROTO: deque(maxlen=60)}

    def publish(self, mode, telemetry):
        t = dict(telemetry)
        t.setdefault("ts", time.time())
        with self._lock:
            self._latest[mode][t["car_id"]] = t
            hist = self._history[mode][t["car_id"]]
            hist.append(t)
            while hist and t["ts"] - hist[0]["ts"] > HISTORY_SECONDS:
                hist.popleft()

    def cars(self, mode, online_only=False):
        now = time.time()
        with self._lock:
            cars = [dict(c) for c in self._latest[mode].values()]
        for c in cars:
            c["online"] = now - c["ts"] < config.ONLINE_SECONDS
        if online_only:
            cars = [c for c in cars if c["online"]]
        return sorted(cars, key=lambda c: c["car_id"])

    def car(self, mode, car_id):
        for c in self.cars(mode):
            if c["car_id"] == car_id:
                return c
        return None

    def history(self, mode, car_id, seconds=300):
        cutoff = time.time() - seconds
        with self._lock:
            return [dict(t) for t in self._history[mode].get(car_id, ()) if t["ts"] >= cutoff]

    def set_source(self, mode, label):
        with self._lock:
            self._source[mode] = label

    def source(self, mode):
        with self._lock:
            return self._source[mode]

    def add_detection(self, detection, mode=config.PROTO):
        with self._lock:
            self._detections[mode].appendleft(dict(detection, ts=detection.get("ts", time.time())))

    def detections(self, limit=15, mode=config.PROTO):
        with self._lock:
            return list(self._detections[mode])[:limit]


hub = Hub()
