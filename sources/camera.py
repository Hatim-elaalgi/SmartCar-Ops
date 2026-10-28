"""ESP32-CAM: discovery (dynamic IP), MJPEG stream reading, and relay to the dashboard.

Why a relay? The ESP32-CAM serves ONE stream client at a time. The dashboard server is that one client;
every browser tab then watches /camera/stream.mjpg on the dashboard instead of the camera itself.

Discovery order: IP typed on the Camera page > CAMERA_IP in .env > last IP that worked > subnet scan.
(Inside Docker, mDNS names like esp32cam.local don't resolve, so we scan CAMERA_SCAN_SUBNETS.)
"""
import ipaddress
import logging
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

import config
from core import events, settings
from core.hub import hub
from sources import detector as detectors
from sources.drawing import draw_annotations, placeholder
from sources.perception import Perception

log = logging.getLogger(__name__)

# esp32-camera framesize values (recent firmware)
FRAMESIZES = {"QVGA 320×240": 5, "CIF 400×296": 6, "VGA 640×480": 8, "SVGA 800×600": 9, "XGA 1024×768": 10}
# The ESP32-CAM can only flip / mirror: a camera mounted on its side is turned upright here (degrees, clockwise),
# before the model sees the frame — it was trained on upright images.
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# "Record frames" on the Camera page: upright frames without overlay, for (re)training the toy model.
RECORD_DIR = config.BASE_DIR / "datasets" / "toy" / "esp32"
RECORD_FPS = 3          # neighbouring frames are near-duplicates: a few per second is plenty


def _port_open(ip, port, timeout=0.35):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe(ip, timeout=0.8):
    """True if `ip` looks like an ESP32-CAM (CameraWebServer /status, or an MJPEG stream)."""
    try:
        r = requests.get(config.CAMERA_STATUS_URL.format(ip=ip), timeout=timeout)
        if r.ok and "framesize" in r.text:
            return True
    except requests.RequestException:
        pass
    try:
        r = requests.get(config.CAMERA_STREAM_URL.format(ip=ip), stream=True, timeout=timeout)
        ok = "multipart" in r.headers.get("Content-Type", "")
        r.close()
        return ok
    except requests.RequestException:
        return False


class CameraService(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="esp32-cam")
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._reset = False
        self.status, self.ip, self.method, self.error = "starting", None, None, None
        self.fps, self.resolution, self.last_frame_ts, self.frames = 0.0, None, 0.0, 0
        self.connected_since = None
        self.scan_info = ""
        self._jpeg, self._frame, self._frame_id = None, None, None
        self._annotated_cache = (None, None)
        self._placeholder = (None, None)
        self.detector = None
        self.perception = None
        self._work = threading.Event()
        self.rotation = int(settings.get("camera_rotate", config.CAMERA_ROTATE))
        self.recording, self.rec_dir, self.rec_frames, self._rec_last = False, None, 0, 0.0

    # --- public API (used by pages and Flask routes) -------------------------------
    @property
    def connected(self):
        return self.status == "connected" and time.time() - self.last_frame_ts < 5

    def info(self):
        with self._lock:
            return {
                "status": "connected" if self.connected else self.status, "ip": self.ip, "method": self.method,
                "error": self.error, "fps": round(self.fps, 1), "resolution": self.resolution,
                "last_frame_ts": self.last_frame_ts, "frames": self.frames, "scan_info": self.scan_info,
                "connected_since": self.connected_since, "rotation": self.rotation,
                "recording": self.recording, "rec_frames": self.rec_frames,
                "rec_dir": self.rec_dir.relative_to(config.BASE_DIR).as_posix() if self.rec_dir else None,
                "detector": self.detector.name if self.detector else "—",
                "detector_error": self.detector.error if self.detector else None,
                "markers_seen": self.perception.markers_seen if self.perception else 0,
            }

    def latest_jpeg(self, annotated=True):
        """(jpeg_bytes, frame_id). Falls back to a placeholder image when offline."""
        with self._lock:
            jpeg, frame, fid = self._jpeg, self._frame, self._frame_id
        if jpeg is None or not self.connected:
            text = {"searching": "Searching for the ESP32-CAM…", "connecting": "Connecting…"}.get(
                self.status, "Camera offline")
            sub = f"Last known IP {self.ip}" if self.ip else "Set the IP on the Camera page or check CAMERA_SCAN_SUBNETS"
            key = f"ph-{text}-{sub}"
            if self._placeholder[0] != key:
                self._placeholder = (key, placeholder(text, sub))
            return self._placeholder[1], key
        annotations = self.perception.annotations if (annotated and self.perception) else []
        if not annotations:
            return jpeg, fid
        cached_id, cached = self._annotated_cache
        if cached_id == fid:
            return cached, fid
        ok, buf = cv2.imencode(".jpg", draw_annotations(frame.copy(), annotations), [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._annotated_cache = (fid, buf.tobytes())
        return self._annotated_cache[1], fid

    def set_manual_ip(self, ip):
        settings.set("camera_ip", ip or "")
        self.restart_discovery()

    def set_recording(self, on):
        """Start / stop saving frames to datasets/toy/esp32/<date_time>/ (a new folder per recording)."""
        with self._lock:
            if on and not self.recording:
                self.rec_dir = RECORD_DIR / time.strftime("%Y%m%d_%H%M%S")
                self.rec_dir.mkdir(parents=True, exist_ok=True)
                self.rec_frames = 0
            self.recording = bool(on)

    def set_rotation(self, degrees):
        settings.set("camera_rotate", int(degrees))
        self.rotation = int(degrees)

    def restart_discovery(self):
        with self._lock:
            self.ip, self._reset = None, True
        self._wake.set()

    def control(self, var, val):
        if not self.ip:
            return False, "Camera not connected"
        try:
            r = requests.get(config.CAMERA_CONTROL_URL.format(ip=self.ip, var=var, val=val), timeout=2)
            return r.ok, ("OK" if r.ok else f"Camera answered {r.status_code}")
        except requests.RequestException as exc:
            return False, str(exc)

    def save_snapshot(self):
        jpeg, _ = self.latest_jpeg(annotated=True)
        if not self.connected or not jpeg:
            return None
        name = f"snap_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.jpg"
        (config.SNAPSHOT_DIR / name).write_bytes(jpeg)
        return name

    # --- thread -----------------------------------------------------------------------
    def run(self):
        self.detector = detectors.load(config.DETECTOR_PROTO, ip_getter=lambda: self.ip,
                                       default_profile=detectors.PROTO_PROFILE)
        self.perception = Perception(self.detector)
        threading.Thread(target=self._perception_loop, daemon=True, name="perception").start()
        failures = 0
        while True:
            self._wake.clear()
            if not self.ip:
                self._set(status="searching", error=None)
                ip, method = self._discover()
                if not ip:
                    self._set(status="offline", error="ESP32-CAM not found")
                    self._wake.wait(config.CAMERA_RESCAN_SECONDS)
                    continue
                self._set(ip=ip, method=method)
                settings.set("last_camera_ip", ip)
                events.log("system", "info", "camera", f"ESP32-CAM found at {ip} ({method})")
            try:
                self._stream()
                failures = 0 if self._reset else failures + 1
            except Exception as exc:
                failures += 1
                self._set(error=str(exc)[:200])
            if self.status == "connected":
                events.log("system", "warning", "camera", "ESP32-CAM stream lost")
                self.perception.on_disconnect()
            self._set(status="offline")
            hub.set_source(config.PROTO, "offline")
            if self._reset:
                self._reset = False
                continue
            if failures >= 3:
                self._set(ip=None)
                failures = 0
            self._wake.wait(min(10, 2 * failures))

    def _set(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _discover(self):
        for method, ip in (("set on dashboard", settings.get("camera_ip")), ("CAMERA_IP", config.CAMERA_IP),
                           ("last known IP", settings.get("last_camera_ip"))):
            if ip and probe(ip):
                return ip, method
        return self._scan(), "network scan"

    def _scan(self):
        hosts = []
        for subnet in config.CAMERA_SCAN_SUBNETS:
            try:
                hosts += [str(h) for h in ipaddress.ip_network(subnet, strict=False).hosts()]
            except ValueError:
                log.warning("Bad subnet in CAMERA_SCAN_SUBNETS: %s", subnet)
        stream_port = urlparse(config.CAMERA_STREAM_URL.format(ip="x")).port or 80
        self._set(scan_info=f"Scanning {len(hosts)} addresses in {', '.join(config.CAMERA_SCAN_SUBNETS)}")

        def check(ip):
            return ip if (_port_open(ip, stream_port) or _port_open(ip, 80)) and probe(ip) else None

        with ThreadPoolExecutor(max_workers=96) as pool:
            futures = [pool.submit(check, ip) for ip in hosts]
            for f in as_completed(futures):
                if f.result():
                    for other in futures:
                        other.cancel()
                    self._set(scan_info=f"Found at {f.result()}")
                    return f.result()
        self._set(scan_info=f"Nothing found in {len(hosts)} addresses — last scan {time.strftime('%H:%M:%S')}")
        return None

    def _stream(self):
        url = config.CAMERA_STREAM_URL.format(ip=self.ip)
        self._set(status="connecting")
        with requests.get(url, stream=True, timeout=(4, 8)) as r:
            r.raise_for_status()
            buf = b""
            for chunk in r.iter_content(chunk_size=16384):
                if self._reset:
                    return
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                while start != -1 and end != -1:
                    self._on_frame(buf[start:end + 2])
                    buf = buf[end + 2:]
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if len(buf) > 4_000_000:
                    buf = b""

    def _on_frame(self, jpeg):
        now = time.time()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        if self.rotation in ROTATIONS:
            frame = cv2.rotate(frame, ROTATIONS[self.rotation])
            jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        with self._lock:
            if self.last_frame_ts:
                inst = 1.0 / max(1e-3, now - self.last_frame_ts)
                self.fps = inst if self.fps == 0 else 0.85 * self.fps + 0.15 * inst
            if self.status != "connected":
                self.status, self.error, self.connected_since = "connected", None, now
                first = True
            else:
                first = False
            self.last_frame_ts, self.frames = now, self.frames + 1
            self._jpeg, self._frame, self._frame_id = jpeg, frame, self.frames
            self.resolution = f"{frame.shape[1]}×{frame.shape[0]}"
            record = self.recording and now - self._rec_last >= 1 / RECORD_FPS
            if record:
                self._rec_last, self.rec_frames = now, self.rec_frames + 1
                path = self.rec_dir / f"esp32_{self.rec_frames:05d}.jpg"
        if record:
            path.write_bytes(jpeg)
        if first:
            events.log("system", "info", "camera", f"ESP32-CAM stream connected ({self.resolution})")
            hub.set_source(config.PROTO, "camera")
        self._work.set()

    def _perception_loop(self):
        """Runs the (possibly slow) model on the newest frame only, so the stream never lags behind."""
        done_id = None
        while True:
            self._work.wait(1.0)
            self._work.clear()
            with self._lock:
                frame, fid, fps = self._frame, self._frame_id, self.fps
            if frame is None or fid == done_id or not self.connected:
                continue
            done_id = fid
            try:
                self.perception.process(frame, time.time(), fps)
                hub.set_source(config.PROTO, "camera")
            except Exception:
                log.exception("Perception failed")
