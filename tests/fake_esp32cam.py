"""A fake ESP32-CAM (CameraWebServer API) for testing and demos without the hardware.

  port 80: /status (JSON with framesize), /control?var=..&val=..
  port 81: /stream (MJPEG) — a synthetic floor view with the tape line, the ArUco marker of each checkpoint
           of the active trajectory passing by, and a red box (an "obstacle") from time to time.

Run it next to the dashboard:
    docker compose --profile fakecam up        (and set CAMERA_IP=fakecam in .env)
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import config
from core import network
from sources.camera import ROTATIONS

W, H, FPS = 640, 480, 12
SECONDS_PER_CHECKPOINT = 4.0


def _route_markers():
    traj = network.active_trajectory()
    net = network.proto_network()
    nodes = traj["nodes"] if traj else list(net.nodes)
    return [net.nodes[n]["marker"] for n in nodes if net.nodes[n]["marker"] is not None]


def make_frame(t, markers, dictionary):
    img = np.full((H, W, 3), (58, 60, 62), np.uint8)
    rng = np.random.default_rng(int(t * FPS))
    # moving floor texture so the dashboard sees motion
    offset = int(t * 120) % 40
    for y in range(-40 + offset, H, 40):
        cv2.rectangle(img, (0, y), (W, y + 20), (80, 82, 84), -1)
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)
    cv2.line(img, (W // 2, H), (W // 2 + 30, 0), (20, 20, 20), 26)          # black tape
    phase = t % SECONDS_PER_CHECKPOINT
    if markers and phase < 1.6:                                               # a checkpoint marker passes by
        k = int(t // SECONDS_PER_CHECKPOINT) % len(markers)
        side = 150
        marker = cv2.aruco.generateImageMarker(dictionary, markers[k], side)
        y0 = int(40 + phase * 150)
        canvas = np.full((side + 30, side + 30), 255, np.uint8)
        canvas[15:15 + side, 15:15 + side] = marker
        img[y0:y0 + side + 30, 60:60 + side + 30] = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    if 2.0 < phase < 3.2:                                                     # a red "obstacle" in the path
        s = int(60 + (phase - 2.0) * 90)
        cv2.rectangle(img, (W // 2 - s // 2, H - 60 - s), (W // 2 + s // 2, H - 60), (40, 40, 210), -1)
    cv2.putText(img, f"FAKE ESP32-CAM {time.strftime('%H:%M:%S')}", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1, cv2.LINE_AA)
    back = (360 - config.CAMERA_ROTATE) % 360                                 # mounted on its side like the real one
    return cv2.rotate(img, ROTATIONS[back]) if back in ROTATIONS else img


class ControlHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body = json.dumps({"framesize": 8, "quality": 12, "vflip": 0, "hmirror": 0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif self.path.startswith("/control"):
            body = b"OK"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
        else:
            body = b"<html><body>Fake ESP32-CAM</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/stream"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=123456789000000000000987654321")
        self.end_headers()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
        start = time.time()
        markers = _route_markers()
        try:
            while True:
                ok, jpg = cv2.imencode(".jpg", make_frame(time.time() - start, markers, dictionary),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.wfile.write(b"\r\n--123456789000000000000987654321\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes())
                time.sleep(1 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def serve(host="0.0.0.0", control_port=80, stream_port=81):
    servers = [ThreadingHTTPServer((host, control_port), ControlHandler),
               ThreadingHTTPServer((host, stream_port), StreamHandler)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    return servers


if __name__ == "__main__":
    serve()
    print("Fake ESP32-CAM running: http://<host>/status and http://<host>:81/stream")
    while True:
        time.sleep(3600)
