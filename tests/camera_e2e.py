"""End-to-end test of the real camera path against the fake ESP32-CAM:
discovery -> MJPEG stream -> ArUco checkpoints -> position -> detector -> track obstacles -> telemetry -> relay.

    docker compose run --rm --no-deps dashboard python -m tests.camera_e2e
"""
import os
import sys
import tempfile
import time

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="camera_e2e_")   # never touch the dashboard's data / camera IP

import config  # noqa: E402

config.CAMERA_IP = "127.0.0.1"
config.CAMERA_SCAN_SUBNETS = []
config.DETECTOR_PROTO = "python:color_blob_detector:ColorBlobDetector"

from core import db, obstacles as track, settings  # noqa: E402
from core.hub import hub  # noqa: E402
from sources.camera import CameraService  # noqa: E402
from tests import fake_esp32cam  # noqa: E402


def main():
    db.init()
    track.ensure_track_obstacles()
    settings.set("camera_ip", "")
    settings.set("last_camera_ip", "")
    fake_esp32cam.serve(host="127.0.0.1")
    cam = CameraService()
    cam.start()
    t0 = time.time()
    while time.time() - t0 < 30:
        time.sleep(1)
        info = cam.info()
        car = hub.car(config.PROTO, config.PROTO_CAR_ID) or {}
        print(f"{time.time() - t0:4.0f}s status={info['status']:<10} fps={info['fps']:5.1f} "
              f"markers={info['markers_seen']} detector={info['detector']!r} "
              f"pos={'yes' if car.get('position_known') else 'no '} state={car.get('state')} "
              f"last_cp={car.get('last_marker')} source={hub.source(config.PROTO)}")
        if info["markers_seen"] >= 3 and car.get("position_known"):
            break
    # Detections update the track's own obstacles (no new obstacle); going around one is logged with a picture.
    met = db.query("SELECT e.cls, o.snapshot FROM encounters e JOIN obstacles o ON o.id = e.obstacle_id "
                   "WHERE e.source='camera' AND e.ts >= ?", (t0,))
    n_obstacles = db.query_one("SELECT COUNT(*) n FROM obstacles WHERE mode=?", (config.PROTO,))["n"]
    jpeg, _ = cam.latest_jpeg(annotated=True)
    checks = {
        "found camera by CAMERA_IP": info["method"] == "CAMERA_IP",
        "stream connected": info["status"] == "connected" and info["fps"] > 3,
        "checkpoints read from ArUco markers": info["markers_seen"] >= 3,
        "position estimated on the track": bool(car.get("position_known")),
        "track obstacle gone around, with a camera snapshot": any((m["snapshot"] or "").startswith("cam_")
                                                                   for m in met),
        "no obstacle created besides the track's": n_obstacles == len(track.network.proto_network().obstacles),
        "data source switched to camera": hub.source(config.PROTO) == "camera",
        "relay serves JPEG frames": jpeg[:2] == b"\xff\xd8",
    }
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"  encounters logged: {len(met)} · prototype obstacles: {n_obstacles}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
