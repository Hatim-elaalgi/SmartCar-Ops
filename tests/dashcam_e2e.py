"""End-to-end test of the real-world dashcam: video -> ONNX models -> obstacles ahead of CAR-01 -> car reaction.

    docker compose run --rm --no-deps dashboard python -m tests.dashcam_e2e [video folder or file]
"""
import os
import sys
import tempfile
import time

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="dashcam_e2e_")   # never touch the dashboard's data

import config  # noqa: E402

config.CAMERA_SCAN_SUBNETS = []
if len(sys.argv) > 1:
    config.REAL_VIDEO_SOURCE = sys.argv[1]

from core import db, settings  # noqa: E402
from core.hub import hub  # noqa: E402
from sources.camera import CameraService  # noqa: E402
from sources.dashcam import DashcamService  # noqa: E402
from sources.sim_real import RealFleetSimulator  # noqa: E402


def main():
    db.init()
    settings.set("real_video_source", config.REAL_VIDEO_SOURCE)
    fleet = RealFleetSimulator()
    dash = DashcamService(CameraService(), fleet)
    fleet.dashcam = dash
    fleet.start()
    dash.start()
    t0 = time.time()
    states, n_dets = set(), 0
    while time.time() - t0 < 45:
        time.sleep(3)
        info = dash.info()
        car = hub.car(config.REAL, config.DASHCAM_CAR_ID) or {}
        states.add(car.get("state"))
        dets = hub.detections(60, mode=config.REAL)
        n_dets = len(dets)
        models = ", ".join(f"{m['id']} {m['ms']:.0f} ms" for m in info["models"])
        print(f"{time.time() - t0:4.0f}s {info['status']:<9} clip={info['clip']} video={info['read_fps']:.0f}fps "
              f"AI={info['det_fps']:.1f}fps [{models}] dets={n_dets} car={car.get('state')} "
              f"classes={sorted({d['cls'] for d in dets})}")
    rows = db.query("SELECT cls, level, distance, snapshot, car_id FROM obstacles WHERE source='camera' "
                    "AND mode='real' AND ts_first >= ?", (t0,))
    jpeg, _ = dash.latest_jpeg()
    checks = {
        "video playing": info["status"] == "playing",
        # the hazards model only exists after training/train_road_obstacles.ipynb has run
        "ONNX models loaded": len(info["models"]) >= 1,
        "models run in real time (>= 3 analysed frames/s)": info["det_fps"] >= 3,
        "detections produced": n_dets > 0,
        "obstacles stored ahead of CAR-01 with snapshots": any(r["snapshot"] and r["car_id"] == "CAR-01" for r in rows),
        "CAR-01 reacted (avoiding / stopped)": bool(states & {"AVOIDING", "STOPPED"}),
        "relay serves JPEG": jpeg[:2] == b"\xff\xd8",
    }
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print(f"  obstacles stored: {len(rows)} -> {sorted({r['cls'] for r in rows})}")
    print(f"  models: {[m['id'] for m in info['models']]} · detector: {info['detector']}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
