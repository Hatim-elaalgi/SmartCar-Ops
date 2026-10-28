"""Central settings. Everything can be overridden with environment variables (see .env.example)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RESOURCES_DIR = BASE_DIR / "resources"
MODELS_DIR = BASE_DIR / "models"

DB_PATH = DATA_DIR / "dashboard.db"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
SETTINGS_PATH = DATA_DIR / "settings.json"
TRACK_PATH = DATA_DIR / "track.json"                 # editable copy of the prototype track
DEFAULT_TRACK_PATH = RESOURCES_DIR / "track_default.json"
ROAD_GRAPH_PATH = RESOURCES_DIR / "road_graph.json"  # real road network (built by tools/build_road_graph.py)
MODEL_REGISTRY_PATH = MODELS_DIR / "registry.json"

PORT = int(os.getenv("PORT", "8050"))
DEBUG = os.getenv("DEBUG", "0") == "1"

# Modes: the same pages show either the simulated real-world fleet or the toy-car prototype.
REAL = "real"
PROTO = "proto"
MODE_LABELS = {REAL: "Real-world (simulated)", PROTO: "Prototype"}

# --- ESP32-CAM ---------------------------------------------------------------
CAMERA_IP = os.getenv("CAMERA_IP", "").strip()
CAMERA_SCAN_SUBNETS = [
    s.strip() for s in os.getenv(
        "CAMERA_SCAN_SUBNETS",
        "192.168.1.0/24,192.168.0.0/24,192.168.43.0/24,192.168.137.0/24,172.20.10.0/28,10.0.0.0/24",
    ).split(",") if s.strip()
]
CAMERA_STREAM_URL = os.getenv("CAMERA_STREAM_URL", "http://{ip}:81/stream")
CAMERA_STATUS_URL = "http://{ip}/status"
CAMERA_CONTROL_URL = "http://{ip}/control?var={var}&val={val}"
CAMERA_RESCAN_SECONDS = 30
# Our ESP32-CAM is mounted on its side: its image is turned 90° clockwise (0 / 90 / 180 / 270, also on the Camera page)
CAMERA_ROTATE = int(os.getenv("CAMERA_ROTATE", "90"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "4.0"))  # mean pixel change that means "car is moving"

PROTO_DEMO_WHEN_OFFLINE = os.getenv("PROTO_DEMO_WHEN_OFFLINE", "1") == "1"
PROTO_CAR_ID = "PROTO-1"
# Toy car speed on straight tape with MOTOR_SPEED = 190 in the Arduino sketch (core/toy_car.py). Estimate:
# 190/255 = 75 % PWM of the ~4 V the L298N leaves from 4 x AA -> TT 1:48 motor ~95 rpm under load x 65 mm wheel
# (20.4 cm per turn) ≈ 30 cm/s. Measure it (time 1 m of straight tape) and set PROTO_SPEED_CM_S; with ArUco markers
# on the track the camera also measures the real speed between checkpoints.
PROTO_SPEED_CM_S = float(os.getenv("PROTO_SPEED_CM_S", "30"))
ARUCO_DICT = "DICT_4X4_50"       # printed checkpoint markers

# --- Models ------------------------------------------------------------------
# "auto" = models/proto_toy.json produced by training/train_toy_obstacles.ipynb (our toy-track obstacles)
DETECTOR_PROTO = os.getenv("DETECTOR_PROTO", "auto")
# "auto" = models/real_road.json produced by training/train_road_obstacles.ipynb (2 ONNX models)
DETECTOR_REAL = os.getenv("DETECTOR_REAL", "auto")
REAL_IMAGES_DIR = os.getenv("REAL_IMAGES_DIR", "").strip()

# --- Real-world dashcam (real video -> real models -> obstacles on the map) ---------------
REAL_VIDEO_SOURCE = os.getenv("REAL_VIDEO_SOURCE", "demo/videos")   # folder | file | http(s)/rtsp URL | esp32
DASHCAM_CAR_ID = "CAR-01"
DASHCAM_HORIZON = float(os.getenv("DASHCAM_HORIZON", "0.5"))      # horizon line, fraction of image height
DASHCAM_HEIGHT_M = float(os.getenv("DASHCAM_HEIGHT_M", "1.3"))    # camera height above the road
DASHCAM_HFOV_DEG = float(os.getenv("DASHCAM_HFOV_DEG", "90"))     # horizontal field of view

# --- Simulation ----------------------------------------------------------------
REAL_FLEET_SIZE = int(os.getenv("REAL_FLEET_SIZE", "3"))
REAL_CRUISE_MS = 8.3             # ~30 km/h, used for ETAs

REAL_CLASSES = ["pedestrian", "cyclist", "vehicle", "animal", "barrier", "roadwork", "pothole", "debris", "cone"]
PROTO_CLASSES = ["column", "box"]       # the obstacles of our toy track (toy model classes)

# How dangerous each class is (0..1). Editable on the System page.
DEFAULT_CLASS_WEIGHTS = {
    "pedestrian": 1.0, "cyclist": 0.95, "animal": 0.9, "vehicle": 0.85, "barrier": 0.75, "roadwork": 0.5,
    "pothole": 0.6, "debris": 0.5, "cone": 0.4,
    "column": 0.7, "box": 0.6,
}

# Units per mode: real distances in metres, prototype distances in centimetres.
UNITS = {REAL: {"dist": "m", "speed": "km/h"}, PROTO: {"dist": "cm", "speed": "cm/s"}}
ONLINE_SECONDS = 5               # a car is "online" if we heard from it in the last N seconds
