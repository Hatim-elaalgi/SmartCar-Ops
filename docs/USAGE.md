# SmartCar Ops — IoT dashboard

Dashboard for our obstacle-aware smart car. One app, two modes (switch in the top bar):

| Mode | Data |
|---|---|
| **Real-world (simulated)** | A simulated fleet driving on a real street map (OpenStreetMap), with 14 days of fake history. Shows how the system would work for real cars. |
| **Prototype** | Our toy car on the tape track. Live data comes from the **ESP32-CAM**. While the camera is offline, a clearly labelled **demo** car drives instead. |

## Run it

You only need Docker Desktop.

```bash
cp .env.example .env          # optional: adjust settings
docker compose up --build     # first build takes a few minutes
```

Open **http://localhost:8050**. The first start generates demo data automatically.

- Code is mounted into the container: after editing, run `docker compose restart` (or set `DEBUG=1` in `.env` for auto-reload).
- Data (SQLite, snapshots, track, settings) lives in the Docker volume `iotdash-data`. "Reset demo data" is on **Devices & logs → Settings**.

## Pages

| Page | What it does |
|---|---|
| **Overview** | Live map, key numbers, open obstacles that need attention, recent events, camera thumbnail (prototype). |
| **Tracking** | Vehicle list (problems first) · live map with trails · selected vehicle (trip progress ring). Trip history with **replay** (play/scrub/speed) and CSV export. |
| **Camera** | ESP32-CAM stream relayed by the dashboard, with detection boxes and a HUD. Connection status, IP, resolution/quality/flash/flip, snapshots. |
| **Route planner** | Pick start + destination (dropdown or click on the map). Shows the **shortest**, the **safest** and the **recommended** route for your priority. Prototype mode adds your manual **trajectories** ranked by the same cost, a **track editor**, and printable **ArUco markers**. |
| **Obstacles** | Safety inbox: Needs review / Acknowledged / Closed. Sorted by priority, with details and snapshot. Mark as acknowledged, resolved or **false positive** (removed from route risk; labelled data for retraining). |
| **Live stats** | KPIs, speed, vehicle state timeline, obstacles per hour / by object / severity mix, trips per day. |
| **AI models** | Offline metrics from `models/registry.json` (precision, recall, mAP, per-class AP, confusion matrix) + behaviour on the road (confidence, false-positive rate). |
| **Devices & logs** | Health of ESP32-CAM, Arduino, simulator, models, database · full event log · severity weights. |

## How the prototype data works (no Arduino ↔ ESP32 link)

The Arduino UNO has no Wi-Fi and is not wired to the ESP32-CAM, so ultrasonic / IR / motor readings never leave the car.
Everything the dashboard knows about the live toy car comes **from the camera image**:

```
ESP32-CAM ──MJPEG──▶ dashboard (Docker) ──▶ ArUco markers → which checkpoint, position, speed between checkpoints
                                       ├──▶ frame differences → moving / stopped
                                       ├──▶ your model → obstacles (class, confidence, proximity, in path)
                                       └──▶ relay /camera/stream.mjpg → every browser tab
```

1. **Print the markers**: Route planner → Prototype → *Download markers*. Print each ~5 cm wide and stick it next to the tape at its checkpoint, facing the camera's direction of travel.
2. **Pick the run**: Route planner → *Your trajectories* → select → *Use selected for the next run*. Position is estimated along that trajectory.
3. A run starts when the camera sees the first checkpoint and ends at the last one.

## ESP32-CAM and its dynamic IP

The ESP32-CAM accepts **one** stream viewer at a time, so the dashboard is that single viewer and relays the stream to all browser tabs.

The IP is found in this order: IP typed on the Camera page → `CAMERA_IP` in `.env` → last IP that worked → **network scan**.
Docker can't see `.local` names, so put your network first in `CAMERA_SCAN_SUBNETS`: run `ipconfig` on the laptop; if your IPv4 address is `192.168.1.23`, use `192.168.1.0/24`. Phone hotspots (`192.168.43.x`, `172.20.10.x`) and the Windows hotspot (`192.168.137.x`) are already in the defaults.

Expected firmware: the standard **CameraWebServer** sketch (`/status` and `/control` on port 80, `/stream` on port 81). For a different stream URL, set `CAMERA_STREAM_URL`.

**No camera at hand?** Run the fake one, which streams moving floor, checkpoint markers and red "obstacles":

```bash
# in .env: CAMERA_IP=fakecam   and   DETECTOR_PROTO=python:color_blob_detector:ColorBlobDetector
docker compose --profile fakecam up
```

## Plug in your model

Set in `.env`, then `docker compose up -d`:

| `DETECTOR_PROTO` / `DETECTOR_REAL` | Meaning |
|---|---|
| `none` | stream only |
| `yolo:models/best.pt` | Ultralytics YOLO weights in `models/`. Build with `WITH_YOLO=1 docker compose build` (installs CPU torch). |
| `python:my_detector:MyDetector` | Any framework: `models/my_detector.py` with a class whose `detect(frame_bgr)` returns `[{"cls", "conf", "box": [x1, y1, x2, y2]}]`. See `models/color_blob_detector.py`. |
| `esp32` | Inference runs **on the camera**: the dashboard polls `http://<cam-ip>/detections`, which must return the same JSON list. |

**Real road model in the Real-world mode**: put some real road photos in a folder of the project and set
`DETECTOR_REAL=yolo:models/real_road.pt` + `REAL_IMAGES_DIR=datasets/road_samples`. Simulated encounters then use your model's class, confidence and the annotated photo.

Each class has a danger weight (editable in **Devices & logs → Settings**). Defaults exist for the toy-track classes (`column`, `box`) and the real-road classes. Any other class name gets 0.5; add it to `DEFAULT_CLASS_WEIGHTS` in `config.py` to give it its own weight.
Then put your real evaluation numbers in `models/registry.json` and set `"example": false`.

## Real road obstacle detection (deep learning)

The Real-world mode has a **real** perception pipeline: **CAR-01's dashcam** plays real road videos (or a live phone / ESP32 stream), and two YOLO models detect obstacles in real time. Each detection is placed on the map ahead of CAR-01 at its estimated distance, is saved as an obstacle with a snapshot, and makes the car slow down or brake.

| Model | Classes | Data | |
|---|---|---|---|
| Road users | pedestrian, cyclist, vehicle, animal | COCO 2017 (118k images) | YOLO26n pretrained |
| Road hazards | pothole, cone, barrier, roadwork sign | Potholes dashcam dataset (Bučko et al., *Sensors* 2022, CC BY 4.0) + ROADWork (CMU, ICCV 2025, ODC-By) | YOLO26s **fine-tuned overnight in the notebook** |

**Train overnight** (GPU, official `ultralytics/ultralytics` image + JupyterLab) — about 6.5 h: ~20 min of downloads (≈ 8 GB, first time only), **6 h of training**, ~10 min of evaluation / export / clips.

```powershell
docker compose stop dashboard            # free CPU/RAM (the script also does it); stop other big containers too
pwsh -ExecutionPolicy Bypass -File training/run_training.ps1             # 6 h, live progress in the terminal
pwsh -ExecutionPolicy Bypass -File training/run_training.ps1 -Hours 8    # longer
pwsh -ExecutionPolicy Bypass -File training/run_training.ps1 -Resume     # continue after an interruption
```

The script keeps Windows awake while it runs (keep the laptop plugged in, lid open), shows every line of the notebook live (epoch progress bars, validation mAP after each epoch), saves the outputs into the notebook, and restarts the dashboard with the new models at the end.
…or run it **in VS Code on the Docker kernel** (GPU + all libraries live in the `trainer` container):

1. `docker compose --profile train up -d trainer` (Jupyter kernel server on http://localhost:8888, token `smartcar`)
2. Open `training/train_road_obstacles.ipynb` → **Select Kernel** (top right) → **Select Another Kernel…** → **Existing Jupyter Server…** → paste `http://localhost:8888/?token=smartcar` → name it `smartcar-docker` → **Python 3 (ipykernel)**
3. **Run All** (or ▶ each cell). Stop the server afterwards with `docker compose --profile train stop trainer`.
4. To follow the night (even after closing VS Code — training keeps running in the container): open `training/monitor_training.ipynb` on the same kernel and run it (curves, best mAP, time left; the 2nd cell refreshes every minute).

(The same server works in the browser: http://localhost:8888, token `smartcar`.)

Settings worth knowing (first cell of the notebook): `TRAIN_HOURS = 6`, `MODEL = "yolo26s.pt"`, `BATCH = 4` (largest that fits the 4 GB GTX 1650 Ti in full precision; AMP is automatically disabled on GTX 16xx), all pothole frames, 5,000 / 600 / 600 ROADWork images.

The notebook downloads only the needed images straight out of the remote archives, trains, evaluates on a held-out test split, exports both models to ONNX (`models/road_users.onnx`, `models/road_hazards.onnx`, `models/real_road.json`), writes the **real metrics** to `models/registry.json`, and cuts the real dashcam demo clips into `demo/videos/` (sources in `demo/videos/CREDITS.md`). The dashboard then runs the models on CPU with onnxruntime — no PyTorch needed.

**Use a real camera**: Camera page (Real-world) → source `http://<phone-ip>:8080/video` with the free *IP Webcam* app on a phone fixed on a dashboard, or `esp32`.

## Severity and priority

`score = 100 × object danger × (0.5 + 0.5 × proximity) × confidence × (1 if in the car's path else 0.5)`
→ Critical ≥ 60, High ≥ 40, Medium ≥ 20, Low. The queue shows open obstacles first, then score, then recency. Repeated detections within 25 m (real) / 30 cm (prototype) are merged.

Route risk per segment = sum of `score/100 × status weight × recency` of its obstacles. Status weight: open = 1, resolved = 0.3, false positive = 0. Recency halves every 7 days (real) / 2 days (prototype).
Route cost = `length + priority × risk × (150 m | 60 cm)`.

## Use your own city

```bash
docker compose run --rm dashboard python -m tools.build_road_graph --bbox 48.842,2.340,48.852,2.356 --name "Our campus"
```

`--bbox` is south,west,north,east (about 1 km × 1 km works well). Then **Reset demo data**.

## Tests

```bash
docker compose run --rm --no-deps dashboard python -m tests.smoke_test   # every page callback, both modes
docker compose run --rm --no-deps dashboard python -m tests.camera_e2e   # real camera path vs. fake ESP32-CAM
docker compose run --rm --no-deps dashboard python -m tests.dashcam_e2e  # real video -> models -> map -> car reaction
```

## Project layout

```
app.py                 Dash app shell (sidebar, mode switch), camera relay route
config.py              all settings (env variables)
pages/                 one file per page
components/            theme (palette, Plotly template), UI pieces, map/track figures
core/                  database, obstacles + severity, trips, events, road/track network + routing, planner
sources/               live data: real-world fleet simulator, dashcam (real video + models), prototype demo,
                       ESP32-CAM service, perception, ONNX detectors
training/              train_road_obstacles.ipynb (GPU training), run_training.ps1 (live terminal run),
                       Dockerfile (ultralytics GPU image + JupyterLab), remote_zip.py, fetch_demo_clips.py
demo/videos/           real dashcam clips used by the Real-world demo (CREDITS.md = sources + licences)
tools/                 seed (demo data), build_road_graph (OpenStreetMap)
resources/             road_graph.json (real streets), track_default.json (tape track)
models/                road_users.onnx, road_hazards.onnx, real_road.json, registry.json (real metrics)
assets/                CSS (liquid-glass theme), Bootstrap + icons (bundled, works offline)
tests/                 smoke test, camera + dashcam end-to-end tests, fake ESP32-CAM
```

Map tiles need internet; everything else works offline (useful on a phone hotspot without data).
