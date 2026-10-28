"""Generates train_toy_obstacles.ipynb — the obstacle detector of OUR toy track (Prototype mode).
Run:  python build_toy_notebook.py"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "id": f"t{len(cells):02d}", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text, tags=None):
    cells.append({"cell_type": "code", "id": f"t{len(cells):02d}", "metadata": {"tags": tags or []},
                  "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Toy-track obstacle detector — training notebook (Prototype / MVP car)

Detects the obstacles of **our own track** in the ESP32-CAM image, so the toy car (Prototype mode of the dashboard) reacts to them.

| Class | Object |
|---|---|
| `column` | striped taped column with a black top |
| `box` | cream "Mutesa" box (red logo on one face) |

### Data — our own video
* **249 frames** filmed at car height while following the black tape (`datasets/toy/raw/frames`).
* **Automatic labels with SAM 2** (`toy_autolabel.py`): one click on each obstacle in a reference frame, SAM 2 follows it through the video and gives its exact outline → 141 boxes, every one checked visually. Frames without an obstacle are kept as *empty track* examples.
* **Split in time blocks** (`toy_dataset.py`): the video is one continuous drive, neighbouring frames are almost identical, so whole blocks go to validation / test, with gaps — no near-duplicate frame is shared between splits.
* **The ESP32-CAM streams 320 × 240 (4:3)** while the video is 832 × 464 (16:9, phone): the model is trained at **320 px**, and training images are turned into **ESP32-CAM-style frames** (random 4:3 crop → 320 × 240, boxes adjusted, softer image, colour cast, noise, strong JPEG).
* **Copy-paste** (training only): obstacles cut out with their SAM outline and pasted on empty-track frames at new places and distances.
* **Two test sets**: the real test frames, and the same frames converted to 320 × 240 ESP32-CAM style (`test_esp32`).

### ▶ How to run it
Same Docker kernel as the other notebook: `docker compose --profile train up -d trainer`, then *Select Kernel → Existing Jupyter Server* → `http://localhost:8888/?token=smartcar` → **Run All**. About **15–20 min** on the GTX 1650 Ti (320 px images train fast).

⚠️ The test set is small (26 frames from the same drive): numbers are a first indication. For a solid result, film a **second drive** (other obstacle positions / light) and use it as the test set.
""")

code("""
# Parameters
MODEL = "yolo26n.pt"      # nano: analyses the ESP32-CAM stream live on the laptop CPU in a few ms
EPOCHS = 80
IMGSZ = 320               # = the ESP32-CAM resolution (320 x 240): train at the size the model will see
BATCH = 16
PATIENCE = 30
CONF = 0.4                # confidence threshold used live in the dashboard
SYNTH_IMAGES = 400        # copy-paste training images
DEGRADE_P = 0.6           # share of training images turned into 320 x 240 ESP32-CAM-style frames
SEED = 0
RELABEL = False           # True = run SAM 2 auto-labelling again (after changing toy_autolabel.py)
""", tags=["parameters"])

code("""
# Check that this notebook runs on the Docker kernel
from pathlib import Path
if not Path("/workspace/training").exists():
    raise SystemExit("Wrong kernel: Select Kernel -> Existing Jupyter Server -> http://localhost:8888/?token=smartcar")
print("OK — running on the Docker trainer kernel")
""")

code("""
import json, shutil, sys, time
from datetime import date

import cv2
import matplotlib.pyplot as plt
import torch
import ultralytics
from IPython.display import Image, display
from ultralytics import YOLO

sys.path.insert(0, "/workspace/training")
import importlib
import toy_autolabel, toy_dataset as td
td, toy_autolabel = importlib.reload(td), importlib.reload(toy_autolabel)   # always use the latest version on disk

DS = Path("/datasets/toy_obstacles")         # prepared dataset (Docker volume)
RUNS = Path("/workspace/training/runs")
MODELS = Path("/workspace/models")
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"ultralytics {ultralytics.__version__} · GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

def show(img, title="", w=16):
    plt.figure(figsize=(w, w * img.shape[0] / img.shape[1])); plt.imshow(img[..., ::-1]); plt.axis("off"); plt.title(title); plt.show()
""")

md("## 1 · Labels (SAM 2 video tracking)")

code("""
if RELABEL or not any(td.LABELS.glob("*.txt")):
    toy_autolabel.main()
labels = list(td.LABELS.glob("*.txt"))
with_box = sum(1 for f in labels if f.read_text().strip())
print(f"{len(labels)} labelled frames · {with_box} with an obstacle · {len(labels) - with_box} empty track")
for f in sorted((td.ROOT / "raw" / "review").glob("auto_*.jpg")):
    print(f.name); display(Image(filename=str(f), width=1100))
""")

md("## 2 · Dataset: time-block split, ESP32-CAM look, copy-paste")

code("""
counts, n_pieces = td.build(DS, degrade_p=DEGRADE_P, synth_n=SYNTH_IMAGES, seed=SEED)
print(f"clean obstacle cut-outs used for copy-paste: {n_pieces}")
for split, c in counts.items():
    print(f"{split:>5}: {c}")
show(td.preview(DS, "train", n=12), "training images (real, ESP32-CAM look, copy-paste)")
show(td.preview(DS, "test", n=8), "test images (real frames, never used for training)")
show(td.preview(DS, "test_esp32", n=8), "the same test frames as 320 x 240 ESP32-CAM-style images")
""")

md("## 3 · Train on the GPU")

code("""
model = YOLO(MODEL)
t0 = time.time()
model.train(data=str(DS / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, patience=PATIENCE,
            seed=SEED, cos_lr=True, close_mosaic=10, plots=True, project=str(RUNS), name="toy_obstacles", exist_ok=True)
TRAIN_MINUTES = (time.time() - t0) / 60
BEST = RUNS / "toy_obstacles" / "weights" / "best.pt"
print(f"training took {TRAIN_MINUTES:.0f} min · {model.trainer.epoch + 1} epochs")
display(Image(filename=str(RUNS / "toy_obstacles" / "results.png"), width=1000))
""")

md("## 4 · Evaluate on the held-out test frames (real, and as 320 × 240 ESP32-CAM frames)")

code("""
best = YOLO(BEST)

def evaluate(split):
    m = best.val(data=str(DS / "data.yaml"), split=split, imgsz=IMGSZ, batch=BATCH, device=DEVICE, plots=True,
                 project=str(RUNS), name=f"toy_obstacles_{split}", exist_ok=True)
    P, R = float(m.box.mp), float(m.box.mr)
    return m, {"precision": round(P, 3), "recall": round(R, 3), "map50": round(float(m.box.map50), 3),
               "map50_95": round(float(m.box.map), 3), "f1": round(2 * P * R / max(P + R, 1e-9), 3),
               "inference_ms": round(float(m.speed["inference"]), 1)}

m, metrics = evaluate("test")
m_esp, metrics_esp32 = evaluate("test_esp32")
per_class = {m.names[int(c)]: round(float(ap), 3) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
# Confusion matrix at the live threshold CONF (val's default conf=0.001 would count every weak guess as a detection)
m_cm = best.val(data=str(DS / "data.yaml"), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE, conf=CONF,
                plots=True, project=str(RUNS), name="toy_obstacles_test_cm", exist_ok=True)
cm = m_cm.confusion_matrix.matrix              # rows = predicted, cols = true (last = background)
confusion = {"labels": td.CLASSES + ["background"], "matrix": cm.T.astype(int).tolist(), "conf": CONF, "imgsz": IMGSZ}
print("test (real frames)        :", json.dumps(metrics)); print("AP@50 per class:", per_class)
print("test (ESP32-CAM 320 x 240):", json.dumps(metrics_esp32))
for p in (RUNS / "toy_obstacles_test" / "BoxPR_curve.png", RUNS / "toy_obstacles_test_cm" / "confusion_matrix.png",
          RUNS / "toy_obstacles_test" / "val_batch0_pred.jpg"):
    if p.exists():
        display(Image(filename=str(p), width=800))
""")

md("## 5 · Export for the dashboard (Prototype mode)")

code("""
onnx_path = Path(best.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True))
shutil.copy(onnx_path, MODELS / "toy_obstacles.onnx")
shutil.copy(BEST, MODELS / "toy_obstacles.pt")
profile = {"name": "Toy track obstacles", "models": [
    {"id": "toy-obstacles", "path": "models/toy_obstacles.onnx", "imgsz": IMGSZ, "conf": CONF,
     "classes": {"column": "column", "box": "box"}}]}
(MODELS / "proto_toy.json").write_text(json.dumps(profile, indent=2))

reg_path = MODELS / "registry.json"
registry = json.loads(reg_path.read_text())
entry = {
    "id": "toy-obstacles", "mode": "proto", "name": "Toy track obstacle detector",
    "framework": f"{MODEL.split('.')[0].upper()} fine-tuned (Ultralytics {ultralytics.__version__})", "version": "1.0",
    "trained": date.today().isoformat(), "input": f"{IMGSZ} × {IMGSZ}",
    "dataset": {"images": sum(c["real"] for c in counts.values()),
                "split": f"{counts['train']['real']} / {counts['val']['real']} / {counts['test']['real']} real frames",
                "sources": ["Our toy-track video (249 frames), labels from SAM 2 video tracking, checked visually",
                            f"+ {counts['train']['esp32_look']} ESP32-CAM-look copies and {counts['train']['pasted']} "
                            "copy-paste images (training only)"]},
    "example": False, "metrics_source": "held-out test frames (time-block split of the same drive)",
    "metrics": metrics, "per_class": per_class, "confusion": confusion,
    "metrics_variants": [{"label": "Same test frames as 320 × 240 ESP32-CAM-style images", "metrics": metrics_esp32}],
    "training": {"epochs": model.trainer.epoch + 1, "minutes": round(TRAIN_MINUTES, 1),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
}
registry["models"] = [m_ for m_ in registry["models"] if m_.get("id") not in ("toy-car-v1", "toy-obstacles")] + [entry]
reg_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
print("exported models/toy_obstacles.onnx + models/proto_toy.json · registry updated")
""")

md("## 6 · Sanity check: predictions on test frames")

code("""
files = sorted((DS / "images" / "test").glob("*.jpg"))[::3][:8]
res = best.predict([str(f) for f in files], conf=CONF, verbose=False)
tiles = [cv2.resize(r.plot(), (416, 232)) for r in res]
import numpy as np
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
show(np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]), "test frames — model predictions")
print("All done. Restart the dashboard to use the model in Prototype mode:  docker compose restart dashboard")
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
out = Path(__file__).with_name("train_toy_obstacles.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
