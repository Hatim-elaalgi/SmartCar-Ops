"""Confusion matrices at the confidence threshold the dashboard really uses.

Ultralytics' validation builds its confusion matrix from every prediction above conf=0.001 (the setting needed for
mAP), so the "background" line counts thousands of weak guesses the dashboard never shows, and the diagonal is the
recall when keeping everything. This recomputes it on the TEST split at each model's live setting (input size and
confidence from its profile) and stores it in models/registry.json (+ the report figures of the road-hazards model).

    docker compose --profile train exec trainer python confusion_matrices.py
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS = Path("/workspace/models")
REPORT = Path("/workspace/training/report")
JOBS = [  # registry id, weights, dataset, profile with the live setting
    ("road-hazards", MODELS / "road_hazards.pt", Path("/datasets/road_hazards/data.yaml"), MODELS / "real_road.json"),
    ("toy-obstacles", MODELS / "toy_obstacles.pt", Path("/datasets/toy_obstacles/data.yaml"), MODELS / "proto_toy.json"),
]


def main():
    reg_path = MODELS / "registry.json"
    reg = json.loads(reg_path.read_text())
    for mid, weights, data, profile in JOBS:
        missing = [p for p in (weights, data, profile) if not p.exists()]
        if missing:
            print(f"skip {mid}: missing {missing[0]}")
            continue
        live = next(m for m in json.loads(profile.read_text())["models"] if m["id"] == mid)
        r = YOLO(weights).val(data=str(data), split="test", imgsz=live["imgsz"], conf=live["conf"], batch=8, device=0,
                              plots=True, verbose=False, project="/tmp/confusion", name=mid, exist_ok=True)
        names = [r.names[i] for i in sorted(r.names)]
        cm = r.confusion_matrix.matrix                      # rows = predicted, cols = true (last = background)
        entry = next(m for m in reg["models"] if m["id"] == mid)
        entry["confusion"] = {"labels": names + ["background"], "matrix": cm.T.astype(int).tolist(),   # rows = true
                              "conf": live["conf"], "imgsz": live["imgsz"]}
        print(f"\n{mid} · test split · {live['imgsz']} px · confidence >= {live['conf']}")
        for i, n in enumerate(names):
            print(f"  {n:>14}: {int(cm[i, i])} found · {int(cm[-1, i])} missed · "
                  f"{int(cm[:-1, i].sum() - cm[i, i])} wrong class · {int(cm[i, -1])} false detections")
        if mid == "road-hazards":
            for f in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
                shutil.copy(Path(r.save_dir) / f, REPORT / f)
    reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
    print("\nmodels/registry.json updated")


if __name__ == "__main__":
    main()
