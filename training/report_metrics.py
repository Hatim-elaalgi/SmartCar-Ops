"""Report pack: evaluates the trained hazards model on the held-out TEST split with
  (a) the real-time setting used live in the dashboard (640 px), and
  (b) the best inference setting chosen on VALIDATION by tune_inference.py (e.g. 960 px + TTA),
then writes tables + figures to training/report/ and both results to models/registry.json.

    docker compose --profile train exec trainer python report_metrics.py
"""
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
from ultralytics import YOLO

WEIGHTS = "/workspace/training/runs/road_hazards/weights/best.pt"
DATA = "/datasets/road_hazards/data.yaml"
OUT = Path("/workspace/training/report")
TUNING = Path("/workspace/training/runs/inference_tuning.json")
REGISTRY = Path("/workspace/models/registry.json")


CONFS = (0.25, 0.4, 0.5, 0.6, 0.7)


def operating_points(r):
    """Mean precision / recall over classes at fixed confidence thresholds (from the test PR curves)."""
    x = np.linspace(0, 1, 1000)
    out = []
    for c in CONFS:
        j = int(np.searchsorted(x, c))
        out.append((c, float(r.box.p_curve[:, j].mean()), float(r.box.r_curve[:, j].mean())))
    return out


def evaluate(model, imgsz, augment, name, single_cls=False):
    r = model.val(data=DATA, split="test", imgsz=imgsz, batch=4, device=0, augment=augment, plots=True,
                  single_cls=single_cls, verbose=False, project="/tmp/report", name=name, exist_ok=True)
    evaluate.last = r
    b = r.box
    overall = {"precision": round(float(b.mp), 3), "recall": round(float(b.mr), 3),
               "f1": round(2 * float(b.mp) * float(b.mr) / max(float(b.mp + b.mr), 1e-9), 3),
               "map50": round(float(b.map50), 3), "map50_95": round(float(b.map), 3),
               "inference_ms": round(float(r.speed["inference"]), 1)}
    per_class = {}
    for i, c in enumerate(b.ap_class_index):
        p, rc = float(b.p[i]), float(b.r[i])
        per_class[r.names[int(c)]] = {"objects": int(r.nt_per_class[c]), "precision": round(p, 3),
                                      "recall": round(rc, 3), "f1": round(2 * p * rc / max(p + rc, 1e-9), 3),
                                      "map50": round(float(b.ap50[i]), 3), "map50_95": round(float(b.ap[i]), 3)}
    cm = r.confusion_matrix.matrix
    names = [r.names[i] for i in sorted(r.names)]
    return overall, per_class, {"labels": names + ["background"], "matrix": cm.T.astype(int).tolist()}, \
        Path(r.save_dir)


def table(per_class, overall):
    lines = ["| Class | Objects | Precision | Recall | F1 | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|---|---|"]
    for k, v in per_class.items():
        lines.append(f"| {k} | {v['objects']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                     f"{v['map50']:.3f} | {v['map50_95']:.3f} |")
    total = sum(v["objects"] for v in per_class.values())
    lines.append(f"| **all** | **{total}** | **{overall['precision']:.3f}** | **{overall['recall']:.3f}** | "
                 f"**{overall['f1']:.3f}** | **{overall['map50']:.3f}** | **{overall['map50_95']:.3f}** |")
    return "\n".join(lines)


def main():
    tuning = json.loads(TUNING.read_text())
    best = tuning["options"]["best (offline)"]
    size, tta = best["imgsz"], best["tta"]
    setting = f"{size} px" + (" + TTA" if tta else "")
    model = YOLO(WEIGHTS)
    live, live_pc, _, _ = evaluate(model, 640, False, "live")
    top, top_pc, top_cm, top_dir = evaluate(model, size, tta, "best")
    top_ops = operating_points(evaluate.last)
    # Class-agnostic view: every prediction / label is just "obstacle" (is there a hazard here, whatever its type?)
    agn_live, _, _, _ = evaluate(model, 640, False, "live_agnostic", single_cls=True)
    agn_top, _, _, _ = evaluate(model, size, tta, "best_agnostic", single_cls=True)
    agn_ops = operating_points(evaluate.last)

    OUT.mkdir(parents=True, exist_ok=True)
    # confusion_matrix*.png come from confusion_matrices.py (at the live confidence threshold, not val's 0.001)
    for f in ("BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
              "val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch2_pred.jpg"):
        if (top_dir / f).exists():
            shutil.copy(top_dir / f, OUT / f)
    for f in ("results.png", "labels.jpg"):
        src = Path("/workspace/training/runs/road_hazards") / f
        if src.exists():
            shutil.copy(src, OUT / f"training_{f}")

    reg = json.loads(REGISTRY.read_text())
    h = next(m for m in reg["models"] if m["id"] == "road-hazards")
    ds = h["dataset"]
    md = f"""# Road hazards detector — evaluation results

*Generated {date.today().isoformat()} by `training/report_metrics.py`.*

## Method
- **Model:** {h['framework']}, fine-tuned from COCO-pretrained weights for {h['training']['epochs']} epochs
  ({h['training']['minutes'] / 60:.1f} h on an {h['training']['gpu']}), input 640 × 640.
- **Data:** {ds['images']:,} real dashcam images — {'; '.join(ds['sources'])}.
- **Split (train / validation / test):** {ds['split']}. Potholes: each drive split in time (80 / 10 / 10 %, with a gap);
  ROADWork: official geographic split (validation and test come from places not seen in training).
- **Classes:** pothole, cone (cones, tubular markers), barrier (drums, barricades, barriers, fences, vertical panels),
  roadwork sign (temporary traffic-control signs, arrow boards, message boards).
- **Protocol:** the inference setting (input size, test-time augmentation) was selected on the **validation** split;
  all numbers below are measured on the **test** split, never used for training or selection.
  Precision / recall are given at the single confidence that maximises the mean F1 (Ultralytics convention);
  mAP@50 uses IoU ≥ 0.5, mAP@50-95 averages IoU 0.5 → 0.95.

## Results on the test split ({sum(v['objects'] for v in top_pc.values()):,} objects in 815 images)

### Best inference setting: {setting}
{table(top_pc, top)}

### Real-time setting used in the dashboard: 640 px (CPU, ONNX)
{table(live_pc, live)}

| Setting | Precision | Recall | mAP@50 | mAP@50-95 | GPU inference |
|---|---|---|---|---|---|
| 640 px (real time) | {live['precision']:.3f} | {live['recall']:.3f} | {live['map50']:.3f} | {live['map50_95']:.3f} | {live['inference_ms']:.0f} ms |
| {setting} (best) | {top['precision']:.3f} | {top['recall']:.3f} | {top['map50']:.3f} | {top['map50_95']:.3f} | {top['inference_ms']:.0f} ms |

Recommended per-class confidence thresholds for deployment (best F1 per class on validation, not used in the tables above): {best['thresholds']}.

## Additional views (same model, same test split)

### Hazard detection, class-agnostic — "is there an obstacle?"
All four classes are merged into a single *obstacle* class, for predictions and labels alike: a cone predicted as a
barrier counts as a correct detection. This measures what matters most for a car — finding the hazard — independently
of naming its exact type.

| Setting | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| 640 px (real time) | {agn_live['precision']:.3f} | {agn_live['recall']:.3f} | {agn_live['f1']:.3f} | {agn_live['map50']:.3f} | {agn_live['map50_95']:.3f} |
| {setting} (best) | {agn_top['precision']:.3f} | {agn_top['recall']:.3f} | {agn_top['f1']:.3f} | {agn_top['map50']:.3f} | {agn_top['map50_95']:.3f} |

### Operating points — precision / recall trade-off ({setting})
A detector can be tuned for fewer false alarms (higher confidence threshold → higher precision, lower recall) or for
missing fewer obstacles (lower threshold). Values are read from the test precision–recall curves; *per-class* = mean over
the four classes, *agnostic* = single obstacle class.

| Confidence threshold | Precision (per-class) | Recall (per-class) | Precision (agnostic) | Recall (agnostic) |
|---|---|---|---|---|
""" + "\n".join(f"| {c:.2f} | {p:.3f} | {rc:.3f} | {ap:.3f} | {ar:.3f} |"
                for (c, p, rc), (_, ap, ar) in zip(top_ops, agn_ops)) + """

## Figures (this folder)
`BoxPR_curve.png` precision–recall curves · `BoxF1_curve.png` F1 vs confidence ·
`confusion_matrix_normalized.png` share of each true class (real-time setting: 640 px, confidence ≥ 0.35, from
`confusion_matrices.py`; *background* = missed objects / false detections) ·
`val_batch*_pred.jpg` predictions on test images · `training_results.png` training curves · `training_labels.jpg` label statistics.
"""
    (OUT / "metrics.md").write_text(md, encoding="utf-8")
    json.dump({"setting_best": setting, "best": top, "best_per_class": top_pc, "live": live, "live_per_class": live_pc,
               "class_agnostic": {"live": agn_live, "best": agn_top},
               "operating_points_best": [{"conf": c, "precision": p, "recall": r} for c, p, r in top_ops],
               "operating_points_best_agnostic": [{"conf": c, "precision": p, "recall": r} for c, p, r in agn_ops]},
              open(OUT / "metrics.json", "w"), indent=2)

    h["metrics_best"] = dict(top, setting=setting)
    h["per_class_best"] = {k: v["map50"] for k, v in top_pc.items()}
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
    print(md)


if __name__ == "__main__":
    main()
