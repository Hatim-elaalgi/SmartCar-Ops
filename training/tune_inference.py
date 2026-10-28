"""Better results WITHOUT retraining: tune how the trained model is *used*.

  1. input size at inference (the model was trained at 640; larger sizes help small/far objects)
  2. test-time augmentation (TTA: flipped + rescaled copies merged)
  3. per-class confidence thresholds (each class gets the threshold with its best F1)

Settings are CHOSEN on the validation split and REPORTED on the test split (never used for choosing).

    docker compose --profile train exec trainer python tune_inference.py
"""
import json
import sys

import numpy as np
from ultralytics import YOLO

WEIGHTS = "/workspace/training/runs/road_hazards/weights/best.pt"
DATA = "/datasets/road_hazards/data.yaml"
SIZES = [640, 768, 896, 960]


def evaluate(model, split, imgsz, augment=False):
    r = model.val(data=DATA, split=split, imgsz=imgsz, batch=4, device=0, augment=augment, plots=False,
                  verbose=False, project="/tmp/tune", name=f"{split}_{imgsz}_{int(augment)}", exist_ok=True)
    return r


def summary(r):
    b = r.box
    return {"P": float(b.mp), "R": float(b.mr), "mAP50": float(b.map50), "mAP50-95": float(b.map)}


def per_class_thresholds(r):
    """Confidence with the best F1 for each class, read from the validation F1 curves."""
    x = np.linspace(0, 1, 1000)
    return {r.names[int(c)]: round(float(x[int(np.argmax(r.box.f1_curve[i]))]), 3)
            for i, c in enumerate(r.box.ap_class_index)}


def pr_at(r, thresholds):
    """Precision / recall per class on this split when each class uses its own threshold."""
    x = np.linspace(0, 1, 1000)
    out = {}
    for i, c in enumerate(r.box.ap_class_index):
        name = r.names[int(c)]
        j = int(np.searchsorted(x, thresholds[name]))
        out[name] = (float(r.box.p_curve[i][j]), float(r.box.r_curve[i][j]))
    return out


def main():
    model = YOLO(WEIGHTS)
    rows = []
    print("\n== 1+2 · input size and TTA (chosen on VALIDATION) ==", flush=True)
    for size in SIZES:
        for aug in (False, True):
            try:
                s = summary(evaluate(model, "val", size, aug))
            except Exception as exc:                       # TTA is not supported by every model head
                print(f"  {size} TTA={aug}: not available ({str(exc)[:80]})")
                continue
            rows.append((size, aug, s))
            print(f"  imgsz {size:4d} TTA {'on ' if aug else 'off'}: P {s['P']:.3f}  R {s['R']:.3f}  "
                  f"mAP50 {s['mAP50']:.3f}  mAP50-95 {s['mAP50-95']:.3f}", flush=True)
    options = {"best (offline)": max(rows, key=lambda t: t[2]["mAP50-95"]),
               "real-time (no TTA)": max((t for t in rows if not t[1]), key=lambda t: t[2]["mAP50-95"])}
    for label, (size, aug, _) in options.items():
        print(f"  -> {label}: imgsz {size}, TTA {'on' if aug else 'off'}")

    print("\n== TEST split (never used for choosing) ==", flush=True)
    base = evaluate(model, "test", 640, False)
    b = summary(base)
    print(f"  baseline (640, threshold 0.35 for all): P {b['P']:.3f}  R {b['R']:.3f}  mAP50 {b['mAP50']:.3f}  "
          f"mAP50-95 {b['mAP50-95']:.3f}")
    report = {"validation": [{"imgsz": s, "tta": a, **m} for s, a, m in rows], "test_baseline": b, "options": {}}
    for label, (size, aug, _) in options.items():
        thr = per_class_thresholds(evaluate(model, "val", size, aug))      # thresholds chosen on validation
        tuned = evaluate(model, "test", size, aug)
        t = summary(tuned)
        print(f"\n  {label} — imgsz {size}, TTA {'on' if aug else 'off'}, per-class thresholds {thr}")
        print(f"    test: P {t['P']:.3f}  R {t['R']:.3f}  mAP50 {t['mAP50']:.3f}  mAP50-95 {t['mAP50-95']:.3f}")
        base_pr, tuned_pr = pr_at(base, {k: 0.35 for k in thr}), pr_at(tuned, thr)
        print(f"    {'class':<14}{'P now':>7}{'R now':>7}{'thr':>6}{'P new':>7}{'R new':>7}")
        for name in thr:
            print(f"    {name:<14}{base_pr[name][0]:>7.3f}{base_pr[name][1]:>7.3f}{thr[name]:>6.2f}"
                  f"{tuned_pr[name][0]:>7.3f}{tuned_pr[name][1]:>7.3f}")
        report["options"][label] = {"imgsz": size, "tta": aug, "thresholds": thr, "test": t,
                                    "per_class_test": {k: {"P": v[0], "R": v[1]} for k, v in tuned_pr.items()}}
    json.dump(report, open("/workspace/training/runs/inference_tuning.json", "w"), indent=2)
    print("\nsaved training/runs/inference_tuning.json")


if __name__ == "__main__":
    sys.exit(main())
