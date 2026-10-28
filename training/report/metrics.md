# Road hazards detector — evaluation results

*Generated 2026-10-23 by `training/report_metrics.py`.*

## Method
- **Model:** YOLO26S fine-tuned (Ultralytics 8.4.184), fine-tuned from COCO-pretrained weights for 36 epochs
  (8.0 h on an NVIDIA GeForce GTX 1650 Ti), input 640 × 640.
- **Data:** 7,894 real dashcam images — Potholes under challenging conditions (Bučko et al., Sensors 2022) — CC BY 4.0; ROADWork (Ghosh et al., ICCV 2025, CMU) — ODC-By 1.0.
- **Split (train / validation / test):** 6324 / 755 / 815. Potholes: each drive split in time (80 / 10 / 10 %, with a gap);
  ROADWork: official geographic split (validation and test come from places not seen in training).
- **Classes:** pothole, cone (cones, tubular markers), barrier (drums, barricades, barriers, fences, vertical panels),
  roadwork sign (temporary traffic-control signs, arrow boards, message boards).
- **Protocol:** the inference setting (input size, test-time augmentation) was selected on the **validation** split;
  all numbers below are measured on the **test** split, never used for training or selection.
  Precision / recall are given at the single confidence that maximises the mean F1 (Ultralytics convention);
  mAP@50 uses IoU ≥ 0.5, mAP@50-95 averages IoU 0.5 → 0.95.

## Results on the test split (5,105 objects in 815 images)

### Best inference setting: 960 px + TTA
| Class | Objects | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| pothole | 299 | 0.626 | 0.268 | 0.375 | 0.330 | 0.119 |
| cone | 2358 | 0.681 | 0.763 | 0.720 | 0.744 | 0.464 |
| barrier | 1977 | 0.688 | 0.594 | 0.638 | 0.634 | 0.369 |
| roadwork_sign | 471 | 0.630 | 0.567 | 0.597 | 0.588 | 0.430 |
| **all** | **5105** | **0.656** | **0.548** | **0.597** | **0.574** | **0.346** |

### Real-time setting used in the dashboard: 640 px (CPU, ONNX)
| Class | Objects | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| pothole | 299 | 0.601 | 0.254 | 0.357 | 0.312 | 0.108 |
| cone | 2358 | 0.712 | 0.715 | 0.714 | 0.694 | 0.416 |
| barrier | 1977 | 0.692 | 0.539 | 0.606 | 0.578 | 0.317 |
| roadwork_sign | 471 | 0.638 | 0.495 | 0.557 | 0.520 | 0.367 |
| **all** | **5105** | **0.661** | **0.501** | **0.570** | **0.526** | **0.302** |

| Setting | Precision | Recall | mAP@50 | mAP@50-95 | GPU inference |
|---|---|---|---|---|---|
| 640 px (real time) | 0.661 | 0.501 | 0.526 | 0.302 | 10 ms |
| 960 px + TTA (best) | 0.656 | 0.548 | 0.574 | 0.346 | 50 ms |

Recommended per-class confidence thresholds for deployment (best F1 per class on validation, not used in the tables above): {'pothole': 0.302, 'cone': 0.441, 'barrier': 0.467, 'roadwork_sign': 0.546}.

## Additional views (same model, same test split)

### Hazard detection, class-agnostic — "is there an obstacle?"
All four classes are merged into a single *obstacle* class, for predictions and labels alike: a cone predicted as a
barrier counts as a correct detection. This measures what matters most for a car — finding the hazard — independently
of naming its exact type.

| Setting | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| 640 px (real time) | 0.726 | 0.597 | 0.655 | 0.630 | 0.364 |
| 960 px + TTA (best) | 0.711 | 0.646 | 0.677 | 0.677 | 0.412 |

### Operating points — precision / recall trade-off (960 px + TTA)
A detector can be tuned for fewer false alarms (higher confidence threshold → higher precision, lower recall) or for
missing fewer obstacles (lower threshold). Values are read from the test precision–recall curves; *per-class* = mean over
the four classes, *agnostic* = single obstacle class.

| Confidence threshold | Precision (per-class) | Recall (per-class) | Precision (agnostic) | Recall (agnostic) |
|---|---|---|---|---|
| 0.25 | 0.520 | 0.612 | 0.563 | 0.720 |
| 0.40 | 0.673 | 0.534 | 0.707 | 0.647 |
| 0.50 | 0.763 | 0.474 | 0.781 | 0.579 |
| 0.60 | 0.869 | 0.396 | 0.853 | 0.495 |
| 0.70 | 0.916 | 0.300 | 0.893 | 0.370 |

## Figures (this folder)
`BoxPR_curve.png` precision–recall curves · `BoxF1_curve.png` F1 vs confidence ·
`confusion_matrix_normalized.png` share of each true class (real-time setting: 640 px, confidence ≥ 0.35, from
`confusion_matrices.py`; *background* = missed objects / false detections) ·
`val_batch*_pred.jpg` predictions on test images · `training_results.png` training curves · `training_labels.jpg` label statistics.
