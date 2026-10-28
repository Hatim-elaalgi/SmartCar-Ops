"""Obstacle severity: one explainable score from 0 to 100.

    score = 100 x class_weight x (0.5 + 0.5 x proximity) x confidence x (1 if in path else 0.5)

class_weight  how dangerous the object is (pedestrian 1.0 ... cone 0.4), editable on the System page
proximity     0 = far, 1 = touching (from the ultrasonic distance or the bounding-box height)
confidence    model confidence 0..1
in path       the obstacle is in front of the car, not on the side
"""
from core import settings

LEVELS = [("critical", 60), ("high", 40), ("medium", 20), ("low", 0)]
LEVEL_ORDER = {name: i for i, (name, _) in enumerate(LEVELS)}

# Status colours are fixed and always shown with an icon + label, never colour alone.
LEVEL_META = {
    "critical": {"label": "Critical", "color": "#d03b3b", "icon": "bi-exclamation-octagon-fill"},
    "high":     {"label": "High",     "color": "#ec835a", "icon": "bi-exclamation-triangle-fill"},
    "medium":   {"label": "Medium",   "color": "#fab219", "icon": "bi-exclamation-circle-fill"},
    "low":      {"label": "Low",      "color": "#0ca30c", "icon": "bi-info-circle-fill"},
}

STATUS_META = {
    "new": "New", "ack": "Acknowledged", "resolved": "Resolved", "false_positive": "False positive",
    "avoided": "Avoided",           # prototype track obstacles: the car always goes around them
}


def score(cls, confidence, proximity, in_path, weights=None):
    weights = weights or settings.class_weights()
    w = weights.get(cls, 0.5)
    proximity = max(0.0, min(1.0, proximity or 0.0))
    confidence = max(0.0, min(1.0, confidence or 0.0))
    return round(100 * w * (0.5 + 0.5 * proximity) * confidence * (1.0 if in_path else 0.5), 1)


def level_for(value):
    for name, threshold in LEVELS:
        if value >= threshold:
            return name
    return "low"
