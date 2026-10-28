"""Small JSON settings file (camera IP, active trajectory, class weights...)."""
import json
import threading

import config

_lock = threading.Lock()


def _load():
    try:
        return json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get(key, default=None):
    with _lock:
        return _load().get(key, default)


def set(key, value):  # noqa: A001 - mirrors get()
    with _lock:
        data = _load()
        data[key] = value
        config.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def class_weights():
    weights = dict(config.DEFAULT_CLASS_WEIGHTS)
    weights.update(get("class_weights", {}) or {})
    return weights
