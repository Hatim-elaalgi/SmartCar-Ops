"""Detected obstacles: recording (with de-duplication), listing by priority, review status, route risk.

Prototype mode is different: the test track has a fixed set of obstacles (track JSON "obstacles"), each one a single
obstacle row with status "avoided" (the car always goes around them, so they are never "open"). Detections update
those rows and every avoidance is logged in `encounters` for the statistics."""
import math
import shutil
import time

import config
from core import db, events, geo, network, settings
from core.severity import LEVEL_META, STATUS_META, level_for, score

# The same object seen again within this radius is merged into one obstacle ("seen 5x").
MERGE_RADIUS = {config.REAL: 25.0, config.PROTO: 30.0}          # metres / centimetres
# Old obstacles count less in route risk: weight halves every HALF_LIFE seconds.
HALF_LIFE = {config.REAL: 7 * 86400, config.PROTO: 2 * 86400}
STATUS_WEIGHT = {"new": 1.0, "ack": 1.0, "avoided": 0.3, "resolved": 0.3, "false_positive": 0.0}
OPEN = ("new", "ack")
AVOIDED = "avoided"            # prototype track obstacles: known, always gone around by the car
SHOWN = OPEN + (AVOIDED,)      # drawn on the maps
TRACK_PHOTOS = config.RESOURCES_DIR / "track_obstacles"     # {cls}.jpg: picture shown for each track obstacle


def record(mode, cls, confidence, proximity, in_path, x, y, *, car_id=None, trip_id=None, edge=None,
           distance=None, source="sim", snapshot=None, ts=None):
    """Store a detection. Returns (obstacle_id, is_new)."""
    ts = ts or time.time()
    sev = score(cls, confidence, proximity, in_path)

    match = None
    if x is not None and y is not None:
        radius = MERGE_RADIUS[mode]
        dx, dy = geo.deg_box(mode, y, radius)
        candidates = db.query(
            "SELECT id, x, y, severity, confidence, proximity FROM obstacles "
            "WHERE mode=? AND cls=? AND status IN ('new','ack') AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?",
            (mode, cls, x - dx, x + dx, y - dy, y + dy))
        candidates = [c for c in candidates if geo.distance(mode, x, y, c["x"], c["y"]) <= radius]
        if candidates:
            match = min(candidates, key=lambda c: geo.distance(mode, x, y, c["x"], c["y"]))
    if match is not None:
        best = max(sev, match["severity"] or 0)
        db.execute(
            "UPDATE obstacles SET ts_last=?, seen=seen+1, severity=?, level=?, confidence=MAX(confidence, ?), "
            "proximity=MAX(proximity, ?), distance=COALESCE(?, distance), snapshot=COALESCE(snapshot, ?) "
            "WHERE id=?",
            (ts, best, level_for(best), confidence, proximity, distance, snapshot, match["id"]))
        return match["id"], False

    lvl = level_for(sev)
    oid = db.execute(
        "INSERT INTO obstacles(mode, car_id, trip_id, ts_first, ts_last, cls, confidence, proximity, distance, "
        "in_path, x, y, edge, severity, level, status, snapshot, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mode, car_id, trip_id, ts, ts, cls, confidence, proximity, distance, int(bool(in_path)), x, y, edge,
         sev, lvl, "new", snapshot, source))
    if lvl in ("critical", "high"):
        where = f" by {car_id}" if car_id else ""
        events.log(mode, "critical" if lvl == "critical" else "warning", "obstacle",
                   f"{LEVEL_META[lvl]['label']} obstacle: {cls.replace('_', ' ')}{where} (score {sev:.0f})", ts)
    return oid, True


# --- prototype track obstacles ----------------------------------------------------------------------------------
def ensure_track_obstacles(now=None):
    """Prototype: exactly one obstacle row (status "avoided") per obstacle placed on the test track, at its place
    on the track; every other prototype obstacle is removed with its snapshot.
    Returns the obstacle ids, in the order of the track's obstacle list."""
    now = now or time.time()
    net = network.proto_network()
    rows = db.query("SELECT id, cls, status, snapshot, x, y FROM obstacles WHERE mode=?", (config.PROTO,))
    ids = []
    for o in net.obstacles:
        free = [r for r in rows if r["cls"] == o["cls"] and r["status"] == AVOIDED and r["id"] not in ids]
        row = min(free, key=lambda r: math.hypot((r["x"] or 0) - o["x"], (r["y"] or 0) - o["y"]), default=None)
        snapshot = row["snapshot"] if row else None
        photo = TRACK_PHOTOS / f"{o['cls']}.jpg"
        if not snapshot and photo.exists():
            snapshot = f"track_{o['cls']}.jpg"
            shutil.copy(photo, config.SNAPSHOT_DIR / snapshot)
        edge = net.nearest_edge(o["x"], o["y"])
        if row:
            db.execute("UPDATE obstacles SET x=?, y=?, edge=?, in_path=1, snapshot=? WHERE id=?",
                       (o["x"], o["y"], edge, snapshot, row["id"]))
            ids.append(row["id"])
        else:
            sev = score(o["cls"], 0.9, 1.0, True)          # on the tape, passed at the ultrasonic distance (13 cm)
            ids.append(db.execute(
                "INSERT INTO obstacles(mode, car_id, ts_first, ts_last, cls, confidence, proximity, distance, in_path, "
                "x, y, edge, severity, level, status, snapshot, source, seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                (config.PROTO, config.PROTO_CAR_ID, now, now, o["cls"], 0.9, 1.0, 13.0, 1, o["x"], o["y"], edge,
                 sev, level_for(sev), AVOIDED, snapshot, "track")))
    stale = [r for r in rows if r["id"] not in ids]
    for r in stale:
        if r["snapshot"] and not r["snapshot"].startswith("track_"):
            (config.SNAPSHOT_DIR / r["snapshot"]).unlink(missing_ok=True)
    db.executemany("DELETE FROM obstacles WHERE id=?", [(r["id"],) for r in stale])
    db.execute("DELETE FROM encounters WHERE mode=? AND obstacle_id NOT IN (SELECT id FROM obstacles WHERE mode=?)",
               (config.PROTO, config.PROTO))
    return ids


def track_sighting(cls, confidence, proximity, *, trip_id=None, source="camera", ts=None, encounter=False,
                   snapshot=None, near=None):
    """Prototype: one of the track obstacles was seen — update it instead of creating a new obstacle.
    near=(x, y) picks among obstacles of the same class; without it, the one met longest ago (they come in turn).
    encounter=True when the car goes around it (logged for the statistics). Returns its id, or None when there is
    no such obstacle on the track (the detection is ignored)."""
    ts = ts or time.time()
    rows = db.query("SELECT o.id, o.severity, o.snapshot, o.x, o.y, "
                    "(SELECT MAX(e.ts) FROM encounters e WHERE e.obstacle_id = o.id) AS last_met "
                    "FROM obstacles o WHERE o.mode=? AND o.cls=? AND o.status=?", (config.PROTO, cls, AVOIDED))
    if not rows:
        return None
    if near is not None:            # the caller knows where it is (demo car): the obstacle at that place
        row = min(rows, key=lambda r: math.hypot(r["x"] - near[0], r["y"] - near[1]))
    else:                           # the camera alone: on a loop the car meets them in turn -> the one met longest ago
        row = min(rows, key=lambda r: r["last_met"] or 0)
    best = max(score(cls, confidence, proximity, True), row["severity"] or 0)
    db.execute("UPDATE obstacles SET ts_last=?, seen=seen+1, confidence=MAX(confidence, ?), "
               "proximity=MAX(proximity, ?), severity=?, level=?, car_id=?, trip_id=COALESCE(?, trip_id), source=?, "
               "snapshot=COALESCE(?, snapshot) WHERE id=?",
               (ts, confidence, proximity, best, level_for(best), config.PROTO_CAR_ID, trip_id, source, snapshot,
                row["id"]))
    if snapshot and row["snapshot"] and row["snapshot"].startswith("cam_"):
        (config.SNAPSHOT_DIR / row["snapshot"]).unlink(missing_ok=True)      # keep only the latest camera picture
    if encounter:
        db.execute("INSERT INTO encounters(mode, ts, obstacle_id, cls, trip_id, source) VALUES (?,?,?,?,?,?)",
                   (config.PROTO, ts, row["id"], cls, trip_id, source))
    return row["id"]


def expire(mode, source, max_age_s, now=None):
    """Delete unreviewed obstacles of a source not seen for max_age_s, with their snapshots. Returns how many."""
    now = now or time.time()
    rows = db.query("SELECT id, snapshot FROM obstacles WHERE mode=? AND source=? AND status='new' AND ts_last < ?",
                    (mode, source, now - max_age_s))
    for r in rows:
        if r["snapshot"]:
            (config.SNAPSHOT_DIR / r["snapshot"]).unlink(missing_ok=True)
    db.executemany("DELETE FROM obstacles WHERE id=?", [(r["id"],) for r in rows])
    return len(rows)


def set_snapshot(obstacle_id, filename):
    db.execute("UPDATE obstacles SET snapshot=? WHERE id=?", (filename, obstacle_id))


def set_status(ids, status, mode=None):
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    db.execute(f"UPDATE obstacles SET status=? WHERE id IN ({marks})", (status, *ids))
    if mode:
        events.log(mode, "info", "obstacle", f"{len(ids)} obstacle(s) marked as {STATUS_META[status].lower()}")


def listing(mode, since=None, levels=None, statuses=None, classes=None, limit=2000):
    sql = "SELECT * FROM obstacles WHERE mode = ?"
    params = [mode]
    if since:
        sql += " AND ts_last >= ?"
        params.append(since)
    for col, values in (("level", levels), ("status", statuses), ("cls", classes)):
        if values:
            sql += f" AND {col} IN ({','.join('?' * len(values))})"
            params += list(values)
    # Priority: open first, then highest severity, then most recent.
    sql += (" ORDER BY CASE status WHEN 'new' THEN 0 WHEN 'ack' THEN 1 WHEN 'avoided' THEN 2 WHEN 'resolved' THEN 3"
            " ELSE 4 END,"
            " severity DESC, ts_last DESC LIMIT ?")
    params.append(limit)
    return db.query_df(sql, params)


def get(obstacle_id):
    return db.query_one("SELECT * FROM obstacles WHERE id = ?", (obstacle_id,))


def classes(mode):
    return [r["cls"] for r in db.query("SELECT DISTINCT cls FROM obstacles WHERE mode=? ORDER BY cls", (mode,))]


def edge_risk(mode, now=None):
    """Risk per road/track segment from the obstacle history (recent + open obstacles weigh more)."""
    now = now or time.time()
    half = HALF_LIFE[mode]
    rows = db.query("SELECT edge, severity, status, ts_last FROM obstacles "
                    "WHERE mode=? AND edge IS NOT NULL AND status != 'false_positive' AND ts_last > ?",
                    (mode, now - 6 * half))
    risk = {}
    for r in rows:
        age = max(0.0, now - r["ts_last"])
        w = (r["severity"] or 0) / 100 * STATUS_WEIGHT.get(r["status"], 1.0) * 0.5 ** (age / half)
        risk[r["edge"]] = risk.get(r["edge"], 0.0) + w
    return risk


def recompute_severity():
    weights = settings.class_weights()
    rows = db.query("SELECT id, cls, confidence, proximity, in_path FROM obstacles")
    updates = []
    for r in rows:
        s = score(r["cls"], r["confidence"], r["proximity"], r["in_path"], weights)
        updates.append((s, level_for(s), r["id"]))
    db.executemany("UPDATE obstacles SET severity=?, level=? WHERE id=?", updates)
    return len(updates)
