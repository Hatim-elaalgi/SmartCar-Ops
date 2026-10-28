"""Trips (one drive from A to B, or one run of the toy car on a trajectory) and their GPS-like points."""
import threading
import time
import uuid

from core import db

_buffer = []
_buffer_lock = threading.Lock()


def start(mode, car_id, route_name=None, ts=None, source="sim"):
    trip_id = f"{mode[0].upper()}{uuid.uuid4().hex[:7]}"
    db.execute("INSERT INTO trips(id, mode, car_id, route_name, start_ts, status, source) VALUES (?,?,?,?,?,?,?)",
               (trip_id, mode, car_id, route_name, ts or time.time(), "active", source))
    return trip_id


def add_point(trip_id, ts, x, y, speed, state):
    """Buffered: points are written in batches by flush()."""
    with _buffer_lock:
        _buffer.append((trip_id, ts, x, y, speed, state))
        if len(_buffer) < 20:
            return
    flush()


def flush():
    with _buffer_lock:
        rows = _buffer[:]
        _buffer.clear()
    db.executemany("INSERT INTO trip_points(trip_id, ts, x, y, speed, state) VALUES (?,?,?,?,?,?)", rows)


def bump(trip_id, obstacles=0, avoid=0):
    if trip_id:
        db.execute("UPDATE trips SET n_obstacles = n_obstacles + ?, n_avoid = n_avoid + ? WHERE id = ?",
                   (obstacles, avoid, trip_id))


def finish(trip_id, distance, ts=None, status="done"):
    flush()
    db.execute("UPDATE trips SET end_ts = ?, distance = ?, status = ? WHERE id = ?",
               (ts or time.time(), distance, status, trip_id))


def close_dangling(mode, source):
    """Mark trips left 'active' by a previous run of the app as interrupted."""
    db.execute("UPDATE trips SET status='interrupted', end_ts=COALESCE(end_ts, start_ts) "
               "WHERE mode=? AND source=? AND status='active'", (mode, source))


def recent(mode, since=None, limit=300):
    sql = "SELECT * FROM trips WHERE mode = ?"
    params = [mode]
    if since:
        sql += " AND start_ts >= ?"
        params.append(since)
    sql += " ORDER BY start_ts DESC LIMIT ?"
    params.append(limit)
    return db.query_df(sql, params)


def points(trip_id):
    flush()
    return db.query_df("SELECT ts, x, y, speed, state FROM trip_points WHERE trip_id = ? ORDER BY ts", (trip_id,))


def get(trip_id):
    return db.query_one("SELECT * FROM trips WHERE id = ?", (trip_id,))
