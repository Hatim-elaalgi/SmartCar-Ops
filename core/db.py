"""SQLite storage. One short-lived connection per call keeps it safe across threads.

Positions are stored as x/y everywhere:
  real mode  -> x = longitude, y = latitude (GeoJSON order)
  proto mode -> x, y in centimetres on the track
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS trips (
    id          TEXT PRIMARY KEY,
    mode        TEXT NOT NULL,
    car_id      TEXT NOT NULL,
    route_name  TEXT,
    start_ts    REAL NOT NULL,
    end_ts      REAL,
    distance    REAL DEFAULT 0,
    n_obstacles INTEGER DEFAULT 0,
    n_avoid     INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'active',      -- active | done | interrupted
    source      TEXT DEFAULT 'sim'
);
CREATE INDEX IF NOT EXISTS ix_trips_mode_start ON trips(mode, start_ts);

CREATE TABLE IF NOT EXISTS trip_points (
    trip_id TEXT NOT NULL,
    ts      REAL NOT NULL,
    x       REAL,
    y       REAL,
    speed   REAL,
    state   TEXT
);
CREATE INDEX IF NOT EXISTS ix_points_trip ON trip_points(trip_id, ts);

CREATE TABLE IF NOT EXISTS obstacles (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    mode       TEXT NOT NULL,
    car_id     TEXT,
    trip_id    TEXT,
    ts_first   REAL NOT NULL,
    ts_last    REAL NOT NULL,
    cls        TEXT NOT NULL,
    confidence REAL,
    proximity  REAL,                       -- 0 (far) .. 1 (touching)
    distance   REAL,                       -- m (real) / cm (proto), when known
    in_path    INTEGER DEFAULT 1,
    x          REAL,
    y          REAL,
    edge       TEXT,                       -- road / track segment id
    severity   REAL,
    level      TEXT,
    status     TEXT DEFAULT 'new',         -- new | ack | resolved | false_positive | avoided
    seen       INTEGER DEFAULT 1,
    snapshot   TEXT,
    source     TEXT DEFAULT 'sim'          -- seed | sim | demo | camera
);
CREATE INDEX IF NOT EXISTS ix_obs_mode_status ON obstacles(mode, status);
CREATE INDEX IF NOT EXISTS ix_obs_mode_ts ON obstacles(mode, ts_last);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    mode    TEXT NOT NULL,                 -- real | proto | system
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,                 -- info | warning | critical
    kind    TEXT NOT NULL,                 -- trip | obstacle | camera | checkpoint | system
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);

-- Each time a car goes around a known obstacle (prototype track obstacles), for the statistics
CREATE TABLE IF NOT EXISTS encounters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode        TEXT NOT NULL,
    ts          REAL NOT NULL,
    obstacle_id INTEGER,
    cls         TEXT NOT NULL,
    trip_id     TEXT,
    source      TEXT                       -- seed | demo | camera
);
CREATE INDEX IF NOT EXISTS ix_enc_mode_ts ON encounters(mode, ts);
"""


def connect():
    con = sqlite3.connect(config.DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    return con


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with session() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)


def query(sql, params=()):
    with session() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def query_one(sql, params=()):
    rows = query(sql, params)
    return rows[0] if rows else None


def query_df(sql, params=()):
    with session() as con:
        return pd.read_sql_query(sql, con, params=params)


def execute(sql, params=()):
    with session() as con:
        return con.execute(sql, params).lastrowid


def executemany(sql, rows):
    if not rows:
        return
    with session() as con:
        con.executemany(sql, rows)


def clear_all():
    with session() as con:
        for table in ("trip_points", "trips", "obstacles", "events", "encounters"):
            con.execute(f"DELETE FROM {table}")


def clear_mode(mode):
    """Delete the history of one mode only (the other mode's data is kept)."""
    with session() as con:
        con.execute("DELETE FROM trip_points WHERE trip_id IN (SELECT id FROM trips WHERE mode = ?)", (mode,))
        for table in ("trips", "obstacles", "events", "encounters"):
            con.execute(f"DELETE FROM {table} WHERE mode = ?", (mode,))
