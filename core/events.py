"""Event log (trips, obstacles, camera, checkpoints...)."""
import time

from core import db


def log(mode, level, kind, message, ts=None):
    db.execute("INSERT INTO events(mode, ts, level, kind, message) VALUES (?,?,?,?,?)",
               (mode, ts or time.time(), level, kind, message))


def recent(mode, limit=200, levels=None, kinds=None):
    sql = "SELECT * FROM events WHERE mode IN (?, 'system')"
    params = [mode]
    if levels:
        sql += f" AND level IN ({','.join('?' * len(levels))})"
        params += list(levels)
    if kinds:
        sql += f" AND kind IN ({','.join('?' * len(kinds))})"
        params += list(kinds)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    return db.query(sql, params)
