"""Small aggregate queries used by several pages."""
import time

from core import db

DAY = 86400


def count_obstacles(mode, since=None, statuses=None, levels=None):
    sql, params = "SELECT COUNT(*) n FROM obstacles WHERE mode=?", [mode]
    if since:
        sql += " AND ts_first >= ?"
        params.append(since)
    for col, values in (("status", statuses), ("level", levels)):
        if values:
            sql += f" AND {col} IN ({','.join('?' * len(values))})"
            params += list(values)
    return db.query_one(sql, params)["n"]


def count_encounters(mode, since):
    """How many times a car went around a known (track) obstacle since `since`."""
    return db.query_one("SELECT COUNT(*) n FROM encounters WHERE mode=? AND ts >= ?", (mode, since))["n"]


def open_by_level(mode):
    rows = db.query("SELECT level, COUNT(*) n FROM obstacles WHERE mode=? AND status IN ('new','ack') GROUP BY level",
                    (mode,))
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    counts.update({r["level"]: r["n"] for r in rows})
    return counts


def trips_since(mode, since):
    row = db.query_one("SELECT COUNT(*) n, COALESCE(SUM(distance), 0) dist, COALESCE(SUM(n_avoid), 0) avoid "
                       "FROM trips WHERE mode=? AND start_ts >= ?", (mode, since))
    return row


def last_24h():
    return time.time() - DAY
