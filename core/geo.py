"""Small geometry helpers that work for both modes.

real  -> points are (lon, lat), distances in metres (haversine)
proto -> points are (x, y) in cm, distances in cm (euclidean)
"""
import bisect
import math
import random

import config

EARTH_RADIUS_M = 6_371_000.0


def haversine(x1, y1, x2, y2):
    p1, p2 = math.radians(y1), math.radians(y2)
    dp, dl = p2 - p1, math.radians(x2 - x1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def euclid(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def distance(mode, x1, y1, x2, y2):
    return haversine(x1, y1, x2, y2) if mode == config.REAL else euclid(x1, y1, x2, y2)


def bearing(mode, x1, y1, x2, y2):
    """Heading in degrees, 0 = north / up, clockwise."""
    dx, dy = x2 - x1, y2 - y1
    if mode == config.REAL:
        dx *= math.cos(math.radians((y1 + y2) / 2))
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def cumulative(mode, points):
    """Cumulative distance at each vertex of a polyline."""
    cum = [0.0]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        cum.append(cum[-1] + distance(mode, x1, y1, x2, y2))
    return cum


def point_along(mode, points, cum, s):
    """Position and heading at distance s along a polyline (cum from cumulative())."""
    if len(points) == 1:
        return points[0][0], points[0][1], 0.0
    s = max(0.0, min(s, cum[-1]))
    i = max(1, min(bisect.bisect_right(cum, s), len(points) - 1))
    (x1, y1), (x2, y2) = points[i - 1], points[i]
    seg = cum[i] - cum[i - 1]
    t = 0.0 if seg <= 0 else (s - cum[i - 1]) / seg
    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, bearing(mode, x1, y1, x2, y2)


def offset_point(mode, x, y, max_offset):
    """Return a nearby point (used to scatter simulated obstacles a little off the centre line)."""
    if mode == config.REAL:
        k = max_offset / 111_320
        return x + random.uniform(-k, k) / max(0.2, math.cos(math.radians(y))), y + random.uniform(-k, k)
    return x + random.uniform(-max_offset, max_offset), y + random.uniform(-max_offset, max_offset)


def deg_box(mode, y, radius):
    """Half-size of a bounding box (dx, dy) that contains a circle of `radius` (m or cm)."""
    if mode == config.REAL:
        dy = radius / 111_320
        return dy / max(0.2, math.cos(math.radians(y))), dy
    return radius, radius


def segment_distance(mode, px, py, x1, y1, x2, y2):
    """Distance from a point to a segment (planar approximation, fine at city scale)."""
    if mode == config.REAL:
        k = math.cos(math.radians(py))
        px, x1, x2 = px * k, x1 * k, x2 * k
        scale = 111_320
    else:
        scale = 1
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1) * scale
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)) * scale
