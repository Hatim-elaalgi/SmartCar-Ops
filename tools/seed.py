"""Fake history so every page has data on first launch:
  real mode  -> 14 days of trips for the simulated fleet on the real road network, with obstacle hotspots
  proto mode -> 7 days of toy-car runs on the tape track, going around the track's obstacles (inside of the loop)

Run again from the System page ("Reset demo data") or:
    docker compose run --rm dashboard python -m tools.seed
"""
import math
import random
import time
import uuid

import networkx as nx

import config
from core import db, events, geo, network, obstacles, settings, toy_car
from core.severity import LEVEL_META, level_for, score

DAY = 86400

REAL_HOTSPOT_PROFILES = [
    ("Roadworks", {"roadwork": 5, "cone": 4, "debris": 2}),
    ("School zone", {"pedestrian": 6, "vehicle": 1}),
    ("Pothole cluster", {"pothole": 8, "debris": 1}),
    ("Market street", {"pedestrian": 4, "vehicle": 3, "debris": 1}),
    ("Double parking", {"vehicle": 6, "pedestrian": 1}),
    ("Park edge", {"animal": 4, "pedestrian": 2}),
]
REAL_BACKGROUND = {"pothole": 3, "vehicle": 3, "pedestrian": 2, "debris": 2, "cone": 1, "animal": 1, "roadwork": 1}

PROTO_ROUTE_WEIGHTS = {"Full loop": 4, "Reverse loop": 2, "Start to top (right side)": 1, "Start to top (left side)": 1}


def _pick(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


class ObstacleSink:
    """Collects simulated detections, merging repeats of the same class on the same segment."""

    def __init__(self, mode, merge_window):
        self.mode, self.merge_window = mode, merge_window
        self.rows, self.open = [], {}
        self.weights = settings.class_weights()

    def add(self, cls, conf, prox, in_path, x, y, edge, car_id, trip_id, ts, distance):
        sev = score(cls, conf, prox, in_path, self.weights)
        i = self.open.get((cls, edge))
        if i is not None and ts - self.rows[i]["ts_last"] < self.merge_window:
            r = self.rows[i]
            r.update(ts_last=ts, seen=r["seen"] + 1, severity=max(r["severity"], sev),
                     confidence=max(r["confidence"], conf), proximity=max(r["proximity"], prox))
            return False
        self.rows.append(dict(mode=self.mode, car_id=car_id, trip_id=trip_id, ts_first=ts, ts_last=ts, cls=cls,
                              confidence=conf, proximity=prox, distance=distance, in_path=int(in_path), x=x, y=y,
                              edge=edge, severity=sev, seen=1))
        self.open[(cls, edge)] = len(self.rows) - 1
        return True

    def finalize(self, now, rng):
        for r in self.rows:
            age = now - r["ts_last"]
            if r["confidence"] < 0.62 and rng.random() < 0.45 and age > DAY / 4:
                status = "false_positive"
            elif age > 4 * DAY:
                status = rng.choices(["resolved", "ack", "false_positive"], [0.82, 0.08, 0.10])[0]
            elif age > DAY:
                status = rng.choices(["ack", "resolved", "new", "false_positive"], [0.4, 0.3, 0.2, 0.1])[0]
            else:
                status = rng.choices(["new", "ack"], [0.75, 0.25])[0]
            r["status"] = status
            r["level"] = level_for(r["severity"])

    def write(self):
        cols = ["mode", "car_id", "trip_id", "ts_first", "ts_last", "cls", "confidence", "proximity", "distance",
                "in_path", "x", "y", "edge", "severity", "level", "status", "seen"]
        db.executemany(f"INSERT INTO obstacles({','.join(cols)}, source) VALUES ({','.join('?' * len(cols))}, 'seed')",
                       [tuple(r[c] for c in cols) for r in self.rows])


def _simulate_trip(mode, net, path, car_id, start_ts, *, cruise, slow, dt, hotspots, background, base_rate,
                   hot_rate, rng, sink, route_name):
    pts = net.path_points(path)
    cum = geo.cumulative(mode, pts)
    total = cum[-1]
    edge_ids = net.path_edges(path)
    edge_ends = []
    acc = 0.0
    for e in edge_ids:
        acc += net.edges[e]["length"]
        edge_ends.append(acc)

    def edge_at(s):
        for e, end in zip(edge_ids, edge_ends):
            if s <= end:
                return e
        return edge_ids[-1]

    trip_id = f"{mode[0].upper()}{uuid.uuid4().hex[:7]}"
    ts, s, state, avoid_until = start_ts, 0.0, "CRUISING", 0.0
    n_obs = n_avoid = 0
    points = []
    while s < total:
        speed = (slow if state == "AVOIDING" else cruise) * rng.uniform(0.85, 1.1)
        ds = speed * dt
        eid = edge_at(s)
        rate = base_rate + (hot_rate if eid in hotspots else 0.0)
        if state == "CRUISING" and rng.random() < 1 - math.exp(-rate * ds):
            cls = _pick(rng, hotspots.get(eid, background))
            conf = rng.uniform(0.5, 0.98)
            d = rng.uniform(0.1, 0.9) * (45 if mode == config.REAL else 60)
            prox = max(0.0, 1 - d / (50 if mode == config.REAL else 70))
            in_path = rng.random() < 0.65
            x, y, _ = geo.point_along(mode, pts, cum, min(total, s + d))
            x, y = geo.offset_point(mode, x, y, 4 if mode == config.REAL else 6)
            if sink.add(cls, conf, prox, in_path, x, y, edge_at(min(total, s + d)), car_id, trip_id, ts, round(d, 1)):
                n_obs += 1
            if in_path:
                state, avoid_until = "AVOIDING", ts + rng.uniform(3, 6)
                n_avoid += 1
        if state == "AVOIDING" and ts >= avoid_until:
            state = "CRUISING"
        x, y, _ = geo.point_along(mode, pts, cum, s)
        points.append((trip_id, ts, x, y, round(speed, 2), state))
        s += ds
        ts += dt
    x, y, _ = geo.point_along(mode, pts, cum, total)
    points.append((trip_id, ts, x, y, 0.0, "PARKED"))
    trip = (trip_id, mode, car_id, route_name, start_ts, ts, round(total, 1), n_obs, n_avoid, "done", "seed")
    return trip, points


def _write_trips(trips, points):
    db.executemany("INSERT INTO trips(id, mode, car_id, route_name, start_ts, end_ts, distance, n_obstacles, n_avoid,"
                   " status, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)", trips)
    db.executemany("INSERT INTO trip_points(trip_id, ts, x, y, speed, state) VALUES (?,?,?,?,?,?)", points)


def _obstacle_events(mode, sink, now):
    rows = []
    for r in sink.rows:
        if r["level"] in ("critical", "high") and now - r["ts_first"] < 3 * DAY:
            rows.append((mode, r["ts_first"], "critical" if r["level"] == "critical" else "warning", "obstacle",
                         f"{LEVEL_META[r['level']]['label']} obstacle: {r['cls'].replace('_', ' ')} by {r['car_id']}"
                         f" (score {r['severity']:.0f})"))
    db.executemany("INSERT INTO events(mode, ts, level, kind, message) VALUES (?,?,?,?,?)", rows)


def seed_real(now, rng):
    net = network.real_network()
    nodes = list(net.nodes)
    # Hotspots on busy streets (high betweenness) so the route planner has something to avoid.
    centrality = nx.edge_betweenness_centrality(net.G, k=min(80, len(nodes)), weight="length", seed=1)
    busy = sorted(centrality, key=centrality.get, reverse=True)[:40]
    chosen = rng.sample(busy, k=min(12, len(busy)))
    hotspots = {}
    for i, (u, v) in enumerate(chosen):
        hotspots[network.edge_key(u, v)] = REAL_HOTSPOT_PROFILES[i % len(REAL_HOTSPOT_PROFILES)][1]

    sink = ObstacleSink(config.REAL, merge_window=2 * DAY)
    all_trips, all_points, ev = [], [], []
    for c in range(config.REAL_FLEET_SIZE):
        car_id = f"CAR-{c + 1:02d}"
        for day in range(14, -1, -1):
            day_start = now - day * DAY - (now % DAY)
            starts = sorted(day_start + rng.uniform(7, 21) * 3600 for _ in range(rng.randint(3, 5)))
            busy_until = 0
            for start in starts:
                if start < busy_until or start > now - 600:
                    continue
                src = rng.choice(nodes)
                far = [n for n in nodes if 500 < geo.distance(config.REAL, *net.xy(src), *net.xy(n)) < 1800]
                if not far:
                    continue
                dst = rng.choice(far)
                path = net.best_path(src, dst)
                if len(path) < 2:
                    continue
                name = f"{net.nodes[src]['label']} → {net.nodes[dst]['label']}"
                trip, pts = _simulate_trip(config.REAL, net, path, car_id, start, cruise=9.0, slow=3.0, dt=5,
                                           hotspots=hotspots, background=REAL_BACKGROUND, base_rate=1 / 1200,
                                           hot_rate=1 / 50, rng=rng, sink=sink, route_name=name)
                if trip[5] > now:
                    continue
                busy_until = trip[5] + 300
                all_trips.append(trip)
                all_points.extend(pts)
                ev.append((config.REAL, trip[5], "info", "trip",
                           f"{car_id} arrived at {net.nodes[dst]['label']} — {trip[6] / 1000:.1f} km, {trip[7]} obstacle(s)"))
    sink.finalize(now, rng)
    sink.write()
    _write_trips(all_trips, all_points)
    db.executemany("INSERT INTO events(mode, ts, level, kind, message) VALUES (?,?,?,?,?)", ev)
    _obstacle_events(config.REAL, sink, now)
    return len(all_trips), len(sink.rows)


def _simulate_proto_run(net, traj, start_ts, rng):
    """One run of the toy car, driven like the Arduino sketch (core/toy_car.py): slower on bends, and each track
    obstacle on the way is gone around on the inside of the loop. Returns (trip, points, encounters)."""
    path, mode = traj["nodes"], config.PROTO
    pts = net.path_points(path)
    cum = geo.cumulative(mode, pts)
    total = cum[-1]
    edge_ids = net.path_edges(path)
    edge_ends, acc = [], 0.0
    for e in edge_ids:
        acc += net.edges[e]["length"]
        edge_ends.append(acc)
    ahead = net.obstacles_along(path)
    trip_id = f"P{uuid.uuid4().hex[:7]}"
    ts, s, points, met = start_ts, 0.0, [], []
    while s < total:
        if ahead and ahead[0]["s"] - s <= toy_car.OBSTACLE_CM:      # the ultrasonic sensor sees it: detour
            o = ahead.pop(0)
            met.append((ts, o["index"], trip_id))
            for k in range(math.ceil(toy_car.AVOID_S)):
                along, lateral, _, speed = toy_car.avoid_pose(k)
                x, y, h = geo.point_along(mode, pts, cum, min(total, s + along))
                r, lateral = math.radians(h), lateral * o["side"]
                points.append((trip_id, ts, x - lateral * math.cos(r), y + lateral * math.sin(r), round(speed, 2),
                               "AVOIDING"))
                ts += 1
            s += toy_car.ALONG_CM
            continue
        eid = next((e for e, end in zip(edge_ids, edge_ends) if s <= end), edge_ids[-1])
        bend = len(net.edges[eid]["geometry"]) > 2
        speed = toy_car.SPEED_CM_S * (toy_car.CURVE_FACTOR if bend else 1.0) * rng.uniform(0.92, 1.06)
        x, y, _ = geo.point_along(mode, pts, cum, s)
        points.append((trip_id, ts, x, y, round(speed, 2), "CRUISING"))
        s += speed
        ts += 1
    x, y, _ = geo.point_along(mode, pts, cum, total)
    points.append((trip_id, ts, x, y, 0.0, "PARKED"))
    trip = (trip_id, mode, config.PROTO_CAR_ID, traj["name"], start_ts, ts, round(total, 1), len(met), len(met),
            "done", "seed")
    return trip, points, met


def seed_proto(now, rng):
    net = network.proto_network()
    trajectories = [t for t in net.trajectories if net.is_valid_path(t["nodes"])]
    ids = obstacles.ensure_track_obstacles(now)
    if not trajectories:
        return 0, len(ids)
    weights = {t["name"]: PROTO_ROUTE_WEIGHTS.get(t["name"], 1) for t in trajectories}
    by_name = {t["name"]: t for t in trajectories}
    all_trips, all_points, ev, met = [], [], [], []
    for day in range(7, -1, -1):
        day_start = now - day * DAY - (now % DAY)
        session = day_start + rng.uniform(9, 17) * 3600
        for _ in range(rng.randint(3, 6)):
            if session > now - 120:
                break
            trip, pts, enc = _simulate_proto_run(net, by_name[_pick(rng, weights)], session, rng)
            all_trips.append(trip)
            all_points.extend(pts)
            met.extend(enc)
            ev.extend((config.PROTO, ts, "info", "obstacle",
                       f"Went around the {net.obstacles[i]['cls']} (inside of the track, "
                       f"{net.obstacles[i]['label'].lower()})") for ts, i, _ in enc)
            ev.append((config.PROTO, trip[5], "info", "trip",
                       f"Run on '{trip[3]}' finished — {trip[6]:.0f} cm, went around {trip[7]} obstacle(s)"))
            session = trip[5] + rng.uniform(60, 600)
        ev.append((config.PROTO, day_start + 8.5 * 3600, "info", "camera", "ESP32-CAM connected"))
    _write_trips(all_trips, all_points)
    db.executemany("INSERT INTO encounters(mode, ts, obstacle_id, cls, trip_id, source) VALUES (?,?,?,?,?,?)",
                   [(config.PROTO, ts, ids[i], net.obstacles[i]["cls"], tid, "seed") for ts, i, tid in met])
    for i, oid in enumerate(ids):
        mine = sorted(e for e in met if e[1] == i)
        if mine:
            db.execute("UPDATE obstacles SET ts_first=?, ts_last=?, seen=?, trip_id=?, source='seed' WHERE id=?",
                       (mine[0][0], mine[-1][0], len(mine), mine[-1][2], oid))
    db.executemany("INSERT INTO events(mode, ts, level, kind, message) VALUES (?,?,?,?,?)", ev)
    return len(all_trips), len(ids)


def reseed_proto(now=None, rng_seed=7):
    """Regenerate the prototype history only (the real-world data is kept)."""
    for r in db.query("SELECT snapshot FROM obstacles WHERE mode=? AND snapshot IS NOT NULL", (config.PROTO,)):
        if not r["snapshot"].startswith("track_"):
            (config.SNAPSHOT_DIR / r["snapshot"]).unlink(missing_ok=True)
    db.clear_mode(config.PROTO)
    return seed_proto(now or time.time(), random.Random(rng_seed))


def seed_all(now=None, rng_seed=42):
    now = now or time.time()
    rng = random.Random(rng_seed)
    db.clear_all()
    real = seed_real(now, rng)
    proto = seed_proto(now, rng)
    events.log("system", "info", "system", "Demo data generated")
    return {"real": real, "proto": proto}


def ensure():
    """Seed only when the database is empty (first launch)."""
    if not db.query_one("SELECT id FROM trips LIMIT 1"):
        return seed_all()
    return None


if __name__ == "__main__":
    db.init()
    t0 = time.time()
    result = seed_all()
    print(f"Seeded {result} in {time.time() - t0:.1f}s")
