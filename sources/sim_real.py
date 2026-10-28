"""Simulated real-world fleet: cars drive between random intersections of the real road network,
meet obstacles (more often on segments with a bad history) and slow down / stop to avoid them."""
import logging
import math
import random
import threading
import time

import config
from core import events, geo, network, obstacles, trips
from core.hub import hub
from core.severity import level_for, score
from sources import real_model

log = logging.getLogger(__name__)
MODE = config.REAL
CLASS_MIX = {"pothole": 3, "vehicle": 3, "pedestrian": 3, "debris": 2, "cone": 1, "roadwork": 1, "animal": 1}
TARGET_SPEED = {"AVOIDING": 3.0, "STOPPED": 0.0}


class _Car:
    def __init__(self, car_id, node, net):
        self.id, self.node = car_id, node
        self.x, self.y = net.xy(node)
        self.heading, self.speed = 0.0, 0.0
        self.state, self.state_until = "PARKED", time.time() + random.uniform(1, 6)
        self.battery = random.uniform(55, 100)
        self.cruise = random.uniform(8, 12)
        self.trip_id, self.dest_label = None, ""
        self.path, self.points, self.cum, self.edges, self.edge_ends = [], [], [0.0], [], []
        self.s = self.trip_dist = 0.0
        self.front, self.front_until = None, 0.0
        self.last_point = 0.0

    def edge_at(self, s):
        for e, end in zip(self.edges, self.edge_ends):
            if s <= end:
                return e
        return self.edges[-1] if self.edges else None


class RealFleetSimulator(threading.Thread):
    def __init__(self, n_cars=config.REAL_FLEET_SIZE, tick=1.0):
        super().__init__(daemon=True, name="real-fleet-sim")
        self.n_cars, self.tick = n_cars, tick
        self.cars = []
        self._risk, self._risk_ts = {}, 0.0
        self.running = False
        self._lock = threading.RLock()
        self.dashcam = None      # set by services: real video + models drive CAR-01's obstacles

    def risk(self, now):
        if now - self._risk_ts > 60:
            self._risk, self._risk_ts = obstacles.edge_risk(MODE, now), now
        return self._risk

    def run(self):
        trips.close_dangling(MODE, "sim")
        net = network.real_network()
        nodes = list(net.nodes)
        self.cars = [_Car(f"CAR-{i + 1:02d}", random.choice(nodes), net) for i in range(self.n_cars)]
        self.running = True
        last = time.time()
        while True:
            now = time.time()
            dt = min(now - last, 3.0)
            last = now
            for car in self.cars:
                try:
                    with self._lock:
                        self._step(net, car, now, dt)
                except Exception:
                    log.exception("Simulation step failed for %s", car.id)
            hub.set_source(MODE, "simulated")
            time.sleep(max(0.05, self.tick - (time.time() - now)))

    # --- one car, one tick ------------------------------------------------------------
    def _step(self, net, car, now, dt):
        if car.state == "CHARGING":
            car.speed = 0.0
            car.battery = min(100.0, car.battery + 1.5 * dt)
            if car.battery >= 99:
                car.state, car.state_until = "PARKED", now + 3
        elif car.state == "PARKED":
            car.speed = 0.0
            if now >= car.state_until:
                self._new_trip(net, car, now)
        else:
            target = TARGET_SPEED.get(car.state, car.cruise)
            car.speed = max(0.0, car.speed + max(-4 * dt, min(2.5 * dt, target - car.speed)))
            ds = car.speed * dt
            car.s += ds
            car.trip_dist += ds
            car.battery = max(0.0, car.battery - 0.01 * dt - 0.0025 * ds)
            if car.state in ("AVOIDING", "STOPPED") and now >= car.state_until:
                if car.state == "STOPPED":
                    car.state, car.state_until = "AVOIDING", now + 3
                else:
                    car.state = "CRUISING"
            eid = car.edge_at(car.s)
            if car.state == "CRUISING" and eid and not self._has_live_camera(car):
                # Risky segments produce more obstacles, capped so risk can't snowball.
                rate = 1 / 1500 + min(self.risk(now).get(eid, 0.0), 2.0) / 1200
                if random.random() < 1 - math.exp(-rate * ds):
                    self._encounter(net, car, now)
            car.x, car.y, car.heading = geo.point_along(MODE, car.points, car.cum, car.s)
            if now - car.last_point >= 3:
                trips.add_point(car.trip_id, now, car.x, car.y, round(car.speed, 2), car.state)
                car.last_point = now
            if car.s >= car.cum[-1]:
                self._finish_trip(net, car, now)
        self._publish(car, now)

    # --- hooks used by the dashcam (real video + trained models) -------------------------------
    def _has_live_camera(self, car):
        return bool(self.dashcam and self.dashcam.active and car.id == config.DASHCAM_CAR_ID)

    def point_ahead(self, car_id, distance):
        """Map position `distance` metres ahead of the car on its route: (x, y, edge, trip_id) or None."""
        with self._lock:
            car = next((c for c in self.cars if c.id == car_id), None)
            if car is None:
                return None
            if not car.trip_id or len(car.points) < 2:
                return car.x, car.y, None, None
            s = min(car.cum[-1], car.s + distance)
            x, y, _ = geo.point_along(MODE, car.points, car.cum, s)
            return x, y, car.edge_at(s), car.trip_id

    def react(self, car_id, state, seconds, front):
        """A real detection makes the car slow down (AVOIDING) or brake (STOPPED)."""
        with self._lock:
            car = next((c for c in self.cars if c.id == car_id), None)
            if car is None or car.state not in ("CRUISING", "AVOIDING", "STOPPED"):
                return
            now = time.time()
            if state == "STOPPED" or car.state == "CRUISING":
                if car.state == "CRUISING":
                    trips.bump(car.trip_id, avoid=1)
                car.state = state
            car.state_until = max(car.state_until, now + seconds)
            car.front, car.front_until = round(front, 1), now + 3

    def _publish(self, car, now):
        total = car.cum[-1] if car.trip_id else 0
        hub.publish(MODE, {
            "car_id": car.id, "ts": now, "x": car.x, "y": car.y, "heading": car.heading,
            "speed": car.speed, "state": car.state, "battery": round(car.battery, 1),
            "trip_id": car.trip_id, "dest": car.dest_label if car.trip_id else "",
            "progress": (car.s / total) if total else None, "trip_dist": car.trip_dist if car.trip_id else 0,
            "front": car.front if now < car.front_until else None, "source": "sim",
        })

    def _new_trip(self, net, car, now):
        if car.battery < 20:
            car.state = "CHARGING"
            events.log(MODE, "warning", "vehicle", f"{car.id} battery low ({car.battery:.0f}%) — charging")
            return
        src = car.node
        far = [n for n in net.nodes if 500 < geo.distance(MODE, *net.xy(src), *net.xy(n)) < 1800]
        if not far:
            car.state_until = now + 5
            return
        dst = random.choice(far)
        path = net.best_path(src, dst, self.risk(now), alpha=1.0)   # the fleet uses the safety-aware planner
        if len(path) < 2:
            car.state_until = now + 5
            return
        car.path, car.points = path, net.path_points(path)
        car.cum = geo.cumulative(MODE, car.points)
        car.edges = net.path_edges(path)
        acc, car.edge_ends = 0.0, []
        for e in car.edges:
            acc += net.edges[e]["length"]
            car.edge_ends.append(acc)
        car.s = car.trip_dist = 0.0
        car.state, car.dest_label = "CRUISING", net.nodes[dst]["label"]
        car.trip_id = trips.start(MODE, car.id, route_name=f"{net.nodes[src]['label']} → {car.dest_label}", ts=now)
        events.log(MODE, "info", "trip", f"{car.id} started a trip to {car.dest_label} ({car.cum[-1] / 1000:.1f} km)")

    def _finish_trip(self, net, car, now):
        trips.add_point(car.trip_id, now, car.x, car.y, 0.0, "PARKED")
        trips.finish(car.trip_id, round(car.trip_dist, 1), now)
        info = trips.get(car.trip_id) or {}
        events.log(MODE, "info", "trip", f"{car.id} arrived at {car.dest_label} — {car.trip_dist / 1000:.1f} km, "
                                         f"{info.get('n_obstacles', 0)} obstacle(s)")
        car.node = car.path[-1]
        car.trip_id = None
        # The dashcam car keeps driving while its video plays (a parked car with a moving video would look odd).
        car.state, car.state_until = "PARKED", now + (1 if self._has_live_camera(car) else random.uniform(5, 15))

    def _encounter(self, net, car, now):
        mix_cls = random.choices(list(CLASS_MIX), weights=list(CLASS_MIX.values()))[0]
        d = random.uniform(4, 45)
        det = {"cls": mix_cls, "conf": random.uniform(0.55, 0.98), "proximity": max(0.0, 1 - d / 50),
               "in_path": random.random() < 0.65, "snapshot": None}
        model = real_model.get()
        if model.enabled:
            det.update(model.sample() or {})
        ahead = min(car.cum[-1], car.s + d)
        x, y, _ = geo.point_along(MODE, car.points, car.cum, ahead)
        x, y = geo.offset_point(MODE, x, y, 4)
        oid, is_new = obstacles.record(MODE, det["cls"], det["conf"], det["proximity"], det["in_path"], x, y,
                                       car_id=car.id, trip_id=car.trip_id, edge=car.edge_at(ahead),
                                       distance=round(d, 1), source="sim", snapshot=det["snapshot"], ts=now)
        car.front, car.front_until = round(d, 1), now + 4
        avoided = 0
        if det["in_path"]:
            avoided = 1
            if level_for(score(det["cls"], det["conf"], det["proximity"], True)) == "critical":
                car.state, car.state_until = "STOPPED", now + 2.5
            else:
                car.state, car.state_until = "AVOIDING", now + random.uniform(3, 6)
        trips.bump(car.trip_id, obstacles=int(is_new), avoid=avoided)
