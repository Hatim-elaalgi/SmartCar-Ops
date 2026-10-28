"""Demo toy car: follows the active trajectory on the tape track while the ESP32-CAM is offline,
so the Prototype pages can be developed and shown without the hardware. Clearly labelled "DEMO" in the UI.

It drives like the Arduino sketch (core/toy_car.py): full speed on straight tape, slower on bends. The only obstacles
are the ones placed on the test track (track JSON "obstacles"): the camera sees each one coming, and at 13 cm (the
ultrasonic sensor) the car goes around it on the INSIDE of the loop with the sketch's timed detour, then back to the
tape. It never stops for an obstacle."""
import logging
import random
import threading
import time

import config
from core import events, geo, network, obstacles, toy_car, trips
from core.hub import hub

log = logging.getLogger(__name__)
MODE = config.PROTO
SEE_AHEAD_CM = 60                       # the camera starts to see an obstacle this far ahead


class ProtoDemoSimulator(threading.Thread):
    def __init__(self, camera, tick=0.5):
        super().__init__(daemon=True, name="proto-demo-sim")
        self.camera, self.tick = camera, tick
        self.state, self.state_until = "PARKED", time.time() + 2
        self.speed, self.s, self.trip_id = 0.0, 0.0, None
        self.route, self.route_name, self.points, self.cum, self.edges, self.node_s = [], "", [], [0.0], [], []
        self.curved = {}
        self.last_marker, self.x, self.y, self.heading = None, 0.0, 0.0, 0.0
        self.ahead = []                         # track obstacles still ahead on this run (network.obstacles_along)
        self.seen = set()                       # ... already spotted by the camera on this run
        self.avoiding = None                    # the obstacle being gone around
        self.avoid_path = None                  # its estimated detour path, drawn on the maps
        self.avoid_t0 = self.avoid_s0 = 0.0
        self.lateral, self.rel_heading = 0.0, 0.0
        self._last_point = 0.0

    def run(self):
        trips.close_dangling(MODE, "demo")
        last = time.time()
        while True:
            now = time.time()
            dt = min(now - last, 2.0)
            last = now
            try:
                if self.camera.connected:
                    self._abort(now)            # real data takes over
                elif config.PROTO_DEMO_WHEN_OFFLINE:
                    hub.set_source(MODE, "demo")
                    self._step(now, dt)
                else:
                    self._abort(now)
                    hub.set_source(MODE, "offline")
            except Exception:
                log.exception("Prototype demo step failed")
            time.sleep(self.tick)

    def _abort(self, now):
        if self.trip_id:
            trips.finish(self.trip_id, round(self.s, 1), now, status="interrupted")
            self.trip_id, self.state = None, "PARKED"

    def _start_run(self, now):
        net = network.proto_network()
        traj = network.active_trajectory()
        if not traj or not net.is_valid_path(traj["nodes"]):
            self.state_until = now + 5
            return
        self.route, self.route_name = traj["nodes"], traj["name"]
        self.points = net.path_points(self.route)
        self.cum = geo.cumulative(MODE, self.points)
        self.edges = net.path_edges(self.route)
        self.curved = {e: len(net.edges[e]["geometry"]) > 2 for e in self.edges}    # rounded corners (via points)
        acc, self.node_s = 0.0, [0.0]
        for e in self.edges:
            acc += net.edges[e]["length"]
            self.node_s.append(acc)
        self.s, self.state, self.last_marker = 0.0, "CRUISING", self.route[0]
        self.ahead, self.seen, self.avoiding = net.obstacles_along(self.route), set(), None
        self.lateral, self.rel_heading = 0.0, 0.0
        self.trip_id = trips.start(MODE, config.PROTO_CAR_ID, route_name=self.route_name, ts=now, source="demo")

    def _edge_at(self, s):
        for e, end in zip(self.edges, self.node_s[1:]):
            if s <= end:
                return e
        return self.edges[-1] if self.edges else None

    def _step(self, now, dt):
        if self.state == "PARKED":
            self.speed = 0.0
            if now >= self.state_until:
                self._start_run(now)
        elif self.state == "AVOIDING":
            self._avoid_step(now)
        else:
            eid = self._edge_at(self.s)
            target = toy_car.SPEED_CM_S * (toy_car.CURVE_FACTOR if self.curved.get(eid) else 1.0)
            target *= random.uniform(0.95, 1.05)
            self.speed = max(0.0, self.speed + max(-60 * dt, min(40 * dt, target - self.speed)))
            self.s += self.speed * dt
            if self.ahead:
                o = self.ahead[0]
                d = o["s"] - self.s
                if d <= SEE_AHEAD_CM and id(o) not in self.seen:
                    self._spot(now, o, d)
                if d <= toy_car.OBSTACLE_CM:            # the HC-SR04 sees it: the sketch's detour starts
                    late = (toy_car.OBSTACLE_CM - d) / max(self.speed, 1.0)   # crossed 13 cm during this step
                    self.s = o["s"] - toy_car.OBSTACLE_CM
                    self._start_avoid(now - late, self.ahead.pop(0))
                    self._avoid_step(now)
        if self.state != "PARKED":
            for node, ns in zip(self.route, self.node_s):
                if ns <= self.s:
                    self.last_marker = node
            self._place()
            if now - self._last_point >= 1:
                trips.add_point(self.trip_id, now, self.x, self.y, round(self.speed, 1), self.state)
                self._last_point = now
            if self.s >= self.cum[-1]:
                loop = self.route[0] == self.route[-1]
                if not loop:
                    trips.add_point(self.trip_id, now, self.x, self.y, 0.0, "PARKED")
                trips.finish(self.trip_id, round(self.cum[-1], 1), now)
                info = trips.get(self.trip_id) or {}
                events.log(MODE, "info", "trip", f"Demo {'lap' if loop else 'run'} on '{self.route_name}' finished — "
                                                 f"{self.cum[-1]:.0f} cm, went around {info.get('n_obstacles', 0)} "
                                                 f"obstacle(s)")
                self.trip_id, self.avoiding, self.lateral, self.rel_heading = None, None, 0.0, 0.0
                self.state, self.state_until = "PARKED", now + 8
                if loop:
                    self._start_run(now)        # a loop: the car drives straight on into the next lap
        total = self.cum[-1] if self.trip_id else 0
        hub.publish(MODE, {
            "car_id": config.PROTO_CAR_ID, "ts": now, "x": self.x, "y": self.y, "heading": self.heading,
            "speed": self.speed, "state": self.state, "trip_id": self.trip_id, "route": self.route_name,
            "last_marker": self.last_marker, "progress": (min(self.s, total) / total) if total else None,
            "trip_dist": self.s if self.trip_id else 0, "fps": round(random.uniform(11, 14), 1),
            "position_known": True, "source": "demo",
            "avoid_path": self.avoid_path if self.state == "AVOIDING" else None,
        })

    def _place(self):
        """Position on the tape, moved to the inside of the loop while going around an obstacle."""
        self.x, self.y, h = toy_car.track_point(self.points, self.cum, self.s, self.lateral)
        self.heading = (h + self.rel_heading) % 360

    def _spot(self, now, o, d):
        """The camera sees the obstacle coming."""
        self.seen.add(id(o))
        conf = random.uniform(0.85, 0.97)
        obstacles.track_sighting(o["cls"], conf, max(0.0, 1 - d / 70), trip_id=self.trip_id, source="demo", ts=now,
                                 near=(o["x"], o["y"]))
        hub.add_detection({"cls": o["cls"], "conf": conf, "proximity": max(0.0, 1 - d / 70), "in_path": True,
                           "ts": now, "source": "demo"})

    def _start_avoid(self, now, o):
        self.state, self.avoid_t0, self.avoid_s0, self.avoiding = "AVOIDING", now, self.s, o
        self.avoid_path = toy_car.avoid_track(self.points, self.cum, self.s, o["side"])
        obstacles.track_sighting(o["cls"], random.uniform(0.9, 0.98), 1.0, trip_id=self.trip_id, source="demo",
                                 ts=now, encounter=True, near=(o["x"], o["y"]))
        trips.bump(self.trip_id, obstacles=1, avoid=1)
        events.log(MODE, "info", "obstacle", f"Demo car went around the {o['cls']} "
                                             f"(inside of the track, {o['label'].lower()})", now)

    def _avoid_step(self, now):
        t = now - self.avoid_t0
        along, lateral, rel_heading, self.speed = toy_car.avoid_pose(t)
        side = self.avoiding["side"]            # +1: inside of the loop on the left, -1: on the right
        self.lateral, self.rel_heading = lateral * side, rel_heading * side
        self.s = min(self.cum[-1], self.avoid_s0 + along)
        if t >= toy_car.AVOID_S:                # back on the tape: line following again
            self.state, self.avoiding, self.lateral, self.rel_heading = "CRUISING", None, 0.0, 0.0
            self.avoid_path = None
