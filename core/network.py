"""Road network (real mode) and tape track (prototype mode) as graphs, plus route planning.

Route cost of a segment = length + alpha x RISK_UNIT x risk(segment)
  alpha = 0  -> pure shortest path
  alpha up   -> avoid segments with a history of obstacles
RISK_UNIT says how much extra distance one unit of risk is "worth".
"""
import json
import math
import random
import threading

import networkx as nx

import config
from core import geo, settings, toy_car

RISK_UNIT = {config.REAL: 150.0, config.PROTO: 60.0}          # metres / centimetres
CRUISE_SPEED = {config.REAL: config.REAL_CRUISE_MS, config.PROTO: config.PROTO_SPEED_CM_S}


def edge_key(u, v):
    u, v = str(u), str(v)
    return f"{u}~{v}" if u <= v else f"{v}~{u}"


class Network:
    def __init__(self, mode, name, nodes, edges, trajectories=None, meta=None, obstacles=None):
        self.mode = mode
        self.name = name
        self.meta = meta or {}
        self.trajectories = trajectories or []
        # Prototype: obstacles placed on the test track ({cls, x, y, label}); the car always goes around them.
        self.obstacles = [{"cls": str(o["cls"]), "x": float(o["x"]), "y": float(o["y"]),
                           "label": o.get("label") or str(o["cls"])} for o in (obstacles or [])]
        self.nodes = {}
        for n in nodes:
            marker = n.get("marker")
            self.nodes[str(n["id"])] = {
                "x": float(n["x"]), "y": float(n["y"]),
                "label": n.get("label") or str(n["id"]),
                "marker": int(marker) if marker not in (None, "") else None,
            }
        self.markers = {n["marker"]: nid for nid, n in self.nodes.items() if n["marker"] is not None}

        self.edges = {}
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        for e in edges:
            u, v = str(e["u"]), str(e["v"])
            if u == v or u not in self.nodes or v not in self.nodes:
                continue
            geom = [tuple(p) for p in (e.get("geometry") or [])]
            if not geom:
                geom = [self.xy(u)] + [tuple(p) for p in (e.get("via") or [])] + [self.xy(v)]
            length = float(e.get("length") or geo.cumulative(mode, geom)[-1])
            eid = edge_key(u, v)
            if eid in self.edges and self.edges[eid]["length"] <= length:
                continue
            self.edges[eid] = {"id": eid, "u": u, "v": v, "length": length, "geometry": geom,
                               "name": e.get("name", "")}
            self.G.add_edge(u, v, eid=eid, length=length)

    # --- geometry -------------------------------------------------------------------
    def xy(self, nid):
        n = self.nodes[str(nid)]
        return n["x"], n["y"]

    def edge_geometry(self, u, v):
        e = self.edges[edge_key(u, v)]
        return e["geometry"] if e["u"] == str(u) else list(reversed(e["geometry"]))

    def path_edges(self, path):
        return [edge_key(a, b) for a, b in zip(path, path[1:])]

    def path_points(self, path):
        if not path:
            return []
        pts = [self.xy(path[0])]
        for a, b in zip(path, path[1:]):
            pts.extend(self.edge_geometry(a, b)[1:])
        return pts

    def path_length(self, path):
        return sum(self.edges[e]["length"] for e in self.path_edges(path))

    def is_valid_path(self, path):
        return len(path) >= 2 and all(self.G.has_edge(a, b) for a, b in zip(path, path[1:]))

    def bounds(self):
        xs = [n["x"] for n in self.nodes.values()]
        ys = [n["y"] for n in self.nodes.values()]
        return min(xs), max(xs), min(ys), max(ys)

    def center(self):
        x0, x1, y0, y1 = self.bounds()
        return (x0 + x1) / 2, (y0 + y1) / 2

    def zoom(self):
        """Map zoom level that fits the whole network (real mode)."""
        x0, x1, y0, y1 = self.bounds()
        span = max(x1 - x0, (y1 - y0) * 1.6, 1e-4)
        return max(10.0, min(17.0, math.log2(360 / span) - 0.3))

    def nearest_node(self, x, y):
        return min(self.nodes, key=lambda nid: geo.distance(self.mode, x, y, *self.xy(nid)))

    def nearest_edge(self, x, y):
        best, best_d = None, float("inf")
        for eid, e in self.edges.items():
            g = e["geometry"]
            for (x1, y1), (x2, y2) in zip(g, g[1:]):
                d = geo.segment_distance(self.mode, x, y, x1, y1, x2, y2)
                if d < best_d:
                    best, best_d = eid, d
        return best

    def inside_side(self, x, y, heading):
        """+1 when the inside of the loop (centre of the checkpoints) is on the left of a car at (x, y) driving
        towards `heading` (degrees, 0 = up, clockwise), -1 when it is on the right."""
        cx = sum(n["x"] for n in self.nodes.values()) / max(1, len(self.nodes))
        cy = sum(n["y"] for n in self.nodes.values()) / max(1, len(self.nodes))
        h = math.radians(heading)
        return 1 if -math.cos(h) * (cx - x) + math.sin(h) * (cy - y) > 0 else -1

    def obstacles_along(self, path, max_offset=8.0):
        """Prototype: the track obstacles lying on this route, in driving order:
        [{"s": distance along the route, "index" (in the track's list), "cls", "label", "x", "y", "side"}].
        side = inside_side() there: the car goes around on the inside of the loop."""
        pts = self.path_points(path)
        cum = geo.cumulative(self.mode, pts)
        found = []
        for index, o in enumerate(self.obstacles):
            best = None
            for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts, pts[1:])):
                seg2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
                t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((o["x"] - x1) * (x2 - x1) + (o["y"] - y1) * (y2 - y1)) / seg2))
                px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                d = math.hypot(o["x"] - px, o["y"] - py)
                if d <= max_offset and (best is None or d < best[0]):
                    side = self.inside_side(px, py, geo.bearing(self.mode, x1, y1, x2, y2))
                    best = (d, cum[i] + t * math.sqrt(seg2), side)
            if best:
                found.append({"s": best[1], "index": index, "cls": o["cls"], "label": o["label"], "x": o["x"],
                              "y": o["y"], "side": best[2]})
        return sorted(found, key=lambda f: f["s"])

    # --- routing --------------------------------------------------------------------
    def best_path(self, src, dst, risk=None, alpha=0.0):
        risk = risk or {}
        unit = RISK_UNIT[self.mode]

        def weight(u, v, d):
            return d["length"] + alpha * unit * risk.get(d["eid"], 0.0)

        try:
            return nx.shortest_path(self.G, str(src), str(dst), weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def path_risk(self, path, risk):
        return sum(risk.get(e, 0.0) for e in self.path_edges(path))

    def path_cost(self, path, risk, alpha):
        return self.path_length(path) + alpha * RISK_UNIT[self.mode] * self.path_risk(path, risk)

    def eta_seconds(self, path):
        if self.mode == config.PROTO:           # the toy car slows down on bends (one wheel stops in turns)
            return sum(self.edges[e]["length"] / (CRUISE_SPEED[self.mode] *
                                                  (toy_car.CURVE_FACTOR if len(self.edges[e]["geometry"]) > 2 else 1))
                       for e in self.path_edges(path))
        return self.path_length(path) / CRUISE_SPEED[self.mode]


# --- loading -------------------------------------------------------------------------
_lock = threading.Lock()
_real = None
_proto = None
_proto_stamp = None


def synthetic_grid(center=(2.3488, 48.8500), n=10, spacing_m=110):
    """Fallback street grid used when resources/road_graph.json is missing."""
    rnd = random.Random(7)
    cx, cy = center
    dlat = spacing_m / 111_320
    dlon = dlat / math.cos(math.radians(cy))
    nodes, edges = [], []
    for i in range(n):
        for j in range(n):
            nodes.append({"id": f"{i}-{j}", "label": f"Street {i + 1} / Avenue {j + 1}",
                          "x": cx + (j - n / 2) * dlon + rnd.uniform(-0.15, 0.15) * dlon,
                          "y": cy + (i - n / 2) * dlat + rnd.uniform(-0.15, 0.15) * dlat})
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                edges.append({"u": f"{i}-{j}", "v": f"{i}-{j + 1}", "name": f"Street {i + 1}"})
            if i + 1 < n and rnd.random() > 0.08:
                edges.append({"u": f"{i}-{j}", "v": f"{i + 1}-{j}", "name": f"Avenue {j + 1}"})
    return {"name": "Synthetic grid", "nodes": nodes, "edges": edges}


def real_network():
    global _real
    with _lock:
        if _real is None:
            if config.ROAD_GRAPH_PATH.exists():
                data = json.loads(config.ROAD_GRAPH_PATH.read_text(encoding="utf-8"))
            else:
                data = synthetic_grid()
            _real = Network(config.REAL, data.get("name", "Road network"), data["nodes"], data["edges"],
                            meta={"bbox": data.get("bbox"), "source": data.get("source", "synthetic")})
        return _real


def _track_path():
    return config.TRACK_PATH if config.TRACK_PATH.exists() else config.DEFAULT_TRACK_PATH


def load_track_data():
    return json.loads(_track_path().read_text(encoding="utf-8"))


def proto_network():
    global _proto, _proto_stamp
    path = _track_path()
    stamp = (str(path), path.stat().st_mtime_ns)
    with _lock:
        if _proto is None or _proto_stamp != stamp:
            data = json.loads(path.read_text(encoding="utf-8"))
            _proto = Network(config.PROTO, data.get("name", "Track"), data["nodes"], data["edges"],
                             trajectories=data.get("trajectories", []), meta={"units": "cm"},
                             obstacles=data.get("obstacles", []))
            _proto_stamp = stamp
        return _proto


def get_network(mode):
    return real_network() if mode == config.REAL else proto_network()


def validate_track(data):
    errors = []
    ids, markers = set(), set()
    for n in data.get("nodes", []):
        nid = str(n.get("id", "")).strip()
        if not nid:
            errors.append("A node has no id.")
            continue
        if nid in ids:
            errors.append(f"Node id '{nid}' is used twice.")
        ids.add(nid)
        try:
            float(n.get("x")), float(n.get("y"))
        except (TypeError, ValueError):
            errors.append(f"Node '{nid}' needs numeric x and y (cm).")
        m = n.get("marker")
        if m not in (None, ""):
            try:
                m = int(m)
                if not 0 <= m < 50:
                    raise ValueError
                if m in markers:
                    errors.append(f"Marker {m} is used twice.")
                markers.add(m)
            except (TypeError, ValueError):
                errors.append(f"Node '{nid}': marker must be a number from 0 to 49.")
    adjacency = set()
    for e in data.get("edges", []):
        u, v = str(e.get("u", "")).strip(), str(e.get("v", "")).strip()
        if u not in ids or v not in ids:
            errors.append(f"Segment {u or '?'} - {v or '?'} uses an unknown node.")
        elif u == v:
            errors.append(f"Segment {u} - {v} connects a node to itself.")
        adjacency.add(edge_key(u, v))
    names = set()
    for t in data.get("trajectories", []):
        name = str(t.get("name", "")).strip()
        if not name:
            errors.append("A trajectory has no name.")
        elif name in names:
            errors.append(f"Trajectory name '{name}' is used twice.")
        names.add(name)
        path = t.get("nodes", [])
        if len(path) < 2:
            errors.append(f"Trajectory '{name}' needs at least 2 nodes.")
        for a, b in zip(path, path[1:]):
            if edge_key(a, b) not in adjacency:
                errors.append(f"Trajectory '{name}': no segment between {a} and {b}.")
    for o in data.get("obstacles", []):
        try:
            float(o.get("x")), float(o.get("y"))
            if not str(o.get("cls", "")).strip():
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Each track obstacle needs a class and numeric x and y (cm).")
    if not ids:
        errors.append("The track has no nodes.")
    return errors


def save_track(data):
    errors = validate_track(data)
    if not errors:
        config.TRACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.TRACK_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return errors


def active_trajectory():
    net = proto_network()
    if not net.trajectories:
        return None
    wanted = settings.get("active_trajectory")
    for t in net.trajectories:
        if t["name"] == wanted:
            return t
    return net.trajectories[0]
