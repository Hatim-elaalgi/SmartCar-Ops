"""Route suggestions: shortest vs safest vs the recommended trade-off for the chosen priority."""
import config
from core import network, obstacles

SAFEST_ALPHA = 8.0
# Risk thresholds used to colour risky segments on the map (the toy track is short, so it accumulates faster).
RISK_LEVELS = {config.REAL: (("critical", 0.8), ("high", 0.3)), config.PROTO: (("critical", 1.2), ("high", 0.5))}


def _summary(net, path, risk, open_df):
    edges = set(net.path_edges(path))
    on_route = open_df[open_df["edge"].isin(edges)] if len(open_df) else open_df
    avoided = len(net.obstacles_along(path)) if net.mode == config.PROTO else 0
    return {
        "path": path,
        "points": net.path_points(path),
        "length": net.path_length(path),
        "eta": net.eta_seconds(path),
        "risk": net.path_risk(path, risk),
        "obstacles": int(len(on_route)),
        "avoided": avoided,             # prototype: track obstacles the car goes around on this route
        "by_level": on_route["level"].value_counts().to_dict() if len(on_route) else {},
    }


def plan(mode, start, end, alpha):
    """Returns (routes, risk). Each route: name, also (other names with the same path), metrics."""
    net = network.get_network(mode)
    risk = obstacles.edge_risk(mode)
    open_df = obstacles.listing(mode, statuses=obstacles.OPEN)
    routes, by_path = [], {}
    for name, a in (("Recommended", alpha), ("Shortest", 0.0), ("Safest", SAFEST_ALPHA)):
        path = net.best_path(start, end, risk, a)
        if len(path) < 2:
            continue
        key = tuple(path)
        if key in by_path:
            by_path[key]["also"].append(name)
            continue
        r = _summary(net, path, risk, open_df)
        r.update(name=name, also=[])
        by_path[key] = r
        routes.append(r)
    return routes, risk


def rank_trajectories(alpha):
    """Prototype: score the manual trajectories with the same cost as the planner."""
    net = network.proto_network()
    risk = obstacles.edge_risk(net.mode)
    open_df = obstacles.listing(net.mode, statuses=obstacles.OPEN)
    rows = []
    for t in net.trajectories:
        if not net.is_valid_path(t["nodes"]):
            continue
        s = _summary(net, t["nodes"], risk, open_df)
        rows.append({"name": t["name"], "path": " → ".join(t["nodes"]), "length": s["length"], "eta": s["eta"],
                     "risk": s["risk"], "obstacles": s["avoided"],
                     "cost": net.path_cost(t["nodes"], risk, alpha)})
    rows.sort(key=lambda r: r["cost"])
    return rows


def risk_layers(risk, mode):
    (_, very), (_, risky) = RISK_LEVELS[mode]
    layers = {"critical": [], "high": []}
    for eid, value in risk.items():
        if value >= very:
            layers["critical"].append(eid)
        elif value >= risky:
            layers["high"].append(eid)
    return layers


def recommendation(routes):
    if not routes:
        return "No route found between these points."
    if len(routes) == 1:
        return "One route wins on every criterion: no safer detour exists here."
    rec = routes[0]
    shortest = next((r for r in routes if r["name"] == "Shortest" or "Shortest" in r["also"]), None)
    if shortest is None or shortest is rec:
        return "The shortest route is also the best one at this priority."
    extra = (rec["length"] / shortest["length"] - 1) * 100 if shortest["length"] else 0
    less_risk = (1 - rec["risk"] / shortest["risk"]) * 100 if shortest["risk"] else 0
    avoided = shortest["obstacles"] - rec["obstacles"]
    parts = [f"{extra:+.0f}% distance vs. shortest", f"{less_risk:.0f}% less risk"]
    if avoided > 0:
        parts.append(f"avoids {avoided} open obstacle{'s' if avoided > 1 else ''}")
    return ", ".join(parts) + "."
