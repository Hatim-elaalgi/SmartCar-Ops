"""Map (real mode, MapLibre tiles) and track (prototype mode, plain x/y in cm) figures with one API."""
import plotly.graph_objects as go

import config
from components import theme
from components.ui import fmt_speed
from core.severity import LEVEL_META

TRACK_TAPE = "#34342f"


def _is_real(mode):
    return mode == config.REAL


def _line(data_mode, xs, ys, **kw):
    """data_mode = real | proto (not to be confused with Plotly's trace `mode`)."""
    return go.Scattermap(lon=xs, lat=ys, mode="lines", **kw) if _is_real(data_mode) else \
        go.Scatter(x=xs, y=ys, mode="lines", **kw)


def _points(data_mode, xs, ys, **kw):
    return go.Scattermap(lon=xs, lat=ys, **kw) if _is_real(data_mode) else go.Scatter(x=xs, y=ys, **kw)


def _segments(net, edge_ids=None):
    """All edges as one polyline with None breaks (fast to draw)."""
    xs, ys = [], []
    for eid in (edge_ids if edge_ids is not None else net.edges):
        for x, y in net.edges[eid]["geometry"]:
            xs.append(x)
            ys.append(y)
        xs.append(None)
        ys.append(None)
    return xs, ys


def empty_figure(message="No data yet", height=None):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=theme.MUTED, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height)
    return fig


def network_figure(mode, net, *, cars=(), trails=None, selected=None, obstacles=None, routes=(), start=None,
                   end=None, clickable_nodes=False, show_edges=None, risk_edges=None, uirevision="net",
                   height=None, showlegend=False, car_positions_only=False):
    """cars: hub telemetry dicts; trails: {car_id: [(x, y), ...]}; obstacles: DataFrame with x, y, level...;
    routes: [{"name", "points", "color", "width"}]; risk_edges: {level: [edge ids]}."""
    real = _is_real(mode)
    fig = go.Figure()

    # Track / road graph
    if not real:
        xs, ys = _segments(net)
        fig.add_trace(_line(mode, xs, ys, line=dict(color=TRACK_TAPE, width=14), hoverinfo="skip",
                            showlegend=False))
        fig.add_trace(_line(mode, xs, ys, line=dict(color=theme.BASELINE, width=1, dash="dot"), hoverinfo="skip",
                            showlegend=False))
    elif show_edges:
        xs, ys = _segments(net)
        fig.add_trace(_line(mode, xs, ys, line=dict(color="#4a4a46", width=1.5), hoverinfo="skip",
                            showlegend=False))

    for level, label in (("high", "Risky segment"), ("critical", "Very risky segment")):
        ids = (risk_edges or {}).get(level)
        if ids:
            xs, ys = _segments(net, ids)
            fig.add_trace(_line(mode, xs, ys, name=label, line=dict(color=LEVEL_META[level]["color"], width=5),
                                opacity=0.55, hoverinfo="skip", showlegend=showlegend))

    # Routes (planner)
    for r in routes:
        xs = [p[0] for p in r["points"]]
        ys = [p[1] for p in r["points"]]
        fig.add_trace(_line(mode, xs, ys, name=r["name"], line=dict(color=r["color"], width=r.get("width", 4)),
                            opacity=r.get("opacity", 1.0), hoverinfo="name", showlegend=showlegend))

    # Nodes / checkpoints
    if not real or clickable_nodes:
        ids = list(net.nodes)
        nx_ = [net.nodes[i]["x"] for i in ids]
        ny_ = [net.nodes[i]["y"] for i in ids]
        if real:
            fig.add_trace(_points(mode, nx_, ny_, mode="markers", name="Intersections",
                                  marker=dict(size=7, color=theme.MUTED, opacity=0.55), customdata=ids,
                                  text=[net.nodes[i]["label"] for i in ids],
                                  hovertemplate="%{text}<extra>Click to select</extra>", showlegend=False))
        else:
            labels = [i for i in ids]
            hover = [f"{net.nodes[i]['label']}<br>Marker #{net.nodes[i]['marker']}" if net.nodes[i]["marker"]
                     is not None else net.nodes[i]["label"] for i in ids]
            fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text", text=labels, customdata=ids,
                                     textfont=dict(color=theme.INK_1, size=11), textposition="middle center",
                                     marker=dict(size=24, color=theme.SURFACE_2,
                                                 line=dict(color=theme.MUTED, width=1.5)),
                                     hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                                     showlegend=False, name="Checkpoints"))

    # Start / end
    for nid, label, color in ((start, "Start", theme.INK_1), (end, "Destination", theme.SERIES[3])):
        if nid and nid in net.nodes:
            x, y = net.xy(nid)
            fig.add_trace(_points(mode, [x], [y], mode="markers", name=label,
                                  marker=dict(size=18 if real else 30, color=color, opacity=0.9),
                                  hovertemplate=f"{label}: {net.nodes[nid]['label']}<extra></extra>",
                                  showlegend=showlegend))

    # Obstacles
    if obstacles is not None and len(obstacles):
        df = obstacles.dropna(subset=["x", "y"])
        known = df[df["status"] == "avoided"]           # prototype track obstacles: the car always goes around them
        if len(known):
            hover = [f"<b>{r.cls.replace('_', ' ').capitalize()}</b> · avoided by the car<br>"
                     f"{LEVEL_META[r.level]['label']} danger ({r.severity:.0f}) · gone around {r.seen}×"
                     for r in known.itertuples()]
            fig.add_trace(_points(mode, list(known["x"]), list(known["y"]), mode="markers+text", name="Avoided obstacle",
                                  text=[c.replace("_", " ") for c in known["cls"]],
                                  textposition=["middle left" if x < sum(n["x"] for n in net.nodes.values())
                                                / len(net.nodes) else "middle right" for x in known["x"]],
                                  textfont=dict(color=theme.INK_1, size=12),
                                  marker=dict(size=16, color=theme.INK_2, symbol="square",
                                              line=dict(color=theme.SURFACE, width=2)),
                                  customdata=list(known["id"]), hovertext=hover,
                                  hovertemplate="%{hovertext}<extra></extra>", showlegend=showlegend))
        df = df[df["status"] != "avoided"]
        for level in ("low", "medium", "high", "critical"):
            sub = df[df["level"] == level]
            if not len(sub):
                continue
            meta = LEVEL_META[level]
            hover = [f"<b>{r.cls.replace('_', ' ')}</b> · {meta['label']} ({r.severity:.0f})<br>"
                     f"Seen {r.seen}× · {r.status.replace('_', ' ')}" for r in sub.itertuples()]
            fig.add_trace(_points(mode, list(sub["x"]), list(sub["y"]), mode="markers", name=meta["label"],
                                  marker=dict(size=11 if real else 13, color=meta["color"],
                                              **({} if real else {"symbol": "diamond",
                                                                  "line": dict(color=theme.SURFACE, width=2)})),
                                  customdata=list(sub["id"]), hovertext=hover,
                                  hovertemplate="%{hovertext}<extra></extra>", showlegend=showlegend))

    # Trails
    for car_id, pts in (trails or {}).items():
        if len(pts) < 2:
            continue
        is_sel = car_id == selected
        fig.add_trace(_line(mode, [p[0] for p in pts], [p[1] for p in pts], hoverinfo="skip", showlegend=False,
                            line=dict(color=theme.ACCENT if is_sel else theme.INK_2, width=3 if is_sel else 2),
                            opacity=0.9 if is_sel else 0.35))

    # Estimated detour path while a car goes around an obstacle (prototype)
    for c in cars:
        path = c.get("avoid_path")
        if path and not real:
            fig.add_trace(_line(mode, [p[0] for p in path], [p[1] for p in path], hoverinfo="skip", showlegend=False,
                                name="Estimated detour",
                                line=dict(color=theme.DODGE_PATH, width=3, dash="dot", shape="spline", smoothing=1.1)))

    # Cars
    cars = [c for c in cars if c.get("x") is not None]
    if cars:
        if selected:
            sel = [c for c in cars if c["car_id"] == selected]
            if sel:
                fig.add_trace(_points(mode, [sel[0]["x"]], [sel[0]["y"]], mode="markers", hoverinfo="skip",
                                      showlegend=False, marker=dict(size=34 if real else 40, color=theme.ACCENT,
                                                                    opacity=0.28)))
        hover = []
        for c in cars:
            m = theme.state_meta(c.get("state") if c.get("online", True) else "OFFLINE")
            extra = f"<br>Battery {c['battery']:.0f}%" if c.get("battery") is not None else ""
            hover.append(f"<b>{c['car_id']}</b><br>{m['label']} · {fmt_speed(mode, c.get('speed'))}{extra}")
        colors = [theme.state_meta(c.get("state") if c.get("online", True) else "OFFLINE")["color"] for c in cars]
        marker = dict(size=16, color=colors) if real else dict(
            size=22, color=colors, symbol="triangle-up", angle=[c.get("heading", 0) for c in cars],
            line=dict(color=theme.PLANE, width=2))
        fig.add_trace(_points(mode, [c["x"] for c in cars], [c["y"] for c in cars], mode="markers",
                              marker=marker, customdata=[c["car_id"] for c in cars], hovertext=hover,
                              hovertemplate="%{hovertext}<extra></extra>", showlegend=False, name="Vehicles"))

    # Layout
    if real:
        cx, cy = net.center()
        fig.update_layout(map=dict(style="carto-darkmatter", center=dict(lon=cx, lat=cy), zoom=net.zoom()),
                          margin=dict(l=0, r=0, t=0, b=0))
    else:
        x0, x1, y0, y1 = net.bounds()
        pad = 30
        fig.update_xaxes(range=[x0 - pad, x1 + pad], showgrid=True, gridcolor=theme.GRID, dtick=50,
                         showticklabels=False, showline=False, zeroline=False, constrain="domain")
        fig.update_yaxes(range=[y0 - pad, y1 + pad], showgrid=True, gridcolor=theme.GRID, dtick=50,
                         showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1)
        fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    fig.update_layout(uirevision=uirevision, height=height, showlegend=showlegend,
                      legend=dict(y=0.99, yanchor="top", x=0.01, bgcolor="rgba(13,13,13,0.7)"))
    return fig
