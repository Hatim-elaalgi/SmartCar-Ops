"""Overview: mission control — what is happening and what needs attention right now."""
import statistics
import time

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

import config
from components import theme, ui
from components.figures import network_figure
from core import events, metrics, network, obstacles
from core.hub import hub
from core.severity import LEVEL_META

dash.register_page(__name__, path="/", name="Overview", order=0,
                   description="What is happening now, and what needs attention first.")

EVENT_ICON = {"info": ("bi-info-circle", theme.MUTED), "warning": ("bi-exclamation-triangle-fill", theme.WARNING),
              "critical": ("bi-exclamation-octagon-fill", theme.CRITICAL)}

layout = html.Div([
    html.Div(id="ov-kpis", className="stat-grid"),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=8, children=ui.panel(
            "Live map", dcc.Graph(id="ov-map", config={"displayModeBar": False}, className="map-tall"),
            actions=html.Div(id="ov-legend", className="legend"), flush=True)),
        dbc.Col(lg=4, children=[
            html.Div(id="ov-cam-wrap", children=ui.panel(
                "Camera", html.Img(id="ov-cam", className="video-thumb"),
                actions=dcc.Link("Open camera", href="/camera", className="link-sm"), flush=True)),
            ui.panel("Needs attention", html.Div(id="ov-attention"), subtitle="Open obstacles, most severe first",
                     actions=dcc.Link("All obstacles", href="/obstacles", className="link-sm")),
        ]),
    ]),
    ui.panel("Recent events", html.Div(id="ov-events"),
             actions=dcc.Link("Event log", href="/system", className="link-sm")),
])


def _kpis(mode, cars, open_counts):
    since = metrics.last_24h()
    obs_24h = metrics.count_obstacles(mode, since=since)
    obs_1h = metrics.count_obstacles(mode, since=time.time() - 3600)
    crit = ui.stat("Open critical", str(open_counts["critical"]), f"{open_counts['high']} high",
                   LEVEL_META["critical"]["icon"], theme.CRITICAL)
    if mode == config.PROTO:      # fixed track obstacles: count each time the car went around one
        obs = ui.stat("Avoided · 24 h", ui.compact(metrics.count_encounters(mode, since)),
                      f"{metrics.count_encounters(mode, time.time() - 3600)} in the last hour", "bi-shuffle")
    else:
        obs = ui.stat("Obstacles · 24 h", ui.compact(obs_24h), f"{obs_1h} in the last hour", "bi-cone-striped")
    if mode == config.REAL:
        online = [c for c in cars if c["online"]]
        moving = [c for c in online if c.get("speed", 0) > 0.5]
        on_trip = [c for c in online if c.get("trip_id")]
        progress = [c["progress"] for c in on_trip if c.get("progress") is not None]
        batteries = [c for c in online if c.get("battery") is not None]
        low = min(batteries, key=lambda c: c["battery"]) if batteries else None
        return [
            ui.stat("Vehicles online", f"{len(online)}/{len(cars)}", f"{len(moving)} moving", "bi-broadcast",
                    theme.GOOD if online else theme.MUTED),
            ui.stat("Active trips", str(len(on_trip)),
                    f"avg. {statistics.mean(progress) * 100:.0f}% done" if progress else "no trip running",
                    "bi-signpost-split-fill"),
            obs, crit,
            ui.stat("Average speed", ui.fmt_speed(mode, statistics.mean(c["speed"] for c in moving))
                    if moving else "—", "moving vehicles", "bi-speedometer2", live=True),
            ui.stat("Average battery", f"{statistics.mean(c['battery'] for c in batteries):.0f}%" if batteries
                    else "—", f"lowest {low['car_id']} {low['battery']:.0f}%" if low else None,
                    "bi-battery-half", live=True),
        ]
    car = cars[0] if cars else None
    online = bool(car and car["online"])
    state = car.get("state") if online else "OFFLINE"
    runs = metrics.trips_since(mode, since)
    return [
        ui.stat("Car state", ui.state_pill(state),
                f"last seen {ui.ago(car['ts'])}" if car else "never seen", "bi-car-front-fill"),
        ui.stat("Speed (est.)", ui.fmt_speed(mode, car.get("speed")) if online else "—",
                f"checkpoint {car.get('last_marker') or '—'}" if car else None, "bi-speedometer2", live=True),
        ui.stat("Run progress", f"{car['progress'] * 100:.0f}%" if online and car.get("progress") is not None
                else "—", (car.get("route") or "no active run") if car else None, "bi-signpost-split-fill",
                live=True),
        ui.stat("Runs · 24 h", str(runs["n"]), ui.fmt_dist(mode, runs["dist"]) + " driven", "bi-arrow-repeat"),
        obs, crit,
    ]


@callback(Output("ov-cam", "src"), Input("mode-switch", "value"))
def ov_camera(mode):
    """Prototype: ESP32-CAM · Real-world: dashcam of CAR-01 with the trained models."""
    source = "proto" if mode == config.PROTO else "real"
    return f"/camera/stream.mjpg?source={source}&overlay=1&t={int(time.time())}"


@callback(Output("ov-kpis", "children"), Output("ov-map", "figure"), Output("ov-legend", "children"),
          Input("tick", "n_intervals"), Input("mode-switch", "value"))
def ov_live(_, mode):
    mode = mode or config.REAL
    net = network.get_network(mode)
    cars = hub.cars(mode)
    shown = obstacles.listing(mode, statuses=obstacles.SHOWN)
    fig = network_figure(mode, net, cars=cars, obstacles=shown, uirevision=f"ov-{mode}")
    states = ("CRUISING", "AVOIDING") if mode == config.PROTO else ("CRUISING", "AVOIDING", "STOPPED")
    legend = [ui.legend_item(theme.STATE_META[s]["label"], theme.STATE_META[s]["color"]) for s in states]
    if mode == config.PROTO:
        legend += [ui.legend_item("Avoided obstacle", theme.INK_2, "square"),
                   ui.legend_item("Estimated detour", f"repeating-linear-gradient(90deg, {theme.DODGE_PATH} 0 3px, "
                                                      "transparent 3px 6px)", "line")]
    else:
        legend.append(ui.legend_item("Open obstacle", theme.SERIOUS, "diamond"))
    return _kpis(mode, cars, metrics.open_by_level(mode)), fig, legend


@callback(Output("ov-attention", "children"), Output("ov-events", "children"),
          Input("tick-slow", "n_intervals"), Input("mode-switch", "value"))
def ov_lists(_, mode):
    mode = mode or config.REAL
    df = obstacles.listing(mode, statuses=obstacles.OPEN, limit=6)
    if len(df):
        attention = [html.Div(className="list-row", children=[
            html.I(className=f"bi {LEVEL_META[r.level]['icon']} icon", style={"color": LEVEL_META[r.level]["color"]}),
            html.Div(className="grow", children=[
                html.Div(f"{r.cls.replace('_', ' ').capitalize()} · {LEVEL_META[r.level]['label']} ({r.severity:.0f})",
                         className="title"),
                html.Div(f"{r.car_id or 'unknown car'} · seen {r.seen}× · {ui.status_text(r.status)}",
                         className="meta"),
            ]),
            html.Span(ui.ago(r.ts_last), className="when"),
        ]) for r in df.itertuples()]
    elif mode == config.PROTO:
        n = len(network.proto_network().obstacles)
        attention = ui.empty(f"Nothing open: the car goes around the {n} obstacles of the track.", "bi-check2-circle")
    else:
        attention = ui.empty("Nothing open. All obstacles are reviewed.", "bi-check2-circle")

    rows = events.recent(mode, limit=8)
    feed = [html.Div(className="list-row", children=[
        html.I(className=f"bi {EVENT_ICON[e['level']][0]} icon", style={"color": EVENT_ICON[e["level"]][1]}),
        html.Div(e["message"], className="grow title", style={"fontWeight": 400}),
        html.Span(ui.ago(e["ts"]), className="when"),
    ]) for e in rows] or ui.empty("No events yet")
    return attention, feed
