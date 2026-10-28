"""Live stats: KPIs, speed, state timeline and obstacle analytics (real fleet or prototype)."""
import statistics
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import config
from components import theme, ui
from components.figures import empty_figure
from core import db, metrics, network
from core.hub import hub
from core.severity import LEVEL_META, LEVELS

dash.register_page(__name__, path="/stats", name="Live stats", order=5,
                   description="Real-time numbers for the vehicles, their behaviour and the obstacles they meet.")

GRAPH = {"config": {"displayModeBar": False}}
TIMELINE_STATES = ["CRUISING", "AVOIDING", "STOPPED", "LINE_LOST", "PARKED", "CHARGING"]

layout = html.Div([
    html.Div(id="st-kpis", className="stat-grid"),
    html.Div(id="st-note"),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=7, children=ui.panel(html.Span(id="st-speed-title"), dcc.Graph(id="st-speed", className="chart", **GRAPH))),
        dbc.Col(lg=5, children=ui.panel(html.Span(id="st-table-title"), html.Div(id="st-table"))),
    ]),
    ui.panel("Vehicle state · last 15 min", dcc.Graph(id="st-timeline", className="chart-sm", **GRAPH),
             subtitle="When each vehicle was cruising, avoiding, stopped…"),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=6, children=ui.panel("Obstacles per hour · last 24 h", dcc.Graph(id="st-hourly", className="chart", **GRAPH))),
        dbc.Col(lg=6, children=ui.panel("Obstacles by object · last 7 days", dcc.Graph(id="st-classes", className="chart", **GRAPH))),
    ]),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=6, children=ui.panel(html.Span(id="st-trips-title"), dcc.Graph(id="st-trips", className="chart", **GRAPH))),
        dbc.Col(lg=6, children=ui.panel("Severity mix · last 7 days", dcc.Graph(id="st-severity", className="chart", **GRAPH))),
    ]),
])


def _local(epoch_seconds):
    """Epoch seconds -> naive local datetimes (same clock as the live charts)."""
    return pd.to_datetime(epoch_seconds.map(datetime.fromtimestamp)) if len(epoch_seconds) else \
        pd.Series([], dtype="datetime64[ns]")


def _segments(history):
    """Compress a telemetry history into (state, start, end) segments."""
    segs = []
    for t in history:
        state = t.get("state", "OFFLINE")
        if segs and segs[-1][0] == state and t["ts"] - segs[-1][2] < 6:
            segs[-1][2] = t["ts"]
        else:
            if segs:
                segs[-1][2] = t["ts"]
            segs.append([state, t["ts"], t["ts"]])
    return segs


def _timeline(mode, cars):
    fig = go.Figure()
    shown = set()
    for state in TIMELINE_STATES:
        m = theme.STATE_META[state]
        ys, bases, widths, hover = [], [], [], []
        for c in cars:
            for s, t0, t1 in _segments(hub.history(mode, c["car_id"], 900)):
                if s != state or t1 <= t0:
                    continue
                ys.append(c["car_id"])
                bases.append(datetime.fromtimestamp(t0))
                widths.append((t1 - t0) * 1000)
                hover.append(f"{c['car_id']} · {m['label']} · {ui.fmt_dur(t1 - t0)}")
        if ys:
            shown.add(state)
            fig.add_trace(go.Bar(orientation="h", y=ys, x=widths, base=bases, name=m["label"],
                                 marker=dict(color=m["color"], line=dict(color=theme.SURFACE, width=1)),
                                 hovertext=hover, hovertemplate="%{hovertext}<extra></extra>"))
    if not shown:
        return empty_figure("Waiting for live data…")
    fig.update_layout(barmode="overlay", bargap=0.35, margin=dict(l=70, r=12, t=30, b=24),
                      xaxis=dict(type="date", tickformat="%H:%M:%S"), yaxis=dict(categoryorder="category descending"))
    return fig


@callback(Output("st-kpis", "children"), Output("st-speed", "figure"), Output("st-speed-title", "children"),
          Output("st-table", "children"), Output("st-table-title", "children"), Output("st-timeline", "figure"),
          Output("st-note", "children"),
          Input("tick", "n_intervals"), Input("mode-switch", "value"))
def st_live(_, mode):
    mode = mode or config.REAL
    cars = hub.cars(mode)
    online = [c for c in cars if c["online"]]
    since = metrics.last_24h()
    trips = metrics.trips_since(mode, since)
    obs_24h = metrics.count_obstacles(mode, since=since)
    factor = 3.6 if mode == config.REAL else 1.0

    # speed chart (one line per vehicle, fixed categorical order)
    speed = go.Figure()
    for i, c in enumerate(cars):
        hist = hub.history(mode, c["car_id"], 300)
        if hist:
            speed.add_trace(go.Scatter(x=[datetime.fromtimestamp(t["ts"]) for t in hist],
                                       y=[(t.get("speed") or 0) * factor for t in hist], mode="lines",
                                       name=c["car_id"], line=dict(color=theme.SERIES[i % 8], width=2)))
    speed.update_layout(hovermode="x unified", showlegend=len(cars) > 1, margin=dict(l=44, r=12, t=30, b=28),
                        xaxis=dict(tickformat="%H:%M:%S"),
                        yaxis=dict(title=dict(text="km/h" if mode == config.REAL else "cm/s"), rangemode="tozero"))
    if not speed.data:
        speed = empty_figure("Waiting for live data…")

    if mode == config.REAL:
        moving = [c for c in online if c.get("speed", 0) > 0.5]
        batteries = [c["battery"] for c in online if c.get("battery") is not None]
        kpis = [
            ui.stat("Vehicles online", f"{len(online)}/{len(cars)}", f"{len(moving)} moving", "bi-broadcast"),
            ui.stat("Distance · 24 h", ui.fmt_dist(mode, trips["dist"]), f"{trips['n']} trips", "bi-signpost-split-fill"),
            ui.stat("Average speed", ui.fmt_speed(mode, statistics.mean(c["speed"] for c in moving)) if moving else "—",
                    "moving vehicles", "bi-speedometer2", live=True),
            ui.stat("Obstacles per hour", f"{obs_24h / 24:.1f}", "24 h average", "bi-cone-striped"),
            ui.stat("Avoidance manoeuvres", ui.compact(trips["avoid"]), "last 24 h", "bi-shuffle"),
            ui.stat("Average battery", f"{statistics.mean(batteries):.0f}%" if batteries else "—", None,
                    "bi-battery-half", live=True),
        ]
        rows = [html.Tr([html.Td(html.Span(c["car_id"], className="mono")),
                         html.Td(ui.state_pill(c["state"] if c["online"] else "OFFLINE")),
                         html.Td(ui.fmt_speed(mode, c.get("speed"))),
                         html.Td(f"{c['battery']:.0f}%" if c.get("battery") is not None else "—"),
                         html.Td(f"{c['progress'] * 100:.0f}%" if c.get("progress") is not None else "—"),
                         html.Td(ui.ago(c["ts"]))]) for c in cars]
        table = html.Table(className="tbl", children=[
            html.Thead(html.Tr([html.Th(h) for h in ("Car", "State", "Speed", "Battery", "Trip", "Last seen")])),
            html.Tbody(rows)]) if rows else ui.empty("Waiting for data…")
        return (kpis, speed, "Speed · last 5 min", table, "Fleet", _timeline(mode, cars), None)

    car = cars[0] if cars else None
    is_on = bool(car and car["online"])
    source = hub.source(mode)
    kpis = [
        ui.stat("Car state", ui.state_pill(car["state"] if is_on else "OFFLINE"),
                f"last seen {ui.ago(car['ts'])}" if car else "never seen", "bi-car-front-fill"),
        ui.stat("Data source", {"camera": "Live camera", "demo": "Demo", "offline": "Offline"}.get(source, source),
                {"camera": "ESP32-CAM stream", "demo": "camera offline — simulated"}.get(source),
                theme.SOURCE_META.get(source, theme.SOURCE_META["offline"])["icon"]),
        ui.stat("Speed (est.)", ui.fmt_speed(mode, car.get("speed")) if is_on else "—", "between checkpoints",
                "bi-speedometer2", live=True),
        ui.stat("Camera", f"{car.get('fps') or 0:.0f} fps" if is_on else "—", "frames per second", "bi-camera-video-fill",
                live=True),
        ui.stat("Runs · 24 h", str(trips["n"]), ui.fmt_dist(mode, trips["dist"]) + " driven", "bi-arrow-repeat"),
        ui.stat("Track obstacles", str(len(network.proto_network().obstacles)),
                f"gone around {trips['avoid']}× in 24 h", "bi-cone-striped"),
    ]
    runs = db.query("SELECT route_name, start_ts, end_ts, distance, n_obstacles, status FROM trips "
                    "WHERE mode=? ORDER BY start_ts DESC LIMIT 8", (mode,))
    table = html.Table(className="tbl", children=[
        html.Thead(html.Tr([html.Th(h) for h in ("Started", "Trajectory", "Time", "Distance", "Obstacles", "Status")])),
        html.Tbody([html.Tr([html.Td(ui.ago(r["start_ts"])), html.Td(r["route_name"] or "—"),
                             html.Td("—" if r["status"] == "interrupted" else
                                     ui.fmt_dur(r["end_ts"] - r["start_ts"]) if r["end_ts"] else "running"),
                             html.Td(ui.fmt_dist(mode, r["distance"])), html.Td(r["n_obstacles"]),
                             html.Td(r["status"])]) for r in runs]),
    ]) if runs else ui.empty("No runs yet")
    return (kpis, speed, "Estimated speed · last 5 min", table, "Recent runs", _timeline(mode, cars), None)


@callback(Output("st-hourly", "figure"), Output("st-classes", "figure"), Output("st-trips", "figure"),
          Output("st-trips-title", "children"), Output("st-severity", "figure"),
          Input("tick-slow", "n_intervals"), Input("mode-switch", "value"))
def st_history(_, mode):
    mode = mode or config.REAL
    now = time.time()

    if mode == config.PROTO:      # the track's obstacles are fixed: count each time the car goes around one
        obs = db.query_df("SELECT e.ts AS ts_first, e.cls, o.level FROM encounters e JOIN obstacles o "
                          "ON o.id = e.obstacle_id WHERE e.mode=? AND e.ts >= ?", (mode, now - 7 * 86400))
    else:
        obs = db.query_df("SELECT ts_first, cls, level FROM obstacles WHERE mode=? AND ts_first >= ?",
                          (mode, now - 7 * 86400))
    obs["t"] = _local(obs["ts_first"])

    # obstacles per hour, last 24 h (single series)
    last = obs[obs["ts_first"] >= now - 86400] if len(obs) else obs
    if len(last):
        hours = pd.date_range(end=pd.Timestamp(datetime.fromtimestamp(now)).floor("h"), periods=24, freq="h")
        counts = last.set_index("t").resample("h").size().reindex(hours, fill_value=0)
        hourly = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=theme.ACCENT,
                                  hovertemplate="%{x|%H:00}: %{y} obstacles<extra></extra>"))
        hourly.update_layout(margin=dict(l=36, r=8, t=12, b=28), xaxis=dict(tickformat="%H:%M"))
    else:
        hourly = empty_figure("No obstacles in the last 24 h")

    # by object (single series, sorted)
    if len(obs):
        by_cls = obs["cls"].value_counts().sort_values()
        classes = go.Figure(go.Bar(orientation="h", y=[c.replace("_", " ") for c in by_cls.index], x=by_cls.values,
                                   marker_color=theme.ACCENT, text=by_cls.values, textposition="outside",
                                   textfont=dict(color=theme.INK_2), cliponaxis=False,
                                   hovertemplate="%{y}: %{x}<extra></extra>"))
        classes.update_layout(margin=dict(l=90, r=32, t=12, b=28), bargap=0.4)
    else:
        classes = empty_figure("No obstacles in the last 7 days")

    # trips per day
    days = 14 if mode == config.REAL else 7
    trips = db.query_df("SELECT start_ts, distance FROM trips WHERE mode=? AND start_ts >= ?", (mode, now - days * 86400))
    if len(trips):
        trips["day"] = _local(trips["start_ts"]).dt.floor("D")
        per_day = trips.groupby("day").size()
        trips_fig = go.Figure(go.Bar(x=per_day.index, y=per_day.values, marker_color=theme.ACCENT,
                                     hovertemplate="%{x|%d %b}: %{y}<extra></extra>"))
        trips_fig.update_layout(margin=dict(l=36, r=8, t=12, b=28), xaxis=dict(tickformat="%d %b"))
    else:
        trips_fig = empty_figure("No trips yet")
    trips_title = f"{'Trips' if mode == config.REAL else 'Runs'} per day · last {days} days"

    # severity mix per day (stacked, status colours, legend)
    if len(obs):
        obs["day"] = obs["t"].dt.floor("D")
        sev = go.Figure()
        for name, _ in reversed(LEVELS):
            per = obs[obs["level"] == name].groupby("day").size()
            sev.add_trace(go.Bar(x=per.index, y=per.values, name=LEVEL_META[name]["label"],
                                 marker=dict(color=LEVEL_META[name]["color"], line=dict(color=theme.SURFACE, width=2)),
                                 hovertemplate="%{x|%d %b}: %{y} " + LEVEL_META[name]["label"].lower() + "<extra></extra>"))
        sev.update_layout(barmode="stack", margin=dict(l=36, r=8, t=30, b=28), bargap=0.4,
                          xaxis=dict(tickformat="%d %b"))
    else:
        sev = empty_figure("No obstacles in the last 7 days")
    return hourly, classes, trips_fig, trips_title, sev
