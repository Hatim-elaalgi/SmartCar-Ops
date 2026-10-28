"""Tracking: live position (roster | map | selected vehicle) and trip history with replay."""
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

import config
from components import theme, ui
from components.figures import empty_figure, network_figure
from core import db, geo, network, trips
from core.hub import hub

dash.register_page(__name__, path="/tracking", name="Tracking", order=1,
                   description="Where every vehicle is now, and where it has been.")

# Exceptions first in the roster.
STATE_RANK = {"STOPPED": 0, "LINE_LOST": 1, "AVOIDING": 2, "OFFLINE": 3, "CRUISING": 4, "CHARGING": 5, "PARKED": 6}
# A longer step between two telemetry samples is a reset (demo car back at the start), not a drive.
TRAIL_JUMP = {config.REAL: 150.0, config.PROTO: 40.0}     # metres / centimetres

TRIP_COLUMNS = [
    {"name": "Start", "id": "start"}, {"name": "Car", "id": "car_id"}, {"name": "Route", "id": "route_name"},
    {"name": "Duration", "id": "duration"}, {"name": "Distance", "id": "distance_txt"},
    {"name": "Obstacles", "id": "n_obstacles", "type": "numeric"},
]

layout = html.Div([
    dcc.Store(id="tr-selected"),
    dcc.Store(id="tr-replay-data"),
    dcc.Interval(id="tr-replay-int", interval=250, disabled=True),
    dcc.Download(id="tr-download"),
    dbc.Row(className="g-3", children=[
        dbc.Col(xl=3, lg=4, children=ui.panel("Vehicles", html.Div(id="tr-roster"), subtitle="Problems first")),
        dbc.Col(xl=6, lg=8, children=ui.panel(
            "Live position", dcc.Graph(id="tr-map", config={"displayModeBar": False}, className="map-tall"),
            actions=dbc.Switch(id="tr-trails", label="Trails (2 min)", value=True), flush=True)),
        dbc.Col(xl=3, lg=12, children=ui.panel("Selected vehicle", html.Div(id="tr-detail"))),
    ]),
    ui.panel("Trip history", subtitle="Select a trip to replay it", children=dbc.Row(className="g-3", children=[
        dbc.Col(lg=6, children=dash_table.DataTable(
            id="tr-trips", columns=TRIP_COLUMNS, data=[], row_selectable="single",
            style_cell_conditional=[{"if": {"column_id": "route_name"}, "maxWidth": "200px", "overflow": "hidden",
                                     "textOverflow": "ellipsis"}],
            **ui.table_style(page_size=10))),
        dbc.Col(lg=6, children=[
            dcc.Graph(id="tr-replay-map", config={"displayModeBar": False}, className="map-md"),
            html.Div(className="d-flex align-items-center gap-3 mt-2", children=[
                dbc.Button(html.I(className="bi bi-play-fill"), id="tr-play", color="primary", size="sm",
                           title="Play / pause"),
                html.Div(dcc.Slider(id="tr-slider", min=0, max=1, step=1, value=0, marks=None,
                                    updatemode="drag", allow_direct_input=False), style={"flex": 1}),
                dbc.Select(id="tr-speed", value="4", size="sm", style={"width": "86px"},
                           options=[{"label": f"{s}×", "value": str(s)} for s in (1, 4, 10, 25)]),
                dbc.Button([html.I(className="bi bi-download me-1"), "CSV"], id="tr-export", size="sm",
                           color="secondary", outline=True),
            ]),
            html.Div(id="tr-replay-info", className="hint mt-1"),
            dcc.Graph(id="tr-replay-speed", config={"displayModeBar": False}, className="chart-sm"),
        ]),
    ])),
])


# --- live --------------------------------------------------------------------------------------
@callback(Output("tr-selected", "data"),
          Input({"type": "tr-car", "id": ALL}, "n_clicks"), Input("tr-map", "clickData"),
          prevent_initial_call=True)
def tr_select(_clicks, click):
    trigger = ctx.triggered_id
    if isinstance(trigger, dict):
        if not ctx.triggered[0]["value"]:
            raise PreventUpdate
        return trigger["id"]
    if trigger == "tr-map" and click:
        car = click["points"][0].get("customdata")
        if isinstance(car, str):
            return car
    raise PreventUpdate


def _roster_item(mode, c, selected):
    state = c["state"] if c["online"] else "OFFLINE"
    m = theme.state_meta(state)
    return html.Button(
        id={"type": "tr-car", "id": c["car_id"]}, n_clicks=0,
        className="roster-item" + (" selected" if c["car_id"] == selected else "") + ("" if c["online"] else " stale"),
        children=[
            html.I(className=f"bi {m['icon']}", style={"color": m["color"], "fontSize": "18px"}),
            html.Div(className="grow", children=[
                html.Div(c["car_id"], className="name"),
                html.Div(f"{m['label']} · {ui.ago(c['ts'])}", className="meta"),
            ]),
            html.Div(ui.fmt_speed(mode, c.get("speed")) if c["online"] else "—", className="speed"),
        ])


def _detail(mode, c):
    if not c:
        return ui.empty("No vehicle is reporting yet", "bi-broadcast")
    state = c["state"] if c["online"] else "OFFLINE"
    progress = c.get("progress")
    rows = [ui.kv("Speed", ui.fmt_speed(mode, c.get("speed"))),
            ui.kv("Heading", f"{c.get('heading', 0):.0f}°")]
    if mode == config.REAL:
        rows += [ui.kv("Battery", f"{c['battery']:.0f}%" if c.get("battery") is not None else "—"),
                 ui.kv("Destination", c.get("dest") or "—"),
                 ui.kv("Front obstacle", ui.fmt_dist(mode, c.get("front")) if c.get("front") else "none")]
    else:
        rows += [ui.kv("Route", c.get("route") or "—"), ui.kv("Last checkpoint", c.get("last_marker") or "—"),
                 ui.kv("Camera", f"{c.get('fps') or 0:.0f} fps")]
    rows += [ui.kv("Trip distance", ui.fmt_dist(mode, c.get("trip_dist")) if c.get("trip_id") else "—"),
             ui.kv("Trip", c.get("trip_id") or "—", mono=True),
             ui.kv("Last seen", ui.ago(c["ts"])), ui.kv("Source", c.get("source", "—"))]
    warn = None
    if mode == config.PROTO and c.get("source") == "camera" and not c.get("position_known"):
        warn = ui.info_note("Position unknown until the camera sees a checkpoint marker.", "bi-crosshair")
    return html.Div([
        html.Div(className="car-head", children=[html.Span(c["car_id"], className="name"), ui.state_pill(state)]),
        ui.ring(progress, f"{progress * 100:.0f}%" if progress is not None else "—",
                "trip progress" if progress is not None else "no active trip"),
        warn, *rows,
    ])


@callback(Output("tr-roster", "children"), Output("tr-map", "figure"), Output("tr-detail", "children"),
          Input("tick", "n_intervals"), Input("mode-switch", "value"), Input("tr-selected", "data"),
          Input("tr-trails", "value"))
def tr_live(_, mode, selected, show_trails):
    mode = mode or config.REAL
    cars = hub.cars(mode)
    ids = [c["car_id"] for c in cars]
    if selected not in ids:
        selected = ids[0] if ids else None
    ordered = sorted(cars, key=lambda c: (STATE_RANK.get(c["state"] if c["online"] else "OFFLINE", 9), c["car_id"]))
    roster = [_roster_item(mode, c, selected) for c in ordered] or ui.empty("Waiting for data…", "bi-hourglass")
    trails = {}
    if show_trails:
        for c in cars:
            pts, last_trip = [], None
            for t in hub.history(mode, c["car_id"], 120):
                if t.get("x") is None:
                    continue
                jump = pts and pts[-1][0] is not None and \
                    geo.distance(mode, *pts[-1], t["x"], t["y"]) > TRAIL_JUMP[mode]
                if pts and (t.get("trip_id") != last_trip or jump):
                    pts.append((None, None))  # new trip or a reset: don't draw a line from the previous point
                pts.append((t["x"], t["y"]))
                last_trip = t.get("trip_id")
            trails[c["car_id"]] = pts
    fig = network_figure(mode, network.get_network(mode), cars=cars, trails=trails, selected=selected,
                         uirevision=f"tr-{mode}")
    current = next((c for c in cars if c["car_id"] == selected), None)
    return roster, fig, _detail(mode, current)


# --- history -----------------------------------------------------------------------------------
def _duration(r):
    if r.status == "interrupted":
        return "interrupted"
    return ui.fmt_dur(r.end_ts - r.start_ts) if pd.notna(r.end_ts) else "running"


# Selection is tracked by row index (a selection made by a callback doesn't update `selected_row_ids`),
# and re-mapped to the same trip each time the list refreshes.
@callback(Output("tr-trips", "data"), Output("tr-trips", "selected_rows"),
          Input("tick-slow", "n_intervals"), Input("mode-switch", "value"),
          State("tr-trips", "selected_rows"), State("tr-trips", "data"))
def tr_trips(_, mode, selected_rows, previous):
    mode = mode or config.REAL
    df = trips.recent(mode, limit=300)
    data = [{"id": r.id, "start": ui.fmt_time(r.start_ts), "car_id": r.car_id, "route_name": r.route_name or "—",
             "duration": _duration(r), "distance_txt": ui.fmt_dist(mode, r.distance), "n_obstacles": r.n_obstacles,
             "status": r.status} for r in df.itertuples()]
    prev_id = None
    if selected_rows and previous and selected_rows[0] < len(previous):
        prev_id = previous[selected_rows[0]]["id"]
    index = {d["id"]: i for i, d in enumerate(data)}
    if prev_id in index:
        rows = [index[prev_id]]
    else:  # start with the latest finished trip, ready to replay
        rows = [i for i, d in enumerate(data) if d["status"] == "done"][:1]
    return data, rows


@callback(Output("tr-replay-data", "data"), Output("tr-slider", "max"), Output("tr-slider", "value"),
          Input("tr-trips", "selected_rows"), State("tr-trips", "data"), State("tr-replay-data", "data"),
          State("mode-switch", "value"))
def tr_load(selected_rows, table, current, mode):
    if not selected_rows or not table or selected_rows[0] >= len(table):
        return None, 1, 0
    trip_id = table[selected_rows[0]]["id"]
    if current and current.get("trip_id") == trip_id:
        raise PreventUpdate
    pts = trips.points(trip_id)
    if not len(pts):
        return {"trip_id": trip_id, "mode": mode, "points": []}, 1, 0
    if mode == config.PROTO:      # the track's obstacles stand there on every run
        obs = db.query_df("SELECT * FROM obstacles WHERE mode = ? AND status = 'avoided'", (mode,))
    else:
        obs = db.query_df("SELECT * FROM obstacles WHERE trip_id = ?", (trip_id,))
    route = (trips.get(trip_id) or {}).get("route_name")
    return ({"trip_id": trip_id, "mode": mode, "route": route, "points": pts.values.tolist(),
             "obstacles": obs.to_dict("records")}, len(pts) - 1, 0)


@callback(Output("tr-replay-int", "disabled"), Output("tr-slider", "value", allow_duplicate=True),
          Output("tr-play", "children"),
          Input("tr-play", "n_clicks"), Input("tr-replay-int", "n_intervals"),
          State("tr-replay-int", "disabled"), State("tr-slider", "value"), State("tr-slider", "max"),
          State("tr-speed", "value"), prevent_initial_call=True)
def tr_playback(_clicks, _n, disabled, value, vmax, speed):
    play_icon, pause_icon = html.I(className="bi bi-play-fill"), html.I(className="bi bi-pause-fill")
    if ctx.triggered_id == "tr-play":
        if disabled:  # start (from the beginning if we are at the end)
            return False, 0 if value >= vmax else no_update, pause_icon
        return True, no_update, play_icon
    nxt = min(vmax, (value or 0) + max(1, int(speed or 1) // 2))
    if nxt >= vmax:
        return True, vmax, play_icon
    return no_update, nxt, no_update


@callback(Output("tr-replay-map", "figure"), Output("tr-replay-speed", "figure"), Output("tr-replay-info", "children"),
          Input("tr-slider", "value"), Input("tr-replay-data", "data"))
def tr_render(i, data):
    if not data or not data.get("points"):
        msg = "Select a trip in the table to replay it." if not data else "This trip has no recorded points."
        return empty_figure(msg), empty_figure(""), ""
    mode = data["mode"]
    pts = data["points"]  # ts, x, y, speed, state
    i = max(0, min(int(i or 0), len(pts) - 1))
    ts, x, y, speed, state = pts[i]
    obs = pd.DataFrame(data.get("obstacles") or [])
    path = [(p[1], p[2]) for p in pts]
    routes = [{"name": "Trip", "points": path, "color": theme.INK_2, "width": 3, "opacity": 0.4},
              {"name": "Driven", "points": path[:i + 1], "color": theme.ACCENT, "width": 4}]
    heading = 0.0
    if i > 0:
        heading = geo.bearing(mode, pts[i - 1][1], pts[i - 1][2], x, y)
    car = {"car_id": "replay", "x": x, "y": y, "speed": speed, "state": state, "heading": heading, "online": True}
    fig = network_figure(mode, network.get_network(mode), cars=[car], routes=routes,
                         obstacles=obs if len(obs) else None, uirevision=f"replay-{data['trip_id']}")

    times = [datetime.fromtimestamp(p[0]) for p in pts]
    factor = 3.6 if mode == config.REAL else 1.0
    sp = go.Figure(go.Scatter(x=times, y=[(p[3] or 0) * factor for p in pts], mode="lines",
                              line=dict(color=theme.ACCENT, width=2), fill="tozeroy",
                              fillcolor="rgba(57,135,229,0.10)", hovertemplate="%{y:.0f}<extra></extra>"))
    sp.add_vline(x=times[i], line=dict(color=theme.INK_2, width=1))
    sp.update_layout(margin=dict(l=40, r=8, t=8, b=24), hovermode="x unified", xaxis=dict(tickformat="%H:%M:%S"),
                     yaxis=dict(title=dict(text="km/h" if mode == config.REAL else "cm/s")))
    m = theme.state_meta(state)
    info = (f"{data.get('route') or data['trip_id']} · {ui.fmt_time(ts, with_date=False)} · {m['label']} · "
            f"{ui.fmt_speed(mode, speed)} · point {i + 1}/{len(pts)}")
    return fig, sp, info


@callback(Output("tr-download", "data"), Input("tr-export", "n_clicks"), State("tr-replay-data", "data"),
          prevent_initial_call=True)
def tr_export(_, data):
    if not data or not data.get("points"):
        raise PreventUpdate
    df = pd.DataFrame(data["points"], columns=["ts", "x", "y", "speed", "state"])
    df.insert(0, "time", pd.to_datetime(df["ts"], unit="s"))
    if data["mode"] == config.REAL:
        df = df.rename(columns={"x": "lon", "y": "lat", "speed": "speed_m_s"})
    else:
        df = df.rename(columns={"x": "x_cm", "y": "y_cm", "speed": "speed_cm_s"})
    return dcc.send_data_frame(df.to_csv, f"trip_{data['trip_id']}.csv", index=False)
