"""Devices & logs: health of every part of the system, the event log, and settings."""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html

import config
from components import theme, ui
from core import db, events, network, obstacles, settings, toy_car
from core.hub import hub
from sources import real_model
from sources.services import camera, dashcam, real_sim
from tools import seed

dash.register_page(__name__, path="/system", name="Devices & logs", order=7,
                   description="Health of the car, camera, models and data — plus the full event log.")

LEVEL_ICON = {"info": ("bi-info-circle", theme.MUTED), "warning": ("bi-exclamation-triangle-fill", theme.WARNING),
              "critical": ("bi-exclamation-octagon-fill", theme.CRITICAL)}

layout = html.Div([
    dbc.Tabs(id="sys-tabs", active_tab="devices", children=[
        dbc.Tab(label="Devices", tab_id="devices", children=html.Div(id="sys-devices", className="device-grid")),
        dbc.Tab(label="Event log", tab_id="events", children=ui.panel(None, flush=True, children=[
            html.Div(className="filter-bar", style={"padding": "12px 16px 0"}, children=[
                dbc.Checklist(id="sys-ev-levels", value=["info", "warning", "critical"], inline=True,
                              className="check-pills",
                              options=[{"label": l.capitalize(), "value": l} for l in ("info", "warning", "critical")]),
                dcc.Dropdown(id="sys-ev-kinds", multi=True, placeholder="All kinds", style={"minWidth": "240px"},
                             options=[{"label": k, "value": k} for k in
                                      ("trip", "obstacle", "camera", "checkpoint", "vehicle", "system")]),
            ]),
            dash_table.DataTable(id="sys-events", data=[], columns=[
                {"name": "Time", "id": "time"}, {"name": "Level", "id": "level_md", "presentation": "markdown"},
                {"name": "Kind", "id": "kind"}, {"name": "Message", "id": "message"}],
                **ui.table_style(page_size=15)),
        ])),
        dbc.Tab(label="Settings", tab_id="settings", children=dbc.Row(className="g-3", children=[
            dbc.Col(lg=6, children=ui.panel("Object danger weights", subtitle="0 = harmless, 1 = most dangerous. "
                                            "Used by the severity score.", children=[
                dash_table.DataTable(id="sys-weights", editable=True, data=[], columns=[
                    {"name": "Object", "id": "cls", "editable": False},
                    {"name": "Weight", "id": "weight", "type": "numeric"}], **ui.table_style(page_size=12)),
                html.Div(className="d-flex gap-2 mt-3", children=[
                    dbc.Button("Save weights", id="sys-weights-save", color="primary", size="sm"),
                    dbc.Button("Recompute all scores", id="sys-recompute", color="secondary", outline=True, size="sm"),
                ]),
                html.Div(id="sys-weights-msg", className="mt-2"),
            ])),
            dbc.Col(lg=6, children=ui.panel("Demo data", children=[
                html.P("Regenerates 14 days of simulated fleet history and 7 days of prototype runs. "
                       "Everything in the database is replaced (live data keeps flowing).", className="hint"),
                dbc.Button([html.I(className="bi bi-arrow-repeat me-2"), "Reset demo data"], id="sys-reset",
                           color="danger", outline=True, size="sm"),
                dcc.ConfirmDialog(id="sys-reset-confirm",
                                  message="Replace ALL trips, obstacles and events with fresh demo data?"),
                html.Div(id="sys-reset-msg", className="mt-2"),
            ])),
        ])),
    ]),
])


def _device(title, icon, status_pill, rows, note=None):
    return ui.panel(html.Span([html.I(className=f"bi {icon} me-2", style={"color": theme.MUTED}), title]),
                    actions=status_pill, children=[*rows, html.Div(note, className="hint mt-2") if note else None])


@callback(Output("sys-devices", "children"), Input("tick-slow", "n_intervals"), Input("sys-tabs", "active_tab"))
def sys_devices(_, tab):
    if tab != "devices":
        return dash.no_update
    cam = camera.info()
    cam_ok = cam["status"] == "connected"
    proto_source = hub.source(config.PROTO)
    real_cars = hub.cars(config.REAL)
    net = network.real_network()
    track = network.proto_network()
    counts = {t: db.query_one(f"SELECT COUNT(*) n FROM {t}")["n"] for t in ("trips", "trip_points", "obstacles", "events")}
    size_mb = config.DB_PATH.stat().st_size / 1e6 if config.DB_PATH.exists() else 0
    model = real_model.get()
    dash_info = dashcam.info()
    dash_ok = dash_info["status"] == "playing"
    return [
        _device(f"Dashcam · {config.DASHCAM_CAR_ID}", "bi-film",
                ui.pill("Playing" if dash_ok else dash_info["status"].capitalize(),
                        theme.GOOD if dash_ok else theme.WARNING),
                [ui.kv("Source", dash_info["source"], mono=True), ui.kv("Clip", dash_info["clip"] or "—"),
                 ui.kv("Video", f"{dash_info['read_fps']:.0f} fps" if dash_ok else "—"),
                 ui.kv("Analysed", f"{dash_info['det_fps']:.1f} frames / s" if dash_ok else "—"),
                 ui.kv("Detector", dash_info["detector"])],
                dash_info["error"]),
        _device("ESP32-CAM", "bi-camera-video-fill",
                ui.pill("Online" if cam_ok else cam["status"].capitalize(), theme.GOOD if cam_ok else theme.CRITICAL),
                [ui.kv("IP address", cam["ip"] or "—", mono=True), ui.kv("Found by", cam["method"] or "—"),
                 ui.kv("Frame rate", f"{cam['fps']:.1f} fps" if cam_ok else "—"),
                 ui.kv("Resolution", cam["resolution"] or "—"), ui.kv("Frames received", ui.compact(cam["frames"])),
                 ui.kv("Last frame", ui.ago(cam["last_frame_ts"]))],
                cam["scan_info"] if not cam_ok else None),
        _device("Arduino UNO", "bi-cpu", ui.pill("On-board only", theme.MUTED, "bi-link-45deg"),
                [ui.kv("Ultrasonic", "HC-SR04 · Trig D12 / Echo D13"), ui.kv("Line sensors", "2 × IR · D2 / D3 (1 = tape)"),
                 ui.kv("Motors", "L298N · IN1–IN4 on D8–D11"),
                 ui.kv("Motor speed", f"PWM {toy_car.MOTOR_PWM}/255 ({toy_car.DUTY:.0%}) ≈ "
                                      f"{toy_car.SPEED_CM_S:.0f} cm/s"),
                 ui.kv("Obstacle", f"detour when ≤ {toy_car.OBSTACLE_CM} cm"),
                 ui.kv("Detour", f"left {toy_car.AVOID_STEPS[1][1] / 1000:.1f} s · right "
                                 f"{toy_car.AVOID_STEPS[3][1] / 1000:.1f} s · right → tape (≈ {toy_car.AVOID_S:.1f} s)"),
                 ui.kv("Wireless link", "none")],
                "Values from firmware/arduino_car/arduino_car.ino (speed estimated from the PWM). The UNO has no "
                "Wi-Fi and is not wired to the ESP32-CAM, so its readings never reach the dashboard: prototype data "
                "comes from the camera, and the demo car drives with these values."),
        _device("Prototype data", "bi-car-front-fill", ui.source_pill(proto_source),
                [ui.kv("Track", track.name), ui.kv("Checkpoints", str(len(track.nodes))),
                 ui.kv("Markers", f"{len(track.markers)} ({config.ARUCO_DICT})"),
                 ui.kv("Active trajectory", (network.active_trajectory() or {}).get("name", "—")),
                 ui.kv("Demo when camera offline", "on" if config.PROTO_DEMO_WHEN_OFFLINE else "off")]),
        _device("Real-world simulator", "bi-globe2",
                ui.pill("Running" if real_sim.running else "Stopped", theme.GOOD if real_sim.running else theme.MUTED),
                [ui.kv("Vehicles", str(len(real_cars))), ui.kv("Road network", net.name),
                 ui.kv("Intersections", ui.compact(len(net.nodes))), ui.kv("Road segments", ui.compact(len(net.edges))),
                 ui.kv("Map data", net.meta.get("source", "—"))]),
        _device("Models", "bi-bounding-box", None,
                [ui.kv("Prototype detector", cam["detector"]), ui.kv("Real-world (dashcam)", dash_info["detector"]),
                 ui.kv("Real-world (simulated cars)", model.status), ui.kv("Registry", "models/registry.json")],
                cam["detector_error"]),
        _device("Database", "bi-database", ui.pill(f"{size_mb:.1f} MB", theme.MUTED),
                [ui.kv("Trips", ui.compact(counts["trips"])), ui.kv("Track points", ui.compact(counts["trip_points"])),
                 ui.kv("Obstacles", ui.compact(counts["obstacles"])), ui.kv("Events", ui.compact(counts["events"]))]),
    ]


@callback(Output("sys-events", "data"), Input("tick-slow", "n_intervals"), Input("mode-switch", "value"),
          Input("sys-ev-levels", "value"), Input("sys-ev-kinds", "value"))
def sys_events(_, mode, levels, kinds):
    rows = events.recent(mode or config.REAL, limit=500, levels=levels or ["none"], kinds=kinds or None)
    return [{"time": ui.fmt_time(r["ts"]), "kind": r["kind"], "message": r["message"],
             "level_md": f'<span class="lvl" style="color:{LEVEL_ICON[r["level"]][1]}">●</span> {r["level"].capitalize()}'}
            for r in rows]


@callback(Output("sys-weights", "data"), Input("sys-tabs", "active_tab"))
def sys_weights(_tab):
    return [{"cls": c, "weight": w} for c, w in settings.class_weights().items()]


@callback(Output("sys-weights-msg", "children"), Input("sys-weights-save", "n_clicks"),
          Input("sys-recompute", "n_clicks"), State("sys-weights", "data"), prevent_initial_call=True)
def sys_save_weights(_s, _r, rows):
    if ctx.triggered_id == "sys-recompute":
        n = obstacles.recompute_severity()
        return dbc.Alert(f"Recomputed the score of {n} obstacles.", color="success", className="mb-0")
    try:
        weights = {r["cls"]: float(r["weight"]) for r in rows}
    except (TypeError, ValueError):
        return dbc.Alert("Weights must be numbers.", color="danger", className="mb-0")
    if any(not 0 <= w <= 1 for w in weights.values()):
        return dbc.Alert("Weights must be between 0 and 1.", color="danger", className="mb-0")
    settings.set("class_weights", weights)
    return dbc.Alert("Saved. New detections use these weights; click “Recompute” to update existing ones.",
                     color="success", className="mb-0")


@callback(Output("sys-reset-confirm", "displayed"), Input("sys-reset", "n_clicks"), prevent_initial_call=True)
def sys_reset_ask(_):
    return True


@callback(Output("sys-reset-msg", "children"), Input("sys-reset-confirm", "submit_n_clicks"), prevent_initial_call=True)
def sys_reset(_):
    result = seed.seed_all()
    return dbc.Alert(f"Demo data regenerated: {result['real'][0]} fleet trips, {result['proto'][0]} prototype runs.",
                     color="success", className="mb-0")
