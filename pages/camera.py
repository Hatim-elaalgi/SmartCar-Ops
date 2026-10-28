"""Camera: Prototype = ESP32-CAM stream; Real-world = dashcam of CAR-01 (real video + trained models).
Both streams are relayed by the dashboard with detections, a HUD, and connection / source controls."""
import ipaddress
import re
import time

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

import config
from components import theme, ui
from core.hub import hub
from core.severity import LEVEL_META
from sources.camera import FRAMESIZES
from sources.services import camera, dashcam

dash.register_page(__name__, path="/camera", name="Camera", order=2,
                   description="Live video with the models' detections — dashcam (real-world) or ESP32-CAM (prototype).")

STATUS = {"connected": ("Connected", theme.GOOD, "bi-broadcast"),
          "playing": ("Playing", theme.GOOD, "bi-broadcast"),
          "connecting": ("Connecting", theme.WARNING, "bi-arrow-repeat"),
          "searching": ("Searching the network", theme.WARNING, "bi-search"),
          "starting": ("Starting", theme.MUTED, "bi-hourglass"),
          "no source": ("No video", theme.WARNING, "bi-camera-video-off"),
          "error": ("Error", theme.CRITICAL, "bi-exclamation-octagon"),
          "offline": ("Offline", theme.CRITICAL, "bi-wifi-off")}

# --- Real-world: dashcam of CAR-01 ----------------------------------------------------------------
real_section = html.Div(id="cam-real", children=[
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=8, children=ui.panel(
            f"Dashcam · {config.DASHCAM_CAR_ID}", subtitle="Real road video → road users + road hazards models, "
                                                          "in real time", flush=True,
            actions=dbc.Switch(id="cam-real-overlay", label="Detections overlay", value=True),
            children=html.Div(className="video-frame wide", children=[
                html.Img(id="cam-real-img", alt="Dashcam stream"),
                html.Div(id="cam-real-hud", className="hud"),
            ]))),
        dbc.Col(lg=4, children=[
            ui.panel("Video source", [
                html.Div(id="cam-real-src"),
                html.Label("Source", className="field-label"),
                dbc.InputGroup([dbc.Input(id="cam-real-input", placeholder="demo/videos · http://phone:8080/video · esp32",
                                          debounce=True),
                                dbc.Button("Use", id="cam-real-use", color="primary")], size="sm"),
                html.Div(className="d-flex gap-2 mt-2", children=[
                    dbc.Button([html.I(className="bi bi-film me-1"), "Demo clips"], id="cam-real-demo", size="sm",
                               color="secondary", outline=True),
                    dbc.Button([html.I(className="bi bi-camera-video me-1"), "ESP32-CAM"], id="cam-real-esp",
                               size="sm", color="secondary", outline=True),
                ]),
                html.Div(id="cam-real-msg", className="hint mt-2"),
            ]),
            ui.panel("Models", html.Div(id="cam-real-models"), subtitle="ONNX, running on the laptop CPU"),
        ]),
    ]),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=7, children=ui.panel("Live detections", html.Div(id="cam-real-dets"),
                                        subtitle="Every box the models draw, with estimated distance and danger")),
        dbc.Col(lg=5, children=ui.panel("How it works", html.Div([
            html.P(["Detections in the car's path make ", html.B(config.DASHCAM_CAR_ID), " ",
                    html.B("slow down or brake"), " (people and animals from 30 m, a vehicle ahead below 15 m, "
                    "potholes / cones / barriers from 25 m). ", html.B("Static hazards"), " are mapped ahead of the "
                    "car at their estimated distance, with a snapshot; ", html.B("moving road users"),
                    " are mapped only when they are in the path and closer than 15 m."], className="hint"),
            html.P(["Distance ≈ typical real height × focal ÷ box height in pixels (person 1.7 m, car 1.5 m, "
                    f"cone 0.7 m…, {config.DASHCAM_HFOV_DEG:.0f}° lens); flat potholes use the road geometry "
                    f"(camera {config.DASHCAM_HEIGHT_M} m high). In path = in the car's own lane."], className="hint"),
            html.P([html.B("Use a real camera: "), "install the free “IP Webcam” app on a phone, fix it on a "
                    "dashboard, and enter its address (e.g. http://192.168.1.50:8080/video) as the source — or use "
                    "“ESP32-CAM”."], className="hint mb-0"),
        ]))),
    ]),
])

# --- Prototype: ESP32-CAM -------------------------------------------------------------------------
proto_section = html.Div(id="cam-proto", children=[
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=8, children=ui.panel(
            "Live stream · ESP32-CAM", flush=True,
            actions=dbc.Switch(id="cam-overlay", label="Detections overlay", value=True),
            children=html.Div(className="video-frame", children=[
                html.Img(id="cam-img", alt="ESP32-CAM stream"),
                html.Div(id="cam-hud", className="hud"),
            ]))),
        dbc.Col(lg=4, children=[
            ui.panel("Connection", [
                html.Div(id="cam-conn"),
                html.Label("Camera IP (optional — leave empty for auto-discovery)", className="field-label"),
                dbc.InputGroup([dbc.Input(id="cam-ip", placeholder="e.g. 192.168.1.42", debounce=True),
                                dbc.Button("Connect", id="cam-connect", color="primary")], size="sm"),
                html.Div(className="d-flex gap-2 mt-2", children=[
                    dbc.Button([html.I(className="bi bi-search me-1"), "Rescan network"], id="cam-rescan",
                               size="sm", color="secondary", outline=True),
                    dbc.Button([html.I(className="bi bi-camera me-1"), "Snapshot"], id="cam-snap", size="sm",
                               color="secondary", outline=True),
                    dbc.Button([html.I(className="bi bi-record-circle me-1"), "Record frames"], id="cam-record",
                               size="sm", color="secondary", outline=True),
                ]),
                html.Div(id="cam-msg", className="hint mt-2"),
                html.Div(id="cam-rec-msg", className="hint mt-1"),
            ]),
            ui.panel("Camera settings", [
                html.Label("Resolution", className="field-label"),
                dbc.Select(id="cam-res", size="sm", placeholder="Choose a resolution…",
                           options=[{"label": k, "value": v} for k, v in FRAMESIZES.items()]),
                html.Label("JPEG quality (lower = sharper, slower)", className="field-label"),
                dcc.Slider(id="cam-quality", min=10, max=63, step=1, value=12, marks={10: "10", 36: "36", 63: "63"}),
                html.Label("Flash LED", className="field-label"),
                dcc.Slider(id="cam-flash", min=0, max=255, step=5, value=0, marks={0: "off", 255: "max"}),
                html.Div(className="d-flex gap-3 mt-2", children=[
                    dbc.Switch(id="cam-vflip", label="Flip vertically", value=False),
                    dbc.Switch(id="cam-hmirror", label="Mirror", value=False),
                ]),
                html.Label("Rotation (camera mounted on its side)", className="field-label"),
                dbc.RadioItems(id="cam-rotate", inline=True, options=[{"label": f"{d}°", "value": d}
                                                                      for d in (0, 90, 180, 270)],
                               className="segmented", inputClassName="btn-check", labelClassName="btn btn-segment",
                               labelCheckedClassName="active"),
                html.Div(id="cam-set-msg", className="hint mt-1"),
            ]),
        ]),
    ]),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=6, children=ui.panel("Live detections", html.Div(id="cam-dets"),
                                        subtitle="Latest model outputs on the stream")),
        dbc.Col(lg=6, children=ui.panel("Snapshots", html.Div(id="cam-snaps"),
                                        subtitle="Saved frames and obstacle snapshots")),
    ]),
])

layout = html.Div([real_section, proto_section])


@callback(Output("cam-real", "style"), Output("cam-proto", "style"), Output("cam-real-img", "src"),
          Output("cam-img", "src"), Output("cam-rotate", "value"),
          Input("mode-switch", "value"), Input("cam-real-overlay", "value"), Input("cam-overlay", "value"))
def cam_mode(mode, real_overlay, proto_overlay):
    """Only the visible stream is connected (each <img> holds a live HTTP connection)."""
    t = int(time.time())
    if mode == config.PROTO:
        return {"display": "none"}, {}, "", \
            f"/camera/stream.mjpg?source=proto&overlay={int(bool(proto_overlay))}&t={t}", camera.rotation
    return {}, {"display": "none"}, f"/camera/stream.mjpg?source=real&overlay={int(bool(real_overlay))}&t={t}", "", \
        camera.rotation


# --- real-world callbacks -------------------------------------------------------------------------
def _det_row(d, mode):
    meta = LEVEL_META.get(d.get("level") or "low", LEVEL_META["low"])
    parts = [f"{d['conf']:.0%}" if d.get("conf") is not None else None,
             f"≈ {ui.fmt_dist(mode, d['distance'])}" if d.get("distance") is not None else None,
             "in path" if d.get("in_path") else "side", "demo" if d.get("source") == "demo" else None]
    return html.Div(className="list-row", children=[
        html.I(className=f"bi {meta['icon']} icon", style={"color": meta["color"]}),
        html.Div(className="grow", children=[
            html.Div(d["cls"].replace("_", " "), className="title"),
            html.Div(" · ".join(p for p in parts if p), className="meta")]),
        html.Span(ui.ago(d["ts"]), className="when"),
    ])


@callback(Output("cam-real-src", "children"), Output("cam-real-hud", "children"), Output("cam-real-dets", "children"),
          Output("cam-real-models", "children"),
          Input("tick", "n_intervals"), State("mode-switch", "value"))
def cam_real_live(_, mode):
    if mode == config.PROTO:
        raise PreventUpdate
    info = dashcam.info()
    label, color, icon = STATUS.get(info["status"], STATUS["offline"])
    src = [
        html.Div(className="car-head", children=[ui.pill(label, color, icon),
                                                 html.Span(info["source"], className="mono hint")]),
        ui.kv("Clip", info["clip"] or "—"),
        ui.kv("Credit", html.Span(info["credit"] or "—", style={"whiteSpace": "normal"})),
        ui.kv("Resolution", info["resolution"] or "—"),
        ui.kv("Video", f"{info['read_fps']:.0f} fps" if info["status"] == "playing" else "—"),
        ui.kv("Models", f"{info['det_fps']:.1f} analysed frames / s" if info["status"] == "playing" else "—"),
    ]
    if info["error"]:
        src.append(html.Div(info["error"], className="hint mt-2", style={"color": theme.SERIOUS}))

    car = hub.car(config.REAL, config.DASHCAM_CAR_ID)
    live = info["status"] == "playing"
    hud = [html.Div([html.Span(className="rec") if live else html.I(className="bi bi-camera-video-off"),
                     f"{config.DASHCAM_CAR_ID} · {'LIVE' if live else 'NO VIDEO'}",
                     html.Span(f"{info['det_fps']:.1f} fps AI", style={"color": theme.MUTED}) if live else None],
                    className="tl")]
    if car and car.get("online"):
        hud.append(html.Div(ui.state_pill(car.get("state")), className="tr"))
        hud.append(html.Div([html.I(className="bi bi-speedometer2"), ui.fmt_speed(config.REAL, car.get("speed")),
                             html.Span(f"front {car['front']:.0f} m" if car.get("front") else "",
                                       style={"color": theme.WARNING})], className="bl"))
    hud.append(html.Div(time.strftime("%H:%M:%S"), className="br"))

    dets = hub.detections(12, mode=config.REAL)
    det_rows = [_det_row(d, config.REAL) for d in dets] or ui.empty(
        "No detections yet" if live else "Waiting for video…", "bi-bounding-box")

    if info["models"]:
        models = [html.Div(className="list-row", children=[
            html.I(className="bi bi-cpu icon", style={"color": theme.ACCENT}),
            html.Div(className="grow", children=[
                html.Div(f"{m['id']} · {m['file']}", className="title"),
                html.Div(", ".join(c.replace("_", " ") for c in m["classes"]), className="meta",
                         style={"whiteSpace": "normal"})]),
            html.Span(f"{m['ms']:.0f} ms", className="when"),
        ]) for m in info["models"]]
    else:
        models = [html.Div(info["detector"], className="title mb-1"),
                  html.Div(info.get("detector_error") or "Train the models with training/train_road_obstacles.ipynb, "
                           "then restart the dashboard.", className="hint")]
    return src, hud, det_rows, models


@callback(Output("cam-real-msg", "children"), Input("cam-real-use", "n_clicks"), Input("cam-real-input", "n_submit"),
          Input("cam-real-demo", "n_clicks"), Input("cam-real-esp", "n_clicks"), State("cam-real-input", "value"),
          prevent_initial_call=True)
def cam_real_source(_u, _s, _d, _e, value):
    if ctx.triggered_id == "cam-real-demo":
        source = config.REAL_VIDEO_SOURCE
    elif ctx.triggered_id == "cam-real-esp":
        source = "esp32"
    else:
        source = (value or "").strip()
        if not source:
            raise PreventUpdate
        if not (re.match(r"^(https?|rtsp)://", source) or source == "esp32" or (config.BASE_DIR / source).exists()):
            return "Use a folder/file inside the project, an http(s):// or rtsp:// URL, or esp32."
    dashcam.set_source(source)
    return f"Switching to {source}…"


# --- prototype callbacks --------------------------------------------------------------------------
def _hud(info, car):
    live = info["status"] == "connected"
    source = hub.source(config.PROTO)
    left = [html.Span(className="rec") if live else html.I(className="bi bi-wifi-off"),
            f"LIVE  {info['fps']:.1f} fps" if live else ("DEMO SIMULATION" if source == "demo" else "NO SIGNAL")]
    if live and info["resolution"]:
        left.append(html.Span(info["resolution"], style={"color": theme.MUTED}))
    parts = [html.Div(left, className="tl")]
    if car and car.get("online"):
        parts.append(html.Div(ui.state_pill(car.get("state")), className="tr"))
        parts.append(html.Div([html.I(className="bi bi-crosshair"),
                               f"Checkpoint {car.get('last_marker') or '—'} · {car.get('route') or 'no route'}"],
                              className="bl"))
    parts.append(html.Div(time.strftime("%H:%M:%S"), className="br"))
    return parts


@callback(Output("cam-conn", "children"), Output("cam-hud", "children"), Output("cam-dets", "children"),
          Input("tick", "n_intervals"), State("mode-switch", "value"))
def cam_live(_, mode):
    if mode == config.REAL:
        raise PreventUpdate
    info = camera.info()
    label, color, icon = STATUS.get(info["status"], STATUS["offline"])
    since = info["connected_since"]
    conn = [
        html.Div(className="car-head", children=[ui.pill(label, color, icon),
                                                 html.Span(info["ip"] or "no IP yet", className="mono")]),
        ui.kv("Found by", info["method"] or "—"),
        ui.kv("Frame rate", f"{info['fps']:.1f} fps" if info["status"] == "connected" else "—"),
        ui.kv("Resolution", info["resolution"] or "—"),
        ui.kv("Last frame", ui.ago(info["last_frame_ts"]) if info["last_frame_ts"] else "—"),
        ui.kv("Connected for", ui.fmt_dur(time.time() - since) if since and info["status"] == "connected" else "—"),
        ui.kv("Detector", info["detector"]),
        ui.kv("Checkpoints seen", str(info["markers_seen"])),
    ]
    if info["status"] != "connected" and info["scan_info"]:
        conn.append(html.Div(info["scan_info"], className="hint mt-2"))
    if info["error"] and info["status"] != "connected":
        conn.append(html.Div(info["error"], className="hint mt-1", style={"color": theme.SERIOUS}))
    if info["detector_error"]:
        conn.append(html.Div(f"Model error: {info['detector_error']}", className="hint mt-1",
                             style={"color": theme.SERIOUS}))
    dets = hub.detections(10)
    det_rows = [_det_row(d, config.PROTO) for d in dets] or ui.empty(
        "No detections yet. Set DETECTOR_PROTO in .env to plug in your model.", "bi-bounding-box")
    return conn, _hud(info, hub.car(config.PROTO, config.PROTO_CAR_ID)), det_rows


@callback(Output("cam-msg", "children"), Input("cam-connect", "n_clicks"), Input("cam-ip", "n_submit"),
          Input("cam-rescan", "n_clicks"), State("cam-ip", "value"), prevent_initial_call=True)
def cam_connect(_c, _s, _r, ip):
    if ctx.triggered_id == "cam-rescan":
        camera.set_manual_ip("")
        return "Scanning the network for the ESP32-CAM…"
    ip = (ip or "").strip()
    if not ip:
        camera.set_manual_ip("")
        return "Auto-discovery enabled."
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        return "That is not a valid IP address."
    camera.set_manual_ip(ip)
    return f"Connecting to {ip}…"


@callback(Output("cam-record", "children"), Output("cam-record", "color"), Output("cam-record", "outline"),
          Output("cam-rec-msg", "children"),
          Input("cam-record", "n_clicks"), Input("tick", "n_intervals"), State("mode-switch", "value"))
def cam_record(_n, _t, mode):
    """Training data: upright camera frames (no overlay) saved a few times per second while recording."""
    if mode == config.REAL:
        raise PreventUpdate
    if ctx.triggered_id == "cam-record":
        camera.set_recording(not camera.recording)
    info = camera.info()
    if info["recording"]:
        return ([html.I(className="bi bi-stop-circle me-1"), f"Stop ({info['rec_frames']})"], "danger", False,
                f"Recording {info['rec_frames']} frames → {info['rec_dir']}"
                + ("" if info["status"] == "connected" else " · waiting for the camera"))
    done = f"Last recording: {info['rec_frames']} frames in {info['rec_dir']}" if info["rec_dir"] else None
    return [html.I(className="bi bi-record-circle me-1"), "Record frames"], "secondary", True, done


@callback(Output("cam-set-msg", "children"),
          Input("cam-res", "value"), Input("cam-quality", "value"), Input("cam-flash", "value"),
          Input("cam-vflip", "value"), Input("cam-hmirror", "value"), Input("cam-rotate", "value"),
          prevent_initial_call=True)
def cam_settings(res, quality, flash, vflip, hmirror, rotate):
    if ctx.triggered_id == "cam-rotate":            # done by the dashboard, not the camera
        if rotate is None or rotate == camera.rotation:
            raise PreventUpdate
        camera.set_rotation(rotate)
        return f"Image turned {rotate}° clockwise before display and detection"
    var, val = {"cam-res": ("framesize", res), "cam-quality": ("quality", quality),
                "cam-flash": ("led_intensity", flash), "cam-vflip": ("vflip", int(bool(vflip))),
                "cam-hmirror": ("hmirror", int(bool(hmirror)))}[ctx.triggered_id]
    if val is None:
        raise PreventUpdate
    ok, msg = camera.control(var, val)
    return f"{var} = {val}: {msg}" if ok else f"Could not change {var}: {msg}"


@callback(Output("cam-snaps", "children"), Input("cam-snap", "n_clicks"), Input("tick-slow", "n_intervals"))
def cam_snapshots(_n, _t):
    if ctx.triggered_id == "cam-snap" and not camera.save_snapshot():
        note = html.Div("Snapshot needs a connected camera.", className="hint mb-2")
    else:
        note = None
    files = sorted(config.SNAPSHOT_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if not files:
        return [note, ui.empty("No snapshots yet", "bi-images")]
    return [note, html.Div(className="snap-grid", children=[
        html.A(html.Img(src=f"/snapshots/{p.name}", alt=p.name), href=f"/snapshots/{p.name}", target="_blank",
               title=p.name) for p in files])]
