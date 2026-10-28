"""Smart Car IoT dashboard — entry point.  Run with:  docker compose up --build  ->  http://localhost:8050"""
import logging
import os
import time

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, callback, dcc, html
from flask import Response, request, send_from_directory

import config
from components import theme, ui
from core import db, obstacles
from core.hub import hub
from tools import seed

theme.register_template()
db.init()
seed.ensure()
obstacles.ensure_track_obstacles()      # prototype: exactly the obstacles placed on the test track

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, title="SmartCar Ops", update_title=None)
server = app.server
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app.index_string = """<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

NAV = [
    (None, [("/", "Overview", "bi-grid-1x2-fill")]),
    ("Live", [("/tracking", "Tracking", "bi-geo-alt-fill"), ("/camera", "Camera", "bi-camera-video-fill")]),
    ("Planning", [("/planner", "Route planner", "bi-signpost-split-fill")]),
    ("Safety", [("/obstacles", "Obstacles", "bi-exclamation-triangle-fill")]),
    ("Analytics", [("/stats", "Live stats", "bi-graph-up"), ("/models", "AI models", "bi-bounding-box")]),
    ("System", [("/system", "Devices & logs", "bi-hdd-network-fill")]),
]


def sidebar():
    groups = []
    for title, links in NAV:
        groups.append(html.Div(className="nav-group", children=[
            html.Div(title, className="nav-group-title") if title else None,
            *[dbc.NavLink([html.I(className=f"bi {icon}"), html.Span(label)], href=href, active="exact",
                          className="nav-item-link") for href, label, icon in links],
        ]))
    return html.Aside(className="sidebar", children=[
        html.Div(className="brand", children=[
            html.Div(html.I(className="bi bi-crosshair"), className="brand-mark"),
            html.Div([html.Div("SmartCar Ops", className="brand-name"),
                      html.Div("Obstacle-aware IoT car", className="brand-sub")]),
        ]),
        html.Nav(groups, className="nav-groups"),
        html.Div(className="sidebar-foot", children=[html.Span(className="live-dot"), html.Span(id="clock")]),
    ])


app.layout = html.Div(className="shell", children=[
    dcc.Location(id="url"),
    dcc.Interval(id="tick", interval=1000),
    dcc.Interval(id="tick-slow", interval=5000),
    sidebar(),
    html.Div(className="main", children=[
        html.Header(className="topbar", children=[
            html.Div(className="topbar-title", children=[
                html.H1(id="page-title"), html.Div(id="page-desc", className="page-desc"),
            ]),
            html.Div(className="topbar-right", children=[
                html.Div(id="source-pill"),
                dbc.RadioItems(
                    id="mode-switch", value=config.REAL, inline=True, persistence=True, persistence_type="local",
                    options=[{"label": html.Span([html.I(className="bi bi-globe2 me-2"), "Real-world"]),
                              "value": config.REAL},
                             {"label": html.Span([html.I(className="bi bi-car-front-fill me-2"), "Prototype"]),
                              "value": config.PROTO}],
                    className="segmented", inputClassName="btn-check", labelClassName="btn btn-segment",
                    labelCheckedClassName="active"),
            ]),
        ]),
        html.Main(dash.page_container, className="content"),
    ]),
])


@callback(Output("page-title", "children"), Output("page-desc", "children"), Input("url", "pathname"))
def page_heading(pathname):
    for page in dash.page_registry.values():
        if page["path"] == pathname:
            return page["name"], page.get("description", "")
    return "Not found", ""


@callback(Output("source-pill", "children"), Output("clock", "children"),
          Input("tick", "n_intervals"), Input("mode-switch", "value"))
def live_status(_, mode):
    return ui.source_pill(hub.source(mode or config.REAL)), time.strftime("%H:%M:%S")


# --- camera relay + files ---------------------------------------------------------------------
@server.route("/camera/stream.mjpg")
def camera_stream():
    """MJPEG relay. ?source=proto (ESP32-CAM, default) or ?source=real (dashcam of CAR-01)."""
    from sources.services import camera, dashcam
    overlay = request.args.get("overlay", "1") == "1"
    feed = dashcam if request.args.get("source") == "real" else camera

    def generate():
        last_id, last_sent = None, 0.0
        while True:
            jpeg, frame_id = feed.latest_jpeg(annotated=overlay)
            now = time.time()
            # Re-send at least once a second: some browsers only paint a part when the next one arrives.
            if frame_id != last_id or now - last_sent > 1.0:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")
                last_id, last_sent = frame_id, now
            time.sleep(0.04)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store"})


@server.route("/snapshots/<path:name>")
def snapshot_file(name):
    return send_from_directory(config.SNAPSHOT_DIR, name, max_age=3600)


if __name__ == "__main__":
    from sources import services
    # With DEBUG=1 Flask's reloader runs this file twice; only start the background services in the child.
    if not config.DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        services.start()
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG, dev_tools_silence_routes_logging=True)
