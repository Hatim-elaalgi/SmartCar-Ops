"""Route planner: shortest vs safest vs recommended. Prototype mode adds your manual trajectories,
a track editor and printable ArUco checkpoint markers."""
import io
import time
import zipfile

import cv2
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate

import config
from components import theme, ui
from components.figures import network_figure
from core import geo, network, obstacles, planner, settings
from core.severity import LEVEL_META

dash.register_page(__name__, path="/planner", name="Route planner", order=3,
                   description="Pick a start and a destination: the shortest, the safest, and the best trade-off.")

ROUTE_COLORS = {"Recommended": theme.SERIES[0], "Shortest": theme.SERIES[1], "Safest": theme.SERIES[2]}
ALPHA_MARKS = {0: "Shortest", 2.5: "Balanced", 5: "Safest"}


def _editor_table(table_id, columns):
    return dash_table.DataTable(id=table_id, columns=columns, data=[], editable=True, row_deletable=True,
                                **ui.table_style(page_size=25))


layout = html.Div([
    dcc.Store(id="pl-track-version", data=0),
    dcc.Download(id="pl-markers-dl"),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=8, children=ui.panel(
            "Route map", flush=True, children=[
                dcc.Graph(id="pl-map", config={"displayModeBar": False}, className="map-xl"),
                html.Div(className="legend map-legend", children=[
                    *[ui.legend_item(name, color, "line") for name, color in ROUTE_COLORS.items()],
                    ui.legend_item("Risky segment", LEVEL_META["high"]["color"], "line"),
                    ui.legend_item("Very risky segment", LEVEL_META["critical"]["color"], "line"),
                    ui.legend_item("Start", theme.INK_1), ui.legend_item("Destination", theme.SERIES[3]),
                    html.Span(id="pl-obs-legend"),
                ]),
            ],
            actions=[
                html.Span("Click sets", className="hint"),
                dbc.RadioItems(id="pl-pick", value="start", inline=True, className="segmented sm",
                               inputClassName="btn-check", labelClassName="btn btn-segment",
                               labelCheckedClassName="active",
                               options=[{"label": "Start", "value": "start"},
                                        {"label": "Destination", "value": "end"}]),
                dbc.Switch(id="pl-risk", label="Risk layer", value=True),
            ])),
        dbc.Col(lg=4, children=[
            ui.panel("Plan a route", [
                html.Label("Start", className="field-label"),
                dcc.Dropdown(id="pl-start", clearable=False, searchable=True),
                html.Label("Destination", className="field-label"),
                dcc.Dropdown(id="pl-end", clearable=False, searchable=True),
                html.Label("Priority", className="field-label"),
                dcc.Slider(id="pl-alpha", min=0, max=5, step=0.5, value=1.5, marks=ALPHA_MARKS),
                html.Div("Cost of a segment = length + priority × risk. Risk comes from the obstacle history "
                         "(recent and open obstacles weigh more).", className="hint mt-2"),
            ]),
            ui.panel("Suggested routes", html.Div(id="pl-results")),
        ]),
    ]),
    html.Div(id="pl-proto", children=[
        ui.panel("Your trajectories", subtitle="Manual routes on the tape track, ranked with the same priority",
                 actions=html.Div(id="pl-active-msg", className="hint"), children=[
                     dash_table.DataTable(
                         id="pl-traj", row_selectable="single", data=[], sort_action="native",
                         columns=[{"name": "Rank", "id": "rank"}, {"name": "Trajectory", "id": "name"},
                                  {"name": "Path", "id": "path"}, {"name": "Length", "id": "length_txt"},
                                  {"name": "Run time", "id": "eta_txt"}, {"name": "Obstacles avoided", "id": "obstacles"},
                                  {"name": "Risk", "id": "risk_txt"}, {"name": "Active", "id": "active"}],
                         **ui.table_style(page_size=8)),
                     html.Div(className="mt-2", children=dbc.Button(
                         [html.I(className="bi bi-flag-fill me-2"), "Use selected for the next run"],
                         id="pl-set-active", color="primary", size="sm")),
                 ]),
        ui.panel("Track editor", subtitle="Positions in cm. Each checkpoint gets an ArUco marker the camera reads.",
                 actions=[dbc.Button([html.I(className="bi bi-qr-code me-2"), "Download markers"],
                                     id="pl-markers", size="sm", color="secondary", outline=True)],
                 children=[
                     dbc.Row(className="g-3", children=[
                         dbc.Col(lg=4, children=[
                             html.Div("Checkpoints", className="field-label"),
                             _editor_table("pl-nodes", [
                                 {"name": "Id", "id": "id"}, {"name": "Label", "id": "label"},
                                 {"name": "x", "id": "x", "type": "numeric"}, {"name": "y", "id": "y", "type": "numeric"},
                                 {"name": "Marker", "id": "marker", "type": "numeric"}]),
                             dbc.Button("+ Checkpoint", id="pl-add-node", size="sm", color="secondary",
                                        outline=True, className="mt-2"),
                         ]),
                         dbc.Col(lg=4, children=[
                             html.Div("Segments", className="field-label"),
                             _editor_table("pl-edges", [
                                 {"name": "From", "id": "u"}, {"name": "To", "id": "v"},
                                 {"name": "Curve points (x,y; x,y)", "id": "via"}]),
                             dbc.Button("+ Segment", id="pl-add-edge", size="sm", color="secondary",
                                        outline=True, className="mt-2"),
                         ]),
                         dbc.Col(lg=4, children=[
                             html.Div("Trajectories", className="field-label"),
                             _editor_table("pl-trajs", [{"name": "Name", "id": "name"},
                                                        {"name": "Checkpoints (A > B > C)", "id": "nodes"}]),
                             dbc.Button("+ Trajectory", id="pl-add-traj", size="sm", color="secondary",
                                        outline=True, className="mt-2"),
                         ]),
                     ]),
                     html.Div(className="d-flex gap-2 mt-3", children=[
                         dbc.Button([html.I(className="bi bi-save me-2"), "Save track"], id="pl-save",
                                    color="primary", size="sm"),
                         dbc.Button("Reset to default track", id="pl-reset", color="secondary", outline=True,
                                    size="sm"),
                     ]),
                     html.Div(id="pl-save-msg", className="mt-2"),
                 ]),
    ]),
])


# --- start / end ----------------------------------------------------------------------------------
def _default_ends(mode, net):
    if mode == config.PROTO:
        for t in net.trajectories:                  # a trajectory that goes somewhere (loops start = end)
            if t["nodes"][0] != t["nodes"][-1]:
                return t["nodes"][0], t["nodes"][-1]
        start = net.trajectories[0]["nodes"][0] if net.trajectories else next(iter(net.nodes))
        return start, max(net.nodes, key=lambda n: geo.distance(mode, *net.xy(start), *net.xy(n)))
    x0, x1, y0, y1 = net.bounds()
    a = min(net.nodes, key=lambda n: geo.distance(mode, *net.xy(n), x0 + (x1 - x0) * 0.15, y0 + (y1 - y0) * 0.2))
    b = min(net.nodes, key=lambda n: geo.distance(mode, *net.xy(n), x0 + (x1 - x0) * 0.85, y0 + (y1 - y0) * 0.8))
    return a, b


@callback(Output("pl-start", "options"), Output("pl-end", "options"), Output("pl-start", "value"),
          Output("pl-end", "value"), Output("pl-proto", "style"),
          Input("mode-switch", "value"), Input("pl-track-version", "data"))
def pl_options(mode, _version):
    mode = mode or config.REAL
    net = network.get_network(mode)
    options = sorted(({"label": n["label"] if mode == config.REAL else f"{nid} — {n['label']}", "value": nid}
                      for nid, n in net.nodes.items()), key=lambda o: o["label"])
    start, end = _default_ends(mode, net)
    return options, options, start, end, ({} if mode == config.PROTO else {"display": "none"})


@callback(Output("pl-start", "value", allow_duplicate=True), Output("pl-end", "value", allow_duplicate=True),
          Output("pl-pick", "value"),
          Input("pl-map", "clickData"), State("pl-pick", "value"), State("mode-switch", "value"),
          prevent_initial_call=True)
def pl_click(click, pick, mode):
    if not click:
        raise PreventUpdate
    node = click["points"][0].get("customdata")
    if not isinstance(node, str) or node not in network.get_network(mode or config.REAL).nodes:
        raise PreventUpdate
    if pick == "start":
        return node, no_update, "end"
    return no_update, node, "start"


# --- routes ---------------------------------------------------------------------------------------
@callback(Output("pl-obs-legend", "children"), Input("mode-switch", "value"))
def pl_obs_legend(mode):
    if mode == config.PROTO:
        return ui.legend_item("Avoided obstacle", theme.INK_2, "square")
    return ui.legend_item("Open obstacle (colour = level)", LEVEL_META["medium"]["color"], "diamond")


def _route_card(mode, r, recommended):
    lv = r["by_level"]
    obstacle_txt = [ui.pill(f"{lv[l]} {LEVEL_META[l]['label'].lower()}", LEVEL_META[l]["color"], LEVEL_META[l]["icon"])
                    for l in ("critical", "high", "medium", "low") if lv.get(l)]
    also = f"Also the {' and the '.join(a.lower() for a in r['also'])} route" if r["also"] else None
    return html.Div(className="route-card" + (" recommended" if recommended else ""), children=[
        html.Div(className="route-head", children=[
            html.Span(className="route-key", style={"background": ROUTE_COLORS[r["name"]]}), r["name"],
            html.Span("Best for your priority", className="badge-soft ms-auto") if recommended else None,
        ]),
        html.Div(also, className="hint mb-2") if also else None,
        html.Div(className="route-metrics", children=[
            html.Div([html.Div("Distance", className="m-label"), html.Div(ui.fmt_dist(mode, r["length"]), className="m-value")]),
            html.Div([html.Div("ETA", className="m-label"), html.Div(ui.fmt_dur(r["eta"]), className="m-value")]),
            html.Div([html.Div("Risk", className="m-label"), html.Div(f"{r['risk']:.2f}", className="m-value")]),
            html.Div([html.Div("Obstacles avoided", className="m-label"), html.Div(str(r["avoided"]), className="m-value")])
            if mode == config.PROTO else
            html.Div([html.Div("Open obstacles", className="m-label"), html.Div(str(r["obstacles"]), className="m-value")]),
        ]),
        html.Div(obstacle_txt, className="d-flex gap-1 flex-wrap mt-2") if obstacle_txt else None,
    ])


@callback(Output("pl-map", "figure"), Output("pl-results", "children"),
          Input("pl-start", "value"), Input("pl-end", "value"), Input("pl-alpha", "value"), Input("pl-risk", "value"),
          Input("pl-track-version", "data"), State("mode-switch", "value"))
def pl_compute(start, end, alpha, show_risk, _version, mode):
    mode = mode or config.REAL
    net = network.get_network(mode)
    if start not in net.nodes or end not in net.nodes:
        raise PreventUpdate
    routes, risk = planner.plan(mode, start, end, alpha or 0)
    # Draw the non-recommended routes first so the recommended one sits on top.
    drawn = [{"name": r["name"], "points": r["points"], "color": ROUTE_COLORS[r["name"]],
              "width": 6 if r["name"] == "Recommended" else 4,
              "opacity": 1.0 if r["name"] == "Recommended" else 0.85} for r in reversed(routes)]
    fig = network_figure(mode, net, routes=drawn, start=start, end=end, clickable_nodes=True,
                         show_edges=True, risk_edges=planner.risk_layers(risk, mode) if show_risk else None,
                         obstacles=obstacles.listing(mode, statuses=obstacles.SHOWN),
                         uirevision=f"pl-{mode}-{_version}")
    if not routes:
        return fig, ui.empty("No route between these two points.", "bi-sign-dead-end")
    cards = [_route_card(mode, r, i == 0) for i, r in enumerate(routes)]
    return fig, [html.Div(planner.recommendation(routes), className="route-note mb-2"), *cards]


# --- prototype: trajectories ---------------------------------------------------------------------
@callback(Output("pl-traj", "data"), Output("pl-traj", "selected_rows"),
          Input("pl-alpha", "value"), Input("pl-track-version", "data"), Input("pl-active-msg", "children"),
          Input("mode-switch", "value"))
def pl_trajectories(alpha, _version, _msg, mode):
    if mode != config.PROTO:
        raise PreventUpdate
    active = network.active_trajectory()
    rows = planner.rank_trajectories(alpha or 0)
    data = [{"id": r["name"], "rank": i + 1, "name": r["name"], "path": r["path"],
             "length_txt": ui.fmt_dist(config.PROTO, r["length"]), "eta_txt": ui.fmt_dur(r["eta"]),
             "obstacles": r["obstacles"], "risk_txt": f"{r['risk']:.2f}",
             "active": "● active" if active and active["name"] == r["name"] else ""} for i, r in enumerate(rows)]
    selected = [i for i, d in enumerate(data) if d["active"]]
    return data, selected


@callback(Output("pl-active-msg", "children"), Input("pl-set-active", "n_clicks"),
          State("pl-traj", "selected_row_ids"), prevent_initial_call=True)
def pl_set_active(_, selected):
    if not selected:
        return "Select a trajectory first."
    settings.set("active_trajectory", selected[0])
    return f"'{selected[0]}' is now the active run ({time.strftime('%H:%M:%S')})."


# --- prototype: track editor ----------------------------------------------------------------------
def _track_to_tables(data):
    nodes = [{"id": n["id"], "label": n.get("label", ""), "x": n["x"], "y": n["y"], "marker": n.get("marker")}
             for n in data["nodes"]]
    edges = [{"u": e["u"], "v": e["v"], "via": "; ".join(f"{p[0]:g},{p[1]:g}" for p in e.get("via", []))}
             for e in data["edges"]]
    trajs = [{"name": t["name"], "nodes": " > ".join(t["nodes"])} for t in data.get("trajectories", [])]
    return nodes, edges, trajs


def _tables_to_track(nodes, edges, trajs):
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    def parse_via(text):
        pts = []
        for part in str(text or "").split(";"):
            if part.strip():
                x, y = part.split(",")
                pts.append([float(x), float(y)])
        return pts

    return {
        "name": network.load_track_data().get("name", "Track"), "units": "cm",
        "nodes": [{"id": str(n.get("id", "")).strip(), "label": n.get("label") or str(n.get("id", "")),
                   "x": num(n.get("x")), "y": num(n.get("y")),
                   "marker": int(n["marker"]) if n.get("marker") not in (None, "") else None} for n in nodes],
        "edges": [{"u": str(e.get("u", "")).strip(), "v": str(e.get("v", "")).strip(), "via": parse_via(e.get("via"))}
                  for e in edges],
        "trajectories": [{"name": str(t.get("name", "")).strip(),
                          "nodes": [p.strip() for p in str(t.get("nodes", "")).split(">") if p.strip()]}
                         for t in trajs],
        "obstacles": network.load_track_data().get("obstacles", []),     # not edited here: kept as they are
    }


@callback(Output("pl-nodes", "data"), Output("pl-edges", "data"), Output("pl-trajs", "data"),
          Input("mode-switch", "value"), Input("pl-track-version", "data"))
def pl_load_track(mode, _version):
    if mode != config.PROTO:
        raise PreventUpdate
    return _track_to_tables(network.load_track_data())


@callback(Output("pl-nodes", "data", allow_duplicate=True), Output("pl-edges", "data", allow_duplicate=True),
          Output("pl-trajs", "data", allow_duplicate=True),
          Input("pl-add-node", "n_clicks"), Input("pl-add-edge", "n_clicks"), Input("pl-add-traj", "n_clicks"),
          State("pl-nodes", "data"), State("pl-edges", "data"), State("pl-trajs", "data"), prevent_initial_call=True)
def pl_add_rows(_a, _b, _c, nodes, edges, trajs):
    if ctx.triggered_id == "pl-add-node":
        used = {n.get("marker") for n in nodes}
        marker = next(m for m in range(50) if m not in used)
        return nodes + [{"id": f"N{len(nodes) + 1}", "label": "", "x": 0, "y": 0, "marker": marker}], no_update, no_update
    if ctx.triggered_id == "pl-add-edge":
        return no_update, edges + [{"u": "", "v": "", "via": ""}], no_update
    return no_update, no_update, trajs + [{"name": f"Trajectory {len(trajs) + 1}", "nodes": ""}]


@callback(Output("pl-save-msg", "children"), Output("pl-track-version", "data"),
          Input("pl-save", "n_clicks"), Input("pl-reset", "n_clicks"),
          State("pl-nodes", "data"), State("pl-edges", "data"), State("pl-trajs", "data"), prevent_initial_call=True)
def pl_save(_s, _r, nodes, edges, trajs):
    if ctx.triggered_id == "pl-reset":
        config.TRACK_PATH.unlink(missing_ok=True)
        obstacles.ensure_track_obstacles()
        return dbc.Alert("Default track restored.", color="info", className="mb-0"), time.time()
    try:
        data = _tables_to_track(nodes, edges, trajs)
    except ValueError:
        return dbc.Alert("Curve points must look like: 120,40; 150,60", color="danger", className="mb-0"), no_update
    errors = network.save_track(data)
    if errors:
        return dbc.Alert([html.Div("Track not saved:"), html.Ul([html.Li(e) for e in errors[:8]])],
                         color="danger", className="mb-0"), no_update
    obstacles.ensure_track_obstacles()
    return dbc.Alert("Track saved. Print new markers if you changed them.", color="success", className="mb-0"), time.time()


@callback(Output("pl-markers-dl", "data"), Input("pl-markers", "n_clicks"), prevent_initial_call=True)
def pl_markers(_):
    """Printable ArUco markers (one PNG per checkpoint). Print them ~5 cm wide next to the tape."""
    net = network.proto_network()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for nid, n in net.nodes.items():
            if n["marker"] is None:
                continue
            marker = cv2.aruco.generateImageMarker(dictionary, n["marker"], 500)
            page = np.full((680, 600), 255, np.uint8)
            page[50:550, 50:550] = marker
            cv2.putText(page, f"Checkpoint {nid}  -  marker #{n['marker']}", (50, 600), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, 0, 2, cv2.LINE_AA)
            cv2.putText(page, f"{config.ARUCO_DICT} - print 5 cm wide", (50, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        90, 1, cv2.LINE_AA)
            ok, png = cv2.imencode(".png", page)
            zf.writestr(f"checkpoint_{nid}_marker_{n['marker']}.png", png.tobytes())
    return dcc.send_bytes(buf.getvalue(), "aruco_checkpoints.zip")
