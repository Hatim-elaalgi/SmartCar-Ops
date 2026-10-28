"""Obstacles: a safety inbox. Detections sorted by priority (open first, then severity, then recency),
with a review workflow: New -> Acknowledged -> Resolved / False positive."""
import time

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html
from dash.exceptions import PreventUpdate

import config
from components import theme, ui
from components.figures import empty_figure, network_figure
from core import metrics, network, obstacles
from core.severity import LEVEL_META, LEVELS

dash.register_page(__name__, path="/obstacles", name="Obstacles", order=4,
                   description="Every detected obstacle, most urgent first. Review them to keep route risk accurate.")

TABS = {"review": ("Needs review", ["new"]), "ack": ("Acknowledged", ["ack"]), "avoided": ("Avoided", ["avoided"]),
        "closed": ("Closed", ["resolved", "false_positive"]), "all": ("All", None)}
RANGES = {"1h": 3600, "24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "all": None}

COLUMNS = [
    {"name": "Level", "id": "level_md", "presentation": "markdown"},
    {"name": "Score", "id": "severity", "type": "numeric", "format": {"specifier": ".0f"}},
    {"name": "Object", "id": "cls_txt"},
    {"name": "Conf.", "id": "conf_txt"},
    {"name": "Distance", "id": "dist_txt"},
    {"name": "In path", "id": "path_txt"},
    {"name": "Seen", "id": "seen", "type": "numeric"},
    {"name": "Last seen", "id": "last_txt"},
    {"name": "Status", "id": "status_txt"},
    {"name": "Car", "id": "car_id"},
]

layout = html.Div([
    dcc.Store(id="ob-refresh", data=0),
    html.Div(className="filter-bar", children=[
        dbc.RadioItems(id="ob-tab", value="review", inline=True, className="segmented",
                       inputClassName="btn-check", labelClassName="btn btn-segment", labelCheckedClassName="active",
                       options=[{"label": label, "value": key} for key, (label, _) in TABS.items()]),
        dbc.Checklist(id="ob-levels", value=[name for name, _ in LEVELS], inline=True, className="check-pills",
                      options=[{"label": LEVEL_META[name]["label"], "value": name} for name, _ in LEVELS]),
        dcc.Dropdown(id="ob-classes", multi=True, placeholder="All objects", style={"minWidth": "220px"}),
        dbc.Select(id="ob-range", value="7d", style={"width": "130px"},
                   options=[{"label": label, "value": key} for key, label in
                            (("1h", "Last hour"), ("24h", "Last 24 h"), ("7d", "Last 7 days"),
                             ("30d", "Last 30 days"), ("all", "All time"))]),
    ]),
    html.Div(id="ob-kpis", className="stat-grid"),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=8, children=ui.panel("Priority queue", flush=True, children=[
            dash_table.DataTable(id="ob-table", columns=COLUMNS, data=[], row_selectable="multi",
                                 sort_action="native", **ui.table_style(page_size=12)),
            html.Div(className="action-bar", children=[
                dbc.Button([html.I(className="bi bi-eye me-1"), "Acknowledge"], id="ob-ack", size="sm",
                           color="secondary", outline=True, disabled=True),
                dbc.Button([html.I(className="bi bi-check2-circle me-1"), "Resolve"], id="ob-resolve", size="sm",
                           color="secondary", outline=True, disabled=True),
                dbc.Button([html.I(className="bi bi-x-circle me-1"), "False positive"], id="ob-fp", size="sm",
                           color="secondary", outline=True, disabled=True),
                dbc.Button([html.I(className="bi bi-arrow-counterclockwise me-1"), "Reopen"], id="ob-reopen",
                           size="sm", color="secondary", outline=True, disabled=True),
                html.Span(id="ob-msg", className="hint ms-2"),
            ]),
        ])),
        dbc.Col(lg=4, children=ui.panel("Details", html.Div(id="ob-detail"))),
    ]),
    dbc.Row(className="g-3", children=[
        dbc.Col(lg=6, children=ui.panel("Where they are", dcc.Graph(id="ob-map", config={"displayModeBar": False},
                                                                     className="map-md"), flush=True)),
        dbc.Col(lg=6, children=ui.panel("By object and level", dcc.Graph(id="ob-bars", config={"displayModeBar": False},
                                                                          className="map-md"))),
    ]),
    ui.panel("How priority is computed", html.Div([
        html.P(["Score = 100 × ", html.B("danger of the object"), " × (0.5 + 0.5 × ", html.B("proximity"), ") × ",
                html.B("model confidence"), " × (1 if ", html.B("in the car's path"), " else 0.5)"]),
        html.Div(className="d-flex gap-2 flex-wrap mb-2", children=[
            ui.pill(f"{LEVEL_META[n]['label']} ≥ {t}", LEVEL_META[n]["color"], LEVEL_META[n]["icon"]) for n, t in LEVELS]),
        html.Div("The queue shows open obstacles first (New, then Acknowledged), then by score, then by most recent. "
                 "Repeated detections of the same object within a few metres are merged (“seen 5×”). "
                 "Marking false positives removes them from route risk and gives you labelled data to retrain the model. "
                 "Object danger weights are editable on Devices & logs → Settings.", className="hint"),
    ])),
])


@callback(Output("ob-classes", "options"), Output("ob-classes", "value"), Input("mode-switch", "value"))
def ob_class_options(mode):
    return [{"label": c.replace("_", " "), "value": c} for c in obstacles.classes(mode or config.REAL)], []


@callback(Output("ob-tab", "value"), Input("mode-switch", "value"))
def ob_default_tab(mode):
    """Prototype obstacles are the track's own ones, all avoided: show them first."""
    return "avoided" if mode == config.PROTO else "review"


@callback(Output("ob-table", "data"), Output("ob-table", "selected_rows"), Output("ob-kpis", "children"),
          Output("ob-map", "figure"), Output("ob-bars", "figure"), Output("ob-tab", "options"),
          Input("tick-slow", "n_intervals"), Input("mode-switch", "value"), Input("ob-tab", "value"),
          Input("ob-levels", "value"), Input("ob-classes", "value"), Input("ob-range", "value"),
          Input("ob-refresh", "data"), State("ob-table", "selected_row_ids"))
def ob_refresh(_, mode, tab, levels, classes, rng, _refresh, selected_ids):
    mode = mode or config.REAL
    seconds = RANGES.get(rng)
    since = time.time() - seconds if seconds else None
    base = obstacles.listing(mode, since=since, levels=levels or ["none"], classes=classes or None)
    statuses = TABS[tab][1]
    df = base[base["status"].isin(statuses)] if statuses else base

    data = [{
        "id": int(r.id), "level_md": ui.level_cell(r.level), "severity": r.severity,
        "cls_txt": r.cls.replace("_", " "), "conf_txt": f"{r.confidence:.0%}" if pd.notna(r.confidence) else "—",
        "dist_txt": ui.fmt_dist(mode, r.distance) if pd.notna(r.distance) else "—",
        "path_txt": "yes" if r.in_path else "side", "seen": int(r.seen), "last_txt": ui.ago(r.ts_last),
        "status_txt": ui.status_text(r.status), "car_id": r.car_id or "—",
    } for r in df.itertuples()]
    index = {d["id"]: i for i, d in enumerate(data)}
    selected_rows = [index[i] for i in (selected_ids or []) if i in index]

    counts = metrics.open_by_level(mode)
    reviewed = base[base["status"].isin(["ack", "resolved", "false_positive"])]
    fp_rate = (reviewed["status"] == "false_positive").mean() * 100 if len(reviewed) else None
    kpis = [ui.stat(f"Open {LEVEL_META[n]['label'].lower()}", str(counts[n]), None, LEVEL_META[n]["icon"],
                    LEVEL_META[n]["color"]) for n, _ in LEVELS]
    if mode == config.PROTO:      # the track obstacles: never open, the car always goes around them
        avoided = base[base["status"] == obstacles.AVOIDED]
        kpis.append(ui.stat("Avoided", str(len(avoided)), f"gone around {int(avoided['seen'].sum())}× by the car",
                            "bi-shuffle", theme.GOOD))
    kpis.append(ui.stat("Reviewed", str(len(reviewed)), "in the selected period", "bi-clipboard-check"))
    if mode == config.REAL:
        kpis.append(ui.stat("False positives", f"{fp_rate:.0f}%" if fp_rate is not None else "—",
                            "of reviewed detections", "bi-x-circle"))

    fig_map = network_figure(mode, network.get_network(mode), obstacles=df if len(df) else None, showlegend=True,
                             uirevision=f"ob-{mode}")

    if len(df):
        order = df["cls"].value_counts().sort_values().index.tolist()
        bars = go.Figure()
        for name, _ in reversed(LEVELS):  # low -> critical, left to right
            sub = df[df["level"] == name]["cls"].value_counts()
            bars.add_trace(go.Bar(orientation="h", y=[c.replace("_", " ") for c in order],
                                  x=[int(sub.get(c, 0)) for c in order], name=LEVEL_META[name]["label"],
                                  marker=dict(color=LEVEL_META[name]["color"], line=dict(color=theme.SURFACE, width=2)),
                                  hovertemplate="%{y}: %{x} " + LEVEL_META[name]["label"].lower() + "<extra></extra>"))
        bars.update_layout(barmode="stack", margin=dict(l=90, r=12, t=36, b=28), bargap=0.35,
                           legend=dict(traceorder="normal"))
    else:
        bars = empty_figure("No obstacles match the filters")

    tab_options = [{"label": f"{label} ({len(base[base['status'].isin(st)]) if st else len(base)})", "value": key}
                   for key, (label, st) in TABS.items()]
    return data, selected_rows, kpis, fig_map, bars, tab_options


@callback(Output("ob-ack", "disabled"), Output("ob-resolve", "disabled"), Output("ob-fp", "disabled"),
          Output("ob-reopen", "disabled"), Input("ob-table", "selected_row_ids"))
def ob_buttons(selected):
    off = not selected
    return off, off, off, off


@callback(Output("ob-refresh", "data"), Output("ob-msg", "children"),
          Input("ob-ack", "n_clicks"), Input("ob-resolve", "n_clicks"), Input("ob-fp", "n_clicks"),
          Input("ob-reopen", "n_clicks"), State("ob-table", "selected_row_ids"), State("mode-switch", "value"),
          prevent_initial_call=True)
def ob_action(_a, _b, _c, _d, selected, mode):
    if not selected:
        raise PreventUpdate
    status = {"ob-ack": "ack", "ob-resolve": "resolved", "ob-fp": "false_positive", "ob-reopen": "new"}[ctx.triggered_id]
    obstacles.set_status([int(i) for i in selected], status, mode)
    return time.time(), f"{len(selected)} marked as {ui.status_text(status).lower()}."


@callback(Output("ob-detail", "children"), Input("ob-table", "selected_row_ids"), Input("ob-table", "active_cell"),
          Input("ob-refresh", "data"), State("mode-switch", "value"))
def ob_detail(selected, active, _refresh, mode):
    mode = mode or config.REAL
    oid = (selected or [None])[-1] or (active or {}).get("row_id")
    if not oid:
        return ui.empty("Select an obstacle to see its details", "bi-hand-index")
    o = obstacles.get(int(oid))
    if not o:
        return ui.empty("This obstacle no longer exists")
    img = (html.Img(src=f"/snapshots/{o['snapshot']}", className="detail-img") if o["snapshot"] else
           html.Div(className="detail-placeholder", children=[html.I(className="bi bi-image"),
                                                              "No snapshot (simulated detection)"]))
    return html.Div([
        img,
        html.Div(className="car-head", children=[html.Span(o["cls"].replace("_", " ").capitalize(), className="name"),
                                                 ui.level_pill(o["level"])]),
        ui.kv("Score", f"{o['severity']:.0f} / 100"),
        ui.kv("Confidence", f"{o['confidence']:.0%}" if o["confidence"] is not None else "—"),
        ui.kv("Proximity", f"{o['proximity']:.0%}" if o["proximity"] is not None else "—"),
        ui.kv("Distance", ui.fmt_dist(mode, o["distance"])),
        ui.kv("In the car's path", "yes" if o["in_path"] else "no (side)"),
        ui.kv("Seen", f"{o['seen']}×"),
        ui.kv("First seen", ui.fmt_time(o["ts_first"])),
        ui.kv("Last seen", ui.ago(o["ts_last"])),
        ui.kv("Status", ui.status_text(o["status"])),
        ui.kv("Car / trip", f"{o['car_id'] or '—'} / {o['trip_id'] or '—'}", mono=True),
        ui.kv("Source", o["source"]),
    ])
