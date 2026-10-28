"""AI models: offline evaluation (from models/registry.json) and live behaviour (from reviewed detections)."""
import json
import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import config
from components import theme, ui
from components.figures import empty_figure
from core import db
from sources.services import camera, dashcam

dash.register_page(__name__, path="/models", name="AI models", order=6,
                   description="How good each model is offline, and how it behaves on the road.")

GRAPH = {"config": {"displayModeBar": False}}

layout = html.Div([html.Div(id="md-cards"), html.Div(id="md-live")])


def _registry():
    try:
        return json.loads(config.MODEL_REGISTRY_PATH.read_text(encoding="utf-8")).get("models", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _per_class_fig(per_class):
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(orientation="h", y=[k.replace("_", " ") for k, _ in items], x=[v for _, v in items],
                           marker_color=theme.ACCENT, text=[f"{v:.2f}" for _, v in items], textposition="outside",
                           textfont=dict(color=theme.INK_2), cliponaxis=False,
                           hovertemplate="%{y}: AP %{x:.2f}<extra></extra>"))
    fig.update_layout(margin=dict(l=90, r=36, t=8, b=28), xaxis=dict(range=[0, 1], tickformat=".1f"), bargap=0.4)
    return fig


def _confusion_fig(conf):
    labels = [l.replace("_", " ") for l in conf["labels"]]
    m = np.array(conf["matrix"], dtype=float)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
                               colorscale=[[i / (len(theme.SEQ_BLUE) - 1), c] for i, c in
                                           enumerate(reversed(theme.SEQ_BLUE))],
                               customdata=m, hovertemplate="true %{y} → predicted %{x}<br>%{customdata:.0f} "
                                                           "(%{z:.0%} of row)<extra></extra>"))
    # Readable labels: light ink on dark cells, dark ink on light cells.
    for i, row in enumerate(norm):
        for j, v in enumerate(row):
            if v >= 0.01:
                fig.add_annotation(x=labels[j], y=labels[i], text=f"{v:.0%}", showarrow=False,
                                   font=dict(size=10, color=theme.PLANE if v > 0.55 else theme.INK_1))
    fig.update_layout(margin=dict(l=90, r=8, t=8, b=70), yaxis=dict(autorange="reversed", title="true"),
                      xaxis=dict(title="predicted", tickangle=-35))
    return fig


def _metric_row(metrics):
    tiles = [("Precision", metrics.get("precision"), "{:.2f}"), ("Recall", metrics.get("recall"), "{:.2f}"),
             ("mAP@50", metrics.get("map50"), "{:.2f}"), ("mAP@50-95", metrics.get("map50_95"), "{:.2f}"),
             ("Inference (GPU)", metrics.get("inference_ms"), "{:.0f} ms")]
    return html.Div(className="metric-row", children=[
        html.Div(className="metric", children=[html.Div(label, className="m-label"),
                                               html.Div(fmt.format(v) if v is not None else "—", className="m-value")])
        for label, v, fmt in tiles])


def _model_card(m):
    metrics = m.get("metrics", {})
    best = m.get("metrics_best")
    ds = m.get("dataset", {})
    images = ds.get("images")
    badge = (html.Span("Example metrics — edit models/registry.json", className="badge-warn") if m.get("example") else
             html.Span("Training pending", className="badge-warn") if m.get("trained") == "pending" else
             html.Span("Evaluated on held-out test data", className="badge-soft") if metrics else
             html.Span("Pretrained", className="badge-soft"))
    if not m.get("per_class") and not m.get("confusion"):
        charts = None
    else:
        charts = dbc.Row(className="g-3", children=[
                dbc.Col(lg=5, children=[html.Div("Average precision per class", className="field-label"),
                                        dcc.Graph(figure=_per_class_fig(m.get("per_class", {})), className="chart",
                                                  **GRAPH) if m.get("per_class") else ui.empty("No per-class AP")]),
                dbc.Col(lg=7, children=[html.Div("Confusion matrix (share of each true class" +
                                                 (f" · confidence ≥ {m['confusion']['conf']}, "
                                                  f"{m['confusion'].get('imgsz', '?')} px" if
                                                  m.get("confusion", {}).get("conf") else "") + ")",
                                                 className="field-label"),
                                        dcc.Graph(figure=_confusion_fig(m["confusion"]), style={"height": "340px"},
                                                  **GRAPH) if m.get("confusion") else ui.empty("No confusion matrix")]),
            ])
    training = m.get("training") or {}
    return ui.panel(
        m["name"], subtitle=f"{m.get('framework', '')} · v{m.get('version', '?')} · trained {m.get('trained', '?')}",
        actions=badge,
        children=[
            html.Div(className="model-meta", children=[
                ui.pill(f"Input {m.get('input', '?')}", theme.MUTED, "bi-aspect-ratio"),
                ui.pill(f"{images:,} images" if isinstance(images, int) else f"{images or '?'} images", theme.MUTED,
                        "bi-images"),
                ui.pill(f"Split {ds.get('split', '?')}", theme.MUTED, "bi-pie-chart"),
                ui.pill(f"{training['epochs']} epochs · {training['minutes']:.0f} min on {training['gpu']}",
                        theme.MUTED, "bi-gpu-card") if training else None,
            ]),
            html.Div(className="mb-2", children=[html.Div(f"Data: {s}", className="hint") for s in ds.get("sources", [])]),
            html.Div(f"Best inference setting · {best['setting']} (test split, see training/report/metrics.md)",
                     className="field-label") if best else None,
            _metric_row(best) if best else None,
            html.Div("Real-time setting · 640 px, used live in the dashboard (test split)", className="field-label")
            if best and metrics else None,
            _metric_row(metrics) if metrics else None,
            *[html.Div([html.Div(v["label"], className="field-label"), _metric_row(v["metrics"])])
              for v in m.get("metrics_variants", [])],
            html.Div(className="model-meta", children=[ui.pill(c, theme.ACCENT, "bi-tag")
                                                       for c in m.get("classes_used", [])]) if m.get("classes_used")
            else None,
            html.Div(f"Metrics: {m['metrics_source']}", className="hint mb-2") if m.get("metrics_source") else None,
            charts,
        ])


@callback(Output("md-cards", "children"), Input("mode-switch", "value"))
def md_cards(mode):
    mode = mode or config.REAL
    models = [m for m in _registry() if m.get("mode") == mode]
    if not models:
        return ui.panel("Models", ui.empty("No model registered for this mode in models/registry.json", "bi-cpu"))
    return [_model_card(m) for m in models]


@callback(Output("md-live", "children"), Input("tick-slow", "n_intervals"), Input("mode-switch", "value"))
def md_live(_, mode):
    mode = mode or config.REAL
    since = time.time() - 7 * 86400
    df = db.query_df("SELECT cls, confidence, status, source FROM obstacles WHERE mode=? AND ts_last >= ?",
                     (mode, since))
    reviewed = df[df["status"].isin(["ack", "resolved", "false_positive"])]
    fp = reviewed[reviewed["status"] == "false_positive"]
    fp_rate = len(fp) / len(reviewed) if len(reviewed) else None
    if mode == config.PROTO:
        runtime, runtime_detail = (camera.detector.name if camera.detector else "—"), "on the ESP32-CAM stream"
    else:
        info = dashcam.info()
        names = [m["id"] for m in info["models"]]
        runtime, runtime_detail = (" + ".join(names), f"dashcam of {config.DASHCAM_CAR_ID}") if names else \
            ("Not configured", "run the training notebook")

    if len(df):
        hist = go.Figure(go.Histogram(x=df["confidence"], xbins=dict(start=0.5, end=1.0, size=0.05),
                                      marker=dict(color=theme.ACCENT, line=dict(color=theme.SURFACE, width=2)),
                                      hovertemplate="%{x}: %{y} detections<extra></extra>"))
        hist.update_layout(margin=dict(l=36, r=8, t=8, b=28), xaxis=dict(tickformat=".0%"), bargap=0.1)
    else:
        hist = empty_figure("No detections in the last 7 days")

    if len(reviewed):
        rate = (reviewed.assign(fp=reviewed["status"] == "false_positive").groupby("cls")["fp"].mean()
                .sort_values())
        fp_fig = go.Figure(go.Bar(orientation="h", y=[c.replace("_", " ") for c in rate.index], x=rate.values,
                                  marker_color=theme.SERIES[1], text=[f"{v:.0%}" for v in rate.values],
                                  textposition="outside", textfont=dict(color=theme.INK_2), cliponaxis=False,
                                  hovertemplate="%{y}: %{x:.0%} false positives<extra></extra>"))
        fp_fig.update_layout(margin=dict(l=90, r=40, t=8, b=28), xaxis=dict(tickformat=".0%", rangemode="tozero"),
                             bargap=0.4)
    else:
        fp_fig = empty_figure("Review obstacles to measure false positives")

    return ui.panel("On the road · last 7 days", subtitle="From detections and your reviews on the Obstacles page",
                    children=[
                        html.Div(className="stat-grid", children=[
                            ui.stat("Detections", ui.compact(len(df)), None, "bi-bounding-box"),
                            ui.stat("Mean confidence", f"{df['confidence'].mean():.0%}" if len(df) else "—", None,
                                    "bi-bullseye"),
                            ui.stat("Reviewed", ui.compact(len(reviewed)), None, "bi-clipboard-check"),
                            ui.stat("False-positive rate", f"{fp_rate:.0%}" if fp_rate is not None else "—",
                                    f"{len(fp)} flagged", "bi-x-circle"),
                            ui.stat("Model in use", html.Span(runtime, style={"fontSize": "15px"}), runtime_detail,
                                    "bi-cpu"),
                            ui.stat("Labelled for retraining", ui.compact(len(fp)), "false positives to relabel",
                                    "bi-tags"),
                        ]),
                        dbc.Row(className="g-3", children=[
                            dbc.Col(lg=6, children=[html.Div("Confidence of detections", className="field-label"),
                                                    dcc.Graph(figure=hist, className="chart", **GRAPH)]),
                            dbc.Col(lg=6, children=[html.Div("False-positive rate by object", className="field-label"),
                                                    dcc.Graph(figure=fp_fig, className="chart", **GRAPH)]),
                        ]),
                    ])
