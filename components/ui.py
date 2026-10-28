"""Reusable UI pieces: panels, stat tiles, status pills, formatting helpers, DataTable styling."""
import time
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import html

import config
from components import theme
from core.severity import LEVEL_META, STATUS_META


# --- formatting ----------------------------------------------------------------------------
def ago(ts):
    if not ts:
        return "—"
    s = max(0, time.time() - ts)
    if s < 60:
        return f"{s:.0f} s ago"
    if s < 3600:
        return f"{s / 60:.0f} min ago"
    if s < 86400:
        return f"{s / 3600:.0f} h ago"
    return f"{s / 86400:.0f} d ago"


def fmt_time(ts, with_date=True):
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%d %b %H:%M" if with_date else "%H:%M:%S")


def fmt_dur(seconds):
    if seconds is None:
        return "—"
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


def fmt_dist(mode, value):
    if value is None:
        return "—"
    if mode == config.REAL:
        return f"{value / 1000:.1f} km" if value >= 1000 else f"{value:.0f} m"
    return f"{value / 100:.1f} m" if value >= 1000 else f"{value:.0f} cm"


def fmt_speed(mode, value):
    if value is None:
        return "—"
    return f"{value * 3.6:.0f} km/h" if mode == config.REAL else f"{value:.0f} cm/s"


def compact(n):
    if n is None:
        return "—"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 10_000:
        return f"{n / 1000:.0f}K"
    if abs(n) >= 1000:
        return f"{n / 1000:.1f}K"
    return f"{n:,.0f}" if isinstance(n, (int, float)) and float(n).is_integer() else f"{n:.1f}"


# --- building blocks -----------------------------------------------------------------------
def panel(title=None, children=None, *, actions=None, subtitle=None, className="", body_class="", flush=False):
    # Note: `is not None` matters — an empty Dash component (e.g. a placeholder filled by a callback) is falsy.
    header = None
    if title is not None or actions is not None:
        header = html.Div(className="panel-head", children=[
            html.Div([html.Div(title, className="panel-title"),
                      html.Div(subtitle, className="panel-sub") if subtitle else None]),
            html.Div(actions, className="panel-actions") if actions is not None else None,
        ])
    return html.Section(className=f"panel {className}", children=[
        header, html.Div(children, className=f"panel-body {'flush' if flush else ''} {body_class}"),
    ])


def stat(label, value, sub=None, icon=None, tone=None, live=False):
    return html.Div(className="stat", children=[
        html.Div(className="stat-top", children=[
            html.Span(label, className="stat-label"),
            html.I(className=f"bi {icon} stat-icon", style={"color": tone or theme.MUTED}) if icon else None,
        ]),
        html.Div(value, className="stat-value" + (" tnum" if live else "")),
        html.Div(sub, className="stat-sub") if sub else None,
    ])


def pill(label, color, icon=None, className=""):
    return html.Span(className=f"pill {className}", children=[
        html.I(className=f"bi {icon}", style={"color": color}) if icon else html.Span(className="dot",
                                                                                     style={"background": color}),
        html.Span(label),
    ])


def state_pill(state):
    m = theme.state_meta(state)
    return pill(m["label"], m["color"], m["icon"])


def level_pill(level):
    m = LEVEL_META.get(level, LEVEL_META["low"])
    return pill(m["label"], m["color"], m["icon"])


def status_text(status):
    return STATUS_META.get(status, status)


def source_pill(source):
    m = theme.SOURCE_META.get(source, theme.SOURCE_META["offline"])
    return pill(m["label"], m["color"], m["icon"], className="pill-source")


def kv(label, value, mono=False):
    return html.Div(className="kv", children=[
        html.Span(label, className="kv-label"),
        html.Span(value, className="kv-value" + (" mono" if mono else "")),
    ])


def empty(text, icon="bi-inbox"):
    return html.Div(className="empty", children=[html.I(className=f"bi {icon}"), html.Div(text)])


def ring(fraction, center_text, caption=None, color=theme.ACCENT):
    """Progress ring (mission / trip progress)."""
    pct = 0 if fraction is None else max(0.0, min(1.0, fraction)) * 100
    return html.Div(className="ring-wrap", children=html.Div(className="ring", children=[
        html.Div(className="ring-track", style={
            "background": f"conic-gradient({color} {pct:.1f}%, rgba(255,255,255,0.08) {pct:.1f}% 100%)"}),
        html.Div(className="ring-inner", children=[
            html.Div(center_text, className="ring-value"),
            html.Div(caption, className="ring-caption") if caption else None,
        ]),
    ]))


def legend_item(label, color, shape="dot"):
    return html.Span(className="legend-item", children=[html.Span(className=f"legend-{shape}",
                                                                  style={"background": color}), label])


def info_note(children, icon="bi-info-circle"):
    return dbc.Alert([html.I(className=f"bi {icon} me-2"), children], color="secondary", className="note")


# --- DataTable dark styling ------------------------------------------------------------------
def table_style(page_size=10):
    return dict(
        page_size=page_size,
        style_as_list_view=True,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "rgba(0,0,0,0)", "color": theme.INK_1, "border": "none",
                    "borderBottom": f"1px solid {theme.GRID}", "fontFamily": theme.FONT, "fontSize": 13,
                    "padding": "8px 10px", "textAlign": "left", "whiteSpace": "nowrap",
                    "fontVariantNumeric": "tabular-nums"},
        style_header={"backgroundColor": "rgba(0,0,0,0)", "color": theme.MUTED, "fontWeight": 600, "fontSize": 12,
                      "border": "none", "borderBottom": f"1px solid {theme.BASELINE}"},
        style_data_conditional=[
            {"if": {"state": "active"}, "backgroundColor": "rgba(255,255,255,0.06)", "border": "none",
             "borderBottom": f"1px solid {theme.GRID}"},
            {"if": {"state": "selected"}, "backgroundColor": "rgba(57,135,229,0.14)", "border": "none",
             "borderBottom": f"1px solid {theme.GRID}"},
        ],
        css=[{"selector": ".dash-spreadsheet-menu", "rule": "display: none"},
             {"selector": "tr:hover td", "rule": "background-color: rgba(255,255,255,0.045) !important"}],
        markdown_options={"html": True},
    )


def level_cell(level):
    m = LEVEL_META.get(level, LEVEL_META["low"])
    return f'<span class="lvl" style="color:{m["color"]}">●</span> {m["label"]}'


def state_cell(state):
    m = theme.state_meta(state)
    return f'<span class="lvl" style="color:{m["color"]}">●</span> {m["label"]}'
