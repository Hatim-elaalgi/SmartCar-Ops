"""Design tokens + Plotly template. Palette = validated dark data-viz palette (categorical order matters
for colour-blind safety: never cycle or reorder). Status colours are reserved for state / severity."""
import plotly.graph_objects as go
import plotly.io as pio

PLANE = "#07080b"
# Panels are translucent "liquid glass"; SURFACE is their effective colour over the dark backdrop
# (used for gaps between bar segments and solid fills). It stays close to the validated dark surface.
SURFACE = "#18191e"
SURFACE_2 = "#22242a"
INK_1 = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#8d8c86"
GRID = "rgba(255,255,255,0.07)"
BASELINE = "rgba(255,255,255,0.16)"
ACCENT = "#3987e5"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
MONO = 'ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace'

# Categorical slots, dark steps, fixed order.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
DODGE_PATH = "#ff8a1f"           # estimated detour path of a car going around an obstacle (dotted line)

STATE_META = {
    "CRUISING":  {"label": "Cruising",  "color": GOOD,     "icon": "bi-play-circle-fill"},
    "AVOIDING":  {"label": "Avoiding",  "color": WARNING,  "icon": "bi-shuffle"},
    "STOPPED":   {"label": "Stopped",   "color": CRITICAL, "icon": "bi-stop-circle-fill"},
    "LINE_LOST": {"label": "Line lost", "color": SERIOUS,  "icon": "bi-question-circle-fill"},
    "PARKED":    {"label": "Parked",    "color": MUTED,    "icon": "bi-p-circle-fill"},
    "CHARGING":  {"label": "Charging",  "color": ACCENT,   "icon": "bi-battery-charging"},
    "OFFLINE":   {"label": "Offline",   "color": MUTED,    "icon": "bi-wifi-off"},
}

SOURCE_META = {
    "simulated": {"label": "Simulated data", "color": ACCENT, "icon": "bi-cpu-fill"},
    "camera":    {"label": "Live · ESP32-CAM", "color": GOOD, "icon": "bi-broadcast"},
    "demo":      {"label": "Demo (camera offline)", "color": WARNING, "icon": "bi-cpu-fill"},
    "offline":   {"label": "Offline", "color": MUTED, "icon": "bi-wifi-off"},
}


def state_meta(state):
    return STATE_META.get(state or "OFFLINE", STATE_META["OFFLINE"])


def register_template():
    axis = dict(gridcolor=GRID, linecolor=BASELINE, zeroline=False, ticks="", showline=False,
                tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=MUTED, size=11)))
    pio.templates["iot_dark"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, color=INK_2, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        xaxis=dict(axis, showline=True), yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=INK_2, size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="rgba(24,25,31,0.94)", bordercolor="rgba(255,255,255,0.18)",
                        font=dict(color=INK_1, family=FONT, size=12)),
        margin=dict(l=44, r=12, t=28, b=32),
        bargap=0.45, barcornerradius=4,
        hovermode="closest",
    ))
    pio.templates.default = "iot_dark"
