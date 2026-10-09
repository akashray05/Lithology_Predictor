"""Vivid colour theme + legend styling for the Lithology Predictor UI.

Applies on top of figures built by src/plots.py and src/plots_extra.py, so the
original plotting code stays untouched.
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.plots import LITH_ORDER, MODEL_COLORS

VIVID_COLORS = {
    "Sandstone": "#FFC800",
    "Sandstone/Shale": "#8BD13F",
    "Shale": "#5E7391",
    "Marl": "#A64DFF",
    "Limestone": "#00D1E0",
    "Chalk": "#E8C9A0",
    "Dolomite": "#1E5BFF",
    "Anhydrite": "#FF6B35",
    "Halite": "#FF2E93",
    "Coal": "#111111",
    "Basement": "#8B4513",
    "Tuff": "#00A86B",
}
assert set(VIVID_COLORS) == set(LITH_ORDER)

# The display name 'Ex-Tree' (internal key W5) keeps W5's colour in every chart.
MODEL_COLORS.setdefault("Ex-Tree", MODEL_COLORS.get("W5", "#1F77B4"))

CORRECT_COLOR = "#2ECC71"
WRONG_COLOR = "#E63946"
_STATUS_COLORS = {"Correct": CORRECT_COLOR, "Wrong": WRONG_COLOR, "No original label": "#FFFFFF"}


def _discrete_scale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale: list[list] = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    return scale


_LITH_SCALE = _discrete_scale([VIVID_COLORS[n] for n in LITH_ORDER])
_ERR_SCALE = [[0, CORRECT_COLOR], [0.5, CORRECT_COLOR], [0.5, WRONG_COLOR], [1, WRONG_COLOR]]


def style_figure(fig: go.Figure, legend_font: int = 15, swatch: int = 20,
                 right_margin: int | None = 190) -> go.Figure:
    """Recolour lithology/error tracks and enlarge the legend."""
    top = len(LITH_ORDER) - 0.5

    def fix(t):
        if t.type == "heatmap" and t.zmax is not None:
            if abs(float(t.zmax) - top) < 1e-6:
                t.colorscale = _LITH_SCALE          # lithology track
            elif float(t.zmax) == 1.0 and float(t.zmin or 0) == 0.0:
                t.colorscale = _ERR_SCALE           # error strip
        elif t.type == "scatter" and t.mode == "markers" and t.x is not None \
                and len(t.x) == 1 and t.x[0] is None:  # legend-only dummy trace
            if t.name in VIVID_COLORS:
                t.marker.color = VIVID_COLORS[t.name]
            elif t.name in _STATUS_COLORS:
                t.marker.color = _STATUS_COLORS[t.name]
            t.marker.size = swatch
            if "open" not in str(t.marker.symbol):
                t.marker.line = dict(width=1.2, color="#222")

    fig.for_each_trace(fix)
    fig.update_layout(
        legend=dict(
            font=dict(size=legend_font), title=dict(font=dict(size=legend_font + 1)),
            itemsizing="constant", tracegroupgap=8,
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#888", borderwidth=1,
        ),
    )
    if right_margin:
        fig.update_layout(margin=dict(r=right_margin))
    return fig
