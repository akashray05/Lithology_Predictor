"""Figure chrome and lithology colours for the Streamlit UI.

Colours follow the FORCE 2020 lithofacies convention rather than a
dashboard palette. Plot styling is applied on top of figures from
src/plots.py and src/plots_extra.py.
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.plots import LITH_ORDER, MODEL_COLORS

# FORCE 2020 lithofacies colours (competition notebook / NPD-style tracks).
LITH_COLORS = {
    "Sandstone": "#F4D03F",
    "Sandstone/Shale": "#F5B041",
    "Shale": "#32567A",
    "Marl": "#7D9B4E",
    "Limestone": "#5DADE2",
    "Chalk": "#F7E7B4",
    "Dolomite": "#5B6EA6",
    "Anhydrite": "#E67E22",
    "Halite": "#D7A6C7",
    "Coal": "#1A1A1A",
    "Basement": "#6C3483",
    "Tuff": "#07760D",
}
assert set(LITH_COLORS) == set(LITH_ORDER)

# Kept as an alias so older tests and imports keep working.
VIVID_COLORS = LITH_COLORS

MODEL_COLORS.setdefault("Ex-Tree", MODEL_COLORS.get("W5", "#1F4E79"))

CORRECT_COLOR = "#2F6B4F"
WRONG_COLOR = "#8B3A2A"
_STATUS_COLORS = {"Correct": CORRECT_COLOR, "Wrong": WRONG_COLOR, "No original label": "#FFFFFF"}

FIGURE_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#F7F3EB",
    plot_bgcolor="#F7F3EB",
    font=dict(family="Iowan Old Style, Palatino Linotype, Palatino, Georgia, serif",
              color="#1E2420", size=12),
    coloraxis_colorbar=dict(outlinewidth=0),
)


def _discrete_scale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale: list[list] = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    return scale


_LITH_SCALE = _discrete_scale([LITH_COLORS[n] for n in LITH_ORDER])
_ERR_SCALE = [[0, CORRECT_COLOR], [0.5, CORRECT_COLOR], [0.5, WRONG_COLOR], [1, WRONG_COLOR]]


def style_figure(fig: go.Figure, legend_font: int = 13, swatch: int = 20,
                 right_margin: int | None = 190) -> go.Figure:
    """Recolour lithology/error tracks and set report-style chrome."""
    top = len(LITH_ORDER) - 0.5

    def fix(t):
        if t.type == "heatmap" and t.zmax is not None:
            if abs(float(t.zmax) - top) < 1e-6:
                t.colorscale = _LITH_SCALE
            elif float(t.zmax) == 1.0 and float(t.zmin or 0) == 0.0:
                t.colorscale = _ERR_SCALE
        elif t.type == "scatter" and t.mode == "markers" and t.x is not None \
                and len(t.x) == 1 and t.x[0] is None:
            if t.name in LITH_COLORS:
                t.marker.color = LITH_COLORS[t.name]
            elif t.name in _STATUS_COLORS:
                t.marker.color = _STATUS_COLORS[t.name]
            t.marker.size = swatch
            if "open" not in str(t.marker.symbol):
                t.marker.line = dict(width=0.8, color="#2a2620")

    fig.for_each_trace(fix)
    fig.update_layout(
        **FIGURE_LAYOUT,
        legend=dict(
            font=dict(size=legend_font), title=dict(font=dict(size=legend_font)),
            itemsizing="constant", tracegroupgap=6,
            bgcolor="rgba(247,243,235,0.94)", bordercolor="#C4BBA8", borderwidth=1,
        ),
    )
    if right_margin:
        fig.update_layout(margin=dict(r=right_margin))
    return fig
