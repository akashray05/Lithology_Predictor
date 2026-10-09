"""Plotly figures for the Lithology Predictor app.

Lithology tracks are drawn as continuous colour columns (heatmaps), like a
real well-log lithology track, instead of thousands of separate markers.
Only the models passed in are plotted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.inference import LITHOLOGY_NAMES

# --------------------------------------------------------------------------
# Colours / ordering (one place, reused by every figure)
# --------------------------------------------------------------------------
LITH_ORDER = [
    "Sandstone", "Sandstone/Shale", "Shale", "Marl", "Limestone", "Chalk",
    "Dolomite", "Anhydrite", "Halite", "Coal", "Basement", "Tuff",
]
assert set(LITH_ORDER) == set(LITHOLOGY_NAMES.values())

LITHOLOGY_COLORS = {
    "Sandstone": "#F2C94C",
    "Sandstone/Shale": "#9BD35A",
    "Shale": "#6F8196",
    "Marl": "#9B59D0",
    "Limestone": "#33C6D6",
    "Chalk": "#D9D2B6",
    "Dolomite": "#2F6FDE",
    "Anhydrite": "#F0845C",
    "Halite": "#E8489A",
    "Coal": "#1C1C1C",
    "Basement": "#8B5A3C",
    "Tuff": "#2E8B57",
}
NAME_TO_IDX = {name: i for i, name in enumerate(LITH_ORDER)}

MODEL_COLORS = {"W5": "#1F77B4", "XGBoost": "#D62728", "LightGBM": "#2CA02C", "True": "#222222"}
CURVE_COLORS = {"GR": "#2E7D32", "RDEP": "#C62828", "RMED": "#EF6C00",
                "DTC": "#1565C0", "RHOB": "#6A1B9A"}
LOG_CURVES = {"RDEP", "RMED"}

ERROR_COLORS = ("#CFE8D5", "#D64545")   # correct, wrong
_ERROR_SCALE = [[0, ERROR_COLORS[0]], [0.5, ERROR_COLORS[0]],
                [0.5, ERROR_COLORS[1]], [1, ERROR_COLORS[1]]]


def _discrete_colorscale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale = []
    for i, color in enumerate(colors):
        scale += [[i / n, color], [(i + 1) / n, color]]
    return scale


_LITH_SCALE = _discrete_colorscale([LITHOLOGY_COLORS[n] for n in LITH_ORDER])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _insert_gaps(depth, arrays, fills, gap_factor: float = 5.0):
    """Insert an empty sample where depth jumps (missing interval) so the
    heatmap does not smear a colour across the gap."""
    depth = np.asarray(depth, dtype=float)
    arrays = [np.asarray(a) for a in arrays]
    if len(depth) < 3:
        return depth, arrays

    steps = np.diff(depth)
    positive = steps[steps > 0]
    typical = float(np.median(positive)) if len(positive) else 1.0
    gaps = np.where(steps > gap_factor * typical)[0]
    if len(gaps) == 0:
        return depth, arrays

    new_depth = np.insert(depth, gaps + 1, depth[gaps] + typical)
    new_arrays = [np.insert(a, gaps + 1, fill) for a, fill in zip(arrays, fills)]
    return new_depth, new_arrays


def _heatmap_track(depth, z, text, colorscale, zmin, zmax, label):
    z = np.asarray(z, dtype=float)
    text = np.asarray(text, dtype=object)
    depth2, (z2, text2) = _insert_gaps(depth, [z, text], [np.nan, ""])
    return go.Heatmap(
        z=z2[:, None],
        x=[0],
        y=depth2,
        text=text2[:, None],
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        showscale=False,
        hoverongaps=False,
        hovertemplate=f"<b>{label}</b>: %{{text}}<br>Depth %{{y:.2f}}<extra></extra>",
    )


def _lithology_track(depth, names, label):
    names = np.asarray(pd.Series(names).where(pd.notna(names), "—"), dtype=object)
    idx = np.array([NAME_TO_IDX.get(n, np.nan) for n in names], dtype=float)
    return _heatmap_track(depth, idx, names, _LITH_SCALE, -0.5, len(LITH_ORDER) - 0.5, label)


def _error_track(depth, true_code, pred_code, label):
    true_code = np.asarray(true_code, dtype=float)
    pred_code = np.asarray(pred_code, dtype=float)
    unlabeled = np.isnan(true_code)
    wrong = np.where(unlabeled, np.nan, (true_code != pred_code).astype(float))
    text = np.where(unlabeled, "unlabelled", np.where(wrong == 1, "Wrong", "Correct"))
    return _heatmap_track(depth, wrong, text, _ERROR_SCALE, 0, 1, label)


def _accuracy(d: pd.DataFrame, key: str) -> float | None:
    if "TRUE_CODE" not in d.columns:
        return None
    labelled = d["TRUE_CODE"].notna()
    if not labelled.any():
        return None
    return float(
        (d.loc[labelled, f"{key}_PREDICTION_CODE"].to_numpy(float)
         == d.loc[labelled, "TRUE_CODE"].to_numpy(float)).mean()
    )


# --------------------------------------------------------------------------
# Main figure: depth tracks
# --------------------------------------------------------------------------
def build_tracks_figure(
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    depth_col: str = "DEPT",
    depth_label: str = "Depth",
    depth_range: tuple[float, float] | None = None,
    show_true: bool = True,
    show_errors: bool = True,
    show_confidence: bool = False,
    curves: list[str] | tuple[str, ...] = (),
    height: int = 950,
    title: str | None = None,
) -> go.Figure:
    """Curves | True | (Prediction [+ Error]) per selected model | Confidence.

    `models` is a list of (display_name, column_key), e.g. ("XGBoost", "XGBOOST").
    Only these models are drawn.
    """
    d = df.copy()
    d[depth_col] = pd.to_numeric(d[depth_col], errors="coerce")
    d = d.dropna(subset=[depth_col]).sort_values(depth_col)
    if depth_range is not None:
        d = d[d[depth_col].between(depth_range[0], depth_range[1])]
    if d.empty:
        raise ValueError("No samples in the selected depth range.")

    depth = d[depth_col].to_numpy(float)
    # Ground truth exists only if at least one visible sample has a decoded label;
    # an empty/"None" label column is NOT ground truth.
    has_true = (
        "TRUE_LITHOLOGY" in d.columns
        and "TRUE_CODE" in d.columns
        and bool(d["TRUE_CODE"].notna().any())
    )
    # Error strips compare model vs original, so they need the original track.
    draw_errors = has_true and show_true and show_errors
    models = [(n, k) for n, k in models if f"{k}_PREDICTION" in d.columns]
    conf_models = [(n, k) for n, k in models if f"{k}_CONFIDENCE" in d.columns]
    curves = [c for c in curves if c in d.columns]

    # ---- track definitions -------------------------------------------------
    tracks: list[dict] = []
    for c in curves:
        tracks.append({"kind": "curve", "title": c, "width": 1.1, "curve": c})
    if has_true and show_true:
        tracks.append({"kind": "true", "title": "True<br><sup>original</sup>", "width": 0.8})
    for name, key in models:
        acc = _accuracy(d, key)
        subtitle = f"<br><sup>{acc:.1%} acc</sup>" if acc is not None else ""
        tracks.append({"kind": "pred", "title": f"{name}{subtitle}", "width": 0.8,
                       "name": name, "key": key})
        if draw_errors:
            tracks.append({"kind": "error", "title": "Err", "width": 0.22,
                           "name": name, "key": key})
    if show_confidence and conf_models:
        tracks.append({"kind": "conf", "title": "Confidence", "width": 1.1})

    if not tracks:
        raise ValueError("Nothing to plot: select at least one model or curve.")

    n_tracks = len(tracks)
    fig = make_subplots(
        rows=1, cols=n_tracks, shared_yaxes=True,
        column_widths=[t["width"] for t in tracks],
        subplot_titles=[t["title"] for t in tracks],
        horizontal_spacing=0.012,
    )

    present: set[str] = set()

    # ---- traces ------------------------------------------------------------
    for col, t in enumerate(tracks, start=1):
        kind = t["kind"]

        if kind == "curve":
            c = t["curve"]
            fig.add_trace(go.Scatter(
                x=d[c].to_numpy(float), y=depth, mode="lines",
                line=dict(color=CURVE_COLORS.get(c, "#444"), width=1),
                name=c, showlegend=False,
                hovertemplate=f"{c}: %{{x:.3f}}<br>Depth %{{y:.2f}}<extra></extra>",
            ), row=1, col=col)
            if c in LOG_CURVES:
                fig.update_xaxes(type="log", row=1, col=col)

        elif kind == "true":
            names = d["TRUE_LITHOLOGY"].to_numpy(object)
            present |= {n for n in names if isinstance(n, str)}
            fig.add_trace(_lithology_track(depth, names, "True"), row=1, col=col)

        elif kind == "pred":
            names = d[f"{t['key']}_PREDICTION"].to_numpy(object)
            present |= set(names)
            fig.add_trace(_lithology_track(depth, names, t["name"]), row=1, col=col)

        elif kind == "error":
            fig.add_trace(_error_track(
                depth, d["TRUE_CODE"].to_numpy(float),
                d[f"{t['key']}_PREDICTION_CODE"].to_numpy(float),
                f"{t['name']} error",
            ), row=1, col=col)

        elif kind == "conf":
            for name, key in conf_models:
                fig.add_trace(go.Scatter(
                    x=d[f"{key}_CONFIDENCE"].to_numpy(float), y=depth, mode="lines",
                    line=dict(color=MODEL_COLORS.get(name, "#444"), width=1),
                    name=f"{name} confidence", legendgroup=f"conf-{name}",
                    hovertemplate=f"{name}: %{{x:.2f}}<br>Depth %{{y:.2f}}<extra></extra>",
                ), row=1, col=col)
            fig.update_xaxes(range=[0, 1], row=1, col=col)

        if kind in ("true", "pred", "error"):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                             row=1, col=col)

    # ---- legend entries (dummy traces) -------------------------------------
    for name in LITH_ORDER:
        if name in present:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker=dict(symbol="square", size=13, color=LITHOLOGY_COLORS[name],
                            line=dict(width=0.5, color="#333")),
            ), row=1, col=1)
    if draw_errors and any(t["kind"] == "error" for t in tracks):
        legend_items = list(zip(("Correct", "Wrong"), ERROR_COLORS))
        if d["TRUE_CODE"].isna().any():
            legend_items.append(("No original label", "#FFFFFF"))
        for label, color in legend_items:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=label,
                marker=dict(symbol="square", size=13, color=color,
                            line=dict(width=0.8, color="#333")),
            ), row=1, col=1)

    # ---- axes / layout -----------------------------------------------------
    y_min, y_max = float(depth.min()), float(depth.max())
    if y_max <= y_min:
        y_max = y_min + 1.0

    fig.update_yaxes(
        range=[y_max, y_min], showgrid=True, gridcolor="rgba(130,145,165,0.22)",
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikethickness=1, spikecolor="#888",
    )
    fig.update_yaxes(title_text=depth_label, row=1, col=1)
    for col in range(2, n_tracks + 1):
        fig.update_yaxes(showticklabels=False, row=1, col=col)

    fig.update_annotations(font_size=11)
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title or "Lithology by depth", x=0.01),
        legend=dict(title="Legend", x=1.01, y=1.0, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=70, r=150, t=90, b=40),
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------------
# Distribution comparison
# --------------------------------------------------------------------------
def build_distribution_figure(
    df: pd.DataFrame, models: list[tuple[str, str]], include_true: bool = True
) -> go.Figure:
    """Grouped bars: share of each lithology for True and each selected model."""
    sources: list[tuple[str, pd.Series]] = []
    labelled_only = False
    if (include_true and "TRUE_LITHOLOGY" in df.columns and "TRUE_CODE" in df.columns
            and df["TRUE_CODE"].notna().any()):
        # Compare like with like: when True is shown, every bar uses only the
        # samples that have an original label (otherwise a partially labelled
        # well would compare different depth intervals).
        df = df[df["TRUE_CODE"].notna()]
        labelled_only = True
        sources.append(("True", df["TRUE_LITHOLOGY"]))
    for name, key in models:
        if f"{key}_PREDICTION" in df.columns:
            sources.append((name, df[f"{key}_PREDICTION"]))

    rows = []
    for source_name, series in sources:
        counts = series.dropna().value_counts()
        total = int(counts.sum())
        for lith in LITH_ORDER:
            n = int(counts.get(lith, 0))
            rows.append({
                "Lithology": lith, "Source": source_name, "Samples": n,
                "Share (%)": 100.0 * n / total if total else 0.0,
            })
    data = pd.DataFrame(rows)
    keep = data.groupby("Lithology")["Samples"].sum()
    order = [lith for lith in LITH_ORDER if keep.get(lith, 0) > 0]
    data = data[data["Lithology"].isin(order)]

    fig = px.bar(
        data, x="Lithology", y="Share (%)", color="Source", barmode="group",
        color_discrete_map=MODEL_COLORS, hover_data={"Samples": ":,", "Share (%)": ":.2f"},
        category_orders={"Lithology": order},
    )
    fig.update_layout(
        template="plotly_white", height=420, margin=dict(t=30, b=40),
        yaxis_title="Share of samples (%)", xaxis_title=None,
        legend=dict(orientation="h", y=1.1, x=0),
    )
    if labelled_only:
        fig.add_annotation(
            text="Labelled samples only", xref="paper", yref="paper",
            x=1, y=1.12, showarrow=False, font=dict(size=11, color="#667"),
        )
    return fig


# --------------------------------------------------------------------------
# Confusion matrix
# --------------------------------------------------------------------------
def build_confusion_figure(counts: np.ndarray, labels: list[int], normalize: bool = True):
    names = [LITHOLOGY_NAMES[label] for label in labels]
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, row_sums, out=np.zeros(counts.shape, dtype=float),
                           where=row_sums > 0)
        fig = px.imshow(matrix, x=names, y=names, zmin=0, zmax=1, aspect="auto",
                        color_continuous_scale="Blues", text_auto=".0%",
                        labels=dict(x="Predicted", y="True", color="Share of true class"))
    else:
        fig = px.imshow(counts, x=names, y=names, aspect="auto",
                        color_continuous_scale="Blues", text_auto="d",
                        labels=dict(x="Predicted", y="True", color="Samples"))
    fig.update_xaxes(side="top")
    fig.update_layout(template="plotly_white", height=560, margin=dict(t=90, b=30))
    return fig
