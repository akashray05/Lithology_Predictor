"""Extra figures and analysis helpers for the upgraded Lithology Predictor UI.

Adds model-comparison, calibration, boundary-error and geology views on top of
src/plots.py. Nothing here changes models, features or preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.inference import LITHOLOGY_NAMES
from src.theme import LITH_COLORS as LITHOLOGY_COLORS
from src.plots import (
    CURVE_COLORS,
    LITH_ORDER,
    MODEL_COLORS,
    _heatmap_track,
    _lithology_track,
)

LOG_CURVES = {"RDEP", "RMED"}


def _sorted(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    d = df.copy()
    d[depth_col] = pd.to_numeric(d[depth_col], errors="coerce")
    return d.dropna(subset=[depth_col]).sort_values(depth_col)


# --------------------------------------------------------------------------
# Consensus / agreement between models
# --------------------------------------------------------------------------
def consensus(df: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    """Majority-vote lithology and the share of models that agree with it.

    Ties resolve to the lower FORCE code (deterministic). This is a display
    aid, not a new model.
    """
    codes = np.vstack([df[f"{k}_PREDICTION_CODE"].to_numpy(int) for _, k in models])
    uniq = np.unique(codes)
    counts = np.stack([(codes == u).sum(axis=0) for u in uniq])
    best = counts.argmax(axis=0)
    out = pd.DataFrame(index=df.index)
    out["CONSENSUS_CODE"] = uniq[best]
    out["CONSENSUS"] = [LITHOLOGY_NAMES[int(c)] for c in out["CONSENSUS_CODE"]]
    out["AGREEMENT"] = counts.max(axis=0) / codes.shape[0]
    return out


def pairwise_agreement(df: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    names = [n for n, _ in models]
    mat = np.eye(len(names))
    for i, (_, ki) in enumerate(models):
        for j, (_, kj) in enumerate(models):
            mat[i, j] = float(
                (df[f"{ki}_PREDICTION_CODE"].to_numpy() == df[f"{kj}_PREDICTION_CODE"].to_numpy()).mean()
            )
    return pd.DataFrame(mat, index=names, columns=names)


def build_pairwise_figure(mat: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        mat, zmin=0, zmax=1, text_auto=".1%", aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Agreement"),
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(t=20, b=20))
    return fig


def build_consensus_figure(
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    depth_col: str = "DEPT",
    depth_label: str = "Depth",
    depth_range: tuple[float, float] | None = None,
    height: int = 760,
) -> go.Figure:
    """GR | consensus lithology | agreement strip (1 = all models agree)."""
    d = _sorted(df, depth_col)
    if depth_range is not None:
        d = d[d[depth_col].between(*depth_range)]
    if d.empty:
        raise ValueError("No samples in the selected depth range.")
    cons = consensus(d, models)
    depth = d[depth_col].to_numpy(float)
    n = len(models)

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True, column_widths=[1.1, 0.8, 0.5],
        subplot_titles=["GR", "Consensus<br><sup>majority vote</sup>", "Agreement"],
        horizontal_spacing=0.015,
    )
    if "GR" in d.columns:
        fig.add_trace(go.Scatter(
            x=d["GR"].to_numpy(float), y=depth, mode="lines",
            line=dict(color=CURVE_COLORS["GR"], width=1), showlegend=False,
            hovertemplate="GR: %{x:.1f}<br>Depth %{y:.2f}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(_lithology_track(depth, cons["CONSENSUS"].to_numpy(object), "Consensus"), row=1, col=2)

    # red = only one model's view, green = all agree
    scale = [[0, "#D64545"], [0.5, "#F2C94C"], [1, "#4CAF7A"]]
    fig.add_trace(
        _heatmap_track(
            depth, cons["AGREEMENT"].to_numpy(float),
            np.array([f"{a:.0%} of models agree" for a in cons["AGREEMENT"]], dtype=object),
            scale, 1.0 / n, 1.0, "Agreement",
        ),
        row=1, col=3,
    )
    for col in (2, 3):
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=col)

    present = set(cons["CONSENSUS"])
    for name in LITH_ORDER:
        if name in present:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker=dict(symbol="square", size=13, color=LITHOLOGY_COLORS[name],
                            line=dict(width=0.5, color="#333")),
            ), row=1, col=1)

    y0, y1 = float(depth.min()), float(depth.max())
    if y1 <= y0:
        y1 = y0 + 1.0
    fig.update_yaxes(range=[y1, y0], showgrid=True, gridcolor="rgba(130,145,165,0.22)")
    fig.update_yaxes(title_text=depth_label, row=1, col=1)
    for col in (2, 3):
        fig.update_yaxes(showticklabels=False, row=1, col=col)
    fig.update_annotations(font_size=11)
    fig.update_layout(template="plotly_white", height=height, hovermode="closest",
                      margin=dict(l=70, r=150, t=70, b=40),
                      legend=dict(title="Legend", x=1.01, y=1.0))
    return fig


def disagreement_intervals(
    df: pd.DataFrame, models: list[tuple[str, str]], depth_col: str = "DEPT", top: int = 8
) -> pd.DataFrame:
    """Thickest contiguous intervals where the selected models do not all agree."""
    d = _sorted(df, depth_col)
    cons = consensus(d, models)
    dis = (cons["AGREEMENT"] < 1).to_numpy()
    cols = ["Top", "Base", "Thickness", "Samples", "Consensus there", "Lowest agreement"]
    if not dis.any():
        return pd.DataFrame(columns=cols)

    run = np.cumsum(np.r_[True, dis[1:] != dis[:-1]])
    tmp = pd.DataFrame({
        "run": run, "dis": dis, "depth": d[depth_col].to_numpy(float),
        "agree": cons["AGREEMENT"].to_numpy(), "cons": cons["CONSENSUS"].to_numpy(),
    })
    g = tmp[tmp["dis"]].groupby("run")
    out = pd.DataFrame({
        "Top": g["depth"].min(), "Base": g["depth"].max(), "Samples": g.size(),
        "Consensus there": g["cons"].agg(lambda s: s.value_counts().index[0]),
        "Lowest agreement": g["agree"].min(),
    })
    out["Thickness"] = out["Base"] - out["Top"]
    return out[cols].sort_values("Thickness", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------
# Confidence / calibration
# --------------------------------------------------------------------------
def build_confidence_hist(df: pd.DataFrame, models: list[tuple[str, str]]) -> go.Figure:
    fig = go.Figure()
    for name, key in models:
        col = f"{key}_CONFIDENCE"
        if col in df.columns:
            fig.add_trace(go.Histogram(
                x=df[col].to_numpy(float), name=name, opacity=0.6, nbinsx=25,
                marker_color=MODEL_COLORS.get(name),
            ))
    fig.update_layout(template="plotly_white", barmode="overlay", height=340,
                      margin=dict(t=20, b=40), xaxis_title="Confidence (probability of predicted class)",
                      yaxis_title="Samples", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(range=[0, 1])
    return fig


def build_reliability_figure(df: pd.DataFrame, models: list[tuple[str, str]], bins: int = 10) -> go.Figure:
    """Accuracy vs. confidence on labelled samples (calibration check)."""
    lab = df[df["TRUE_CODE"].notna()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#999", dash="dash")))
    edges = np.linspace(0, 1, bins + 1)
    for name, key in models:
        col = f"{key}_CONFIDENCE"
        if col not in lab.columns or lab.empty:
            continue
        conf = lab[col].to_numpy(float)
        ok = (lab[f"{key}_PREDICTION_CODE"].to_numpy(float) == lab["TRUE_CODE"].to_numpy(float))
        idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
        xs, ys, ns = [], [], []
        for b in range(bins):
            m = idx == b
            if m.sum() >= 20:  # ignore tiny bins
                xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(int(m.sum()))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=name, customdata=ns,
            line=dict(color=MODEL_COLORS.get(name)),
            hovertemplate=name + ": conf %{x:.2f}, acc %{y:.2f}, n=%{customdata}<extra></extra>",
        ))
    fig.update_layout(template="plotly_white", height=380, margin=dict(t=20, b=40),
                      xaxis_title="Mean confidence", yaxis_title="Observed accuracy",
                      legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
    return fig


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------
def build_rolling_accuracy_figure(
    df: pd.DataFrame, models: list[tuple[str, str]], depth_col: str = "DEPT",
    window: int = 200, height: int = 700,
) -> go.Figure:
    """Rolling accuracy vs depth on labelled samples (where does each model struggle?)."""
    d = _sorted(df[df["TRUE_CODE"].notna()], depth_col)
    fig = go.Figure()
    for name, key in models:
        ok = (d[f"{key}_PREDICTION_CODE"].to_numpy(float) == d["TRUE_CODE"].to_numpy(float)).astype(float)
        roll = pd.Series(ok).rolling(window, center=True, min_periods=max(10, window // 4)).mean()
        fig.add_trace(go.Scatter(
            x=roll.to_numpy(), y=d[depth_col].to_numpy(float), mode="lines", name=name,
            line=dict(color=MODEL_COLORS.get(name), width=1.5),
            hovertemplate=name + ": %{x:.1%}<br>Depth %{y:.1f}<extra></extra>",
        ))
    y = d[depth_col].to_numpy(float)
    fig.update_layout(template="plotly_white", height=height, margin=dict(t=20, b=40),
                      xaxis_title=f"Rolling accuracy ({window} samples)", yaxis_title="Depth",
                      legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[y.max(), y.min()], showgrid=True, gridcolor="rgba(130,145,165,0.22)")
    return fig


def boundary_analysis(
    df: pd.DataFrame, models: list[tuple[str, str]], depth_col: str = "DEPT", tol: int = 5
) -> pd.DataFrame:
    """Accuracy near true lithology boundaries vs. inside beds.

    A sample is 'near a boundary' if the true label changes within +/- `tol`
    samples. Large gaps between the two accuracies mean most errors are
    boundary placement, which is geologically more forgivable than a
    wrong-bed call (and FORCE labels themselves are least certain there).
    """
    d = _sorted(df[df["TRUE_CODE"].notna()], depth_col)
    cols = ["Model", "Accuracy near boundaries", "Accuracy inside beds",
            "Samples near boundaries", "Share of errors near boundaries"]
    if len(d) < 3:
        return pd.DataFrame(columns=cols)
    t = d["TRUE_CODE"].to_numpy(float)
    change = np.r_[False, t[1:] != t[:-1]].astype(int)
    near = np.convolve(change, np.ones(2 * tol + 1), mode="same") > 0

    rows = []
    for name, key in models:
        wrong = d[f"{key}_PREDICTION_CODE"].to_numpy(float) != t
        n_wrong = int(wrong.sum())
        rows.append({
            "Model": name,
            "Accuracy near boundaries": float(1 - wrong[near].mean()) if near.any() else np.nan,
            "Accuracy inside beds": float(1 - wrong[~near].mean()) if (~near).any() else np.nan,
            "Samples near boundaries": float(near.mean()),
            "Share of errors near boundaries": float(wrong[near].sum() / n_wrong) if n_wrong else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


def bed_thickness_table(
    df: pd.DataFrame, models: list[tuple[str, str]], depth_col: str = "DEPT",
    include_true: bool = True,
) -> pd.DataFrame:
    """Number and thickness of contiguous beds per lithology, per source.

    Over-fragmented predictions (many thin beds) vs. the reference are a sign
    of salt-and-pepper noise that a sequence model or smoothing may fix.
    """
    d = _sorted(df, depth_col)
    depth = d[depth_col].to_numpy(float)
    step = float(np.median(np.diff(depth))) if len(depth) > 1 else 1.0
    sources = []
    if include_true and "TRUE_CODE" in d.columns and d["TRUE_CODE"].notna().any():
        sources.append(("True", d["TRUE_CODE"].to_numpy(float)))
    for name, key in models:
        sources.append((name, d[f"{key}_PREDICTION_CODE"].to_numpy(float)))

    rows = []
    for src, codes in sources:
        valid = ~np.isnan(codes)
        run = np.cumsum(np.r_[True, (codes[1:] != codes[:-1]) | (valid[1:] != valid[:-1])])
        tmp = pd.DataFrame({"run": run, "code": codes, "valid": valid})
        g = tmp[tmp["valid"]].groupby("run").agg(code=("code", "first"), n=("code", "size"))
        g["thick"] = g["n"] * step
        for code, sub in g.groupby("code"):
            rows.append({
                "Source": src, "Lithology": LITHOLOGY_NAMES[int(code)],
                "Beds": int(len(sub)), "Mean thickness": float(sub["thick"].mean()),
                "Max thickness": float(sub["thick"].max()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_o"] = out["Lithology"].map({n: i for i, n in enumerate(LITH_ORDER)})
        out = out.sort_values(["_o", "Source"]).drop(columns="_o").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Geology: cross-plots
# --------------------------------------------------------------------------
def build_crossplot(
    df: pd.DataFrame, x: str, y: str, color_col: str | None,
    max_points: int = 8000, seed: int = 0,
) -> go.Figure:
    """Log cross-plot coloured by a lithology-name column (True or a model)."""
    cols = [c for c in (x, y, color_col) if c]
    d = df[cols].dropna()
    if x in LOG_CURVES:
        d = d[d[x] > 0]
    if y in LOG_CURVES:
        d = d[d[y] > 0]
    if d.empty:
        raise ValueError("No valid samples for this cross-plot.")
    if len(d) > max_points:
        d = d.sample(max_points, random_state=seed)

    fig = px.scatter(
        d, x=x, y=y, color=color_col, opacity=0.8,
        color_discrete_map=LITHOLOGY_COLORS,
        category_orders={color_col: LITH_ORDER} if color_col else None,
        log_x=x in LOG_CURVES, log_y=y in LOG_CURVES,
    )
    fig.update_traces(marker=dict(size=5, opacity=1.0, line=dict(width=0.9, color='#111')))
    fig.update_layout(template="plotly_white", height=520, margin=dict(t=20, b=40),
                      legend=dict(title="Lithology"))
    return fig


def build_crossplot_compare(
    df: pd.DataFrame, x: str, y: str, panels: list[tuple[str, str]],
    max_points: int = 8000, seed: int = 0, highlight_errors: bool = True,
    height: int = 560,
) -> go.Figure:
    """Side-by-side cross-plots: the SAME samples and axes, coloured by each source.

    `panels` is [(title, lithology-name column), ...], e.g.
    [("Original", "TRUE_LITHOLOGY"), ("W5", "W5_PREDICTION")].
    Only rows that have a value in every panel column are used, so panels are
    directly comparable. When `highlight_errors` is on, model panels ring the
    points whose lithology differs from the original.
    """
    cols = list(dict.fromkeys([x, y] + [c for _, c in panels]))
    d = df[cols].dropna()
    for ax in (x, y):
        if ax in LOG_CURVES:
            d = d[d[ax] > 0]
    if d.empty:
        raise ValueError("No valid samples for this cross-plot.")

    truth_col = next((c for t, c in panels if c == "TRUE_LITHOLOGY"), None)
    full = df.dropna(subset=[x, y] + ([truth_col] if truth_col else []))
    if len(d) > max_points:
        d = d.sample(max_points, random_state=seed)   # one sample, shared by all panels

    titles = []
    for title, col in panels:
        if truth_col and col != truth_col and col in df.columns:
            both = df.dropna(subset=[col, truth_col])
            acc = float((both[col] == both[truth_col]).mean()) if len(both) else float("nan")
            titles.append(f"{title}<br><sup>{acc:.1%} agree with original</sup>")
        else:
            titles.append(f"{title}<br><sup>reference</sup>" if col == truth_col else title)

    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=titles, horizontal_spacing=0.02)
    present: set[str] = set()
    for j, (title, col) in enumerate(panels, start=1):
        for lith in LITH_ORDER:
            m = d[col] == lith
            if not m.any():
                continue
            present.add(lith)
            fig.add_trace(go.Scattergl(
                x=d.loc[m, x], y=d.loc[m, y], mode="markers", showlegend=False,
                marker=dict(size=5, color=LITHOLOGY_COLORS[lith], opacity=1.0,
                            line=dict(width=0.9, color="#111")),
                hovertemplate=f"{title}: {lith}<br>{x}=%{{x:.3f}}<br>{y}=%{{y:.3f}}<extra></extra>",
            ), row=1, col=j)
        if highlight_errors and truth_col and col != truth_col:
            w = d[(d[col] != d[truth_col])]
            if not w.empty:
                fig.add_trace(go.Scattergl(
                    x=w[x], y=w[y], mode="markers", showlegend=False,
                    marker=dict(symbol="circle-open", size=10, color="#E63946",
                                line=dict(width=1.5, color="#E63946")),
                    hovertemplate=f"{title} disagrees with original<extra></extra>",
                ), row=1, col=j)
        fig.update_xaxes(title_text=x, type="log" if x in LOG_CURVES else None, row=1, col=j)
        fig.update_yaxes(type="log" if y in LOG_CURVES else None, row=1, col=j)

    fig.update_yaxes(title_text=y, row=1, col=1)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")

    for lith in LITH_ORDER:                               # legend-only entries
        if lith in present:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lith,
                                     marker=dict(symbol="square", color=LITHOLOGY_COLORS[lith])),
                          row=1, col=1)
    if highlight_errors and truth_col and len(panels) > 1:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Disagrees with original",
                                 marker=dict(symbol="circle-open", color="#E63946",
                                             line=dict(width=1.6, color="#E63946"))),
                      row=1, col=1)

    fig.update_annotations(font_size=13)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=60, r=190, t=80, b=50),
                      legend=dict(title="Lithology"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(130,145,165,0.22)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(130,145,165,0.22)")
    return fig
