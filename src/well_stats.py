"""Well-level statistics for the Streamlit UI.

None of this retrains or changes the frozen models. It summarises the
uploaded interval: log curves, lithology mix, bed transitions, and
agreement between classifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import cohen_kappa_score

from src.plots import LITH_ORDER
from src.theme import FIGURE_LAYOUT, LITH_COLORS

REQUIRED_CURVES = ["GR", "RDEP", "RMED", "DTC", "RHOB"]


def _sorted(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    d = df.copy()
    d[depth_col] = pd.to_numeric(d[depth_col], errors="coerce")
    return d.dropna(subset=[depth_col]).sort_values(depth_col)


def _sample_step(depth: np.ndarray) -> float:
    if len(depth) < 2:
        return 1.0
    diffs = np.diff(depth.astype(float))
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else 1.0


def _source_columns(
    df: pd.DataFrame, models: list[tuple[str, str]], include_true: bool
) -> list[tuple[str, str]]:
    cols: list[tuple[str, str]] = []
    if include_true and "TRUE_LITHOLOGY" in df.columns and df["TRUE_CODE"].notna().any():
        cols.append(("Original", "TRUE_LITHOLOGY"))
    for name, key in models:
        col = f"{key}_PREDICTION"
        if col in df.columns:
            cols.append((name, col))
    return cols


def curve_summary(df: pd.DataFrame, curves: list[str] | None = None) -> pd.DataFrame:
    """Min / percentiles / mean / missingness for the input logs."""
    curves = curves or [c for c in REQUIRED_CURVES if c in df.columns]
    rows = []
    n = len(df)
    for c in curves:
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.dropna()
        if c in {"RDEP", "RMED"}:
            valid = valid[valid > 0]
        rows.append({
            "Curve": c,
            "Samples": n,
            "Valid": int(len(valid)),
            "Missing %": 100.0 * (1 - len(valid) / n) if n else np.nan,
            "Min": float(valid.min()) if len(valid) else np.nan,
            "P10": float(valid.quantile(0.10)) if len(valid) else np.nan,
            "P50": float(valid.quantile(0.50)) if len(valid) else np.nan,
            "Mean": float(valid.mean()) if len(valid) else np.nan,
            "P90": float(valid.quantile(0.90)) if len(valid) else np.nan,
            "Max": float(valid.max()) if len(valid) else np.nan,
            "Std": float(valid.std()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


def lithology_mix(
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    depth_col: str = "DEPT",
    include_true: bool = True,
) -> pd.DataFrame:
    """Net thickness and sample share per lithology, per source."""
    d = _sorted(df, depth_col)
    step = _sample_step(d[depth_col].to_numpy(float))
    rows = []
    for src, col in _source_columns(d, models, include_true):
        series = d[col]
        if src == "Original":
            series = series.where(d["TRUE_CODE"].notna())
        counts = series.value_counts(dropna=True)
        total = int(counts.sum())
        for lith, n in counts.items():
            rows.append({
                "Source": src,
                "Lithology": lith,
                "Samples": int(n),
                "Net thickness": float(n) * step,
                "Fraction": float(n) / total if total else np.nan,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = out["Lithology"].map({n: i for i, n in enumerate(LITH_ORDER)})
    return out.sort_values(["Source", "_o"]).drop(columns="_o").reset_index(drop=True)


def lithology_entropy(mix: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy (bits) of the lithology mix — higher means more mixed."""
    rows = []
    for src, sub in mix.groupby("Source"):
        p = sub["Fraction"].to_numpy(float)
        p = p[p > 0]
        H = float(-(p * np.log2(p)).sum()) if len(p) else np.nan
        rows.append({
            "Source": src,
            "Classes present": int(len(p)),
            "Entropy (bits)": H,
            "Max entropy (12 classes)": np.log2(12),
        })
    return pd.DataFrame(rows)


def transition_table(
    df: pd.DataFrame, lith_col: str, depth_col: str = "DEPT", min_count: int = 1
) -> pd.DataFrame:
    """Counts of lithology changes from one bed to the next (along depth)."""
    d = _sorted(df, depth_col)
    if lith_col == "TRUE_LITHOLOGY" and "TRUE_CODE" in d.columns:
        d = d[d["TRUE_CODE"].notna()]
    seq = d[lith_col].dropna().to_numpy(object)
    if len(seq) < 2:
        return pd.DataFrame(columns=["From", "To", "Contacts"])
    # collapse runs so each bed is one state
    change = np.r_[True, seq[1:] != seq[:-1]]
    beds = seq[change]
    if len(beds) < 2:
        return pd.DataFrame(columns=["From", "To", "Contacts"])
    frm, to = beds[:-1], beds[1:]
    tab = pd.DataFrame({"From": frm, "To": to}).value_counts().reset_index(name="Contacts")
    tab = tab[tab["Contacts"] >= min_count]
    order = {n: i for i, n in enumerate(LITH_ORDER)}
    tab["_f"] = tab["From"].map(order)
    tab["_t"] = tab["To"].map(order)
    return (tab.sort_values(["Contacts", "_f", "_t"], ascending=[False, True, True])
            .drop(columns=["_f", "_t"]).reset_index(drop=True))


def transition_matrix(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    present = [n for n in LITH_ORDER if n in set(table["From"]) | set(table["To"])]
    mat = table.pivot(index="From", columns="To", values="Contacts").reindex(index=present, columns=present).fillna(0)
    return mat.astype(int)


def pairwise_kappa(df: pd.DataFrame, models: list[tuple[str, str]]) -> pd.DataFrame:
    """Cohen's kappa between each pair of selected models (and original, if present)."""
    series: list[tuple[str, np.ndarray]] = []
    if "TRUE_CODE" in df.columns and df["TRUE_CODE"].notna().any():
        series.append(("Original", df["TRUE_CODE"].to_numpy(float)))
    for name, key in models:
        series.append((name, df[f"{key}_PREDICTION_CODE"].to_numpy(float)))
    names = [n for n, _ in series]
    mat = np.eye(len(names))
    for i, (_, a) in enumerate(series):
        for j, (_, b) in enumerate(series):
            if i == j:
                continue
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 20:
                mat[i, j] = np.nan
            else:
                mat[i, j] = cohen_kappa_score(a[mask].astype(int), b[mask].astype(int))
    return pd.DataFrame(mat, index=names, columns=names)


def curve_correlations(df: pd.DataFrame, curves: list[str] | None = None) -> pd.DataFrame:
    curves = curves or [c for c in REQUIRED_CURVES if c in df.columns]
    block = df[curves].apply(pd.to_numeric, errors="coerce")
    for c in curves:
        if c in {"RDEP", "RMED"}:
            block[c] = np.log10(block[c].where(block[c] > 0))
    return block.corr(method="spearman")


def interval_stats(df: pd.DataFrame, models: list[tuple[str, str]], depth_col: str = "DEPT") -> dict:
    """Compact numbers for the overview strip."""
    d = _sorted(df, depth_col)
    depth = d[depth_col].to_numpy(float)
    step = _sample_step(depth)
    out = {
        "n": int(len(d)),
        "top": float(depth.min()) if len(depth) else np.nan,
        "base": float(depth.max()) if len(depth) else np.nan,
        "span": float(depth.max() - depth.min()) if len(depth) else np.nan,
        "step": step,
    }
    if models:
        key = models[0][1]
        codes = d[f"{key}_PREDICTION_CODE"].to_numpy(float)
        change = np.r_[True, codes[1:] != codes[:-1]]
        out["beds_first_model"] = int(change.sum())
    if "TRUE_CODE" in d.columns and d["TRUE_CODE"].notna().any():
        t = d.loc[d["TRUE_CODE"].notna(), "TRUE_CODE"].to_numpy(float)
        out["beds_original"] = int(np.r_[True, t[1:] != t[:-1]].sum())
    return out


def apply_figure_chrome(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**FIGURE_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="rgba(70,62,50,0.12)", zeroline=False, linecolor="#8a8172")
    fig.update_yaxes(gridcolor="rgba(70,62,50,0.12)", zeroline=False, linecolor="#8a8172")
    return fig


def build_mix_figure(mix: pd.DataFrame) -> go.Figure:
    if mix.empty:
        raise ValueError("No lithology mix to plot.")
    fig = go.Figure()
    sources = list(dict.fromkeys(mix["Source"]))
    for lith in LITH_ORDER:
        sub = mix[mix["Lithology"] == lith]
        if sub.empty:
            continue
        y = [float(sub.loc[sub["Source"] == s, "Fraction"].sum()) if (sub["Source"] == s).any() else 0.0
             for s in sources]
        fig.add_trace(go.Bar(
            name=lith, x=sources, y=y, marker_color=LITH_COLORS[lith],
            marker_line=dict(width=0.4, color="#2a2620"),
            hovertemplate=lith + ": %{y:.1%}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", yaxis_title="Fraction of samples",
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
                      margin=dict(t=16, b=16, l=48, r=16))
    return apply_figure_chrome(fig, 420)


def build_histogram_figure(
    df: pd.DataFrame, curve: str, lith_col: str, log_x: bool = False, max_points: int = 25000
) -> go.Figure:
    cols = [c for c in (curve, lith_col) if c in df.columns]
    d = df[cols].dropna()
    if curve in {"RDEP", "RMED"}:
        d = d[pd.to_numeric(d[curve], errors="coerce") > 0]
    if d.empty:
        raise ValueError(f"No valid {curve} samples for this histogram.")
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)
    fig = go.Figure()
    for lith in LITH_ORDER:
        x = pd.to_numeric(d.loc[d[lith_col] == lith, curve], errors="coerce").dropna()
        if x.empty:
            continue
        fig.add_trace(go.Histogram(
            x=x, name=lith, marker_color=LITH_COLORS[lith], opacity=0.72, nbinsx=40,
            hovertemplate=lith + ": %{x}<extra></extra>",
        ))
    fig.update_layout(barmode="overlay", xaxis_title=curve, yaxis_title="Samples",
                      legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
                      margin=dict(t=16, b=16, l=48, r=16))
    if log_x:
        fig.update_xaxes(type="log")
    return apply_figure_chrome(fig, 400)


def build_depth_composition_figure(
    df: pd.DataFrame, lith_col: str, depth_col: str = "DEPT", bins: int = 24
) -> go.Figure:
    d = _sorted(df, depth_col)
    if lith_col == "TRUE_LITHOLOGY" and "TRUE_CODE" in d.columns:
        d = d[d["TRUE_CODE"].notna()]
    d = d[[depth_col, lith_col]].dropna()
    if d.empty:
        raise ValueError("No samples for depth composition.")
    depth = d[depth_col].to_numpy(float)
    edges = np.linspace(depth.min(), depth.max(), bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure()
    for lith in LITH_ORDER:
        m = d[lith_col].to_numpy(object) == lith
        if not m.any():
            continue
        hist, _ = np.histogram(depth[m], bins=edges)
        tot, _ = np.histogram(depth, bins=edges)
        frac = np.divide(hist, tot, out=np.zeros_like(hist, dtype=float), where=tot > 0)
        fig.add_trace(go.Scatter(
            x=frac, y=centres, name=lith, stackgroup="one", mode="none",
            fillcolor=LITH_COLORS[lith], line=dict(width=0),
            hovertemplate=lith + ": %{x:.1%}<br>Depth %{y:.1f}<extra></extra>",
        ))
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Lithology fraction", yaxis_title="Depth",
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
                      margin=dict(t=16, b=16, l=56, r=16))
    fig.update_yaxes(autorange="reversed")
    return apply_figure_chrome(fig, 640)


def build_correlation_figure(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(float), x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale=[
            [0, "#5c3d2e"], [0.5, "#f3efe6"], [1, "#2c4a3e"],
        ],
        text=np.round(corr.to_numpy(float), 2), texttemplate="%{text}",
        colorbar=dict(title="Spearman"),
    ))
    fig.update_layout(margin=dict(t=16, b=16, l=80, r=16))
    return apply_figure_chrome(fig, 380)


def build_kappa_figure(mat: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=mat.to_numpy(float), x=list(mat.columns), y=list(mat.index),
        zmin=0, zmax=1, colorscale=[[0, "#f3efe6"], [1, "#2c4a3e"]],
        text=np.round(mat.to_numpy(float), 2), texttemplate="%{text:.2f}",
        colorbar=dict(title="κ"),
    ))
    fig.update_layout(margin=dict(t=16, b=16, l=80, r=16))
    return apply_figure_chrome(fig, 360)


def build_transition_figure(mat: pd.DataFrame) -> go.Figure:
    if mat.empty:
        raise ValueError("No lithology contacts in this interval.")
    fig = go.Figure(go.Heatmap(
        z=mat.to_numpy(float), x=list(mat.columns), y=list(mat.index),
        colorscale=[[0, "#faf8f3"], [1, "#8b3a2a"]],
        text=mat.to_numpy(int), texttemplate="%{text}",
        colorbar=dict(title="Contacts"),
    ))
    fig.update_layout(xaxis_title="To", yaxis_title="From",
                      margin=dict(t=16, b=48, l=90, r=16))
    return apply_figure_chrome(fig, 420)
