"""LITHO-ML — upgraded Streamlit UI (run: streamlit run app_v2.py).

Same frozen models and inference path as app.py; only the interface and
analysis views are new.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation import (
    confusion_counts,
    describe_labels,
    overall_metrics,
    per_class_report,
)
from src.inference import (
    LITHOLOGY_NAMES,
    detect_label_column,
    label_diagnostics,
    load_inference_bundle,
    predict_well,
    read_uploaded_file,
)
from src.plots import (
    LITH_ORDER,
    build_confusion_figure,
    build_distribution_figure,
    build_tracks_figure,
)
from src.plots_extra import (
    bed_thickness_table,
    boundary_analysis,
    build_confidence_hist,
    build_consensus_figure,
    build_crossplot,
    build_crossplot_compare,
    build_pairwise_figure,
    build_reliability_figure,
    build_rolling_accuracy_figure,
    consensus,
    disagreement_intervals,
    pairwise_agreement,
)

from src.model_info import render_model_guide
from src.theme import VIVID_COLORS, style_figure

# ============================================================
# CONFIG + STYLE
# ============================================================
st.set_page_config(page_title="LITHO-ML", page_icon="🪨", layout="wide",
                   initial_sidebar_state="expanded")

_css = Path(__file__).parent / "assets" / "style.css"
if _css.is_file():
    st.markdown(f"<style>{_css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.warning("assets/style.css not found - running with default styling.")

# display name -> internal column key (the frozen model is still called W5 internally)
MODEL_OPTIONS = {"Ex-Tree": "W5", "XGBoost": "XGBOOST", "LightGBM": "LIGHTGBM"}
REQUIRED_CURVES = ["GR", "RDEP", "RMED", "DTC", "RHOB"]
CURVE_HELP = {
    "GR": "Gamma ray (API)", "RDEP": "Deep resistivity (ohm.m)", "RMED": "Medium resistivity (ohm.m)",
    "DTC": "Compressional sonic (us/ft)", "RHOB": "Bulk density (g/cc)",
}


@st.cache_resource
def get_bundle():
    return load_inference_bundle()


@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return read_uploaded_file(BytesIO(file_bytes), filename=filename)


@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes, filename: str, label_column: str | None) -> pd.DataFrame:
    return predict_well(load_raw(file_bytes, filename), well_name=Path(filename).stem,
                        label_column=label_column)


def hero(compact: bool = False):
    sub = ("Upload a well-log file, run three frozen lithology models, and compare them against "
           "the original lithology — depth by depth."
           if not compact else "Frozen-model lithology prediction and comparison")
    st.markdown(
        f"""
        <div class="hero">
          <h1>LITHO-ML</h1>
          <p>{sub}</p>
          <div class="chips">
            <span class="chip">12 lithology classes</span>
            <span class="chip">41 engineered features</span>
            <span class="chip">3 models: Ex-Tree · XGBoost · LightGBM</span>
            <span class="chip">FORCE 2020</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODELS
# ============================================================
try:
    bundle = get_bundle()
except Exception as exc:
    hero(compact=True)
    st.error(f"Could not load the trained models: {exc}")
    st.stop()

# ============================================================
# SIDEBAR 1: MODELS
# ============================================================
st.sidebar.markdown("## 🪨 LITHO-ML")
st.sidebar.caption("Predictions are estimates, not verified geological truth.")
PAGE_PRED, PAGE_GUIDE = "🪨  Predictor", "📘  Model guide"
page = st.sidebar.radio("Navigate", [PAGE_PRED, PAGE_GUIDE], label_visibility="collapsed", key="nav")
selected_models = list(MODEL_OPTIONS)
if page == PAGE_PRED:
    with st.sidebar.expander("① Models", expanded=True):
        selected_models = st.multiselect("Models to display", list(MODEL_OPTIONS),
                                         default=list(MODEL_OPTIONS),
                                         help="Only selected models are plotted, compared and exported.")
        st.caption(f"{len(bundle.features)} frozen input features.")

# ============================================================
# UPLOAD
# ============================================================
def _uploader():
    return st.file_uploader(
        "Upload a well-log file (.las or .csv)", type=["las", "csv"], key="well_upload",
        help="Needs GR, RDEP, RMED, DTC, RHOB. Add a lithology column to compare against the original.",
    )


if page == PAGE_GUIDE:
    hero(compact=True)
    with st.expander("📂 Uploaded well file (kept while you read the guide)"):
        uploaded_file = _uploader()
    render_model_guide()
    st.stop()

uploaded_file = _uploader()

if uploaded_file is None:
    hero()
    st.markdown("### How it works")
    cols = st.columns(4)
    steps = [
        ("Upload", "Drop a LAS or CSV well file. A lithology column is optional."),
        ("Predict", "Features are built exactly as in training; all three frozen models run."),
        ("Compare", "Depth tracks, consensus, disagreement zones and confidence side by side."),
        ("Evaluate", "With labels: FORCE score, F1, confusion matrix, boundary and calibration checks."),
    ]
    for c, (i, (t, d)) in zip(cols, enumerate(steps, 1)):
        c.markdown(f'<div class="step-card"><div class="num">{i}</div><h4>{t}</h4><p>{d}</p></div>',
                   unsafe_allow_html=True)

    st.markdown("### Required input curves")
    cc = st.columns(5)
    for c, name in zip(cc, REQUIRED_CURVES):
        c.metric(name, "required", help=CURVE_HELP[name])
        c.caption(CURVE_HELP[name])

    st.markdown("### Lithology classes")
    st.markdown(
        " ".join(
            f'<span class="chip" style="background:{VIVID_COLORS[n]}22;border-color:{VIVID_COLORS[n]}">'
            f'<span class="lith-dot" style="background:{VIVID_COLORS[n]}"></span>{n}</span>'
            for n in LITH_ORDER
        ),
        unsafe_allow_html=True,
    )
    st.info("Without an original lithology column, only predictions and confidence are shown. "
            "Accuracy and F1 are never computed without ground truth.")
    st.stop()

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

file_bytes, filename = uploaded_file.getvalue(), uploaded_file.name
try:
    raw_df = load_raw(file_bytes, filename)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

missing_curves = [c for c in REQUIRED_CURVES if c not in {str(x).strip().upper() for x in raw_df.columns}]
if missing_curves:
    st.error("Missing required input curves: " + ", ".join(missing_curves))
    st.stop()

# ============================================================
# SIDEBAR 2: LABELS
# ============================================================
detected = detect_label_column(raw_df.columns)
with st.sidebar.expander("② Original lithology", expanded=True):
    choice = st.selectbox("True lithology column", ["Auto-detect", "None"] + [str(c) for c in raw_df.columns],
                          help="FORCE codes (e.g. 65000) or names (e.g. Shale).")
    if choice == "Auto-detect":
        label_column = detected
        st.caption(f"Detected: `{detected}`" if detected else "No lithology column detected.")
    elif choice == "None":
        label_column = None
    else:
        label_column = choice

# ============================================================
# INFERENCE
# ============================================================
try:
    with st.spinner("Building features and running the models..."):
        predictions = run_inference(file_bytes, filename, label_column)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

depth_col, depth_label = "DEPT", "Depth"
if predictions["DEPT"].notna().sum() == 0:
    predictions = predictions.assign(DEPT=predictions["SOURCE_ROW"].astype(float))
    depth_label = "Sample index (no depth column found)"
    st.sidebar.warning("No depth column found; using sample index.")

try:
    label_diag = label_diagnostics(raw_df, label_column) if label_column else None
except Exception:
    label_diag = None
label_info = describe_labels(predictions, label_diag, depth_col=depth_col)
has_truth = label_info["labelled"] > 0
models = [(n, MODEL_OPTIONS[n]) for n in selected_models]
well = Path(filename).stem


def show_label_status():
    getattr(st, label_info["level"])(label_info["message"])


# ============================================================
# SIDEBAR 3: VIEW
# ============================================================
with st.sidebar.expander("③ View", expanded=True):
    d_min, d_max = float(predictions[depth_col].min()), float(predictions[depth_col].max())
    if d_max > d_min:
        depth_range = st.slider("Depth interval", d_min, d_max, (d_min, d_max),
                                step=max((d_max - d_min) / 500, 0.01))
    else:
        depth_range = (d_min, d_max)
    show_true = bool(st.checkbox("Original lithology track", value=True, disabled=not has_truth) and has_truth)
    show_errors = bool(st.checkbox("Error strips", value=True, disabled=not show_true) and show_true)
    crop_req = st.checkbox("Limit to labelled interval", value=False,
                           disabled=label_info["status"] != "partial")
    crop_labelled = bool(crop_req and label_info["status"] == "partial")
    show_confidence = st.checkbox("Confidence track", value=False)
    curves = st.multiselect("Log curves to show", REQUIRED_CURVES, default=["GR"])
    plot_height = st.slider("Plot height (px)", 500, 2000, 950, step=50)

if crop_labelled and label_info["depth_min"] is not None:
    depth_range = (max(depth_range[0], label_info["depth_min"]), min(depth_range[1], label_info["depth_max"]))
    if depth_range[0] > depth_range[1]:
        st.warning("The selected depth interval does not overlap the labelled interval.")
        st.stop()

view_df = predictions[predictions[depth_col].between(*depth_range)]
if view_df.empty:
    st.warning("No samples in the selected depth interval.")
    st.stop()
labelled_view = view_df[view_df["TRUE_CODE"].notna()] if has_truth else view_df.iloc[0:0]

# ============================================================
# HEADER + STATUS RIBBON
# ============================================================
hero(compact=True)
pill = {"full": ("ok", "labels: full"), "partial": ("part", "labels: partial")}.get(
    label_info["status"], ("none", "no labels"))
st.markdown(
    f'<div class="ribbon"><b>{well}</b> · {len(raw_df):,} rows · '
    f'depth {d_min:,.1f}–{d_max:,.1f} · {len(models)} model(s)'
    f'<span class="pill {pill[0]}">{pill[1]}</span></div>',
    unsafe_allow_html=True,
)

tab_over, tab_tracks, tab_cmp, tab_eval, tab_geo, tab_data = st.tabs(
    ["🧭 Overview", "📊 Tracks", "🔀 Compare", "✅ Evaluate", "🧪 Geology", "💾 Export"]
)

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted samples", f"{len(view_df):,}")
    c2.metric("Depth interval", f"{depth_range[0]:,.0f} – {depth_range[1]:,.0f}")
    c3.metric("Labelled samples", f"{len(labelled_view):,}" if has_truth else "—")
    if len(models) >= 2:
        cons_v = consensus(view_df, models)
        c4.metric("All models agree", f"{(cons_v['AGREEMENT'] == 1).mean():.1%}",
                  help="Share of samples where every selected model predicts the same lithology.")
    else:
        col = f"{models[0][1]}_CONFIDENCE"
        if col in view_df.columns:
            c4.metric("Mean confidence", f"{view_df[col].mean():.1%}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Lithology distribution")
        st.plotly_chart(style_figure(build_distribution_figure(view_df, models, include_true=has_truth), right_margin=None),
                        use_container_width=True)
    with right:
        st.subheader("Model summary")
        rows = []
        for name, key in models:
            row = {"Model": name, "Classes predicted": view_df[f"{key}_PREDICTION"].nunique()}
            if f"{key}_CONFIDENCE" in view_df.columns:
                row["Mean confidence"] = view_df[f"{key}_CONFIDENCE"].mean()
            if has_truth and len(labelled_view):
                row["Accuracy"] = float((labelled_view[f"{key}_PREDICTION_CODE"].to_numpy(float)
                                         == labelled_view["TRUE_CODE"].to_numpy(float)).mean())
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Mean confidence": st.column_config.NumberColumn(format="%.3f"),
                                    "Accuracy": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Confidence = probability of the predicted class; not guaranteed calibrated "
                   "(see Evaluate → Calibration).")

    if len(models) >= 2:
        st.subheader("Where the models disagree most")
        dis = disagreement_intervals(view_df, models, depth_col)
        if dis.empty:
            st.success("All selected models agree everywhere in this interval.")
        else:
            st.dataframe(dis, hide_index=True, use_container_width=True,
                         column_config={"Top": st.column_config.NumberColumn(format="%.1f"),
                                        "Base": st.column_config.NumberColumn(format="%.1f"),
                                        "Thickness": st.column_config.NumberColumn(format="%.1f"),
                                        "Lowest agreement": st.column_config.NumberColumn(format="%.2f")})

# ------------------------------------------------------------
# TRACKS
# ------------------------------------------------------------
with tab_tracks:
    show_label_status()
    st.caption("Drag to zoom, double-click to reset, scroll to zoom the depth axis.")
    try:
        fig_tracks = style_figure(build_tracks_figure(
            predictions, models, depth_col=depth_col, depth_label=depth_label, depth_range=depth_range,
            show_true=show_true, show_errors=show_errors, show_confidence=show_confidence,
            curves=curves, height=plot_height, title=f"{well}: lithology by depth",
        ))
        st.plotly_chart(fig_tracks, use_container_width=True, config={
            "scrollZoom": True, "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "scale": 2, "filename": f"{well}_lithology_tracks"},
        })
    except ValueError as exc:
        st.warning(str(exc))

# ------------------------------------------------------------
# COMPARE (model vs model — no labels needed)
# ------------------------------------------------------------
with tab_cmp:
    if len(models) < 2:
        st.info("Select at least two models in the sidebar to compare them.")
    else:
        st.caption("Model-vs-model views. They need no labels, and agreement is NOT accuracy — "
                   "models can agree and still be wrong.")
        a, b = st.columns([2, 3])
        with a:
            st.subheader("Pairwise agreement")
            st.plotly_chart(build_pairwise_figure(pairwise_agreement(view_df, models)),
                            use_container_width=True)
            st.subheader("Confidence distribution")
            st.plotly_chart(build_confidence_hist(view_df, models), use_container_width=True)
        with b:
            st.subheader("Consensus and agreement by depth")
            try:
                st.plotly_chart(
                    style_figure(build_consensus_figure(predictions, models, depth_col, depth_label,
                                                        depth_range, height=min(plot_height, 820))),
                    use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
            except ValueError as exc:
                st.warning(str(exc))
            st.caption("Agreement strip: green = all models agree, red = most disagree. "
                       "Consensus is a majority vote for display only; it is not a separate model.")

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------
with tab_eval:
    show_label_status()
    if not has_truth:
        st.caption("Metrics need valid original labels (e.g. `FORCE_2020_LITHOFACIES_LITHOLOGY`). "
                   "Nothing is calculated without them.")
    else:
        st.caption("These are results for THIS well only. They illustrate behaviour; they are not "
                   "a substitute for the well-wise test report.")
        use_view = st.checkbox("Evaluate only the selected depth interval", value=False)
        eval_df = (view_df if use_view else predictions)
        eval_df = eval_df[eval_df["TRUE_CODE"].notna()]

        if eval_df.empty:
            st.warning("No labelled samples to evaluate.")
        else:
            y_true = eval_df["TRUE_CODE"].to_numpy().astype(int)
            preds = {n: eval_df[f"{k}_PREDICTION_CODE"].to_numpy().astype(int) for n, k in models}
            metrics = overall_metrics(y_true, preds)
            best = metrics.loc[metrics["FORCE score"].idxmax()]

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Labelled samples", f"{len(eval_df):,}")
            b2.metric("Best FORCE score", f"{best['FORCE score']:.4f}",
                      help="Negative mean penalty; closer to zero is better.")
            b3.metric("Best model (FORCE)", str(best["Model"]))
            b4.metric("Best Macro F1", f"{metrics['Macro F1'].max():.3f}")

            sub_over, sub_cm, sub_err, sub_cal = st.tabs(
                ["Metrics", "Confusion & classes", "Where errors happen", "Calibration"])

            with sub_over:
                st.dataframe(metrics, hide_index=True, use_container_width=True, column_config={
                    c: st.column_config.NumberColumn(format="%.4f") for c in metrics.columns if c != "Model"})
                st.caption("Macro/Weighted F1 average over classes present in this well's true labels.")

            with sub_cm:
                cm_cols = st.columns([2, 1])
                cm_model = cm_cols[0].selectbox("Model", [n for n, _ in models], key="cm_model")
                cm_norm = cm_cols[1].radio("Show", ["% of true class", "Sample counts"],
                                           horizontal=True) == "% of true class"
                counts, labels = confusion_counts(y_true, preds[cm_model])
                st.plotly_chart(build_confusion_figure(counts, labels, normalize=cm_norm),
                                use_container_width=True)
                st.dataframe(per_class_report(y_true, preds[cm_model]), hide_index=True,
                             use_container_width=True, column_config={
                                 "Precision": st.column_config.NumberColumn(format="%.3f"),
                                 "Recall": st.column_config.NumberColumn(format="%.3f"),
                                 "F1": st.column_config.NumberColumn(format="%.3f")})

            with sub_err:
                st.markdown("**Rolling accuracy by depth**")
                win = st.slider("Window (samples)", 50, 1000, 200, step=50)
                st.plotly_chart(build_rolling_accuracy_figure(eval_df, models, depth_col, window=win),
                                use_container_width=True)
                st.markdown("**Boundary vs. inside-bed accuracy**")
                tol = st.slider("Boundary zone ± samples", 1, 20, 5)
                st.dataframe(boundary_analysis(eval_df, models, depth_col, tol=tol), hide_index=True,
                             use_container_width=True, column_config={
                                 c: st.column_config.NumberColumn(format="%.3f")
                                 for c in ["Accuracy near boundaries", "Accuracy inside beds",
                                           "Samples near boundaries", "Share of errors near boundaries"]})
                st.caption("If most errors sit near true boundaries, the models place contacts slightly "
                           "off. That is geologically milder than calling a whole bed wrong, and "
                           "reference labels are also least certain at contacts.")

            with sub_cal:
                st.plotly_chart(build_reliability_figure(eval_df, models), use_container_width=True)
                st.caption("Points on the dashed line mean confidence ≈ real accuracy. Below the line = "
                           "over-confident. Bins with fewer than 20 samples are hidden.")

# ------------------------------------------------------------
# GEOLOGY
# ------------------------------------------------------------
with tab_geo:
    st.caption("Do the predictions make petrophysical sense? Cross-plots use raw (cleaned) curve values.")
    sources = {}
    if has_truth:
        sources["Original lithology"] = "TRUE_LITHOLOGY"
    for n, k in models:
        sources[n] = f"{k}_PREDICTION"

    r1, r2, r3 = st.columns([2, 1.3, 1.3])
    view_mode = r1.radio("View", ["Original vs model(s)", "Single plot"], horizontal=True,
                         index=0 if has_truth else 1, disabled=not has_truth,
                         help="Side-by-side needs original lithology labels.")
    compare_mode = bool(has_truth and view_mode == "Original vs model(s)")
    pair = r2.selectbox("Cross-plot", ["GR vs RHOB", "DTC vs RHOB", "RDEP vs GR", "RDEP vs RMED"])
    max_pts = r3.slider("Max points (multiples of 500)", 500, 20000, 2000, step=500)
    xname, yname = [t.strip() for t in pair.split(" vs ")]

    try:
        if compare_mode:
            c1, c2 = st.columns([3, 2])
            pick = c1.multiselect("Models to compare with the original", [n for n, _ in models],
                                  default=[models[0][0]])
            ring = c2.checkbox("Show wrong predictions (ringed)", value=False,
                               help="Off by default. When ticked, ONLY the points where the model disagrees with the original are ringed.")
            if not pick:
                st.info("Select at least one model to compare.")
            else:
                panels = [("Original", "TRUE_LITHOLOGY")] + [(n, sources[n]) for n in pick]
                st.plotly_chart(
                    style_figure(build_crossplot_compare(view_df, xname, yname, panels,
                                                         max_points=max_pts, highlight_errors=ring),
                                 right_margin=None),
                    use_container_width=True)
                st.caption("Same samples and same axes in every panel; only the colour (lithology) differs. "
                           "Only labelled samples are shown. Tick \"Show wrong predictions\" to ring only the "
                           "points where the model's call differs from the original.")
        else:
            colour_by = st.selectbox("Colour by", list(sources))
            st.plotly_chart(
                style_figure(build_crossplot(view_df, xname, yname, sources[colour_by],
                                             max_points=max_pts), right_margin=None),
                use_container_width=True)
    except ValueError as exc:
        st.warning(str(exc))

    st.subheader("Bed thickness by lithology")
    thick = bed_thickness_table(view_df, models, depth_col, include_true=has_truth)
    if thick.empty:
        st.info("Not enough data to compute bed thickness.")
    else:
        st.dataframe(thick, hide_index=True, use_container_width=True, column_config={
            "Mean thickness": st.column_config.NumberColumn(format="%.2f"),
            "Max thickness": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Many thin beds in a prediction vs. the reference suggests salt-and-pepper noise "
                   "(a case for using neighbouring-depth context).")

# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------
with tab_data:
    i1, i2, i3 = st.columns(3)
    i1.metric("Input rows", f"{len(raw_df):,}")
    i2.metric("Input columns", f"{len(raw_df.columns):,}")
    i3.metric("Model features", len(bundle.features))
    with st.expander("Preview uploaded data"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    out_cols = [c for c in ["SOURCE_ROW", "DEPT", "GR", "RDEP", "RMED", "DTC", "RHOB", "WELL"]
                if c in predictions.columns]
    if has_truth:
        out_cols += ["TRUE_CODE", "TRUE_LITHOLOGY"]
    for _, key in models:
        out_cols += [f"{key}_PREDICTION_CODE", f"{key}_PREDICTION", f"{key}_CONFIDENCE"]
    out_cols = list(dict.fromkeys(c for c in out_cols if c in predictions.columns))

    export_df = view_df[out_cols].copy()
    if len(models) >= 2:
        cons = consensus(view_df, models)
        export_df["CONSENSUS"] = cons["CONSENSUS"].to_numpy()
        export_df["MODEL_AGREEMENT"] = cons["AGREEMENT"].to_numpy()
    if has_truth:
        for _, key in models:
            export_df[f"{key}_CORRECT"] = np.where(
                export_df["TRUE_CODE"].isna(), np.nan,
                (export_df["TRUE_CODE"] == export_df[f"{key}_PREDICTION_CODE"]).astype(float))

    st.dataframe(export_df.head(1000), use_container_width=True, height=380)
    st.caption("Preview shows up to 1,000 rows; the CSV has every row in the selected interval.")

    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download predictions (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{well}_predictions.csv", mime="text/csv", use_container_width=True)
    if has_truth and len(labelled_view):
        yt = labelled_view["TRUE_CODE"].to_numpy().astype(int)
        mt = overall_metrics(yt, {n: labelled_view[f"{k}_PREDICTION_CODE"].to_numpy().astype(int)
                                  for n, k in models})
        d2.download_button("⬇ Download metrics (CSV)", mt.to_csv(index=False).encode("utf-8"),
                           file_name=f"{well}_metrics.csv", mime="text/csv", use_container_width=True)
    else:
        d2.button("⬇ Metrics unavailable (no labels)", disabled=True, use_container_width=True)
