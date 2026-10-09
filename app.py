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
    detect_label_column,
    label_diagnostics,
    load_inference_bundle,
    predict_well,
    read_uploaded_file,
)
from src.plots import (
    build_confusion_figure,
    build_distribution_figure,
    build_tracks_figure,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Lithology Predictor", page_icon="🪨", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {
        background: rgba(128,140,160,0.10);
        border: 1px solid rgba(128,140,160,0.25);
        padding: 12px 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🪨 Lithology Predictor")
st.caption("Machine-learning lithology prediction from well-log data, with optional comparison to the original lithology.")

MODEL_OPTIONS = {"W5": "W5", "XGBoost": "XGBOOST", "LightGBM": "LIGHTGBM"}
REQUIRED_CURVES = ["GR", "RDEP", "RMED", "DTC", "RHOB"]


# ============================================================
# LOAD TRAINED MODELS
# ============================================================
@st.cache_resource
def get_bundle():
    return load_inference_bundle()


try:
    bundle = get_bundle()
except Exception as exc:
    st.error(f"Could not load the trained models: {exc}")
    st.stop()


# ============================================================
# CACHED DATA / INFERENCE
# ============================================================
@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return read_uploaded_file(BytesIO(file_bytes), filename=filename)


@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes, filename: str, label_column: str | None) -> pd.DataFrame:
    raw = load_raw(file_bytes, filename)
    return predict_well(raw, well_name=Path(filename).stem, label_column=label_column)


# ============================================================
# SIDEBAR: MODELS
# ============================================================
st.sidebar.header("Models")
selected_models = st.sidebar.multiselect(
    "Models to display",
    options=list(MODEL_OPTIONS),
    default=list(MODEL_OPTIONS),
    help="Only the selected models are plotted, compared and exported.",
)
st.sidebar.caption(f"{len(bundle.features)} frozen input features are used by the models.")

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a well-log file",
    type=["las", "csv"],
    help="LAS or CSV with GR, RDEP, RMED, DTC, RHOB. Include a lithology column to compare against the original.",
)

if uploaded_file is None:
    st.info("Upload a LAS or CSV file to begin.")
    st.markdown(
        """
        ### Workflow
        1. Upload your well-log file (optionally with an original lithology column).
        2. Predictions are generated automatically.
        3. Pick the models you want to compare in the sidebar.
        4. Inspect clean depth tracks: **True vs predicted vs error**.
        5. Check metrics, confusion matrix and download results.
        """
    )
    st.stop()

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

file_bytes = uploaded_file.getvalue()
filename = uploaded_file.name

try:
    raw_df = load_raw(file_bytes, filename)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

normalized_columns = {str(c).strip().upper() for c in raw_df.columns}
missing_curves = [c for c in REQUIRED_CURVES if c not in normalized_columns]
if missing_curves:
    st.error("Missing required input curves: " + ", ".join(missing_curves))
    st.stop()


# ============================================================
# SIDEBAR: GROUND TRUTH
# ============================================================
st.sidebar.header("Original lithology")
detected = detect_label_column(raw_df.columns)
label_choice = st.sidebar.selectbox(
    "True lithology column",
    options=["Auto-detect", "None"] + [str(c) for c in raw_df.columns],
    help="Column holding FORCE lithology codes (e.g. 65000) or names (e.g. Shale).",
)
if label_choice == "Auto-detect":
    label_column = detected
    st.sidebar.caption(f"Detected: `{detected}`" if detected else "No lithology column detected.")
elif label_choice == "None":
    label_column = None
else:
    label_column = label_choice


# ============================================================
# RUN INFERENCE (cached; reruns only when file / label column change)
# ============================================================
try:
    with st.spinner("Generating features and running the models..."):
        predictions = run_inference(file_bytes, filename, label_column)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

depth_col, depth_label = "DEPT", "Depth"
if predictions["DEPT"].notna().sum() == 0:
    predictions = predictions.assign(DEPT=predictions["SOURCE_ROW"].astype(float))
    depth_label = "Sample index (no depth column found)"
    st.sidebar.warning("No depth column found; using sample index.")

# A label column that merely EXISTS is not ground truth. Only samples with a
# decoded lithology count; everything else is reported honestly below.
try:
    label_diag = label_diagnostics(raw_df, label_column) if label_column else None
except Exception:
    label_diag = None
label_info = describe_labels(predictions, label_diag, depth_col=depth_col)
has_truth = label_info["labelled"] > 0


def show_label_status():
    """Render the actual label situation (none / empty / partial / full)."""
    getattr(st, label_info["level"])(label_info["message"])


getattr(st.sidebar, label_info["level"])(label_info["message"])

models = [(name, MODEL_OPTIONS[name]) for name in selected_models]


# ============================================================
# SIDEBAR: DISPLAY OPTIONS
# ============================================================
st.sidebar.header("Display")
d_min, d_max = float(predictions[depth_col].min()), float(predictions[depth_col].max())
if d_max > d_min:
    depth_range = st.sidebar.slider(
        "Depth interval", d_min, d_max, (d_min, d_max), step=max((d_max - d_min) / 500, 0.01)
    )
else:
    depth_range = (d_min, d_max)

show_true_req = st.sidebar.checkbox(
    "Show original lithology track",
    value=True,
    disabled=not has_truth,
    help="Predictions are always shown. This only adds/removes the original track."
    if has_truth else "Unavailable: no usable original lithology labels in this file.",
)
# Streamlit returns the stored value for a disabled checkbox, so combine with has_truth.
show_true = bool(show_true_req and has_truth)

show_errors_req = st.sidebar.checkbox(
    "Show error tracks (original vs model)",
    value=True,
    disabled=not show_true,
    help="Needs the original lithology track to be shown."
    if show_true else "Unavailable: turn on the original lithology track (needs valid labels).",
)
show_errors = bool(show_errors_req and show_true)

crop_labelled_req = st.sidebar.checkbox(
    "Limit plots to the labelled interval",
    value=False,
    disabled=label_info["status"] != "partial",
    help="Hide depths above/below where the original lithology exists."
    if label_info["status"] == "partial" else "Only relevant when labels cover part of the well.",
)
crop_labelled = bool(crop_labelled_req and label_info["status"] == "partial")

show_confidence = st.sidebar.checkbox("Show confidence track", value=False)
curves = st.sidebar.multiselect("Log curves to show", REQUIRED_CURVES, default=["GR"])
plot_height = st.sidebar.slider("Plot height (px)", 500, 2000, 950, step=50)
st.sidebar.caption("Predictions are estimates, not verified geological truth.")

if crop_labelled and label_info["depth_min"] is not None:
    depth_range = (
        max(depth_range[0], label_info["depth_min"]),
        min(depth_range[1], label_info["depth_max"]),
    )
    if depth_range[0] > depth_range[1]:
        st.warning("The selected depth interval does not overlap the labelled interval.")
        st.stop()

view_df = predictions[predictions[depth_col].between(*depth_range)]
if view_df.empty:
    st.warning("No samples in the selected depth interval.")
    st.stop()

pred_codes = {name: view_df[f"{key}_PREDICTION_CODE"].to_numpy() for name, key in models}
labelled_view = view_df[view_df["TRUE_CODE"].notna()] if has_truth else view_df.iloc[0:0]


# ============================================================
# TABS
# ============================================================
tab_overview, tab_tracks, tab_eval, tab_data = st.tabs(
    ["Overview", "Lithology tracks", "Evaluation", "Data & export"]
)

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted samples", f"{len(view_df):,}")
    c2.metric("Depth interval", f"{depth_range[0]:,.0f} – {depth_range[1]:,.0f}")
    c3.metric("Labelled samples", f"{len(labelled_view):,}" if has_truth else "—")

    if len(models) >= 2:
        stacked = np.vstack(list(pred_codes.values()))
        agree = float((stacked == stacked[0]).all(axis=0).mean())
        c4.metric("Models agree", f"{agree:.1%}", help="Share of samples where all selected models predict the same lithology.")
    else:
        conf_col = f"{models[0][1]}_CONFIDENCE"
        if conf_col in view_df.columns:
            c4.metric("Mean confidence", f"{view_df[conf_col].mean():.1%}")

    st.subheader("Lithology distribution")
    st.plotly_chart(
        build_distribution_figure(view_df, models, include_true=has_truth),
        use_container_width=True,
    )

    st.subheader("Model summary")
    summary_rows = []
    for name, key in models:
        row = {
            "Model": name,
            "Lithology classes predicted": view_df[f"{key}_PREDICTION"].nunique(),
        }
        conf_col = f"{key}_CONFIDENCE"
        if conf_col in view_df.columns:
            row["Mean confidence"] = view_df[conf_col].mean()
        if has_truth and len(labelled_view):
            row["Accuracy"] = float(
                (labelled_view[f"{key}_PREDICTION_CODE"].to_numpy(float)
                 == labelled_view["TRUE_CODE"].to_numpy(float)).mean()
            )
        summary_rows.append(row)
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Mean confidence": st.column_config.NumberColumn(format="%.3f"),
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption("Confidence is the mean predicted-class probability. It is not a guarantee of accuracy and may not be calibrated.")

# ------------------------------------------------------------
# LITHOLOGY TRACKS
# ------------------------------------------------------------
with tab_tracks:
    show_label_status()
    if show_errors:
        st.caption("Original lithology on the left, then each selected model's prediction with a green/red error strip (blank = no original label). Drag to zoom, double-click to reset.")
    elif show_true:
        st.caption("Original lithology next to each selected model's prediction. Drag to zoom, double-click to reset.")
    else:
        st.caption("Predicted lithology for each selected model, shown without a reference. Drag to zoom, double-click to reset.")

    try:
        fig_tracks = build_tracks_figure(
            predictions,
            models,
            depth_col=depth_col,
            depth_label=depth_label,
            depth_range=depth_range,
            show_true=show_true,
            show_errors=show_errors,
            show_confidence=show_confidence,
            curves=curves,
            height=plot_height,
            title=f"{Path(filename).stem}: lithology by depth",
        )
        st.plotly_chart(
            fig_tracks,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "scale": 2,
                    "filename": f"{Path(filename).stem}_lithology_tracks",
                },
            },
        )
    except ValueError as exc:
        st.warning(str(exc))

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
with tab_eval:
    show_label_status()
    if not has_truth:
        st.caption(
            "Accuracy, FORCE score and the confusion matrix need valid original labels, "
            "for example a `FORCE_2020_LITHOFACIES_LITHOLOGY` (codes) or `LITHOLOGY` "
            "(names) column with values. Nothing is calculated without them."
        )
    else:
        use_view = st.checkbox("Evaluate only the selected depth interval", value=False)
        eval_df = (view_df if use_view else predictions)
        eval_df = eval_df[eval_df["TRUE_CODE"].notna()]

        if eval_df.empty:
            st.warning("No labelled samples to evaluate.")
        else:
            y_true = eval_df["TRUE_CODE"].to_numpy().astype(int)
            preds = {
                name: eval_df[f"{key}_PREDICTION_CODE"].to_numpy().astype(int)
                for name, key in models
            }

            metrics = overall_metrics(y_true, preds)
            best = metrics.loc[metrics["FORCE score"].idxmax()]
            b1, b2, b3 = st.columns(3)
            b1.metric(
                "Labelled samples used",
                f"{len(eval_df):,}",
                help="Only samples with a valid original label are scored; unlabelled samples are excluded.",
            )
            b2.metric("Best FORCE score", f"{best['FORCE score']:.4f}", help="Negative mean penalty; closer to zero is better.")
            b3.metric("Best model (FORCE)", str(best["Model"]))

            st.subheader("Overall metrics")
            st.dataframe(
                metrics,
                hide_index=True,
                use_container_width=True,
                column_config={
                    col: st.column_config.NumberColumn(format="%.4f")
                    for col in metrics.columns if col != "Model"
                },
            )

            st.subheader("Confusion matrix")
            cm_cols = st.columns([2, 1])
            cm_model = cm_cols[0].selectbox("Model", [name for name, _ in models], key="cm_model")
            cm_normalize = cm_cols[1].radio(
                "Show", ["% of true class", "Sample counts"], horizontal=True
            ) == "% of true class"
            counts, labels = confusion_counts(y_true, preds[cm_model])
            st.plotly_chart(
                build_confusion_figure(counts, labels, normalize=cm_normalize),
                use_container_width=True,
            )

            st.subheader("Per-class performance")
            st.dataframe(
                per_class_report(y_true, preds[cm_model]),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Precision": st.column_config.NumberColumn(format="%.3f"),
                    "Recall": st.column_config.NumberColumn(format="%.3f"),
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                },
            )

# ------------------------------------------------------------
# DATA & EXPORT
# ------------------------------------------------------------
with tab_data:
    st.subheader("Input data")
    i1, i2, i3 = st.columns(3)
    i1.metric("Input rows", f"{len(raw_df):,}")
    i2.metric("Input columns", f"{len(raw_df.columns):,}")
    i3.metric("Model features", len(bundle.features))
    with st.expander("Preview uploaded data"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Prediction table")
    output_columns = [
        c for c in ["SOURCE_ROW", "DEPT", "GR", "RDEP", "RMED", "DTC", "RHOB", "WELL"]
        if c in predictions.columns
    ]
    if has_truth:  # never export an empty "original lithology" column as if it were data
        output_columns += ["TRUE_CODE", "TRUE_LITHOLOGY"]
    for name, key in models:
        output_columns += [
            f"{key}_PREDICTION_CODE", f"{key}_PREDICTION", f"{key}_CONFIDENCE"
        ]
    output_columns = list(dict.fromkeys(c for c in output_columns if c in predictions.columns))

    export_df = view_df[output_columns].copy()
    if has_truth:
        for name, key in models:
            export_df[f"{key}_CORRECT"] = (
                np.where(
                    export_df["TRUE_CODE"].isna(),
                    np.nan,
                    (export_df["TRUE_CODE"] == export_df[f"{key}_PREDICTION_CODE"]).astype(float),
                )
            )

    st.dataframe(export_df.head(1000), use_container_width=True, height=420)
    st.caption("Preview shows up to 1,000 rows. The CSV contains every row in the selected depth interval for the selected models.")

    st.download_button(
        "Download predictions as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(filename).stem}_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
