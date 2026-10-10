"""Model guide page content for the LITHO-ML app.

Edit the text in this file to update the page; no other code needs to change.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# CONTENT (edit here)
# --------------------------------------------------------------------------
INTRO = (
    "This project solves a **12-class supervised lithology classification** problem using well-log "
    "measurements. At each depth sample, a model receives geophysical log values and engineered "
    "features, then predicts a lithology code such as Shale, Sandstone, Limestone, or Dolomite."
)

SHARED = [
    "Trained using the official **98 TRAIN wells** (about **1.17 million** labelled depth samples).",
    "Use the same frozen **41-feature** input schema, in the same order.",
    "Predict among the same **12 FORCE 2020** lithology classes.",
    "Evaluated on the same **10 previously held-out BLIND_TEST wells**, not used for model selection.",
]

FEATURE_GROUPS = [
    ("5", "Base log features", "GR, RDEP_LOG10, RMED_LOG10, DTC, RHOB",
     "Gamma ray, deep and medium resistivity (log-transformed), sonic travel time and bulk density."),
    ("30", "Rolling-statistic features", "Mean and std of each base log",
     "Windows of 5, 9 and 21 samples. They capture local log patterns and changes with depth."),
    ("1", "Resistivity separation", "RESISTIVITY_SEPARATION",
     "Difference between the deep and medium resistivity responses."),
    ("5", "Missingness indicators", "One per base log",
     "Lets the models use information about which measurements are missing."),
]
FEATURE_NOTE = ("The 41 features are shared across the models, so the comparison measures differences between "
                "the learning algorithms and their trained configurations, not differences in input schema.")

MODELS = [
    {
        "name": "Ex-Tree", "full": "ExtraTrees Classifier (internal name: W5)",
        "role": "Selected production candidate", "algorithm": "Extremely Randomized Trees (ExtraTrees)",
        "color": "#1F4E79", "icon": "",
        "summary": ("An ensemble that combines predictions from many randomized decision trees. Each tree learns "
                    "different decision rules from the well-log features, and the ensemble combines them to "
                    "classify lithology at each depth."),
        "trained": [
            "Trained on 98 official TRAIN wells from the FORCE 2020 dataset.",
            "Used 41 frozen engineered features and 12 lithology classes.",
            "Used square-root-balanced class weights to give more importance to underrepresented lithologies.",
            "Final configuration: **100 trees**, **max depth 20**, **min 5 samples per leaf**, **random seed 42**.",
            "Evaluated on the separate, locked 10-well BLIND_TEST set.",
        ],
        "solves": ("Depth-wise lithology classification that accounts for local log patterns, nonlinear "
                   "relationships, missing measurements and class imbalance. Selected as the production "
                   "candidate because it achieved the best official FORCE penalty score of the three final models."),
        "best_for": ("The current production prediction workflow, particularly when overall FORCE evaluation "
                     "performance is the selection criterion."),
    },
    {
        "name": "XGBoost", "full": "Tuned Gradient-Boosted Trees",
        "role": "Alternative boosted-tree model", "algorithm": "Extreme Gradient Boosting (XGBoost)",
        "color": "#6B2E22", "icon": "",
        "summary": ("Builds an ensemble of decision trees sequentially. Each new tree learns patterns associated "
                    "with the remaining prediction errors, and the final prediction combines all trees."),
        "trained": [
            "Trained using the official TRAIN split of 98 wells.",
            "Used the same frozen 41 engineered features as Ex-Tree and LightGBM.",
            "Formulated as a 12-class lithology classification problem.",
            "Used encoded class labels internally; predictions are mapped back to FORCE lithology codes.",
            "Evaluated on the same locked 10-well BLIND_TEST set.",
            "The saved model is the tuned configuration. Exact hyperparameters should be read from the saved "
            "model or its metadata.",
        ],
        "solves": ("A boosting-based alternative to randomized tree ensembles. Sequential tree construction can "
                   "capture complex nonlinear interactions between geophysical measurements and depth-dependent "
                   "patterns."),
        "best_for": ("Comparing boosted-tree learning against Ex-Tree and investigating whether sequential error "
                     "correction improves lithology classification."),
    },
    {
        "name": "LightGBM", "full": "Tuned Gradient-Boosted Trees",
        "role": "Alternative boosted-tree model", "algorithm": "Light Gradient Boosting Machine (LightGBM)",
        "color": "#2C4A3E", "icon": "",
        "summary": ("A gradient-boosting framework designed for efficient tree-based learning. It grows trees "
                    "leaf-wise, choosing splits that improve the objective while applying constraints to control "
                    "model complexity."),
        "trained": [
            "Trained using the official TRAIN split of 98 wells.",
            "Used the same frozen 41 engineered features and 12 lithology classes.",
            "Used encoded class labels internally; predictions are mapped back to FORCE lithology codes.",
            "Evaluated on the same locked 10-well BLIND_TEST set.",
            "The saved model is the tuned configuration; exact hyperparameters come from the saved model or metadata.",
        ],
        "solves": ("An efficient boosting-based approach to nonlinear relationships in well-log data. A useful "
                   "comparison with both Ex-Tree and XGBoost, especially for class-balanced performance and "
                   "recognition of less common lithologies."),
        "best_for": ("Comparing LightGBM's efficiency and classification behaviour with alternative tree "
                     "ensembles, particularly when balanced accuracy and macro F1 matter."),
    },
]

# Recorded results from the project's final blind-test evaluation (percent, FORCE penalty as-is).
RESULTS = {
    "Metric": ["Accuracy", "Balanced accuracy", "Macro F1", "Weighted F1", "FORCE penalty score"],
    "Ex-Tree": [73.87, 46.37, 43.39, 71.37, -0.8745],
    "XGBoost": [73.82, 46.60, 43.67, 71.35, -0.9620],
    "LightGBM": [70.53, 48.22, 45.13, 70.34, -0.9701],
}
HOW_TO_PRESENT = [
    ("Ex-Tree", "best official FORCE penalty score and marginally the highest accuracy."),
    ("XGBoost", "slightly higher macro F1 and balanced accuracy than Ex-Tree."),
    ("LightGBM", "highest balanced accuracy and macro F1 of the three."),
]
RESULTS_NOTE = ("Recorded results from the project's final blind-test evaluation (10 held-out wells), "
                "not generic benchmark values. The FORCE penalty score is negative: closer to zero is better.")


# --------------------------------------------------------------------------
# RENDERING
# --------------------------------------------------------------------------
def _model_card(m: dict) -> None:
    st.markdown(
        f'<div class="model-card" style="border-color:{m["color"]}">'
        f'<div class="model-head" style="background:{m["color"]}">'
        f'<span class="model-name">{m["name"]}</span>'
        f'<span class="model-role">{m["role"]}</span></div>'
        f'<div class="model-algo"><b>Algorithm:</b> {m["algorithm"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(m["summary"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**How it was trained**")
        st.markdown("\n".join(f"- {t}" for t in m["trained"]))
    with c2:
        st.markdown("**What it solves**")
        st.markdown(m["solves"])
        st.markdown("**Best suited for**")
        st.markdown(m["best_for"])


def render_model_guide() -> None:
    st.markdown("## Model notes")
    st.markdown(INTRO)

    st.markdown("### What all three models have in common")
    st.markdown("\n".join(f"- {t}" for t in SHARED))

    st.markdown("### The 41 input features")
    cols = st.columns(4)
    for col, (num, title, names, desc) in zip(cols, FEATURE_GROUPS):
        col.markdown(
            f'<div class="feat-card">'
            f'<div class="feat-num">{num}</div>'
            f'<h4>{title}</h4><code>{names}</code><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )
    st.caption(FEATURE_NOTE)

    st.markdown("### Model architectures")
    tabs = st.tabs([m["name"] for m in MODELS])
    for tab, m in zip(tabs, MODELS):
        with tab:
            _model_card(m)

    st.markdown("### Blind-test results")
    df = pd.DataFrame(RESULTS).set_index("Metric")
    def _fmt(metric: str, v: float) -> str:
        return f"{v:.4f}" if metric == "FORCE penalty score" else f"{v:.2f}%"

    # text = pd.DataFrame({c: [_fmt(i, v) for i, v in df[c].items()] for c in df.columns}, index=df.index)
    text = pd.DataFrame(
        {
            str(c): [_fmt(str(i), float(v)) for i, v in df[c].items()]
            for c in df.columns
        },
        index=df.index,
    )
    # best = df.idxmax(axis=1)  # higher is better for every row (FORCE penalty is negative: max = closest to 0)
    best = df.astype(float).idxmax(axis=1).to_dict()
    styled = text.style.apply(
        lambda row: ["background-color:#d5ddd6;font-weight:600;color:#1e2420" if c == best.get(str(row.name)) else ""
                     for c in row.index], axis=1)
    st.dataframe(styled, use_container_width=True)
    st.caption(RESULTS_NOTE + " Highlighted = best in each row.")

    chart = df.drop(index="FORCE penalty score").reset_index().melt(
        id_vars="Metric", var_name="Model", value_name="Score (%)")
    fig = px.bar(chart, x="Metric", y="Score (%)", color="Model", barmode="group", text_auto=False,
                 color_discrete_map={m["name"]: m["color"] for m in MODELS})
    fig.update_layout(template="plotly_white", height=420, margin=dict(t=20, b=40),
                      legend=dict(orientation="h", y=1.12, font=dict(size=14)), xaxis_title=None)
    fig.update_traces(    texttemplate="%{y:.1f}", textposition="outside", cliponaxis=False)



    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**How to present the comparison**")
    st.markdown("\n".join(f"- **{n}:** {t}" for n, t in HOW_TO_PRESENT))
