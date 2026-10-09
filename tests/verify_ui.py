"""Verify the upgraded UI is installed correctly and behaves as expected.

Run from the project root:
    python tests/verify_ui.py                                   # structure + synthetic checks
    python tests/verify_ui.py "data/splits/wells/blind_test/31_2-10.las"   # + real models/LAS
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("PASS  " if ok else "FAIL  ") + msg)


# ---- 1. files exist and contain the newest features --------------------------------
MARKERS = {
    "app_v2.py": ["assets\" / \"style.css", "build_crossplot_compare", "style_figure",
                  "Original vs model(s)", "Statistics", "lithology_mix"],
    "src/plots_extra.py": ["def build_crossplot_compare", "def boundary_analysis", "def consensus"],
    "src/theme.py": ["LITH_COLORS", "VIVID_COLORS", "def style_figure", "open"],
    "src/well_stats.py": ["def curve_summary", "def pairwise_kappa", "def lithology_mix"],
    "assets/style.css": ["aria-selected", "masthead", "well-line"],
    ".streamlit/config.toml": ["primaryColor"],
}
for rel, needles in MARKERS.items():
    p = ROOT / rel
    if not p.is_file():
        check(False, f"{rel} exists")
        continue
    text = p.read_text(encoding="utf-8")
    check(True, f"{rel} exists")
    for n in needles:
        check(n in text, f"{rel} contains '{n}'  (newest version installed)")

# ---- 2. syntax -------------------------------------------------------------------
for rel in ("app_v2.py", "src/plots_extra.py", "src/theme.py", "src/well_stats.py"):
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(True, f"{rel} compiles")
    except Exception as exc:  # noqa: BLE001
        check(False, f"{rel} compiles: {exc}")

# ---- 3. synthetic figure checks ---------------------------------------------------
try:
    import numpy as np
    import pandas as pd

    from src.inference import LITHOLOGY_NAMES
    from src.plots import build_tracks_figure
    from src.plots_extra import (
        boundary_analysis, build_consensus_figure, build_crossplot_compare, consensus)
    from src.theme import VIVID_COLORS, style_figure

    rng = np.random.default_rng(0)
    n = 2000
    codes = list(LITHOLOGY_NAMES)
    true = np.repeat(rng.choice(codes[:6], 40), 50).astype(float)
    df = pd.DataFrame({
        "DEPT": np.arange(n) * 0.15 + 500, "GR": rng.random(n) * 100, "RDEP": rng.random(n) * 50 + .1,
        "RMED": rng.random(n) * 50 + .1, "DTC": rng.random(n) * 50 + 60, "RHOB": rng.random(n) + 2,
        "TRUE_CODE": true})
    df["TRUE_LITHOLOGY"] = df["TRUE_CODE"].map(LITHOLOGY_NAMES)
    models = [("W5", "W5"), ("XGBoost", "XGBOOST")]
    for _, k in models:
        p = np.where(rng.random(n) < .8, true, rng.choice(codes, n))
        df[f"{k}_PREDICTION_CODE"] = p
        df[f"{k}_PREDICTION"] = pd.Series(p).map(LITHOLOGY_NAMES)
        df[f"{k}_CONFIDENCE"] = rng.random(n)

    f = style_figure(build_tracks_figure(df, models, curves=["GR"]))
    hm = [t for t in f.data if t.type == "heatmap"]
    check(hm and hm[0].colorscale[0][1] == VIVID_COLORS["Sandstone"], "tracks use the FORCE lithology palette")
    sw = [t.marker.size for t in f.data if t.type == "scatter" and t.x is not None and len(t.x) == 1]
    check(sw and min(sw) >= 20, "legend swatches are enlarged")

    f2 = build_crossplot_compare(df, "GR", "RHOB", [("Original", "TRUE_LITHOLOGY"), ("W5", "W5_PREDICTION")])
    check(len(f2.layout.annotations) == 2, "side-by-side cross-plot has Original + model panel")
    check("agree with original" in f2.layout.annotations[1].text, "model panel shows agreement with original")
    check(f2.layout.xaxis2.matches == "x", "panels share the same x-axis")

    cons = consensus(df, models)
    check(cons["AGREEMENT"].between(0, 1).all(), "consensus agreement is within 0..1")
    check(len(boundary_analysis(df, models)) == 2, "boundary analysis returns one row per model")
    build_consensus_figure(df, models)
    check(True, "consensus figure builds")

    from src.well_stats import curve_summary, lithology_mix, pairwise_kappa, transition_table
    cs = curve_summary(df)
    check(list(cs["Curve"]) == ["GR", "RDEP", "RMED", "DTC", "RHOB"], "curve summary covers the five logs")
    mix = lithology_mix(df, models, include_true=True)
    check(not mix.empty and set(mix["Source"]) >= {"Original", "W5", "XGBoost"}, "lithology mix has original and models")
    kap = pairwise_kappa(df, models)
    check(kap.shape == (3, 3) and np.allclose(np.diag(kap), 1), "kappa matrix includes original and is 1 on the diagonal")
    trans = transition_table(df, "TRUE_LITHOLOGY")
    check("Contacts" in trans.columns, "transition table lists bed contacts")
except Exception as exc:  # noqa: BLE001
    check(False, f"synthetic figure checks: {type(exc).__name__}: {exc}")

# ---- 4. optional: real models + real LAS -----------------------------------------
if len(sys.argv) > 1:
    try:
        from src.inference import detect_label_column, predict_well, read_uploaded_file
        from src.plots_extra import consensus as _cons

        path = Path(sys.argv[1])
        raw = read_uploaded_file(path)
        label = detect_label_column(raw.columns)
        pred = predict_well(raw, well_name=path.stem, label_column=label)
        check(len(pred) > 0, f"real inference ran: {len(pred):,} rows from {path.name}")
        for key in ("W5", "XGBOOST", "LIGHTGBM"):
            check(f"{key}_PREDICTION" in pred.columns, f"{key} predictions present")
        if "TRUE_CODE" in pred.columns and pred["TRUE_CODE"].notna().any():
            lab = pred[pred["TRUE_CODE"].notna()]
            for key in ("W5", "XGBOOST", "LIGHTGBM"):
                acc = float((lab[f"{key}_PREDICTION_CODE"] == lab["TRUE_CODE"]).mean())
                print(f"      {key}: accuracy on labelled rows = {acc:.3f}  (compare with the app's Evaluate tab)")
        mdl = [("W5", "W5"), ("XGBoost", "XGBOOST"), ("LightGBM", "LIGHTGBM")]
        check(float(_cons(pred, mdl)["AGREEMENT"].min()) >= 1 / 3, "consensus on real predictions is valid")
    except Exception as exc:  # noqa: BLE001
        check(False, f"real-data check: {type(exc).__name__}: {exc}")

failed = [m for ok, m in results if not ok]
print("\n" + ("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED:"))
for m in failed:
    print("  -", m)
sys.exit(1 if failed else 0)
