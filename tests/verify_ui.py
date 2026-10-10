
"""Verify the upgraded UI is installed correctly and behaves as expected.

Run from the project root:
    python tests/verify_ui.py                                   # structure + synthetic checks
    python tests/verify_ui.py "data/splits/wells/blind_test/31_2-10.las"   # + real models/LAS
    python tests/verify_ui.py --no-pytest                       # skip the pytest suite

Exit code 0 = everything passed. A failed "contains" check means that file on THIS computer is older
than the version in the repository; the script says which file to refresh.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RUN_PYTEST = "--no-pytest" not in sys.argv
results: list[tuple[bool, str]] = []
stale: dict[str, list[str]] = {}


def check(ok: object, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("PASS  " if ok else "FAIL  ") + msg)


def traces(fig: Any, kind: str) -> list[Any]:
    """Return the traces of a Plotly figure with the given type.

    Plotly's Figure type stubs make Pylance think `fig.data` iterates over the strings
    'data' / 'layout' / 'frames'; routing through Any avoids those false errors.
    """
    return [t for t in fig.data if getattr(t, "type", None) == kind]


# ---- 1. files exist and contain the newest features --------------------------------
MARKERS = {
    "final_app.py": ["assets\" / \"style.css", "build_crossplot_compare", "style_figure",
                  "Original vs model(s)", "Statistics", "lithology_mix", "build_confusion_grid"],
    "src/plots_extra.py": ["def build_crossplot_compare", "def boundary_analysis", "def consensus",
                           "def build_confusion_grid"],
    "tests/test_confusion_grid.py": ["def test_app_calls_the_grid_inside_the_evaluate_tab"],
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
        ok = n in text
        check(ok, f"{rel} contains '{n}'  (newest version installed)")
        if not ok:
            stale.setdefault(rel, []).append(n)

# ---- 2. syntax -------------------------------------------------------------------
for rel in ("final_app.py", "src/plots_extra.py", "src/theme.py", "src/well_stats.py"):
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        check(True, f"{rel} compiles")
    except Exception as exc:  # noqa: BLE001
        check(False, f"{rel} compiles: {exc}")


def confusion_cross_check(pred_df, models, label: str) -> None:
    """The confusion grid must agree with the numbers the Evaluate tab shows."""
    import numpy as np

    from src.evaluation import overall_metrics
    from src.plots_extra import build_confusion_grid

    lab = pred_df[pred_df["TRUE_CODE"].notna()]
    y = lab["TRUE_CODE"].to_numpy().astype(int)
    preds = {n: lab[f"{k}_PREDICTION_CODE"].to_numpy().astype(int) for n, k in models}
    fig = build_confusion_grid(y, preds, normalize=False)
    panels = traces(fig, "heatmap")
    check(len(panels) == len(models), f"{label}: one confusion panel per selected model")
    acc = overall_metrics(y, preds).set_index("Model")["Accuracy"]
    for t, (name, _) in zip(panels, models):
        z = np.asarray(t.z, dtype=float)
        grid_acc = float(np.trace(z) / z.sum())
        check(abs(grid_acc - float(acc[name])) < 1e-9,
              f"{label}: {name} accuracy from the confusion grid ({grid_acc:.4f}) equals the Evaluate-tab accuracy")
    norm = traces(build_confusion_grid(y, preds, normalize=True), "heatmap")
    row_sums = [np.asarray(t.z, dtype=float).sum(axis=1) for t in norm]
    rows_ok = all(bool(np.allclose(s[s > 0], 1.0)) for s in row_sums)
    check(rows_ok, f"{label}: every non-empty row of the normalised matrix sums to 100%")


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
    hm = traces(f, "heatmap")
    check(bool(hm) and hm[0].colorscale[0][1] == VIVID_COLORS["Sandstone"],
          "tracks use the FORCE lithology palette")
    sw = [t.marker.size for t in traces(f, "scatter") if t.x is not None and len(t.x) == 1]
    check(bool(sw) and min(sw) >= 20, "legend swatches are enlarged")

    f2 = cast(Any, build_crossplot_compare(
        df, "GR", "RHOB", [("Original", "TRUE_LITHOLOGY"), ("W5", "W5_PREDICTION")]))
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
    confusion_cross_check(df, models, "synthetic")
except Exception as exc:  # noqa: BLE001
    check(False, f"synthetic figure checks: {type(exc).__name__}: {exc}")

# ---- 4. optional: real models + real LAS -----------------------------------------
if ARGS:
    try:
        from src.inference import detect_label_column, predict_well, read_uploaded_file
        from src.plots_extra import consensus as _cons

        path = Path(ARGS[0])
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
        mdl = [("Ex-Tree", "W5"), ("XGBoost", "XGBOOST"), ("LightGBM", "LIGHTGBM")]
        check(float(_cons(pred, mdl)["AGREEMENT"].min()) >= 1 / 3, "consensus on real predictions is valid")
        if "TRUE_CODE" in pred.columns and pred["TRUE_CODE"].notna().any():
            confusion_cross_check(pred, mdl, "real well")
        else:
            print("      (no original lithology in this file: confusion-matrix cross-check skipped)")
    except Exception as exc:  # noqa: BLE001
        check(False, f"real-data check: {type(exc).__name__}: {exc}")

if RUN_PYTEST:
    import subprocess

    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    check(proc.returncode == 0, f"pytest suite: {tail}")
    if proc.returncode != 0:
        print(proc.stdout[-1500:], proc.stderr[-500:])

failed = [m for ok, m in results if not ok]
print("\n" + ("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED:"))
for m in failed:
    print("  -", m)
if stale:
    print("\nThese files on this computer are OLDER than the repository version - refresh them "
          "(git pull, or copy the newest download over them), then run this script again:")
    for rel, missing in stale.items():
        print(f"  - {rel}   (missing: {', '.join(repr(m) for m in missing)})")
sys.exit(1 if failed else 0)