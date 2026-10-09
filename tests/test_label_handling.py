"""Tests for original-lithology (reference label) handling.

Run from the project root:

    python -m unittest tests.test_label_handling -v

Uses small synthetic data and FAKE frozen models, so no model files or real
plotly rendering are needed. If plotly is not installed, a tiny stand-in is used
so the track-building logic can still be checked.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------------------
# Use real plotly when available; otherwise a minimal recording stand-in.
# --------------------------------------------------------------------------
try:
    import plotly  # noqa: F401
    STUBBED = False
except ImportError:
    STUBBED = True
    BAR_CALLS: list = []

    class _Trace:
        def __init__(self, type_, **kw):
            self.type = type_
            self.__dict__.update(kw)

        # figure-like no-ops so px.* stand-ins can be post-processed
        def update_layout(self, **kw): pass
        def update_xaxes(self, **kw): pass
        def update_yaxes(self, **kw): pass
        def add_annotation(self, **kw): pass

    class _Fig:
        def __init__(self, titles):
            self.data = []
            self.layout = SimpleNamespace(
                annotations=[SimpleNamespace(text=t) for t in (titles or [])]
            )

        def add_trace(self, trace, row=None, col=None): self.data.append(trace)
        def update_xaxes(self, **kw): pass
        def update_yaxes(self, **kw): pass
        def update_annotations(self, **kw): pass
        def update_layout(self, **kw): pass

    _go = types.ModuleType("plotly.graph_objects")
    _go.Heatmap = lambda **kw: _Trace("heatmap", **kw)
    _go.Scatter = lambda **kw: _Trace("scatter", **kw)
    _sub = types.ModuleType("plotly.subplots")
    _sub.make_subplots = lambda **kw: _Fig(kw.get("subplot_titles"))
    _px = types.ModuleType("plotly.express")
    _px.bar = lambda data, **kw: (BAR_CALLS.append(data), _Trace("bar"))[1]
    _px.imshow = lambda *a, **kw: _Trace("imshow")
    _plotly = types.ModuleType("plotly")
    _plotly.graph_objects, _plotly.subplots, _plotly.express = _go, _sub, _px
    sys.modules.update({
        "plotly": _plotly, "plotly.graph_objects": _go,
        "plotly.subplots": _sub, "plotly.express": _px,
    })

import src.inference as inf  # noqa: E402
from src.evaluation import describe_labels, overall_metrics  # noqa: E402
from src.inference import (  # noqa: E402
    LITHOLOGY_NAMES,
    build_feature_matrix,
    decode_true_labels,
    detect_label_column,
    label_diagnostics,
    predict_well,
)
from src.plots import build_distribution_figure, build_tracks_figure  # noqa: E402

LABEL = "FORCE_2020_LITHOFACIES_LITHOLOGY"
ALL_MODELS = [("W5", "W5"), ("XGBoost", "XGBOOST"), ("LightGBM", "LIGHTGBM")]


# --------------------------------------------------------------------------
# Fake frozen models (deterministic, depend on the GR feature only)
# --------------------------------------------------------------------------
class FakeW5:
    classes_ = np.array(sorted(LITHOLOGY_NAMES), dtype=float)
    n_features_in_ = 41

    def predict(self, X):
        return np.where(X["GR"].to_numpy() < 60, 65000.0, 30000.0)

    def predict_proba(self, X):
        out = np.full((len(X), 12), 0.1 / 11)
        pred = self.predict(X)
        for i, code in enumerate(pred):
            out[i, list(self.classes_).index(code)] = 0.9
        return out


class FakeBoost:
    classes_ = np.arange(12)
    n_features_in_ = 41

    def predict(self, X):  # index 1 -> 65000, index 0 -> 30000
        return np.where(X["GR"].to_numpy() < 60, 1, 0)

    def predict_proba(self, X):
        out = np.full((len(X), 12), 0.1 / 11)
        out[np.arange(len(X)), self.predict(X)] = 0.9
        return out


IDX_TO_CODE = dict(enumerate([30000, 65000, 65030, 70000, 70032, 74000,
                              80000, 86000, 88000, 90000, 93000, 99000]))


def fake_bundle():
    feats = build_feature_matrix(make_raw())[0].columns.tolist()
    return inf.InferenceBundle(
        models={"W5": FakeW5(), "XGBoost": FakeBoost(), "LightGBM": FakeBoost()},
        features=feats, idx_to_code=IDX_TO_CODE,
    )


def make_raw(n: int = 200, label=None) -> pd.DataFrame:
    """Synthetic well: GR alternates in blocks, so lithology follows GR."""
    rng = np.random.default_rng(0)
    gr = np.where((np.arange(n) // 20) % 2 == 0, 40.0, 90.0) + rng.normal(0, 2, n)
    df = pd.DataFrame({
        "DEPT": 1000 + np.arange(n) * 0.5,
        "GR": gr,
        "RDEP": rng.uniform(1, 30, n),
        "RMED": rng.uniform(1, 30, n),
        "DTC": rng.uniform(70, 130, n),
        "RHOB": rng.uniform(2.0, 2.7, n),
    })
    if label is not None:
        df[LABEL] = label
    return df


def true_code_for(raw: pd.DataFrame) -> np.ndarray:
    return np.where(raw["GR"].to_numpy() < 60, 65000, 30000).astype(float)


# --------------------------------------------------------------------------
# Helpers to inspect a built figure (works for real plotly and the stand-in)
# --------------------------------------------------------------------------
def titles(fig):
    return [a.text.replace("<br>", " ") for a in fig.layout.annotations]


def heatmaps(fig):
    return [t for t in fig.data if t.type == "heatmap"]


def hm(fig, key):
    return [t for t in heatmaps(fig) if key in t.hovertemplate]


def legend_names(fig):
    return [t.name for t in fig.data if t.type == "scatter" and t.name]


def has_error_tracks(fig):
    return bool(hm(fig, "error"))


def has_true_track(fig):
    return bool(hm(fig, "<b>True</b>"))


def prediction_tracks(fig):
    return [t for t in heatmaps(fig) if "error" not in t.hovertemplate
            and "<b>True</b>" not in t.hovertemplate]


# --------------------------------------------------------------------------
class LabelDecodingTests(unittest.TestCase):
    def test_numeric_codes_and_names_decode_to_force_codes(self):  # scenario 7
        series = pd.Series([65000, 65000.0, "30000", "Shale", " sandstone ", "Marl",
                            "Sandstone/Shale", 12345, "Granite", None, np.nan, "None"])
        out = decode_true_labels(series).tolist()
        self.assertEqual(out[:7], [65000, 65000, 30000, 65000, 30000, 80000, 65030])
        self.assertTrue(all(np.isnan(v) for v in out[7:]))
        self.assertEqual(LITHOLOGY_NAMES[65000], "Shale")

    def test_detect_label_column(self):
        self.assertEqual(detect_label_column(["DEPT", "gr", " lithology "]), " lithology ")
        self.assertEqual(detect_label_column(["DEPT", LABEL]), LABEL)
        self.assertIsNone(detect_label_column(["DEPT", "GR", "RHOB"]))

    def test_diagnostics_separate_empty_from_unrecognised(self):
        raw = make_raw(10, label=["None", None, np.nan, "Granite", "Granite",
                                  "Shale", 65000, 99999, "", "30000"])
        d = label_diagnostics(raw, LABEL)
        self.assertEqual(d["recognised"], 3)           # Shale, 65000, "30000"
        self.assertEqual(d["unrecognised"], 3)         # Granite x2, 99999
        self.assertEqual(d["empty"], 4)                # "None", None, NaN, ""
        self.assertEqual(d["empty"] + d["recognised"] + d["unrecognised"], 10)
        self.assertEqual(d["unrecognised_examples"]["Granite"], 2)


class StatusTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(inf, "load_inference_bundle", lambda models_dir=None: fake_bundle())
        patcher.start()
        self.addCleanup(patcher.stop)

    def predict(self, label):
        raw = make_raw(200, label=label) if label is not None else make_raw(200)
        return raw, predict_well(raw, label_column=LABEL if label is not None else None)

    def test_scenario1_no_label_column(self):
        raw, res = self.predict(None)
        self.assertIsNone(detect_label_column(raw.columns))
        self.assertNotIn("TRUE_CODE", res.columns)
        self.assertNotIn("TRUE_LITHOLOGY", res.columns)
        info = describe_labels(res)
        self.assertEqual((info["status"], info["labelled"]), ("no_column", 0))

    def test_scenario2_column_present_but_all_missing(self):
        for label in (np.nan, None, "None"):
            with self.subTest(label=label):
                raw, res = self.predict([label] * 200)
                self.assertIn("TRUE_CODE", res.columns)            # column exists...
                self.assertEqual(int(res["TRUE_CODE"].notna().sum()), 0)   # ...no labels
                diag = label_diagnostics(raw, LABEL)
                info = describe_labels(res, diag)
                self.assertEqual((info["status"], info["labelled"]), ("no_valid_labels", 0))
                self.assertIn(LABEL, info["message"])

    def test_scenario2b_only_unrecognised_values(self):
        raw, res = self.predict(["Granite"] * 200)
        info = describe_labels(res, label_diagnostics(raw, LABEL))
        self.assertEqual(info["status"], "no_valid_labels")
        self.assertIn("Granite", info["message"])

    def test_scenario3_partial_labels(self):
        raw0 = make_raw(200)
        label = true_code_for(raw0).astype(object)
        label[:80] = None                                  # top 40 m unlabelled
        raw, res = self.predict(list(label))
        info = describe_labels(res)
        self.assertEqual((info["status"], info["labelled"], info["samples"]), ("partial", 120, 200))
        self.assertAlmostEqual(info["depth_min"], 1000 + 80 * 0.5)
        self.assertAlmostEqual(info["depth_max"], 1000 + 199 * 0.5)
        self.assertIn("120", info["message"])

    def test_scenario4_full_labels(self):
        raw0 = make_raw(200)
        raw, res = self.predict(list(true_code_for(raw0)))
        info = describe_labels(res)
        self.assertEqual((info["status"], info["labelled"]), ("full", 200))
        self.assertIn("TRUE_LITHOLOGY", res.columns)
        self.assertTrue((res["TRUE_LITHOLOGY"].isin(LITHOLOGY_NAMES.values())).all())

    def test_scenario9_predictions_identical_with_and_without_label(self):
        raw0 = make_raw(200)
        _, with_label = self.predict(list(true_code_for(raw0)))
        _, without = self.predict(None)
        common = [c for c in without.columns]
        pd.testing.assert_frame_equal(with_label[common], without[common])
        extra = set(with_label.columns) - set(without.columns)
        self.assertEqual(extra, {"TRUE_CODE", "TRUE_LITHOLOGY"})

    def test_model_schema_unchanged_by_labels(self):
        raw0 = make_raw(200, label=list(true_code_for(make_raw(200))))
        X_with, _ = build_feature_matrix(raw0, label_column=LABEL)
        X_without, _ = build_feature_matrix(raw0.drop(columns=[LABEL]))
        self.assertEqual(X_with.shape[1], 41)
        self.assertEqual(list(X_with.columns), list(X_without.columns))
        pd.testing.assert_frame_equal(X_with, X_without)
        self.assertNotIn("TRUE_CODE", X_with.columns)      # label never a model input

    def test_models_to_run_subset(self):
        res = predict_well(make_raw(200), models_to_run=["XGBoost"])
        self.assertIn("XGBOOST_PREDICTION", res.columns)
        self.assertNotIn("W5_PREDICTION", res.columns)


class PlotLogicTests(unittest.TestCase):
    def table(self, labelled_from: int | None, n: int = 200, wrong_unlabelled=True):
        """Prediction table. Labelled rows (>= labelled_from) are predicted
        correctly; unlabelled rows are predicted deliberately WRONG so that any
        accuracy computed over them would drop below 100%."""
        depth = 1000 + np.arange(n) * 0.5
        true = np.full(n, 65000.0)
        pred = np.full(n, 65000)
        d = pd.DataFrame({"DEPT": depth, "GR": 50.0})
        if labelled_from is None:
            d["TRUE_CODE"] = np.nan
            d["TRUE_LITHOLOGY"] = None
        else:
            mask = np.arange(n) >= labelled_from
            d["TRUE_CODE"] = np.where(mask, true, np.nan)
            d["TRUE_LITHOLOGY"] = [LITHOLOGY_NAMES[int(c)] if c == c else None for c in d["TRUE_CODE"]]
            if wrong_unlabelled:
                pred = np.where(mask, pred, 30000)
        for _, key in ALL_MODELS:
            d[f"{key}_PREDICTION_CODE"] = pred
            d[f"{key}_PREDICTION"] = [LITHOLOGY_NAMES[int(c)] for c in pred]
            d[f"{key}_CONFIDENCE"] = 0.8
        return d

    def test_scenario1_2_no_labels_means_no_true_error_or_accuracy(self):
        d = self.table(labelled_from=None)
        for show_true, show_errors in [(True, True), (True, False), (False, True)]:
            with self.subTest(show_true=show_true, show_errors=show_errors):
                fig = build_tracks_figure(d, ALL_MODELS, show_true=show_true, show_errors=show_errors)
                self.assertFalse(has_true_track(fig))
                self.assertFalse(has_error_tracks(fig))
                self.assertEqual(len(prediction_tracks(fig)), 3)       # predictions still shown
                self.assertFalse(any("acc" in t for t in titles(fig)))
                self.assertFalse({"Correct", "Wrong", "No original label"} & set(legend_names(fig)))

    def test_scenario5_original_off_keeps_predictions_and_drops_errors(self):
        d = self.table(labelled_from=0)
        fig = build_tracks_figure(d, ALL_MODELS, show_true=False, show_errors=True)
        self.assertFalse(has_true_track(fig))
        self.assertFalse(has_error_tracks(fig))          # error strips need the original track
        self.assertEqual(len(prediction_tracks(fig)), 3)

    def test_scenario6_errors_can_be_switched_off_with_labels(self):
        d = self.table(labelled_from=0)
        fig = build_tracks_figure(d, ALL_MODELS, show_true=True, show_errors=False)
        self.assertTrue(has_true_track(fig))
        self.assertFalse(has_error_tracks(fig))

    def test_scenario3_partial_unlabelled_is_blank_not_correct_or_wrong(self):
        d = self.table(labelled_from=80)                 # 80 unlabelled, 120 labelled
        fig = build_tracks_figure(d, ALL_MODELS, show_true=True, show_errors=True)
        self.assertTrue(has_true_track(fig))
        for trace in hm(fig, "error"):
            z = np.asarray(trace.z, dtype=float).ravel()
            self.assertEqual(int(np.isnan(z).sum()), 80)        # unlabelled -> blank
            self.assertEqual(set(z[~np.isnan(z)].tolist()), {0.0})   # labelled rows all correct
        true_z = np.asarray(hm(fig, "<b>True</b>")[0].z, dtype=float).ravel()
        self.assertEqual(int(np.isnan(true_z).sum()), 80)
        self.assertIn("No original label", legend_names(fig))

    def test_scenario8_accuracy_uses_labelled_rows_only(self):
        d = self.table(labelled_from=80)                 # unlabelled rows are all wrong
        fig = build_tracks_figure(d, ALL_MODELS)
        acc_titles = [t for t in titles(fig) if "acc" in t]
        self.assertEqual(len(acc_titles), 3)
        self.assertTrue(all("100.0% acc" in t for t in acc_titles), acc_titles)
        # If unlabelled rows had been counted, accuracy would be 60.0%.
        labelled = d[d["TRUE_CODE"].notna()]
        metrics = overall_metrics(
            labelled["TRUE_CODE"].astype(int).to_numpy(),
            {"W5": labelled["W5_PREDICTION_CODE"].to_numpy()},
        )
        self.assertAlmostEqual(float(metrics.loc[0, "Accuracy"]), 1.0)

    def test_scenario4_full_labels_show_everything(self):
        d = self.table(labelled_from=0)
        fig = build_tracks_figure(d, ALL_MODELS, show_true=True, show_errors=True)
        self.assertTrue(has_true_track(fig))
        self.assertEqual(len(hm(fig, "error")), 3)
        self.assertNotIn("No original label", legend_names(fig))
        self.assertTrue({"Correct", "Wrong"} <= set(legend_names(fig)))

    def test_depth_range_without_labels_hides_reference(self):
        d = self.table(labelled_from=150)                # labels only for the last 50 samples
        fig = build_tracks_figure(d, ALL_MODELS, depth_range=(1000, 1050))   # unlabelled part only
        self.assertFalse(has_true_track(fig))
        self.assertFalse(has_error_tracks(fig))
        self.assertEqual(len(prediction_tracks(fig)), 3)

    def test_only_selected_models_are_plotted(self):
        d = self.table(labelled_from=0)
        fig = build_tracks_figure(d, [("XGBoost", "XGBOOST")], show_errors=False)
        self.assertEqual(len(prediction_tracks(fig)), 1)
        self.assertIn("XGBoost", prediction_tracks(fig)[0].hovertemplate)

    @unittest.skipUnless(STUBBED, "inspects the stand-in's recorded data")
    def test_distribution_compares_same_samples_when_partially_labelled(self):
        BAR_CALLS.clear()
        d = self.table(labelled_from=80)
        build_distribution_figure(d, ALL_MODELS, include_true=True)
        data = BAR_CALLS[-1]
        totals = data.groupby("Source")["Samples"].sum()
        self.assertTrue((totals == 120).all(), totals.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
