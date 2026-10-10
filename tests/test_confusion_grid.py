# """Tests for the side-by-side confusion-matrix grid shown in the Evaluate tab.

# Written before the feature. Run with:  python -m pytest tests/test_confusion_grid.py -q
# """

# from __future__ import annotations

# import ast
# from pathlib import Path

# import numpy as np
# import pytest
# from sklearn.metrics import confusion_matrix

# from src.inference import LITHOLOGY_NAMES

# ROOT = Path(__file__).resolve().parents[1]
# CODES = [30000, 65000, 65030, 70000, 70032, 74000, 80000, 86000, 88000, 90000, 99000]


# def _toy(n=3000, seed=0):
#     rng = np.random.default_rng(seed)
#     y_true = rng.choice(CODES, n, p=[.06, .40, .08, .10, .04, .03, .06, .05, .06, .06, .06])
#     preds = {}
#     for i, name in enumerate(["Ex-Tree", "XGBoost", "LightGBM"]):
#         keep = rng.random(n) < (.75 - .03 * i)
#         preds[name] = np.where(keep, y_true, rng.choice(CODES, n))
#     return y_true, preds


# @pytest.fixture(scope="module")
# def grid():
#     from src.plots_extra import build_confusion_grid
#     return build_confusion_grid


# def _heatmaps(fig):
#     return [t for t in fig.data if t.type == "heatmap"]


# def test_one_panel_per_selected_model(grid):
#     y, p = _toy()
#     fig = grid(y, p)
#     assert len(_heatmaps(fig)) == 3
#     assert [a.text for a in fig.layout.annotations][:3] == ["Ex-Tree", "XGBoost", "LightGBM"]


# def test_single_model_still_works(grid):
#     y, p = _toy()
#     fig = grid(y, {"XGBoost": p["XGBoost"]})
#     assert len(_heatmaps(fig)) == 1


# def test_all_panels_share_the_same_class_axis(grid):
#     y, p = _toy()
#     # class 93000 (Basement) is predicted by one model only; it must appear in every panel
#     p = {k: v.copy() for k, v in p.items()}
#     p["LightGBM"][:5] = 93000
#     fig = grid(y, p)
#     xs = [tuple(t.x) for t in _heatmaps(fig)]
#     ys = [tuple(t.y) for t in _heatmaps(fig)]
#     assert len(set(xs)) == 1 and len(set(ys)) == 1
#     assert "93000" in xs[0]


# def test_default_axis_labels_are_codes_like_the_blind_test_figure(grid):
#     y, p = _toy()
#     t = _heatmaps(grid(y, p))[0]
#     assert list(t.x) == [str(c) for c in sorted(set(y) | set(np.concatenate(list(p.values()))))]


# def test_name_mode_uses_lithology_names(grid):
#     y, p = _toy()
#     t = _heatmaps(grid(y, p, label_mode="name"))[0]
#     assert "Shale" in t.x and "65000" not in t.x


# def test_true_classes_on_y_axis_run_top_to_bottom(grid):
#     y, p = _toy()
#     fig = grid(y, p)
#     assert fig.layout.yaxis.autorange == "reversed"


# def test_normalised_rows_sum_to_one_and_have_no_nan(grid):
#     y, p = _toy()
#     p = {k: v.copy() for k, v in p.items()}
#     p["XGBoost"][:5] = 93000                      # predicted-only class => its true row is empty
#     for t in _heatmaps(grid(y, p, normalize=True)):
#         z = np.asarray(t.z, dtype=float)
#         assert not np.isnan(z).any()
#         sums = z.sum(axis=1)
#         empty = np.isclose(sums, 0)
#         assert empty.sum() == 1                    # only the Basement row (no true samples)
#         assert np.allclose(sums[~empty], 1.0)


# def test_counts_match_scikit_learn(grid):
#     y, p = _toy()
#     fig = grid(y, p, normalize=False)
#     labels = sorted(set(y) | set(np.concatenate(list(p.values()))))
#     for t, (name, pred) in zip(_heatmaps(fig), p.items()):
#         expected = confusion_matrix(y, pred, labels=labels)
#         assert np.array_equal(np.asarray(t.z, dtype=int), expected), name


# def test_hover_carries_counts_and_share(grid):
#     y, p = _toy()
#     t = _heatmaps(grid(y, p))[0]
#     assert t.customdata is not None and np.asarray(t.customdata).shape[-1] == 2
#     assert "n=" in t.hovertemplate or "samples" in t.hovertemplate


# def test_one_shared_colour_bar(grid):
#     y, p = _toy()
#     fig = grid(y, p)
#     assert sum(1 for t in _heatmaps(fig) if t.showscale) <= 1
#     assert fig.layout.coloraxis.colorscale is not None
#     assert fig.layout.coloraxis.cmin == 0 and fig.layout.coloraxis.cmax == 1


# def test_show_values_adds_cell_text(grid):
#     y, p = _toy()
#     plain = _heatmaps(grid(y, p, show_values=False))[0]
#     texty = _heatmaps(grid(y, p, show_values=True))[0]
#     assert plain.texttemplate is None and texty.texttemplate


# def test_rejects_bad_input(grid):
#     with pytest.raises(ValueError):
#         grid(np.array([]), {"A": np.array([])})
#     with pytest.raises(ValueError):
#         grid(np.array([65000, 65000]), {"A": np.array([65000])})
#     with pytest.raises(ValueError):
#         grid(np.array([65000]), {})


# def test_every_code_has_a_name():
#     assert set(CODES) <= set(LITHOLOGY_NAMES)


# # ---- wiring: the grid is actually used inside the Evaluate tab of the app -----------
# def _app_tree():
#     return ast.parse((ROOT / "final_app.py").read_text(encoding="utf-8"))


# def test_app_imports_the_grid():
#     names = {a.name for n in ast.walk(_app_tree()) if isinstance(n, ast.ImportFrom)
#              and n.module == "src.plots_extra" for a in n.names}
#     assert "build_confusion_grid" in names


# def test_app_calls_the_grid_inside_the_evaluate_tab():
#     for node in ast.walk(_app_tree()):
#         if isinstance(node, ast.With) and any(
#                 isinstance(i.context_expr, ast.Name) and i.context_expr.id == "tab_eval" for i in node.items):
#             calls = [c.func.id for c in ast.walk(node)
#                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
#             assert "build_confusion_grid" in calls
#             return
#     pytest.fail("no `with tab_eval:` block found in final_app.py")


"""Tests for the side-by-side confusion-matrix grid shown in the Evaluate tab.

Written before the feature. Run with:  python -m pytest tests/test_confusion_grid.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from src.inference import LITHOLOGY_NAMES

ROOT = Path(__file__).resolve().parents[1]
CODES = [30000, 65000, 65030, 70000, 70032, 74000, 80000, 86000, 88000, 90000, 99000]


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.choice(CODES, n, p=[.06, .40, .08, .10, .04, .03, .06, .05, .06, .06, .06])
    preds = {}
    for i, name in enumerate(["Ex-Tree", "XGBoost", "LightGBM"]):
        keep = rng.random(n) < (.75 - .03 * i)
        preds[name] = np.where(keep, y_true, rng.choice(CODES, n))
    return y_true, preds


@pytest.fixture(scope="module")
def grid():
    from src.plots_extra import build_confusion_grid
    return build_confusion_grid


def _heatmaps(fig):
    return [t for t in fig.data if t.type == "heatmap"]


def test_one_panel_per_selected_model(grid):
    y, p = _toy()
    fig = grid(y, p)
    assert len(_heatmaps(fig)) == 3
    assert [a.text for a in fig.layout.annotations][:3] == ["Ex-Tree", "XGBoost", "LightGBM"]


def test_single_model_still_works(grid):
    y, p = _toy()
    fig = grid(y, {"XGBoost": p["XGBoost"]})
    assert len(_heatmaps(fig)) == 1


def test_all_panels_share_the_same_class_axis(grid):
    y, p = _toy()
    # class 93000 (Basement) is predicted by one model only; it must appear in every panel
    p = {k: v.copy() for k, v in p.items()}
    p["LightGBM"][:5] = 93000
    fig = grid(y, p)
    xs = [tuple(t.x) for t in _heatmaps(fig)]
    ys = [tuple(t.y) for t in _heatmaps(fig)]
    assert len(set(xs)) == 1 and len(set(ys)) == 1
    assert "93000" in xs[0]


def test_default_axis_labels_are_codes_like_the_blind_test_figure(grid):
    y, p = _toy()
    t = _heatmaps(grid(y, p))[0]
    assert list(t.x) == [str(c) for c in sorted(set(y) | set(np.concatenate(list(p.values()))))]


def test_name_mode_uses_lithology_names(grid):
    y, p = _toy()
    t = _heatmaps(grid(y, p, label_mode="name"))[0]
    assert "Shale" in t.x and "65000" not in t.x


def test_true_classes_on_y_axis_run_top_to_bottom(grid):
    y, p = _toy()
    fig = grid(y, p)
    assert fig.layout.yaxis.autorange == "reversed"


def test_normalised_rows_sum_to_one_and_have_no_nan(grid):
    y, p = _toy()
    p = {k: v.copy() for k, v in p.items()}
    p["XGBoost"][:5] = 93000                      # predicted-only class => its true row is empty
    for t in _heatmaps(grid(y, p, normalize=True)):
        z = np.asarray(t.z, dtype=float)
        assert not np.isnan(z).any()
        sums = z.sum(axis=1)
        empty = np.isclose(sums, 0)
        assert empty.sum() == 1                    # only the Basement row (no true samples)
        assert np.allclose(sums[~empty], 1.0)


def test_counts_match_scikit_learn(grid):
    y, p = _toy()
    fig = grid(y, p, normalize=False)
    labels = sorted(set(y) | set(np.concatenate(list(p.values()))))
    for t, (name, pred) in zip(_heatmaps(fig), p.items()):
        expected = confusion_matrix(y, pred, labels=labels)
        assert np.array_equal(np.asarray(t.z, dtype=int), expected), name


def test_hover_carries_counts_and_share(grid):
    y, p = _toy()
    t = _heatmaps(grid(y, p))[0]
    assert t.customdata is not None and np.asarray(t.customdata).shape[-1] == 2
    assert "n=" in t.hovertemplate or "samples" in t.hovertemplate


def test_one_shared_colour_bar(grid):
    y, p = _toy()
    fig = grid(y, p)
    assert sum(1 for t in _heatmaps(fig) if t.showscale) <= 1
    assert fig.layout.coloraxis.colorscale is not None
    assert fig.layout.coloraxis.cmin == 0 and fig.layout.coloraxis.cmax == 1


def test_show_values_adds_cell_text(grid):
    y, p = _toy()
    plain = _heatmaps(grid(y, p, show_values=False))[0]
    texty = _heatmaps(grid(y, p, show_values=True))[0]
    assert plain.texttemplate is None and texty.texttemplate


def test_rejects_bad_input(grid):
    with pytest.raises(ValueError):
        grid(np.array([]), {"A": np.array([])})
    with pytest.raises(ValueError):
        grid(np.array([65000, 65000]), {"A": np.array([65000])})
    with pytest.raises(ValueError):
        grid(np.array([65000]), {})


def test_every_code_has_a_name():
    assert set(CODES) <= set(LITHOLOGY_NAMES)


# ---- wiring: the grid is actually used inside the Evaluate tab of the app -----------
def _app_tree():
    return ast.parse((ROOT / "final_app.py").read_text(encoding="utf-8"))


def test_app_imports_the_grid():
    names = {a.name for n in ast.walk(_app_tree()) if isinstance(n, ast.ImportFrom)
             and n.module == "src.plots_extra" for a in n.names}
    assert "build_confusion_grid" in names


def test_app_calls_the_grid_inside_the_evaluate_tab():
    for node in ast.walk(_app_tree()):
        if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Name) and i.context_expr.id == "tab_eval" for i in node.items):
            calls = [c.func.id for c in ast.walk(node)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
            assert "build_confusion_grid" in calls
            return
    pytest.fail("no `with tab_eval:` block found in final_app.py")


def test_app_shows_the_grid_on_the_metrics_page():
    """`sub_over` is the Evaluate > Metrics sub-tab; that is where users look for it."""
    for node in ast.walk(_app_tree()):
        if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Name) and i.context_expr.id == "sub_over" for i in node.items):
            calls = [c.func.id for c in ast.walk(node)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
            assert "build_confusion_grid" in calls
            return
    pytest.fail("no `with sub_over:` block found in final_app.py")