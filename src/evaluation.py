"""Evaluation helpers: FORCE 2020 penalty score, overall and per-class metrics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.inference import LITHOLOGY_NAMES

# Official FORCE 2020 penalty matrix (same order/values as the blind-test notebook).
FORCE_CLASSES = [
    30000, 65030, 65000, 80000, 74000, 70000,
    70032, 88000, 86000, 99000, 90000, 93000,
]

PENALTY_MATRIX = np.array([
    [0,   2,   3,   3.5, 2.5, 3.5, 3,   3,   3,   3,   3,   3],
    [2,   0,   2,   3.5, 2.5, 3.5, 3,   3,   3,   3,   3,   3],
    [3,   2,   0,   3.5, 2.5, 3.5, 3,   3,   3,   3,   3,   3],
    [3.5, 3.5, 3.5, 0,   3,   2,   2.5, 3,   3,   3,   3,   3],
    [2.5, 2.5, 2.5, 3,   0,   3,   2.5, 3,   3,   3,   3,   3],
    [3.5, 3.5, 3.5, 2,   3,   0,   2,   3,   3,   3,   3,   3],
    [3,   3,   3,   2.5, 2.5, 2,   0,   3,   3,   3,   3,   3],
    [3,   3,   3,   3,   3,   3,   3,   0,   2,   3,   3,   3],
    [3,   3,   3,   3,   3,   3,   3,   3,   0,   3,   3,   3],
    [3,   3,   3,   3,   3,   3,   3,   3,   3,   0,   3,   3],
    [3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   0,   3],
    [3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   0],
], dtype=float)

assert PENALTY_MATRIX.shape == (12, 12)
assert np.all(np.diag(PENALTY_MATRIX) == 0)

_CLASS_TO_INDEX = {code: i for i, code in enumerate(FORCE_CLASSES)}


def describe_labels(result_df: pd.DataFrame, diagnostics: dict | None = None,
                    depth_col: str = "DEPT") -> dict:
    """Say honestly what reference labels are available for a prediction table.

    status:
        "no_column"       no label column selected / detected
        "no_valid_labels" column exists but has no usable (recognised) labels
        "partial"         some predicted samples have a valid label, some do not
        "full"            every predicted sample has a valid label

    Only rows with a decoded TRUE_CODE count as labelled, so a column that is
    present but empty (or "None") never counts as ground truth.
    """
    n = int(len(result_df))
    labelled_mask = (
        result_df["TRUE_CODE"].notna()
        if "TRUE_CODE" in result_df.columns
        else pd.Series(False, index=result_df.index)
    )
    labelled = int(labelled_mask.sum())

    depth_min = depth_max = None
    if labelled and depth_col in result_df.columns:
        depths = pd.to_numeric(result_df.loc[labelled_mask, depth_col], errors="coerce").dropna()
        if len(depths):
            depth_min, depth_max = float(depths.min()), float(depths.max())

    if "TRUE_CODE" not in result_df.columns:
        status, level = "no_column", "info"
        message = (
            "No original lithology column was selected or detected, so there is "
            "nothing to compare against. Predictions are shown without a reference."
        )
    elif labelled == 0:
        status, level = "no_valid_labels", "warning"
        detail = ""
        if diagnostics and diagnostics.get("column"):
            examples = diagnostics.get("unrecognised_examples") or {}
            shown = ", ".join(f"'{k}'" for k in list(examples)[:3])
            detail = (
                f" Column `{diagnostics['column']}` has {diagnostics['empty']:,} empty and "
                f"{diagnostics['unrecognised']:,} unrecognised values"
                + (f" (e.g. {shown})" if shown else "")
                + "."
            )
        message = (
            "The label column contains no usable lithology labels, so no true "
            "track, error tracks or accuracy are shown. Predictions still work." + detail
        )
    elif labelled < n:
        status, level = "partial", "info"
        span = f" ({depth_min:,.1f} to {depth_max:,.1f})" if depth_min is not None else ""
        message = (
            f"Original lithology exists for {labelled:,} of {n:,} predicted samples "
            f"({labelled / n:.1%}){span}. Elsewhere the original track and error "
            "strips are blank; accuracy and metrics use labelled samples only."
        )
    else:
        status, level = "full", "success"
        message = f"Original lithology exists for all {n:,} predicted samples."

    return {
        "status": status, "level": level, "message": message,
        "samples": n, "labelled": labelled,
        "fraction": (labelled / n) if n else 0.0,
        "depth_min": depth_min, "depth_max": depth_max,
    }


def force_score(y_true, y_pred) -> float:
    """Negative mean FORCE 2020 penalty (closer to zero is better)."""
    true_idx = np.array([_CLASS_TO_INDEX[int(v)] for v in y_true])
    pred_idx = np.array([_CLASS_TO_INDEX[int(v)] for v in y_pred])
    return -float(np.mean(PENALTY_MATRIX[true_idx, pred_idx]))


def overall_metrics(y_true, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Accuracy, balanced accuracy, macro/weighted F1 and FORCE score per model.

    F1 averages use only the classes present in the true labels, as in the
    blind-test notebook.
    """
    y_true = np.asarray(y_true).astype(int)
    classes = np.sort(np.unique(y_true))

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, y_pred in predictions.items():
            y_pred = np.asarray(y_pred).astype(int)
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
                "Macro F1": f1_score(
                    y_true, y_pred, labels=classes, average="macro", zero_division=0
                ),
                "Weighted F1": f1_score(
                    y_true, y_pred, labels=classes, average="weighted", zero_division=0
                ),
                "FORCE score": force_score(y_true, y_pred),
            })
    return pd.DataFrame(rows)


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision / recall / F1 / support for every class seen in truth or prediction."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = sorted(set(y_true) | set(y_pred))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
    support = np.array([(y_true == label).sum() for label in labels])
    predicted = np.array([(y_pred == label).sum() for label in labels])

    return pd.DataFrame({
        "Lithology": [LITHOLOGY_NAMES[label] for label in labels],
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True samples": support,
        "Predicted samples": predicted,
    })


def confusion_counts(y_true, y_pred) -> tuple[np.ndarray, list[int]]:
    """Raw confusion-matrix counts (rows = true, columns = predicted) and labels."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = sorted(set(y_true) | set(y_pred))
    return confusion_matrix(y_true, y_pred, labels=labels), labels
