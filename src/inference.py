"""Frozen-model inference for the Lithology_Predictor project.

This module reproduces the 41-feature inference schema used by the frozen
W5 ExtraTrees, XGBoost, and LightGBM artifacts.

It deliberately does NOT impute missing values: the saved engineered dataset
reconstructs originally missing values as NaN before rolling feature creation,
and the frozen evaluation path predicts directly from those engineered
features. Keep this behavior aligned with the project's notebooks.

New in this version
-------------------
* An optional ground-truth ("original") lithology column can be carried
  through the pipeline (``label_column=``) so it can be plotted next to the
  predictions and used for evaluation.
* ``models_to_run=`` lets callers run only the models they need.
* Loaded models are cached, so repeated calls do not re-read the joblib files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lasio
except ImportError:  # CSV inference can still work without lasio installed.
    lasio = None


INPUT_CURVES = ("GR", "RDEP", "RMED", "DTC", "RHOB")
MODEL_BASE_FEATURES = ("GR", "RDEP_LOG10", "RMED_LOG10", "DTC", "RHOB")
ROLLING_WINDOWS = (5, 9, 21)
LABEL_COLUMN = "FORCE_2020_LITHOFACIES_LITHOLOGY"

LITHOLOGY_NAMES = {
    30000: "Sandstone",
    65000: "Shale",
    65030: "Sandstone/Shale",
    70000: "Limestone",
    70032: "Chalk",
    74000: "Dolomite",
    80000: "Marl",
    86000: "Anhydrite",
    88000: "Halite",
    90000: "Coal",
    93000: "Basement",
    99000: "Tuff",
}
NAME_TO_CODE = {name.upper(): code for code, name in LITHOLOGY_NAMES.items()}

# Column names (upper-case) that are auto-detected as the true lithology.
LABEL_COLUMN_CANDIDATES = (
    LABEL_COLUMN,
    "TRUE_LITHOLOGY",
    "LITHOLOGY",
    "LITHOFACIES",
    "FACIES",
    "LABEL",
)

MODEL_FILES = {
    "W5": "lithology_w5_final.joblib",
    "XGBoost": "lithology_xgboost_final.joblib",
    "LightGBM": "lithology_lightgbm_final.joblib",
}
METADATA_FILES = {
    "W5": "w5_feature_metadata.json",
    "XGBoost": "xgboost_feature_metadata.json",
    "LightGBM": "lightgbm_feature_metadata.json",
}


class InferenceError(RuntimeError):
    """Raised when input data or frozen artifacts fail validation."""


@dataclass
class InferenceBundle:
    models: dict
    features: list[str]
    idx_to_code: dict[int, int]


# --------------------------------------------------------------------------
# Input reading
# --------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column labels the same way as the preprocessing notebook."""
    result = df.copy()
    result.columns = [str(col).strip().upper() for col in result.columns]
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise InferenceError(
            "Column names become duplicated after upper-casing: "
            + ", ".join(map(str, duplicates))
        )
    return result


def detect_label_column(columns) -> str | None:
    """Return the original column name that looks like a lithology label."""
    normalized = {str(col).strip().upper(): col for col in columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def read_uploaded_file(file_or_path, filename: str | None = None) -> pd.DataFrame:
    """Read a LAS or CSV input into a DataFrame.

    `file_or_path` may be a filesystem path or a seekable file-like object
    (such as a Streamlit UploadedFile or BytesIO). For file-like inputs, pass
    `filename` so the format can be determined.
    """
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".las":
            if lasio is None:
                raise InferenceError(
                    "Reading LAS files requires lasio. Install it in the project environment."
                )
            las = lasio.read(str(path))
            return las.df().reset_index()
        raise InferenceError("Unsupported file type. Upload a .las or .csv file.")

    name = filename or getattr(file_or_path, "name", "")
    suffix = Path(str(name)).suffix.lower()

    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    if suffix == ".csv":
        return pd.read_csv(file_or_path)

    if suffix == ".las":
        if lasio is None:
            raise InferenceError(
                "Reading LAS files requires lasio. Install it in the project environment."
            )
        content = file_or_path.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        las = lasio.read(StringIO(content))
        return las.df().reset_index()

    raise InferenceError(
        "Could not determine the input format. Provide a .las or .csv filename."
    )


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------
def _decode_true_labels(series: pd.Series) -> pd.Series:
    """Convert a label column to FORCE lithology codes (float, NaN if unknown).

    Accepts numeric FORCE codes (e.g. 65000) or lithology names (e.g. "Shale").
    """
    numeric = pd.to_numeric(series, errors="coerce").round()
    codes = numeric.where(numeric.isin(list(LITHOLOGY_NAMES)), np.nan)
    from_names = series.astype(str).str.strip().str.upper().map(NAME_TO_CODE)
    return codes.fillna(from_names).astype(float)


def _make_base_features(
    raw_df: pd.DataFrame, label_column: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw curves and create the five model base features.

    Returns:
        source: normalized original rows with a stable row number and depth.
        base: the five model base features with invalid values represented NaN.
    """
    source = _normalize_columns(raw_df)
    if source.empty:
        raise InferenceError("The uploaded file contains no rows.")

    source = source.copy()
    source["_SOURCE_ROW"] = np.arange(len(source), dtype=np.int64)

    if "DEPT" in source.columns:
        depth = pd.to_numeric(source["DEPT"], errors="coerce")
    elif "DEPTH" in source.columns:
        depth = pd.to_numeric(source["DEPTH"], errors="coerce")
    else:
        # Depth is for presentation only, never a model feature.
        depth = pd.Series(np.nan, index=source.index, dtype=float)
    source["_OUTPUT_DEPTH"] = depth.to_numpy()

    # Optional ground truth: presentation/evaluation only, never a model feature.
    if label_column:
        label_key = str(label_column).strip().upper()
        if label_key not in source.columns:
            raise InferenceError(f"Label column '{label_column}' not found in the file.")
        if label_key in INPUT_CURVES:
            raise InferenceError(
                f"'{label_column}' is a model input curve and cannot be the label column."
            )
        source["_TRUE_CODE"] = _decode_true_labels(source[label_key]).to_numpy()

    # The project pipeline allows absent curves by representing them as NaN.
    for curve in INPUT_CURVES:
        if curve not in source.columns:
            source[curve] = np.nan
        source[curve] = pd.to_numeric(source[curve], errors="coerce")

    # Exact invalid-value rules documented in 03_preprocessing.ipynb.
    source.loc[source["GR"] < 0, "GR"] = np.nan
    source.loc[source["DTC"] <= 0, "DTC"] = np.nan
    source.loc[source["RHOB"] <= 0, "RHOB"] = np.nan
    source.loc[source["RDEP"] <= 0, "RDEP"] = np.nan
    source.loc[source["RMED"] <= 0, "RMED"] = np.nan

    base = pd.DataFrame(index=source.index)
    base["GR"] = source["GR"]
    with np.errstate(divide="ignore", invalid="ignore"):
        base["RDEP_LOG10"] = np.log10(source["RDEP"])
        base["RMED_LOG10"] = np.log10(source["RMED"])
    base["DTC"] = source["DTC"]
    base["RHOB"] = source["RHOB"]

    # Keep invalid/infinite values from entering the model matrix.
    base = base.replace([np.inf, -np.inf], np.nan)

    return source, base


def build_feature_matrix(
    raw_df: pd.DataFrame,
    well_name: str = "uploaded_well",
    expected_features: list[str] | None = None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen 41-feature matrix for one uploaded well.

    Rows with all five base features missing are removed, matching the
    project's preprocessing decision. Rolling statistics are computed
    within this well using centered windows and min_periods=3, exactly as in
    04_feature_engineering.ipynb.

    Returns:
        X: model input in the exact metadata order.
        row_info: retained source row, depth, original curve values and,
            when `label_column` is given, the decoded TRUE_CODE.
    """
    source, base = _make_base_features(raw_df, label_column=label_column)

    # Missingness indicators are calculated before rolling features.
    feature_df = base.copy()
    for feature in MODEL_BASE_FEATURES:
        feature_df[f"{feature}_MISSING"] = feature_df[feature].isna().astype("int8")

    # The saved engineered dataset reconstructs NaNs from the indicators
    # before calculating rolling statistics. Base values already remain NaN.
    for feature in MODEL_BASE_FEATURES:
        for window in ROLLING_WINDOWS:
            feature_df[f"{feature}_ROLLMEAN_{window}"] = (
                feature_df[feature]
                .rolling(window=window, center=True, min_periods=3)
                .mean()
            )
            feature_df[f"{feature}_ROLLSTD_{window}"] = (
                feature_df[feature]
                .rolling(window=window, center=True, min_periods=3)
                .std()
            )

    feature_df["RESISTIVITY_SEPARATION"] = (
        feature_df["RDEP_LOG10"] - feature_df["RMED_LOG10"]
    )

    # Do not predict rows with no usable base log measurements.
    valid_rows = ~base[list(MODEL_BASE_FEATURES)].isna().all(axis=1)
    feature_df = feature_df.loc[valid_rows].copy()
    source_kept = source.loc[valid_rows].copy()

    # Metadata order is authoritative; do not rely on construction order.
    if expected_features is None:
        expected_features = (
            list(MODEL_BASE_FEATURES)
            + [
                f"{feature}_ROLL{stat}_{window}"
                for feature in MODEL_BASE_FEATURES
                for window in ROLLING_WINDOWS
                for stat in ("MEAN", "STD")
            ]
            + ["RESISTIVITY_SEPARATION"]
            + [f"{feature}_MISSING" for feature in MODEL_BASE_FEATURES]
        )

    missing = [col for col in expected_features if col not in feature_df.columns]
    extra = [col for col in feature_df.columns if col not in expected_features]
    if missing or extra:
        raise InferenceError(
            f"Feature schema mismatch. Missing={missing}; unexpected={extra}"
        )

    X = feature_df.loc[:, expected_features].copy()
    X.index = source_kept["_SOURCE_ROW"].to_numpy()

    # Preserve source curves/depth for output and plotting. Depth is not in X.
    info_columns = ["_SOURCE_ROW", "_OUTPUT_DEPTH"] + list(INPUT_CURVES)
    if "_TRUE_CODE" in source_kept.columns:
        info_columns.append("_TRUE_CODE")
    row_info = source_kept[info_columns].copy()
    row_info = row_info.rename(
        columns={
            "_SOURCE_ROW": "SOURCE_ROW",
            "_OUTPUT_DEPTH": "DEPT",
            "_TRUE_CODE": "TRUE_CODE",
        }
    )
    row_info.index = X.index

    return X, row_info


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _load_bundle_uncached(models_dir: Path) -> InferenceBundle:
    models = {}
    metadata = {}

    for name, filename in MODEL_FILES.items():
        path = models_dir / filename
        if not path.is_file():
            raise InferenceError(f"Required model artifact not found: {path}")
        models[name] = joblib.load(path)

    for name, filename in METADATA_FILES.items():
        path = models_dir / filename
        if not path.is_file():
            raise InferenceError(f"Required feature metadata not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            metadata[name] = json.load(f)

    feature_lists = {}
    for name, data in metadata.items():
        features = data.get("features")
        if not isinstance(features, list) or len(features) != 41:
            raise InferenceError(
                f"{name} metadata must contain exactly 41 ordered features."
            )
        if len(set(features)) != 41:
            raise InferenceError(f"{name} metadata contains duplicate features.")
        feature_lists[name] = features

    reference = feature_lists["W5"]
    for name, features in feature_lists.items():
        if features != reference:
            raise InferenceError(
                f"{name} feature metadata order differs from W5 metadata."
            )

    # Validate metadata against the actual serialized model schema.
    for name, model in models.items():
        model_feature_names = getattr(model, "feature_names_in_", None)
        n_features = getattr(model, "n_features_in_", None)
        if n_features != len(reference):
            raise InferenceError(
                f"{name} expects {n_features} features; metadata lists {len(reference)}."
            )
        if model_feature_names is not None and list(map(str, model_feature_names)) != reference:
            raise InferenceError(
                f"{name} stored feature order does not match its metadata."
            )

    mapping_path = models_dir / "boosted_label_mapping.json"
    if not mapping_path.is_file():
        raise InferenceError(f"Required label mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    idx_to_code = {int(k): int(v) for k, v in mapping["idx_to_code"].items()}
    if set(idx_to_code.values()) != set(LITHOLOGY_NAMES):
        raise InferenceError(
            "boosted_label_mapping.json does not match the expected 12 FORCE codes."
        )

    # Confirm boosted model classes are encoded indices, and W5 uses FORCE codes.
    w5_classes = set(int(round(float(c))) for c in models["W5"].classes_)
    if w5_classes != set(LITHOLOGY_NAMES):
        raise InferenceError(
            "W5 classes_ are not the expected original FORCE lithology codes."
        )

    for name in ("XGBoost", "LightGBM"):
        classes = set(int(c) for c in models[name].classes_)
        if classes != set(idx_to_code):
            raise InferenceError(
                f"{name} classes_ do not match boosted_label_mapping.json indices."
            )

    return InferenceBundle(models=models, features=reference, idx_to_code=idx_to_code)


@lru_cache(maxsize=2)
def _load_bundle_cached(models_dir: str) -> InferenceBundle:
    return _load_bundle_uncached(Path(models_dir))


def load_inference_bundle(models_dir: str | Path | None = None) -> InferenceBundle:
    """Load and validate the frozen models and metadata (cached per directory)."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parents[1] / "models"
    return _load_bundle_cached(str(Path(models_dir).resolve()))


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_well(
    raw_df: pd.DataFrame,
    well_name: str = "uploaded_well",
    models_dir: str | Path | None = None,
    label_column: str | None = None,
    models_to_run: list[str] | None = None,
) -> pd.DataFrame:
    """Run the frozen models and return predictions plus depth/curves.

    Args:
        label_column: optional column holding the original (true) lithology.
            Adds TRUE_CODE and TRUE_LITHOLOGY columns to the result.
        models_to_run: subset of {"W5", "XGBoost", "LightGBM"}; default all.
    """
    bundle = load_inference_bundle(models_dir)

    if models_to_run is None:
        models_to_run = list(bundle.models)
    unknown = [m for m in models_to_run if m not in bundle.models]
    if unknown:
        raise InferenceError(f"Unknown model(s): {', '.join(unknown)}")

    X, row_info = build_feature_matrix(
        raw_df,
        well_name=well_name,
        expected_features=bundle.features,
        label_column=label_column,
    )

    if X.empty:
        raise InferenceError(
            "No rows remain after preprocessing: every row had all five base logs missing."
        )

    result = row_info.copy()
    result["WELL"] = str(well_name)

    if "TRUE_CODE" in result.columns:
        result["TRUE_LITHOLOGY"] = [
            LITHOLOGY_NAMES[int(code)] if pd.notna(code) else None
            for code in result["TRUE_CODE"]
        ]

    for name in models_to_run:
        model = bundle.models[name]
        raw_prediction = np.asarray(model.predict(X)).reshape(-1)

        if len(raw_prediction) != len(X):
            raise InferenceError(
                f"{name} returned {len(raw_prediction)} predictions for {len(X)} rows."
            )

        if name == "W5":
            codes = np.rint(raw_prediction).astype(np.int64)
        else:
            try:
                codes = np.asarray(
                    [bundle.idx_to_code[int(value)] for value in raw_prediction],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise InferenceError(
                    f"{name} returned unknown encoded class index {exc.args[0]}."
                ) from exc

        invalid_codes = sorted(set(codes) - set(LITHOLOGY_NAMES))
        if invalid_codes:
            raise InferenceError(
                f"{name} returned unknown FORCE lithology codes: {invalid_codes}"
            )

        result[f"{name.upper()}_PREDICTION_CODE"] = codes
        result[f"{name.upper()}_PREDICTION"] = [
            LITHOLOGY_NAMES[int(code)] for code in codes
        ]

        # Confidence is the probability assigned to the model's predicted class.
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X))
            classes = list(model.classes_)
            class_to_column = {int(round(float(c))): i for i, c in enumerate(classes)}
            if name == "W5":
                model_labels = codes
            else:
                model_labels = raw_prediction.astype(int)

            confidence = probabilities[
                np.arange(len(model_labels)),
                [class_to_column[int(label)] for label in model_labels],
            ].astype(float)
            result[f"{name.upper()}_CONFIDENCE"] = confidence

    return result.reset_index(drop=True)
