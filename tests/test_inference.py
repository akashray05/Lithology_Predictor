import numpy as np
import pandas as pd

from src.inference import build_feature_matrix, LITHOLOGY_NAMES


def test_builds_exactly_41_features_and_excludes_depth():
    n = 40
    raw = pd.DataFrame({
        "DEPT": np.arange(n, dtype=float) * 0.152,
        "GR": np.linspace(30, 100, n),
        "RDEP": np.linspace(1, 20, n),
        "RMED": np.linspace(1.5, 15, n),
        "DTC": np.linspace(70, 130, n),
        "RHOB": np.linspace(2.0, 2.7, n),
    })
    X, info = build_feature_matrix(raw, "test_well")
    assert X.shape == (n, 41)
    assert "DEPT" not in X.columns
    assert list(info["DEPT"]) == list(raw["DEPT"])
    assert len(set(X.columns)) == 41


def test_invalid_values_become_missing_and_indicators_are_set():
    raw = pd.DataFrame({
        "DEPT": [100.0, 100.152, 100.304, 100.456],
        "GR": [50.0, -1.0, 60.0, 70.0],
        "RDEP": [10.0, 10.0, 0.0, 12.0],
        "RMED": [8.0, 8.0, 8.0, 8.0],
        "DTC": [90.0, 91.0, 92.0, 0.0],
        "RHOB": [2.3, 2.4, 2.5, 2.6],
    })
    X, _ = build_feature_matrix(raw)
    assert X.loc[1, "GR_MISSING"] == 1
    assert X.loc[2, "RDEP_LOG10_MISSING"] == 1
    assert X.loc[3, "DTC_MISSING"] == 1


def test_lithology_mapping_contains_twelve_force_codes():
    assert len(LITHOLOGY_NAMES) == 12
    assert LITHOLOGY_NAMES[30000] == "Sandstone"
    assert LITHOLOGY_NAMES[99000] == "Tuff"
