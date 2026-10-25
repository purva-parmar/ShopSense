"""Focused tests for ShopSense model helper contracts."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import calculate_classification_metrics
from src.features import FEATURE_COLUMNS, IDENTIFIER_COLUMN, TARGET_COLUMN
from src.train_models import _make_pipeline, load_split


def test_metrics_calculate_positive_class_scores() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_probability = [0.1, 0.6, 0.7, 0.9]
    metrics = calculate_classification_metrics(y_true, y_pred, y_probability)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] is not None


def test_model_pipelines_fit_and_predict_probability() -> None:
    X = pd.DataFrame(
        {
            feature: np.linspace(0.0, 1.0, 20) + index
            for index, feature in enumerate(FEATURE_COLUMNS)
        }
    )
    y = pd.Series([0, 1] * 10)
    pipeline = _make_pipeline("logistic_regression", {"C": 1.0, "class_weight": None})
    pipeline.fit(X, y)

    assert pipeline.predict(X).shape == (20,)
    assert pipeline.predict_proba(X).shape == (20, 2)


def test_load_split_uses_only_approved_predictive_features(tmp_path) -> None:
    frame = pd.DataFrame({IDENTIFIER_COLUMN: range(12)})
    for index, feature in enumerate(FEATURE_COLUMNS):
        frame[feature] = np.arange(12, dtype=float) + index
    frame[TARGET_COLUMN] = [0, 1] * 6
    split_path = tmp_path / "train.csv"
    frame.to_csv(split_path, index=False)

    X, y = load_split(split_path, "train")

    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert IDENTIFIER_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == 12
