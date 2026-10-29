"""Train, compare, evaluate, and export ShopSense classification models.

Run from the repository root:
    python -m src.train_models

Inputs are produced by src.features.run_feature_pipeline:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

The validation set is used for hyperparameter/model selection. Test results are
reported only after the choices are fixed; do not change the model based on test
metrics. The selected pipeline is refit on train + validation before final test
scoring, then exported for the Streamlit application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.evaluation import (
    calculate_classification_metrics,
    save_confusion_matrix,
    save_model_comparison_plot,
)
from src.features import FEATURE_COLUMNS, IDENTIFIER_COLUMN, RANDOM_SEED, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "max_depth": [None, 6, 10, 16, 24],
        "min_samples_leaf": [1, 2, 5, 10],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [60, 100, 150],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 3, 5],
        "class_weight": [None, "balanced_subsample"],
        "max_features": ["sqrt", 0.7],
    },
    "knn": {
        "n_neighbors": [5, 11, 21, 31, 41],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
        "gradient_boosting": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [1, 2, 3],
        "min_samples_leaf": [2, 5, 10],
        "subsample": [0.8, 1.0],
    },
}


def _project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root, not the current folder."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_split(path: str | Path, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a prepared split and return only approved features plus its target."""
    csv_path = _project_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {split_name} split: {csv_path}. Run the feature pipeline first."
        )

    frame = pd.read_csv(csv_path)
    required = {IDENTIFIER_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{split_name} split is empty.")
    if frame[TARGET_COLUMN].isna().any() or not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{split_name} target must contain only non-missing 0/1 labels.")

    X = frame.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    y = frame[TARGET_COLUMN].astype("int8")
    if X.isna().any().any():
        # The current feature pipeline promises complete numeric features. Fail
        # clearly if that upstream contract changes; the pipeline imputer still
        # protects future inference inputs and keeps preprocessing self-contained.
        raise ValueError(f"{split_name} predictive features contain missing values.")
    if not y.isin([0, 1]).all():
        raise ValueError(f"{split_name} target must contain labels 0 and 1 only.")
    return X, y


def _make_pipeline(model_name: str, params: dict[str, Any]) -> Pipeline:
    """Create a model-specific pipeline; scaling is used only where needed."""
    if model_name == "logistic_regression":
        estimator: ClassifierMixin = LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED, **params
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=RANDOM_SEED, **params)
        steps = [("imputer", SimpleImputer(strategy="median"))]
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=1, **params
        )
        steps = [("imputer", SimpleImputer(strategy="median"))]
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            random_state=RANDOM_SEED, **params
        )
        steps = [("imputer", SimpleImputer(strategy="median"))]
    elif model_name == "knn":
        estimator = KNeighborsClassifier(n_jobs=-1, **params)
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _probability_for_positive_class(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Return predict_proba for label 1 regardless of estimator class ordering."""
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError("Trained classifier does not contain positive class 1.")
    positive_index = classes.index(1)
    return pd.Series(model.predict_proba(X)[:, positive_index], index=X.index)


def _metric_row(
    model_name: str,
    split_name: str,
    y_true: pd.Series,
    model: Pipeline,
    X: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    prediction = model.predict(X)
    probability = _probability_for_positive_class(model, X)
    metrics = calculate_classification_metrics(y_true, prediction, probability)
    return {
        "model": model_name,
        "best_params": json.dumps(params, sort_keys=True),
        **{f"{split_name}_{name}": value for name, value in metrics.items()},
    }


def _choose_by_validation(metrics: dict[str, float | None]) -> tuple[float, float, float]:
    """Rank by positive-class F1, then ROC-AUC, then recall (validation only)."""
    return (
        float(metrics.get("f1") or 0.0),
        float(metrics.get("roc_auc") or 0.0),
        float(metrics.get("recall") or 0.0),
    )


def run_model_comparison(
    data_directory: str | Path = "data/processed",
    models_directory: str | Path = "models",
    results_directory: str | Path = "results",
) -> dict[str, Any]:
    """Train five models, tune on validation, test once, and export all artifacts."""
    data_dir = _project_path(data_directory)
    model_dir = _project_path(models_directory)
    results_dir = _project_path(results_directory)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(data_dir / "train.csv", "train")
    X_validation, y_validation = load_split(data_dir / "validation.csv", "validation")
    X_test, y_test = load_split(data_dir / "test.csv", "test")

    for name, target in (("train", y_train), ("validation", y_validation), ("test", y_test)):
        if target.nunique() != 2:
            raise ValueError(f"{name} split must contain both target classes 0 and 1.")
    if len(X_train) < 3:
        raise ValueError("At least 3 training customers are required for model fitting.")

    best_by_model: dict[str, dict[str, Any]] = {}
    validation_rows: list[dict[str, Any]] = []

    for model_name, grid in MODEL_GRIDS.items():
        # KNN cannot use more neighbours than there are training rows.
        candidate_grid = dict(grid)
        if model_name == "knn":
            candidate_grid["n_neighbors"] = [
                k for k in grid["n_neighbors"] if k <= len(X_train)
            ]
            if not candidate_grid["n_neighbors"]:
                candidate_grid["n_neighbors"] = [len(X_train)]

        best_candidate: dict[str, Any] | None = None
        best_score: tuple[float, float, float] | None = None
        best_metrics: dict[str, float | None] | None = None

        for params in ParameterGrid(candidate_grid):
            pipeline = _make_pipeline(model_name, params)
            pipeline.fit(X_train, y_train)
            prediction = pipeline.predict(X_validation)
            probability = _probability_for_positive_class(pipeline, X_validation)
            metrics = calculate_classification_metrics(y_validation, prediction, probability)
            score = _choose_by_validation(metrics)
            if best_score is None or score > best_score:
                best_score = score
                best_candidate = {"params": params, "pipeline": pipeline}
                best_metrics = metrics

        assert best_candidate is not None and best_metrics is not None
        best_params = best_candidate["params"]
        validation_rows.append(
            {
                "model": model_name,
                "best_params": json.dumps(best_params, sort_keys=True),
                **{f"validation_{key}": value for key, value in best_metrics.items()},
            }
        )
        best_by_model[model_name] = {
            "params": best_params,
            "validation_metrics": best_metrics,
        }

    # Model selection is complete before touching the test set. Refit each
    # model's validation-selected configuration on train + validation.
    X_train_validation = pd.concat([X_train, X_validation], ignore_index=True)
    y_train_validation = pd.concat([y_train, y_validation], ignore_index=True)
    validation_frame = pd.DataFrame(validation_rows)
    test_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, selection in best_by_model.items():
        pipeline = _make_pipeline(model_name, selection["params"])
        pipeline.fit(X_train_validation, y_train_validation)
        fitted_models[model_name] = pipeline
        joblib.dump(pipeline, model_dir / f"{model_name}_pipeline.joblib")

        row = _metric_row(
            model_name, "test", y_test, pipeline, X_test, selection["params"]
        )
        test_rows.append(row)
        save_confusion_matrix(
            y_test,
            pipeline.predict(X_test),
            results_dir / f"confusion_matrix_{model_name}.png",
            f"{model_name.replace('_', ' ').title()} — held-out test set",
        )

    test_frame = pd.DataFrame(test_rows)
    comparison = validation_frame.merge(
        test_frame, on=["model", "best_params"], how="inner", validate="one_to_one"
    )
    # Keep selection tied to validation F1, not the test results.
    selected_model_name = max(
        best_by_model,
        key=lambda name: _choose_by_validation(best_by_model[name]["validation_metrics"]),
    )
    selected_pipeline = fitted_models[selected_model_name]
    joblib.dump(selected_pipeline, model_dir / "best_model.joblib")

    comparison_path = results_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    save_model_comparison_plot(
        comparison, "f1", "validation", results_dir / "validation_f1_comparison.png"
    )
    save_model_comparison_plot(
        comparison, "f1", "test", results_dir / "test_f1_comparison.png"
    )

    selected_test = comparison.loc[comparison["model"] == selected_model_name].iloc[0]
    metadata = {
        "project": "ShopSense",
        "task": "binary repeat-purchase classification",
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "positive_class_meaning": "At least one qualifying distinct invoice during the defined outcome window.",
        "identifier_column": IDENTIFIER_COLUMN,
        "feature_columns": list(FEATURE_COLUMNS),
        "random_seed": RANDOM_SEED,
        "selection_metric": "validation_f1",
        "selection_rule": "Highest validation F1; ties broken by validation ROC-AUC, then validation recall.",
        "selected_model": selected_model_name,
        "selected_model_parameters": best_by_model[selected_model_name]["params"],
        "selected_model_validation_metrics": best_by_model[selected_model_name]["validation_metrics"],
        "selected_model_test_metrics": {
            key.removeprefix("test_"): (
                None if pd.isna(selected_test[key]) else float(selected_test[key])
            )
            for key in selected_test.index
            if key.startswith("test_")
        },
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
            "refit_train_validation": len(X_train_validation),
        },
        "notes": [
            "CustomerID is an identifier and is excluded from predictive features.",
            "All learned preprocessing is contained in each exported sklearn Pipeline.",
            "Hyperparameters and model selection use training and validation data only.",
            "Test metrics are held-out evaluation results and must not be used to retune or select a model.",
            "The selected pipeline is refit on train plus validation before final test scoring.",
            "ROC-AUC and PR-AUC are undefined if an evaluation split contains only one class.",
        ],
        "artifacts": {
            "best_model": "models/best_model.joblib",
            "model_comparison": "results/model_comparison.csv",
            "validation_f1_plot": "results/validation_f1_comparison.png",
            "test_f1_plot": "results/test_f1_comparison.png",
        },
    }
    metadata_path = results_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "selected_model": selected_model_name,
        "comparison": comparison,
        "metadata_path": metadata_path,
        "best_model_path": model_dir / "best_model.joblib",
    }


def main() -> None:
    result = run_model_comparison()
    print(f"Selected model (validation-based): {result['selected_model']}")
    print("\nModel comparison:")
    print(result["comparison"].to_string(index=False))
    print(f"\nSaved model: {result['best_model_path']}")
    print(f"Saved metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
