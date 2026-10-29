
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.train_models import load_split


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "experiments" / "final_gradient_boosting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing train and validation splits.
X_train, y_train = load_split(SPLIT_DIR / "train.csv", "train")
X_val, y_val = load_split(SPLIT_DIR / "validation.csv", "validation")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", GradientBoostingClassifier(random_state=42)),
])

param_distributions = {
    "classifier__n_estimators": [50, 75, 100, 150, 200, 300],
    "classifier__learning_rate": [
        0.01, 0.02, 0.03, 0.05, 0.075, 0.1
    ],
    "classifier__max_depth": [1, 2, 3],
    "classifier__min_samples_leaf": [2, 5, 10, 15, 20],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__max_features": [None, "sqrt", 0.7],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=40,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=1,
    refit=True,
    verbose=1,
    return_train_score=True,
)

print("Starting final Gradient Boosting search...")
print("40 configurations x 5 folds = up to 200 model fits.")

search.fit(X_train, y_train)

best_model = search.best_estimator_
val_probabilities = best_model.predict_proba(X_val)[:, 1]
val_predictions = (val_probabilities >= 0.5).astype(int)

metrics = {
    "model": "Gradient Boosting",
    "selection_method": "5-fold CV on training split",
    "cv_best_f1": float(search.best_score_),
    "validation_threshold": 0.5,
    "validation_accuracy": float(
        accuracy_score(y_val, val_predictions)
    ),
    "validation_precision": float(
        precision_score(y_val, val_predictions, zero_division=0)
    ),
    "validation_recall": float(
        recall_score(y_val, val_predictions, zero_division=0)
    ),
    "validation_f1": float(
        f1_score(y_val, val_predictions, zero_division=0)
    ),
    "validation_roc_auc": float(
        roc_auc_score(y_val, val_probabilities)
    ),
    "best_params": search.best_params_,
}

pd.DataFrame(search.cv_results_).sort_values(
    "rank_test_score"
).to_csv(OUTPUT_DIR / "cv_search_results.csv", index=False)

with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nBEST PARAMETERS")
print(json.dumps(search.best_params_, indent=2))

print(f"\nBest 5-fold CV F1: {search.best_score_:.4f}")

print("\nVALIDATION METRICS (threshold = 0.5)")
for name in (
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_roc_auc",
):
    print(f"{name}: {metrics[name]:.4f}")

print(f"\nResults saved to: {OUTPUT_DIR}")
