
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.train_models import (
    _make_pipeline,
    _probability_for_positive_class,
    load_split,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the same train/validation splits used by ShopSense.
X_train, y_train = load_split("data/processed/train.csv", "train")
X_val, y_val = load_split("data/processed/validation.csv", "validation")

# Use the already-selected hyperparameters from Experiment 3.
comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
models_to_check = ["random_forest", "gradient_boosting"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

summary_rows = []
threshold_rows = []

for name in models_to_check:
    matching = comparison.loc[comparison["model"] == name]

    if matching.empty:
        raise ValueError(f"No saved model comparison found for {name}.")

    params = json.loads(matching.iloc[0]["best_params"])

    # Out-of-fold probabilities: each training customer is predicted by
    # a model that did not train on that customer's fold.
    template = _make_pipeline(name, params)
    oof_probabilities = cross_val_predict(
        template,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # Select the threshold using training OOF predictions only.
    precision, recall, thresholds = precision_recall_curve(
        y_train, oof_probabilities
    )

    threshold_f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(threshold_f1.argmax())
    chosen_threshold = float(thresholds[best_index])

    # Fit once on the complete training split; evaluate on validation.
    validation_model = _make_pipeline(name, params)
    validation_model.fit(X_train, y_train)
    val_probabilities = _probability_for_positive_class(
        validation_model, X_val
    ).to_numpy()

    for threshold_name, threshold in [
        ("default_0.5", 0.5),
        ("oof_selected", chosen_threshold),
    ]:
        predictions = (val_probabilities >= threshold).astype(int)

        summary_rows.append({
            "model": name,
            "threshold_method": threshold_name,
            "threshold": threshold,
            "validation_accuracy": accuracy_score(y_val, predictions),
            "validation_precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "validation_recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "validation_f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "validation_roc_auc": roc_auc_score(
                y_val, val_probabilities
            ),
        })

    # Save the full OOF threshold curve for later inspection.
    for i, threshold in enumerate(thresholds):
        p = float(precision[i])
        r = float(recall[i])
        f1 = float(threshold_f1[i])

        threshold_rows.append({
            "model": name,
            "threshold": float(threshold),
            "oof_precision": p,
            "oof_recall": r,
            "oof_f1": f1,
        })

    print(f"\n{name.upper()}")
    print(f"OOF-selected threshold: {chosen_threshold:.4f}")
    print(f"Best OOF F1: {threshold_f1[best_index]:.4f}")

summary = pd.DataFrame(summary_rows)
summary_path = RESULTS_DIR / "oof_threshold_validation_comparison.csv"
summary.to_csv(summary_path, index=False)

curve_path = RESULTS_DIR / "oof_threshold_curves.csv"
pd.DataFrame(threshold_rows).to_csv(curve_path, index=False)

print("\nVALIDATION COMPARISON")
print(summary.round(4).to_string(index=False))
print(f"\nSaved comparison: {summary_path}")
print(f"Saved OOF curves: {curve_path}")

