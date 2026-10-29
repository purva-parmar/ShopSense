
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.train_models import (
    MODEL_GRIDS,
    _make_pipeline,
    _probability_for_positive_class,
    load_split,
)

ROOT = Path(__file__).resolve().parent

X_train, y_train = load_split("data/processed/train.csv", "train")
X_val, y_val = load_split("data/processed/validation.csv", "validation")

comparison = pd.read_csv("results/model_comparison.csv")
models_to_check = ["random_forest", "gradient_boosting"]

for name in models_to_check:
    row = comparison.loc[comparison["model"] == name]
    if row.empty:
        raise ValueError(f"No results found for {name}")

    best_params = json.loads(row.iloc[0]["best_params"])
    model = _make_pipeline(name, best_params)
    model.fit(X_train, y_train)

    probabilities = _probability_for_positive_class(model, X_val)
    precision, recall, thresholds = precision_recall_curve(
        y_val, probabilities
    )

    # The final precision/recall entries have no corresponding threshold.
    rows = []
    for i, threshold in enumerate(thresholds):
        p, r = precision[i], recall[i]
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({
            "model": name,
            "threshold": threshold,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    results = pd.DataFrame(rows).sort_values(
        ["f1", "threshold"], ascending=[False, True]
    )
    output = ROOT / "results" / f"{name}_threshold_experiment.csv"
    results.to_csv(output, index=False)

    print(f"\n{name.upper()}")
    print(results.head(10).round(4).to_string(index=False))
    print(f"Saved: {output}")
