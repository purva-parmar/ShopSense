# ShopSense Member 2: supervised models and evaluation

## Files

Copy `src/train_models.py` and `src/evaluation.py` into the repository's `src/` folder. Copy `tests/test_modeling.py` into `tests/`.

## Dependency additions

Add these to the existing `requirements.txt` (do not replace its current entries):

```text
scikit-learn>=1.5,<2.0
joblib>=1.3,<2.0
```

Matplotlib and pandas are already listed in the repository requirements.

## Input contract from Member 1

The feature pipeline should generate these files under `data/processed/`:

- `train.csv`
- `validation.csv`
- `test.csv`

Each CSV must contain `CustomerID`, the eight feature columns declared by `src.features.FEATURE_COLUMNS`, and `repeat_purchase`. `CustomerID` is never used as a predictive feature.

## Run from repository root

```powershell
python -m pip install -r requirements.txt
python -m pytest -v
python -m src.train_models
```

Run the existing preprocessing and feature-generation commands from the README first if the split CSVs do not exist.

## Outputs for Member 4 (UI)

- `models/best_model.joblib`: selected scikit-learn Pipeline, ready for `joblib.load(...)` and `.predict(...)` / `.predict_proba(...)`.
- `models/logistic_regression_pipeline.joblib`, `models/decision_tree_pipeline.joblib`, `models/random_forest_pipeline.joblib`, `models/knn_pipeline.joblib`: each model's best validation-selected configuration, refit on train + validation.
- `results/model_comparison.csv`: validation and held-out test metrics for all four model families.
- `results/model_metadata.json`: feature names, selected model, selection rule, metrics, and artifact paths.
- `results/confusion_matrix_*.png`: held-out test confusion matrices.
- `results/validation_f1_comparison.png` and `results/test_f1_comparison.png`: comparison figures.

For UI prediction, pass a one-row pandas DataFrame whose columns match `feature_columns` in `model_metadata.json`. Do not pass `CustomerID` into the pipeline.

## Evaluation rule

Hyperparameters and the winning model are selected using validation F1 (ties broken by validation ROC-AUC and then recall). The test set is used only for final reporting; do not change the selected model based on test results. Test comparison charts are descriptive and should not be used to retune the model.
