# ShopSense — Retail Analytics Dataset Preparation

ShopSense is a retail analytics project using the UCI Online Retail Excel transaction dataset. Phase 2 delivers transaction-level exploratory data analysis and cleaning; Phase 3 prepares customer-level features, repeat-purchase labels, and reproducible modeling splits. Predictive models and interactive applications remain later work.

## Dataset

Download the **Online Retail** Excel workbook from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it at:

```text
data/raw/Online Retail.xlsx
```

The workbook is an Excel transaction file, not a database. The raw workbook is local-only, intentionally excluded from Git, and never modified by the pipeline. If it is missing, the EDA notebook stops with a clear file error; it does not substitute synthetic data.

## Setup (Windows PowerShell)

Run from the repository root with Python 3.12:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, use `.\.venv\Scripts\python.exe` explicitly without activating the environment.

## Tests

```powershell
python -m pytest -v
```

Tests use synthetic DataFrames and temporary Excel workbooks; they do not require downloading the UCI workbook.

## EDA and preprocessing

From the repository root, open `notebooks/01_eda.ipynb` in Jupyter and run its cells in order:

```powershell
python -m jupyter notebook notebooks/01_eda.ipynb
```

The notebook locates the repository relative to its working directory, loads the real workbook through `src/preprocessing.py`, calculates its numeric findings from the data, displays and saves EDA charts, and writes the three processing artifacts below. It raises a clear error if the source workbook is unavailable.

The pipeline can also be run without Jupyter:

```powershell
python -c "from src.preprocessing import run_preprocessing; run_preprocessing('data/raw/Online Retail.xlsx', 'data/processed')"
```

| Artifact | Purpose |
|---|---|
| `data/processed/cleaned_transactions.csv` | Deduplicated transactions with valid dates; retains returns, cancellation invoices, and rows with missing customer IDs or descriptions. |
| `data/processed/positive_purchase_transactions.csv` | Valid-date, non-cancellation rows with positive quantity and unit price; missing customer IDs and descriptions remain allowed. |
| `data/processed/eda_audit.json` | Input schema, missingness, duplicates, date coverage, distinct entities, cancellation and sign counts, eligibility, and sequential exclusions. |

Exact duplicate rows are removed deterministically. Missing or unparseable dates are audited and excluded from both outputs. The general cleaned set preserves negative-quantity returns and cancellation records; it does not remove rows solely due to missing `CustomerID` or `Description`, unusual values, or statistical outliers. Positive purchases use a separate eligibility policy: invoice numbers beginning with `C` (trimmed and case-insensitive) are treated as cancellations, and only positive `Quantity` and `UnitPrice` are retained. Audit issue categories can overlap; sequential exclusion counts are supplied separately for reconciliation.

Raw input and generated processed data are intentionally excluded from Git. The notebook also saves local charts in `reports/figures/`.

## Project status and phases

- **Phase 1:** project structure, pytest configuration, setup tests, and CI.
- **Phase 2:** transaction EDA, data-quality audit, reusable cleaning, tests, and documentation.
- **Phase 3:** customer observation-window features, repeat-purchase target, customer-level modeling data, and reproducible leakage-free splits.
- **Later phases:** supervised model training and evaluation; customer clustering; Streamlit UI and deployment.

## Phase 3: customer features and repeat-purchase target

The business question is: **Given a customer's qualifying purchase behavior during an observation period, will that customer make at least one qualifying purchase during the subsequent outcome period?**

The default windows are observation **2010-12-01 through 2011-05-31**, followed by outcome **2011-06-01 through 2011-08-31**, with inclusive calendar-day boundaries. The Phase 2 cleaned source covers **2010-12-01 through 2011-12-09**, so the full outcome period is present. `src/features.py` checks coverage bounds at runtime and refuses to label a cohort if they do not cover the requested windows.

Qualifying purchases reuse the Phase 2 rules: a valid invoice date, invoice number not beginning with `C` after trimming and case normalization, positive `Quantity`, and positive `UnitPrice`. A valid, non-missing `CustomerID` is additionally required for customer-level features and labels. Missing descriptions do not disqualify a purchase. Frequency and outcome events count distinct invoice numbers, not line-item rows. A customer is eligible only with at least one qualifying observation-window purchase and complete verified source coverage through the outcome end.

Features are calculated only from observation-window purchases. The reference date for recency is the day after the inclusive observation end (2011-06-01): `recency_days` is the number of calendar days from the last observation purchase date to that reference. `frequency` is distinct observation invoices; `monetary` is the sum of line-level `Quantity * UnitPrice`; `distinct_products` counts non-missing stock codes; `active_purchase_days` counts distinct purchase dates; `average_invoice_value` is the mean of per-invoice line-revenue totals; `average_items_per_invoice` is the mean of per-invoice quantity totals; and `customer_tenure_days` is the first-to-last observation purchase date span. The target `repeat_purchase` is 1 when the eligible customer has at least one distinct qualifying outcome-window invoice, otherwise 0.

Run the feature pipeline from the repository root after Phase 2 has generated `data/processed/cleaned_transactions.csv`:

```powershell
python -c "from src.features import run_feature_pipeline; run_feature_pipeline('data/processed/cleaned_transactions.csv', 'data/processed')"
```

Generated Phase 3 artifacts (all ignored by Git):

| File | Contents |
|---|---|
| `customer_features.csv` | One observation-only feature row per eligible customer, including `CustomerID` for joins and auditing. |
| `repeat_purchase_labels.csv` | One `CustomerID` and `repeat_purchase` label per eligible customer. |
| `customer_modeling_dataset.csv` | Joined identifiers, safe features, and target for the complete eligible cohort. |
| `train.csv` | Training partition. |
| `validation.csv` | Validation partition for model selection. |
| `test.csv` | Held-out partition for final evaluation only. |
| `modeling_metadata.json` | Date coverage/windows, definitions, eligibility, safe feature list, class/split counts, seed, and limitations. |

### Member 2 handoff

- **Files:** `train.csv`, `validation.csv`, and `test.csv`.
- **Target:** `repeat_purchase`.
- **Identifier:** retain `CustomerID` for auditing and split-integrity checks, but do not use it as a predictor.
- **Exact safe feature columns:** `recency_days`, `frequency`, `monetary`, `distinct_products`, `active_purchase_days`, `average_invoice_value`, `average_items_per_invoice`, `customer_tenure_days`. These are also listed in `modeling_metadata.json` under `features.predictive_feature_columns`.
- Use only that explicit feature list to construct the model matrix; do not include identifiers, target, invoice IDs, raw timestamps, or outcome-derived fields.
- Fit imputers, scalers, feature selection, and any other learned transformations on `train.csv` only; apply the fitted transformations unchanged to validation and test.
- Use validation data for model selection. Preserve `test.csv` for final evaluation, and do not oversample or rebalance before splitting.

Splits use a fixed seed (42) and deterministic per-class allocation targeting 70%/15%/15%. Each target class must have at least three customers so it can appear in each partition; otherwise the pipeline raises an actionable error instead of silently producing an unstratified split. Small-class minimum representation can make actual proportions differ slightly; exact counts and proportions are recorded per split in metadata. No scaling, imputation, feature selection, resampling, predictive-model training, or model evaluation is performed in this phase.

Source date bounds prove that the configured outcome window lies within the workbook's observed coverage; the pipeline cannot independently certify that the upstream source contains every transaction. Repeat-purchase labels describe this fixed three-month horizon, not a lifetime behavior. Customer IDs absent from transactions cannot be assigned to customer-level cohorts.

Phase 3 is complete through dataset preparation only. Model training/evaluation, clustering, and UI/deployment remain unimplemented.
