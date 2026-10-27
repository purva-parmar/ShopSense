# ShopSense

ShopSense is a retail analytics and machine-learning project built around transaction history. It explores customer purchasing behaviour, prepares a repeat-purchase prediction dataset, and is intended to support evidence-based customer segmentation and retail decision-making.

## Project status

ShopSense is a retail analytics and machine-learning project that analyzes transaction history to understand customer purchasing behaviour, segment customers, and predict repeat purchases.

**Implemented:** Transaction-level EDA and data-quality auditing; reusable transaction preprocessing; customer-level feature engineering; repeat-purchase labels; reproducible train, validation, and test splits; customer segmentation and supervised model artifacts; Streamlit dashboard; cloud deployment; automated tests through GitHub Actions.

**Application:** The Streamlit dashboard provides an interface for exploring customer purchasing patterns, viewing customer segments, and generating repeat-purchase predictions. The live application is deployed using Streamlit Community Cloud.

**Development status:** The application is deployed and the automated GitHub Actions test workflow passes. Predictive performance and the business usefulness of customer segments should be assessed using the project's actual evaluation results before making claims about model quality.

## Business objectives

- Understand customer purchase behaviour from transaction history.
- Prepare and evaluate repeat-purchase prediction.
- Explore customer segmentation as a separate analytical task.
- Produce defensible retail insights that may support decision-making.

## Data source and governance

The project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), an Excel transaction workbook containing invoice, product, quantity, date, unit-price, customer, and country fields. Obtain the workbook from the official UCI source and place it at the repository-relative path `data/raw/Online Retail.xlsx` for local processing.

Phase 2 records missing values, exact duplicates, invalid dates, cancellation markers, and quantity/price signs. Exact duplicate rows and missing or unparseable dates are removed from cleaned outputs. The general cleaned transaction data retains returns (negative quantities), cancellation records, and rows missing `CustomerID` or `Description`. Positive-purchase transactions separately exclude cancellation invoices (invoice number begins with `C`) and require positive quantity and unit price. Customer-level analysis additionally requires an identifiable customer; transaction rows without customer IDs remain valid for transaction-level analysis.

The raw workbook and generated files under `data/processed/` are intentionally excluded from Git. The raw source is not modified by the pipeline. Source provenance is attributed to UCI; users should review the dataset's current source-page terms and attribution guidance before redistribution or other use. No project software license is present in this repository.

## Technical approach

The implemented data path is:

1. Explore the source workbook and produce transaction-level quality summaries.
2. Load, validate, audit, and clean transactions through reusable preprocessing functions.
3. Aggregate qualifying observation-window transactions into customer-level features.
4. Construct a binary repeat-purchase target from a subsequent outcome window.
5. Export customer modeling data with customer-isolated, deterministic train/validation/test splits.

Planned work includes supervised model comparison and evaluation, customer segmentation and interpretation, and application integration/deployment. Python, pandas, openpyxl, Matplotlib, pytest, and Jupyter/IPython kernel tooling are used by the current project configuration. Specific modeling and UI technologies should be selected and added when those components are implemented.

## Data-science methodology

The current configured observation window is **2010-12-01 through 2011-05-31**, inclusive; the outcome window is **2011-06-01 through 2011-08-31**, inclusive. The feature pipeline checks the input date coverage before labeling. These dates reflect the verified local workbook coverage and current implementation; they are configuration choices, not universal definitions of repeat purchasing.

Customer features are calculated only from qualifying observation-period purchases. The `repeat_purchase` target is 1 if the customer has at least one distinct qualifying invoice during the complete outcome window and 0 otherwise. Invoice counting is by distinct invoice number rather than line-item count. Customers need an identifiable `CustomerID` and at least one qualifying observation-period purchase to enter the cohort.

The feature list is explicit; customer identifiers, target values, invoice IDs, raw transaction timestamps, and outcome-derived values are not predictive features. The current data split randomly partitions customers with target stratification, using a fixed seed and a 70%/15%/15% target ratio. It is **not an out-of-time evaluation**. Keep the test split held out for final evaluation. Any learned imputation, scaling, feature selection, resampling, or other transformation belongs in the model workflow and must be fit on training data only.

## Repository structure

```text
.
├── .github/workflows/
│   └── tests.yml                 # GitHub Actions test workflow
├── data/
│   ├── raw/                      # Local source workbook; ignored by Git
│   └── processed/                # Local generated data; ignored by Git
├── docs/
│   └── PRODUCTION_ROADMAP.md     # Engineering maturity plan
├── notebooks/
│   └── 01_eda.ipynb              # Reproducible transaction EDA
├── src/
│   ├── features.py               # Customer features, labels, splits, metadata
│   └── preprocessing.py          # Transaction loading, audit, cleaning, export
├── tests/
│   ├── test_features.py          # Customer feature and split tests
│   ├── test_preprocessing.py     # Transaction data-contract tests
│   └── test_project_setup.py     # Project setup smoke tests
├── .gitignore
├── pyproject.toml                # Project metadata and pytest configuration
├── README.md
└── requirements.txt              # Runtime, notebook, and test dependencies
```

The notebook also writes local figures under `reports/figures/`; those generated figures are ignored by Git. `data/raw/` and `data/processed/` contain only tracked `.gitkeep` placeholders in a fresh clone.

## Setup and installation

Use Python 3.12. Obtain the Excel workbook from the UCI link above and save it to the repository-relative raw-data path shown in **Data source and governance**. The pipeline raises a clear error if the workbook or Phase 2 cleaned transaction artifact is missing; it does not fabricate replacement data.

**Windows PowerShell** (run from the repository root):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To run without activating the environment, substitute `.\.venv\Scripts\python.exe` for `python`.

**macOS/Linux** (run from the repository root):

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the EDA notebook from the repository root:

```powershell
python -m jupyter notebook notebooks/01_eda.ipynb
```

Generate the Phase 2 transaction outputs:

```powershell
python -c "from src.preprocessing import run_preprocessing; run_preprocessing('data/raw/Online Retail.xlsx', 'data/processed')"
```

Then generate the Phase 3 customer dataset and splits:

```powershell
python -c "from src.features import run_feature_pipeline; run_feature_pipeline('data/processed/cleaned_transactions.csv', 'data/processed')"
```

These commands describe the implemented local data-preparation steps; they do not constitute a deployed or monitored end-to-end service.

## Testing and quality

Run the local automated tests from the repository root:

```powershell
python -m pytest -v
```

The tests use synthetic transactions and temporary files to exercise schema validation, cleaning, date-window boundaries, feature/label behavior, deterministic splits, and artifact exports without requiring the UCI workbook. `.github/workflows/tests.yml` defines a Python 3.12 GitHub Actions workflow for pushes and pull requests. A local test pass does not prove that a GitHub Actions run passed; confirm the workflow check on the relevant pull request.

## Results and evaluation

The repository contains local EDA and modeling metadata artifacts when the pipelines have been run; generated datasets are not committed. The source workbook has been observed to cover 2010-12-01 through 2011-12-09, which contains the configured outcome window. Counts in local metadata are data-run outputs and may vary if the input workbook changes.

No predictive model evaluation is implemented in the current repository. Precision, recall, F1, ROC-AUC, calibration, operational thresholds, and business impact have not been established. The current random customer split does not demonstrate generalization to a later time period.

## Limitations and roadmap

The current implementation prepares transaction and customer-level analytical datasets; it does not independently verify that upstream source coverage is complete, and a fixed repeat-purchase horizon is not a lifetime customer property. Model behavior, segment usefulness, application requirements, privacy controls, deployment, monitoring, and recovery procedures remain to be validated or implemented.

See the [production engineering roadmap](docs/PRODUCTION_ROADMAP.md) for ordered work, acceptance criteria, risks, verification steps, and integration contracts.

## Contribution, licensing, and acknowledgements

Contributions should include focused changes, relevant tests, and documentation updates. Preserve the documented data contracts; changes to schemas, qualifying-purchase rules, target windows, or split semantics should be reviewed and communicated to downstream consumers. Confirm local tests and the applicable pull-request checks before integration.

The repository currently has no software `LICENSE` file; no license for the project code is implied. The transaction dataset is attributed to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail). Consult the UCI dataset page for source attribution and applicable terms.
