# ShopSense — Retail Transaction EDA and Preprocessing

ShopSense is a retail analytics project using the UCI Online Retail Excel transaction dataset. Phase 2 delivers transaction-level exploratory data analysis, quality auditing, and reusable cleaning; customer modeling and interactive applications belong to later phases.

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
- **Later phases:** customer-level RFM features and repeat-purchase labels/splits; supervised modeling; customer clustering; Streamlit UI and deployment.

Phase 3 and all model training, clustering, and UI/deployment work are not implemented.
