# ShopSense — Customer Segmentation & Repeat-Purchase Prediction

ShopSense analyzes retail transaction data to understand customer purchasing behavior, identify customer segments, and predict repeat purchases.

# ShopSense - Customer Personality & Shopping Pattern Analyzer

## Overview

ShopSense is a machine learning project that analyzes retail transactions to understand customer purchasing behavior, identify customer segments, and predict repeat purchases.

## Objectives

- Clean and analyze retail transaction data.
- Identify shopping trends using Exploratory Data Analysis (EDA).
- Segment customers using RFM analysis and K-Means clustering.
- Compare machine learning models for repeat-purchase prediction.
- Build an interactive dashboard to visualize insights.

## Dataset

**UCI Online Retail Dataset:** [View Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

## Planned Technology Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Streamlit.

## Expected Outcome (planned)

An interactive dashboard that presents customer segments, shopping patterns, and repeat-purchase predictions to help retailers better understand customer behavior.

## Development status

The repository currently contains the initial project description and preprocessing module placeholder. Data cleaning, EDA, feature engineering, model training, clustering, and the dashboard are planned work and are not implemented yet.

## Prerequisites

- Python 3.12
- The UCI Online Retail Excel file, obtained from the dataset link above

Keep the downloaded file at `data/raw/Online Retail.xlsx`. Raw data is local-only and ignored by Git; do not commit it.

## Setup (Windows PowerShell)

Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project currently has no runtime package dependencies. `requirements.txt` installs pytest for the project smoke tests.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Project structure

```text
data/
  raw/                 Local source dataset (not committed)
  processed/           Reserved for generated datasets (not committed)
src/
  __init__.py          Minimal Python package marker
  preprocessing.py     Preprocessing module (implementation planned)
tests/
  test_project_setup.py
pyproject.toml          Project metadata and pytest configuration
requirements.txt        Test dependency installation
```

## Planned phases and team responsibilities

- **Member 1:** data audit and EDA, reusable preprocessing, customer RFM features, repeat-purchase target, leakage-free customer splits, tests, and documentation.
- **Member 2:** supervised models, model comparison, and evaluation.
- **Member 3:** K-means customer segmentation.
- **Member 4:** Streamlit interface, integration, and release/deployment.

Data processing and model outputs will be documented when implemented. No generated datasets, model metrics, or deployment claims are made at this foundation stage.
