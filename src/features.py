"""Customer features, repeat-purchase labels, and reproducible data splits."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing import REQUIRED_COLUMNS, clean_transactions

OBSERVATION_START = date(2010, 12, 1)
OBSERVATION_END = date(2011, 5, 31)
OUTCOME_START = date(2011, 6, 1)
OUTCOME_END = date(2011, 8, 31)
RANDOM_SEED = 42
IDENTIFIER_COLUMN = "CustomerID"
TARGET_COLUMN = "repeat_purchase"
FEATURE_COLUMNS = (
    "recency_days",
    "frequency",
    "monetary",
    "distinct_products",
    "active_purchase_days",
    "average_invoice_value",
    "average_items_per_invoice",
    "customer_tenure_days",
)
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
FEATURE_DEFINITIONS = {
    "recency_days": (
        "Calendar days from the last qualifying observation purchase date to "
        "the day after the inclusive observation end date."
    ),
    "frequency": "Distinct qualifying observation-window invoice count.",
    "monetary": (
        "Sum of Quantity * UnitPrice across qualifying observation-window "
        "transaction lines."
    ),
    "distinct_products": (
        "Distinct non-missing StockCode values purchased in the observation window."
    ),
    "active_purchase_days": (
        "Distinct calendar dates with a qualifying observation-window purchase."
    ),
    "average_invoice_value": (
        "Mean of per-invoice sums of Quantity * UnitPrice in the observation window."
    ),
    "average_items_per_invoice": (
        "Mean of per-invoice Quantity sums in the observation window."
    ),
    "customer_tenure_days": (
        "Calendar-day span from the first to last qualifying observation-window "
        "purchase, inclusive of the first date (last date minus first date)."
    ),
}


def _as_date(value: date | str, name: str) -> date:
    """Parse a date-like window boundary and reject invalid values clearly."""
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a valid calendar date.") from error


def _window_mask(
    transactions: pd.DataFrame, start: date, end: date
) -> pd.Series:
    """Select timestamps in an inclusive calendar-date window."""
    dates = transactions["InvoiceDate"]
    return dates.ge(pd.Timestamp(start)) & dates.lt(
        pd.Timestamp(end) + timedelta(days=1)
    )


def _validate_windows(
    observation_start: date,
    observation_end: date,
    outcome_start: date,
    outcome_end: date,
) -> None:
    """Require non-empty, ordered observation and future outcome periods."""
    if observation_start > observation_end:
        raise ValueError("observation_start must be on or before observation_end.")
    if outcome_start > outcome_end:
        raise ValueError("outcome_start must be on or before outcome_end.")
    if outcome_start <= observation_end:
        raise ValueError("The outcome window must start after the observation window.")


def _valid_identifier_mask(identifiers: pd.Series) -> pd.Series:
    """Treat null and blank-string identifiers as missing values."""
    mask = identifiers.notna()
    if pd.api.types.is_string_dtype(identifiers.dtype) or identifiers.dtype == object:
        mask &= identifiers.astype("string").str.strip().ne("").fillna(False)
    return mask


def _invoice_key(invoices: pd.Series) -> pd.Series:
    """Normalize invoice identifiers to trimmed strings for distinct counts."""
    return invoices.astype("string").str.strip()


def _validate_modeling_dataset(dataset: pd.DataFrame) -> None:
    """Enforce unique customers, target integrity, and finite safe features."""
    expected_columns = [IDENTIFIER_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    if list(dataset.columns) != expected_columns:
        raise ValueError(
            "Modeling dataset columns do not match the documented safe schema."
        )
    if dataset.empty:
        raise ValueError("No eligible customers remain for the modeling dataset.")
    if dataset[IDENTIFIER_COLUMN].isna().any():
        raise ValueError("Modeling dataset contains a missing CustomerID.")
    if dataset[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError("Modeling dataset must contain exactly one row per customer.")
    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError("repeat_purchase must not contain missing target values.")
    if not dataset[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("repeat_purchase must contain only binary values 0 and 1.")
    numeric = dataset.loc[:, FEATURE_COLUMNS]
    if numeric.isna().any().any():
        raise ValueError("Predictive feature columns must not contain missing values.")
    if not numeric.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
        raise ValueError("Predictive feature columns must contain only finite values.")


def _class_summary(dataset: pd.DataFrame) -> dict[str, Any]:
    """Describe binary target counts and within-set proportions."""
    total = len(dataset)
    counts = dataset[TARGET_COLUMN].value_counts()
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    return {
        "row_count": int(total),
        "class_counts": {"0": negatives, "1": positives},
        "class_proportions": {
            "0": float(negatives / total) if total else 0.0,
            "1": float(positives / total) if total else 0.0,
        },
    }


def _class_split_counts(class_size: int) -> tuple[int, int, int]:
    """Allocate at least one member of a class to each stratified split."""
    if class_size < len(SPLIT_RATIOS):
        raise ValueError(
            "Stratified train/validation/test splitting requires at least "
            f"{len(SPLIT_RATIOS)} customers in each target class; found "
            f"{class_size}. Add eligible data or revise the evaluation design."
        )

    remaining = class_size - len(SPLIT_RATIOS)
    ratios = list(SPLIT_RATIOS.values())
    exact = [remaining * ratio for ratio in ratios]
    allocated = [1 + int(value) for value in exact]
    leftovers = class_size - sum(allocated)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (exact[index] - int(exact[index]), -index),
        reverse=True,
    )
    for index in order[:leftovers]:
        allocated[index] += 1
    return allocated[0], allocated[1], allocated[2]


def split_customer_dataset(
    dataset: pd.DataFrame, random_seed: int = RANDOM_SEED
) -> dict[str, pd.DataFrame]:
    """Create deterministic, stratified customer-level 70/15/15 splits.

    Each target class must have at least three customers so every split can
    contain that class. For very small cohorts the minimum-per-class constraint
    takes precedence over exact ratios; metadata records actual split sizes.
    """
    if not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")
    _validate_modeling_dataset(dataset)
    class_counts = dataset[TARGET_COLUMN].value_counts()
    if set(class_counts.index.tolist()) != {0, 1}:
        raise ValueError(
            "Both repeat_purchase classes (0 and 1) are required for a "
            "stratified split."
        )
    for class_size in class_counts:
        _class_split_counts(int(class_size))

    split_parts: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for target in (0, 1):
        class_rows = dataset.loc[dataset[TARGET_COLUMN] == target].sample(
            frac=1, random_state=random_seed + target
        )
        train_count, validation_count, test_count = _class_split_counts(
            len(class_rows)
        )
        train_end = train_count
        validation_end = train_end + validation_count
        split_parts["train"].append(class_rows.iloc[:train_end])
        split_parts["validation"].append(
            class_rows.iloc[train_end:validation_end]
        )
        split_parts["test"].append(class_rows.iloc[validation_end:])
        if len(class_rows.iloc[validation_end:]) != test_count:
            raise RuntimeError("Internal stratified split allocation did not reconcile.")

    splits = {}
    for index, name in enumerate(("train", "validation", "test")):
        splits[name] = (
            pd.concat(split_parts[name])
            .sample(frac=1, random_state=random_seed + 100 + index)
            .reset_index(drop=True)
        )

    all_ids = pd.concat(
        [split[IDENTIFIER_COLUMN] for split in splits.values()],
        ignore_index=True,
    )
    if len(all_ids) != len(dataset) or all_ids.nunique() != len(dataset):
        raise RuntimeError("Customer-level split membership failed integrity checks.")
    return splits


def build_customer_modeling_dataset(
    transactions: pd.DataFrame,
    output_directory: str | Path | None = None,
    observation_start: date | str = OBSERVATION_START,
    observation_end: date | str = OBSERVATION_END,
    outcome_start: date | str = OUTCOME_START,
    outcome_end: date | str = OUTCOME_END,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Build observation-only customer features, future labels, splits, metadata.

    ``transactions`` is the Phase 2 general cleaned transaction DataFrame (or
    the same schema in-memory). The Phase 2 ``clean_transactions`` function is
    reused to derive its documented positive-purchase rows and preserves the
    authoritative cancellation, date, quantity, and price conventions.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")
    starts_ends = (
        _as_date(observation_start, "observation_start"),
        _as_date(observation_end, "observation_end"),
        _as_date(outcome_start, "outcome_start"),
        _as_date(outcome_end, "outcome_end"),
    )
    obs_start, obs_end, out_start, out_end = starts_ends
    _validate_windows(*starts_ends)

    phase2 = clean_transactions(transactions)
    cleaned = phase2.cleaned_transactions
    purchases = phase2.positive_purchase_transactions
    if cleaned.empty:
        raise ValueError(
            "Phase 2 cleaned transactions contain no valid dates; coverage "
            "cannot be verified."
        )
    valid_invoice_numbers = _valid_identifier_mask(purchases["InvoiceNo"])
    if not valid_invoice_numbers.all():
        raise ValueError(
            "Phase 2 positive-purchase rows contain missing or blank InvoiceNo "
            "values; invoice-level features and targets cannot be computed."
        )

    coverage_start = cleaned["InvoiceDate"].min().date()
    coverage_end = cleaned["InvoiceDate"].max().date()
    if coverage_start > obs_start or coverage_end < out_end:
        raise ValueError(
            "The verified transaction date coverage "
            f"({coverage_start.isoformat()} through {coverage_end.isoformat()}) "
            "does not fully contain the requested observation and outcome "
            f"windows ({obs_start.isoformat()} through {out_end.isoformat()}). "
            "Do not label customers from an incomplete outcome period."
        )

    customer_purchases = purchases.loc[
        _valid_identifier_mask(purchases[IDENTIFIER_COLUMN])
    ].copy()
    observation_mask = _window_mask(customer_purchases, obs_start, obs_end)
    outcome_mask = _window_mask(customer_purchases, out_start, out_end)
    observation = customer_purchases.loc[observation_mask].copy()
    outcome = customer_purchases.loc[outcome_mask].copy()
    source_observation_mask = _window_mask(purchases, obs_start, obs_end)
    source_outcome_mask = _window_mask(purchases, out_start, out_end)
    outcome_only_customers = set(outcome[IDENTIFIER_COLUMN]) - set(
        observation[IDENTIFIER_COLUMN]
    )
    if observation.empty:
        raise ValueError(
            "No identified-customer qualifying purchases occur in the "
            "observation window."
        )

    observation["LineRevenue"] = observation["Quantity"] * observation["UnitPrice"]
    observation["PurchaseDate"] = observation["InvoiceDate"].dt.normalize()
    observation["InvoiceKey"] = _invoice_key(observation["InvoiceNo"])
    outcome["InvoiceKey"] = _invoice_key(outcome["InvoiceNo"])
    invoice_summary = (
        observation.groupby([IDENTIFIER_COLUMN, "InvoiceKey"], sort=True)
        .agg(
            invoice_value=("LineRevenue", "sum"),
            invoice_items=("Quantity", "sum"),
        )
        .reset_index()
    )
    invoice_customer = invoice_summary.groupby(IDENTIFIER_COLUMN, sort=True).agg(
        average_invoice_value=("invoice_value", "mean"),
        average_items_per_invoice=("invoice_items", "mean"),
    )
    observation_features = observation.groupby(IDENTIFIER_COLUMN, sort=True).agg(
        last_purchase_date=("PurchaseDate", "max"),
        first_purchase_date=("PurchaseDate", "min"),
        frequency=("InvoiceKey", "nunique"),
        monetary=("LineRevenue", "sum"),
        distinct_products=("StockCode", "nunique"),
        active_purchase_days=("PurchaseDate", "nunique"),
    )
    reference_date = pd.Timestamp(obs_end) + timedelta(days=1)
    observation_features["recency_days"] = (
        reference_date - observation_features["last_purchase_date"]
    ).dt.days
    observation_features["customer_tenure_days"] = (
        observation_features["last_purchase_date"]
        - observation_features["first_purchase_date"]
    ).dt.days
    observation_features = observation_features.join(invoice_customer)
    features = observation_features.loc[:, FEATURE_COLUMNS].reset_index()

    outcome_invoice_counts = outcome.groupby(IDENTIFIER_COLUMN)["InvoiceKey"].nunique()
    labels = pd.DataFrame(
        {IDENTIFIER_COLUMN: features[IDENTIFIER_COLUMN]}
    )
    labels[TARGET_COLUMN] = (
        labels[IDENTIFIER_COLUMN]
        .map(outcome_invoice_counts)
        .fillna(0)
        .gt(0)
        .astype("int8")
    )
    modeling_dataset = features.merge(
        labels, on=IDENTIFIER_COLUMN, how="inner", validate="one_to_one"
    )
    modeling_dataset = modeling_dataset.loc[
        :, [IDENTIFIER_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    ]
    _validate_modeling_dataset(modeling_dataset)
    splits = split_customer_dataset(modeling_dataset, random_seed=random_seed)

    output_paths: dict[str, Path] = {}
    if output_directory is not None:
        output_paths = _write_artifacts(
            features, labels, modeling_dataset, splits, output_directory
        )

    class_counts = modeling_dataset[TARGET_COLUMN].value_counts()
    metadata = {
        "source": {
            "dataset": "UCI Online Retail Excel transaction dataset",
            "phase2_input": "data/processed/cleaned_transactions.csv",
            "phase2_cleaning_reused": True,
            "source_coverage": {
                "start": coverage_start.isoformat(),
                "end": coverage_end.isoformat(),
            },
            "phase2_cleaned_row_count": int(len(cleaned)),
            "phase2_positive_purchase_row_count": int(len(purchases)),
        },
        "windows": {
            "observation": {
                "start_inclusive": obs_start.isoformat(),
                "end_inclusive": obs_end.isoformat(),
            },
            "outcome": {
                "start_inclusive": out_start.isoformat(),
                "end_inclusive": out_end.isoformat(),
            },
            "boundaries": (
                "Calendar dates are inclusive; timestamps are selected from "
                "00:00 on start through, but excluding, 00:00 the day after end."
            ),
            "observation_reference_date": reference_date.date().isoformat(),
            "complete_outcome_coverage_verified": True,
        },
        "target": {
            "name": TARGET_COLUMN,
            "definition": (
                "1 if an eligible observation-window customer has at least one "
                "distinct qualifying invoice during the complete outcome window; "
                "otherwise 0."
            ),
            "counting_unit": "distinct InvoiceNo per CustomerID, not line items",
            "eligible_cohort": (
                "At least one qualifying observation-window purchase, a valid "
                "CustomerID, and verified complete source coverage through "
                "outcome_end."
            ),
            "class_counts": {
                "0": int(class_counts.get(0, 0)),
                "1": int(class_counts.get(1, 0)),
            },
            "class_proportions": _class_summary(modeling_dataset)[
                "class_proportions"
            ],
        },
        "purchase_eligibility": {
            "rules": (
                "Reuse Phase 2 positive-purchase semantics: valid date, invoice "
                "not beginning with C after trimming and case normalization, "
                "Quantity > 0, UnitPrice > 0, and a present InvoiceNo for "
                "invoice-level counting. A non-missing CustomerID is additionally "
                "required for customer-level analysis."
            ),
            "missing_customer_ids_excluded_from_customer_analysis": int(
                (~_valid_identifier_mask(purchases[IDENTIFIER_COLUMN])).sum()
            ),
            "missing_description_does_not_exclude_purchase": True,
        },
        "features": {
            "identifier_column": IDENTIFIER_COLUMN,
            "target_column": TARGET_COLUMN,
            "predictive_feature_columns": list(FEATURE_COLUMNS),
            "definitions": FEATURE_DEFINITIONS,
            "feature_source": "Qualifying transactions inside observation window only.",
            "forbidden_as_predictors": [
                IDENTIFIER_COLUMN,
                TARGET_COLUMN,
                "InvoiceNo",
                "InvoiceDate",
                "outcome-derived columns",
            ],
        },
        "splits": {
            "ratios": SPLIT_RATIOS,
            "random_seed": random_seed,
            "stratification": "per-class deterministic allocation",
            "stratification_feasible": True,
            "minimum_customers_per_class": len(SPLIT_RATIOS),
            "sets_are_disjoint_by_customer": True,
            "test_set_policy": (
                "Reserve test.csv for final evaluation; do not use it for model "
                "selection or fitting learned transformations."
            ),
            "transformation_policy": (
                "Fit imputation, scaling, feature selection, and any other "
                "learned preprocessing on train.csv only; apply unchanged to "
                "validation.csv and test.csv."
            ),
        },
        "counts": {
            "source_coverage_rows": int(len(cleaned)),
            "phase2_qualifying_purchase_rows": int(len(purchases)),
            "qualifying_purchase_rows_missing_customer_id": int(
                (~_valid_identifier_mask(purchases[IDENTIFIER_COLUMN])).sum()
            ),
            "observation_window_rows_missing_customer_id": int(
                (
                    source_observation_mask
                    & ~_valid_identifier_mask(purchases[IDENTIFIER_COLUMN])
                ).sum()
            ),
            "outcome_window_rows_missing_customer_id": int(
                (
                    source_outcome_mask
                    & ~_valid_identifier_mask(purchases[IDENTIFIER_COLUMN])
                ).sum()
            ),
            "qualifying_purchase_rows_outside_both_windows": int(
                len(purchases)
                - int(source_observation_mask.sum())
                - int(source_outcome_mask.sum())
            ),
            "outcome_customers_without_observation_purchase": int(
                len(outcome_only_customers)
            ),
            "identified_observation_window_purchase_rows": int(len(observation)),
            "identified_outcome_window_purchase_rows": int(len(outcome)),
            "eligible_customer_count": int(len(modeling_dataset)),
            "overall": _class_summary(modeling_dataset),
            "per_split": {
                name: _class_summary(split) for name, split in splits.items()
            },
        },
        "limitations_and_warnings": [
            "A repeat purchase means at least one distinct qualifying invoice during this fixed outcome window; it is not a lifetime customer property.",
            "CustomerID-missing purchase rows cannot be assigned to customer-level cohorts and are excluded only at this stage.",
            "The source coverage check verifies observed date bounds; it cannot independently certify upstream completeness of every source transaction.",
            "No imputation, scaling, feature selection, resampling, or model fitting is performed.",
        ],
        "artifacts": {name: path.name for name, path in output_paths.items()},
    }
    if output_directory is not None:
        metadata_path = Path(output_directory) / "modeling_metadata.json"
        metadata["artifacts"]["metadata"] = metadata_path.name
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, allow_nan=False)
            metadata_file.write("\n")
        output_paths["metadata"] = metadata_path

    return modeling_dataset, splits, metadata


def _write_artifacts(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    modeling_dataset: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Export customer feature, label, combined, and split CSV artifacts."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    tables = {
        "customer_features": features,
        "repeat_purchase_labels": labels,
        "customer_modeling_dataset": modeling_dataset,
        **splits,
    }
    paths = {}
    for name, table in tables.items():
        path = output_path / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8")
        paths[name] = path
    return paths


def run_feature_pipeline(
    cleaned_transactions_path: str | Path,
    output_directory: str | Path = "data/processed",
    observation_start: date | str = OBSERVATION_START,
    observation_end: date | str = OBSERVATION_END,
    outcome_start: date | str = OUTCOME_START,
    outcome_end: date | str = OUTCOME_END,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Load the Phase 2 cleaned CSV and write the complete Phase 3 artifact set."""
    path = Path(cleaned_transactions_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2 cleaned transactions not found: {path}")
    try:
        transactions = pd.read_csv(
            path,
            parse_dates=["InvoiceDate"],
            dtype={"InvoiceNo": "string"},
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read Phase 2 cleaned CSV: {path}") from error
    return build_customer_modeling_dataset(
        transactions,
        output_directory=output_directory,
        observation_start=observation_start,
        observation_end=observation_end,
        outcome_start=outcome_start,
        outcome_end=outcome_end,
        random_seed=random_seed,
    )
