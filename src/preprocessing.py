"""Load, audit, and clean UCI Online Retail transaction data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import pandas as pd

REQUIRED_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


@dataclass(frozen=True)
class CleaningResult:
    """General cleaned transactions, positive purchases, and audit details."""

    cleaned_transactions: pd.DataFrame
    positive_purchase_transactions: pd.DataFrame
    audit: dict[str, Any]


def validate_required_columns(transactions: pd.DataFrame) -> None:
    """Validate a DataFrame input and its required transaction columns."""
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")
    if not transactions.columns.is_unique:
        raise ValueError("Transaction data must not contain duplicate column names.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in transactions.columns
    ]
    if missing_columns:
        raise ValueError(
            "Transaction data is missing required columns: "
            + ", ".join(missing_columns)
        )


def parse_invoice_dates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Copy data and parse dates, marking unparseable or missing values as NaT.

    The returned InvoiceDate is timezone-naive UTC; timezone-naive inputs are
    interpreted as UTC. Invalid dates remain NaT so the audit can count them
    before cleaning explicitly excludes them.
    """
    validate_required_columns(transactions)
    parsed = transactions.copy(deep=True)
    parsed["InvoiceDate"] = pd.to_datetime(
        parsed["InvoiceDate"], errors="coerce", format="mixed", utc=True
    ).dt.tz_localize(None)
    return parsed


def _normalize_numeric_columns(transactions: pd.DataFrame) -> pd.DataFrame:
    """Copy data and ensure quantity and unit-price columns are numeric."""
    normalized = transactions.copy(deep=True)
    for column in ("Quantity", "UnitPrice"):
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{column} must contain numeric values or missing values."
            ) from error
    return normalized


def _prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with validated, normalized date and numeric columns."""
    validate_required_columns(transactions)
    return _normalize_numeric_columns(parse_invoice_dates(transactions))


def load_transactions(file_path: str | Path) -> pd.DataFrame:
    """Load an Excel workbook from an explicit path and validate its schema."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Online Retail Excel file not found: {path}")

    try:
        transactions = pd.read_excel(path)
    except (OSError, ValueError, ImportError, BadZipFile) as error:
        raise ValueError(f"Could not read Excel transaction file: {path}") from error

    return _prepare_transactions(transactions)


def _count_values(values: pd.Series) -> dict[str, int]:
    """Count negative, zero, and positive values without counting missing data."""
    return {
        "negative": int(values.lt(0).sum()),
        "zero": int(values.eq(0).sum()),
        "positive": int(values.gt(0).sum()),
    }


def _cancellation_mask(transactions: pd.DataFrame) -> pd.Series:
    """Identify invoice numbers whose trimmed, case-normalized value starts C."""
    invoice_numbers = transactions["InvoiceNo"].astype("string")
    return invoice_numbers.str.strip().str.upper().str.startswith("C", na=False)


def _positive_purchase_mask(transactions: pd.DataFrame) -> pd.Series:
    """Identify non-cancellation rows with a valid date and positive values."""
    return (
        transactions["InvoiceDate"].notna()
        & ~_cancellation_mask(transactions)
        & transactions["Quantity"].gt(0).fillna(False)
        & transactions["UnitPrice"].gt(0).fillna(False)
    )


def audit_transactions(transactions: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable transaction quality counts and source metadata.

    Missing-value percentages use all input rows as their denominator. Invalid
    non-null dates are reported separately from source-null dates. Issue
    categories may overlap and must not be summed as unique affected rows.
    """
    validate_required_columns(transactions)
    prepared = _prepare_transactions(transactions)
    row_count, column_count = transactions.shape
    missing_counts = transactions.isna().sum()
    invalid_date_mask = (
        transactions["InvoiceDate"].notna() & prepared["InvoiceDate"].isna()
    )
    missing_date_count = int(transactions["InvoiceDate"].isna().sum())
    valid_dates = prepared["InvoiceDate"].dropna()
    cancellation_rows = _cancellation_mask(prepared)
    cancellation_invoices = prepared.loc[cancellation_rows, "InvoiceNo"]
    missing_customer_count = int(transactions["CustomerID"].isna().sum())
    missing_description_count = int(transactions["Description"].isna().sum())

    return {
        "input_row_count": int(row_count),
        "input_column_count": int(column_count),
        "columns": [str(column) for column in transactions.columns],
        "data_types": {
            str(column): str(dtype)
            for column, dtype in transactions.dtypes.items()
        },
        "exact_duplicate_row_count": int(transactions.duplicated().sum()),
        "missing_values": {
            str(column): {
                "count": int(count),
                "percentage": float(count / row_count * 100) if row_count else 0.0,
            }
            for column, count in missing_counts.items()
        },
        "missing_customer_id_row_count": missing_customer_count,
        "missing_description_row_count": missing_description_count,
        "invalid_date_value_count": int(invalid_date_mask.sum()),
        "missing_date_value_count": missing_date_count,
        "invalid_or_missing_date_count": int(
            invalid_date_mask.sum() + missing_date_count
        ),
        "date_coverage": {
            "start": valid_dates.min().isoformat() if not valid_dates.empty else None,
            "end": valid_dates.max().isoformat() if not valid_dates.empty else None,
        },
        "distinct_non_missing_counts": {
            "customers": int(prepared["CustomerID"].nunique(dropna=True)),
            "invoices": int(prepared["InvoiceNo"].nunique(dropna=True)),
            "products": int(prepared["StockCode"].nunique(dropna=True)),
            "countries": int(prepared["Country"].nunique(dropna=True)),
        },
        "cancellation_invoice_row_count": int(cancellation_rows.sum()),
        "distinct_cancellation_invoice_count": int(
            cancellation_invoices.nunique(dropna=True)
        ),
        "quantity_sign_counts": _count_values(prepared["Quantity"]),
        "unit_price_sign_counts": _count_values(prepared["UnitPrice"]),
        "negative_quantity_potential_return_row_count": int(
            prepared["Quantity"].lt(0).sum()
        ),
        "positive_purchase_eligible_row_count": int(
            _positive_purchase_mask(prepared).sum()
        ),
        "missing_value_percentage_denominator": (
            "input_row_count; issue categories can overlap"
        ),
        "overlap_note": (
            "Issue counts overlap and must not be summed as unique affected rows."
        ),
    }


def _positive_purchase_exclusions(
    transactions: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    """Return eligibility and overlapping exclusion counts for valid-date rows."""
    cancellations = _cancellation_mask(transactions)
    quantity_positive = transactions["Quantity"].gt(0).fillna(False)
    price_positive = transactions["UnitPrice"].gt(0).fillna(False)
    eligible = ~cancellations & quantity_positive & price_positive
    exclusions = {
        "cancellation_invoice_rows": int(cancellations.sum()),
        "distinct_cancellation_invoices": int(
            transactions.loc[cancellations, "InvoiceNo"].nunique(dropna=True)
        ),
        "negative_quantity_rows": int(transactions["Quantity"].lt(0).sum()),
        "zero_quantity_rows": int(transactions["Quantity"].eq(0).sum()),
        "missing_quantity_rows": int(transactions["Quantity"].isna().sum()),
        "negative_unit_price_rows": int(transactions["UnitPrice"].lt(0).sum()),
        "zero_unit_price_rows": int(transactions["UnitPrice"].eq(0).sum()),
        "missing_unit_price_rows": int(transactions["UnitPrice"].isna().sum()),
        "positive_purchase_excluded_unique_rows": int((~eligible).sum()),
    }
    return eligible, exclusions


def clean_transactions(transactions: pd.DataFrame) -> CleaningResult:
    """Create general cleaned transactions and a separate positive-purchase set.

    Exact duplicates are removed first, then rows with invalid or missing
    invoice dates. The general output retains returns, cancellation invoices,
    and missing CustomerID or Description values. Positive purchases exclude
    cancellation invoice numbers and rows without positive Quantity and
    UnitPrice. Missing customer/description values are retained in both sets.
    Exclusion reconciliation is sequential and disjoint; quality categories
    are also reported as overlapping counts for diagnosis.
    """
    validate_required_columns(transactions)
    source = transactions.copy(deep=True)
    initial_audit = audit_transactions(source)

    duplicate_mask = source.duplicated()
    without_duplicates = source.loc[~duplicate_mask].copy()
    after_duplicate_count = len(without_duplicates)

    dated = _prepare_transactions(without_duplicates)
    valid_date_mask = dated["InvoiceDate"].notna()
    cleaned = dated.loc[valid_date_mask].copy()
    invalid_dates_removed = int((~valid_date_mask).sum())

    cancellations = _cancellation_mask(cleaned)
    after_cancellations = cleaned.loc[~cancellations]
    quantity_positive = after_cancellations["Quantity"].gt(0).fillna(False)
    after_quantity = after_cancellations.loc[quantity_positive]
    price_positive = after_quantity["UnitPrice"].gt(0).fillna(False)
    positive_purchases = after_quantity.loc[price_positive].copy()

    _, overlapping_exclusions = _positive_purchase_exclusions(cleaned)
    sequential_exclusions = {
        "cancellation_invoice_rows": int(cancellations.sum()),
        "nonpositive_or_missing_quantity_rows_after_cancellations": int(
            (~quantity_positive).sum()
        ),
        "nonpositive_or_missing_unit_price_rows_after_quantity_filter": int(
            (~price_positive).sum()
        ),
    }
    audit = {
        "before_cleaning": initial_audit,
        "cleaning": {
            "row_counts": {
                "input": len(source),
                "after_exact_duplicate_removal": int(after_duplicate_count),
                "after_invalid_date_removal": len(cleaned),
                "cleaned_transactions": len(cleaned),
                "after_cancellation_exclusion": len(after_cancellations),
                "after_quantity_eligibility": len(after_quantity),
                "positive_purchase_transactions": len(positive_purchases),
            },
            "exact_duplicate_rows_removed": int(duplicate_mask.sum()),
            "invalid_or_missing_dates_removed_after_deduplication": (
                invalid_dates_removed
            ),
            "positive_purchase_exclusions": overlapping_exclusions,
            "positive_purchase_exclusion_reconciliation": sequential_exclusions,
            "positive_purchase_exclusion_counts_overlap": True,
            "positive_purchase_policy": (
                "Require a valid InvoiceDate, exclude trimmed case-insensitive "
                "invoice numbers beginning with C, and require Quantity > 0 "
                "and UnitPrice > 0. Retain missing CustomerID and Description."
            ),
            "date_policy": (
                "Parse InvoiceDate with errors coerced to NaT; exclude null or "
                "unparseable dates from both outputs."
            ),
        },
    }
    return CleaningResult(cleaned, positive_purchases, audit)


def write_cleaning_outputs(
    result: CleaningResult, output_directory: str | Path
) -> dict[str, Path]:
    """Write cleaned CSVs and a JSON audit to a local output directory."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs = {
        "cleaned_transactions": output_path / "cleaned_transactions.csv",
        "positive_purchase_transactions": (
            output_path / "positive_purchase_transactions.csv"
        ),
        "audit": output_path / "eda_audit.json",
    }
    result.cleaned_transactions.to_csv(
        outputs["cleaned_transactions"], index=False, date_format="%Y-%m-%dT%H:%M:%S"
    )
    result.positive_purchase_transactions.to_csv(
        outputs["positive_purchase_transactions"],
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
    )
    with outputs["audit"].open("w", encoding="utf-8") as audit_file:
        json.dump(result.audit, audit_file, indent=2, allow_nan=False)
        audit_file.write("\n")
    return outputs


def run_preprocessing(
    input_path: str | Path, output_directory: str | Path
) -> CleaningResult:
    """Load, clean, and export the documented local preprocessing outputs."""
    result = clean_transactions(load_transactions(input_path))
    write_cleaning_outputs(result, output_directory)
    return result
