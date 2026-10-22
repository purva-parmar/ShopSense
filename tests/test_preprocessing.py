"""Unit tests for reusable transaction preprocessing."""

import json

import pandas as pd
import pytest

from src.preprocessing import (
    REQUIRED_COLUMNS,
    audit_transactions,
    clean_transactions,
    load_transactions,
    parse_invoice_dates,
    write_cleaning_outputs,
)


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Small dataset with duplicates and deliberately varied quality issues."""
    return pd.DataFrame(
        [
            ["1001", "A1", "Widget", 2, "2011-01-01 10:00", 10.0, 1, "UK"],
            ["1001", "A1", "Widget", 2, "2011-01-01 10:00", 10.0, 1, "UK"],
            ["C1002", "A2", "Returned", -2, "2011-01-02 10:00", 10.0, 1, "UK"],
            ["1003", "A3", "Return", -1, "2011-01-03 10:00", 5.0, 2, "France"],
            ["1004", "A4", "Zero quantity", 0, "2011-01-04 10:00", 2.0, 2, "UK"],
            ["1005", "A5", "Zero price", 1, "2011-01-05 10:00", 0.0, 3, "UK"],
            ["1006", "A6", None, 3, "2011-01-06 10:00", 4.0, None, "France"],
            ["1007", "A7", "Bad date", 1, "not-a-date", 5.0, 4, "UK"],
            ["1008", "A8", "Negative price", 1, "2011-01-08 10:00", -1.0, 5, "UK"],
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_parse_invoice_dates_returns_copy_and_marks_invalid_values(
    sample_transactions: pd.DataFrame,
) -> None:
    parsed = parse_invoice_dates(sample_transactions)

    assert pd.api.types.is_datetime64_any_dtype(parsed["InvoiceDate"])
    assert pd.isna(parsed.loc[7, "InvoiceDate"])
    assert sample_transactions.loc[7, "InvoiceDate"] == "not-a-date"
    assert parsed is not sample_transactions


def test_date_parsing_normalizes_timezone_aware_and_naive_values_to_utc(
    sample_transactions: pd.DataFrame,
) -> None:
    transactions = sample_transactions.iloc[[0, 2]].copy()
    transactions["InvoiceDate"] = [
        "2011-01-01 10:00:00+02:00",
        "2011-01-02 10:00:00",
    ]

    parsed = parse_invoice_dates(transactions)

    assert str(parsed["InvoiceDate"].dtype) == "datetime64[ns]"
    assert parsed["InvoiceDate"].tolist() == [
        pd.Timestamp("2011-01-01 08:00:00"),
        pd.Timestamp("2011-01-02 10:00:00"),
    ]


def test_missing_required_columns_raise_clear_error(
    sample_transactions: pd.DataFrame,
) -> None:
    incomplete = sample_transactions.drop(columns=["Country"])

    with pytest.raises(ValueError, match="Country"):
        audit_transactions(incomplete)


def test_missing_input_file_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_transactions(tmp_path / "missing.xlsx")


def test_load_transactions_parses_excel_and_validates_schema(
    tmp_path, sample_transactions
) -> None:
    workbook_path = tmp_path / "transactions.xlsx"
    sample_transactions.to_excel(workbook_path, index=False)

    loaded = load_transactions(workbook_path)

    assert list(loaded.columns) == list(REQUIRED_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(loaded["InvoiceDate"])
    assert pd.api.types.is_numeric_dtype(loaded["Quantity"])
    assert pd.isna(loaded.loc[7, "InvoiceDate"])


def test_unreadable_excel_file_raises_clear_error(tmp_path) -> None:
    workbook_path = tmp_path / "not-an-excel-file.xlsx"
    workbook_path.write_text("not an Excel workbook", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read Excel"):
        load_transactions(workbook_path)


def test_audit_reports_counts_percentages_and_overlap(
    sample_transactions: pd.DataFrame,
) -> None:
    audit = audit_transactions(sample_transactions)

    assert audit["input_row_count"] == 9
    assert audit["input_column_count"] == 8
    assert audit["exact_duplicate_row_count"] == 1
    assert audit["invalid_or_missing_date_count"] == 1
    assert audit["missing_values"]["CustomerID"] == {
        "count": 1,
        "percentage": pytest.approx(100 / 9),
    }
    assert audit["missing_values"]["Description"]["count"] == 1
    assert audit["cancellation_invoice_row_count"] == 1
    assert audit["distinct_cancellation_invoice_count"] == 1
    assert audit["quantity_sign_counts"] == {
        "negative": 2,
        "zero": 1,
        "positive": 6,
    }
    assert audit["unit_price_sign_counts"] == {
        "negative": 1,
        "zero": 1,
        "positive": 7,
    }
    assert audit["negative_quantity_potential_return_row_count"] == 2
    assert audit["date_coverage"]["start"] == "2011-01-01T10:00:00"
    assert audit["date_coverage"]["end"] == "2011-01-08T10:00:00"
    assert "overlap" in audit["overlap_note"]


def test_cleaning_removes_duplicates_and_invalid_dates_but_retains_audit_rows(
    sample_transactions: pd.DataFrame,
) -> None:
    result = clean_transactions(sample_transactions)

    assert len(result.cleaned_transactions) == 7
    assert result.audit["cleaning"]["exact_duplicate_rows_removed"] == 1
    assert (
        result.audit["cleaning"][
            "invalid_or_missing_dates_removed_after_deduplication"
        ]
        == 1
    )
    assert result.cleaned_transactions["InvoiceNo"].tolist() == [
        "1001",
        "C1002",
        "1003",
        "1004",
        "1005",
        "1006",
        "1008",
    ]
    assert "C1002" in result.cleaned_transactions["InvoiceNo"].values
    assert -1 in result.cleaned_transactions["Quantity"].values
    assert result.cleaned_transactions["CustomerID"].isna().sum() == 1
    assert result.cleaned_transactions["Description"].isna().sum() == 1


def test_positive_purchase_output_uses_explicit_eligibility_rules(
    sample_transactions: pd.DataFrame,
) -> None:
    result = clean_transactions(sample_transactions)

    assert result.positive_purchase_transactions["InvoiceNo"].tolist() == [
        "1001",
        "1006",
    ]
    assert result.positive_purchase_transactions["Quantity"].gt(0).all()
    assert result.positive_purchase_transactions["UnitPrice"].gt(0).all()
    assert not result.positive_purchase_transactions["InvoiceNo"].str.startswith(
        "C"
    ).any()
    assert result.positive_purchase_transactions["CustomerID"].isna().sum() == 1
    exclusions = result.audit["cleaning"]["positive_purchase_exclusions"]
    assert exclusions["cancellation_invoice_rows"] == 1
    # The cancellation row also has negative quantity, so issue counts overlap.
    assert exclusions["negative_quantity_rows"] == 2
    assert exclusions["zero_quantity_rows"] == 1
    assert exclusions["zero_unit_price_rows"] == 1
    assert exclusions["negative_unit_price_rows"] == 1
    assert exclusions["positive_purchase_excluded_unique_rows"] == 5


def test_cleaning_row_counts_reconcile_and_input_is_unchanged(
    sample_transactions: pd.DataFrame,
) -> None:
    original = sample_transactions.copy(deep=True)
    result = clean_transactions(sample_transactions)
    row_counts = result.audit["cleaning"]["row_counts"]

    assert row_counts == {
        "input": 9,
        "after_exact_duplicate_removal": 8,
        "after_invalid_date_removal": 7,
        "cleaned_transactions": 7,
        "after_cancellation_exclusion": 6,
        "after_quantity_eligibility": 4,
        "positive_purchase_transactions": 2,
    }
    assert row_counts["input"] - row_counts["after_exact_duplicate_removal"] == 1
    assert row_counts["after_exact_duplicate_removal"] - row_counts[
        "after_invalid_date_removal"
    ] == 1
    assert row_counts["after_cancellation_exclusion"] == 6
    assert row_counts["after_quantity_eligibility"] == 4
    assert row_counts["positive_purchase_transactions"] == 2
    sequential = result.audit["cleaning"][
        "positive_purchase_exclusion_reconciliation"
    ]
    assert sum(sequential.values()) == 5
    pd.testing.assert_frame_equal(sample_transactions, original)


def test_cleaning_is_deterministic(sample_transactions: pd.DataFrame) -> None:
    first = clean_transactions(sample_transactions)
    second = clean_transactions(sample_transactions)

    pd.testing.assert_frame_equal(
        first.cleaned_transactions, second.cleaned_transactions
    )
    pd.testing.assert_frame_equal(
        first.positive_purchase_transactions, second.positive_purchase_transactions
    )
    assert first.audit == second.audit


def test_no_valid_positive_purchases_returns_empty_consistent_output(
    sample_transactions: pd.DataFrame,
) -> None:
    no_purchases = sample_transactions.loc[
        sample_transactions["Quantity"] <= 0
    ].copy()

    result = clean_transactions(no_purchases)

    assert result.positive_purchase_transactions.empty
    assert list(result.positive_purchase_transactions.columns) == list(
        sample_transactions.columns
    )


def test_missing_quantity_or_price_is_audited_and_excluded() -> None:
    transactions = pd.DataFrame(
        [
            ["2001", "B1", "Missing quantity", None, "2011-02-01", 2.0, 10, "UK"],
            ["2002", "B2", "Missing price", 1, "2011-02-02", None, 11, "UK"],
        ],
        columns=REQUIRED_COLUMNS,
    )

    result = clean_transactions(transactions)
    exclusions = result.audit["cleaning"]["positive_purchase_exclusions"]

    assert len(result.cleaned_transactions) == 2
    assert result.positive_purchase_transactions.empty
    assert exclusions["missing_quantity_rows"] == 1
    assert exclusions["missing_unit_price_rows"] == 1
    assert exclusions["positive_purchase_excluded_unique_rows"] == 2


def test_audit_records_column_types_missing_reasons_and_eligibility() -> None:
    transactions = pd.DataFrame(
        [
            ["3001", "C1", None, 1, "bad-date", 2.0, None, "UK"],
            ["C3002", "C2", "cancelled", -1, "2011-02-02", 2.0, 20, "UK"],
            ["3003", "C3", "eligible", 1, "2011-02-03", 2.0, 21, "UK"],
        ],
        columns=REQUIRED_COLUMNS,
    )

    audit = audit_transactions(transactions)

    assert audit["columns"] == list(REQUIRED_COLUMNS)
    assert audit["data_types"]["InvoiceDate"] == "object"
    assert audit["invalid_date_value_count"] == 1
    assert audit["missing_date_value_count"] == 0
    assert audit["missing_customer_id_row_count"] == 1
    assert audit["missing_description_row_count"] == 1
    assert audit["positive_purchase_eligible_row_count"] == 1


def test_empty_frame_audit_is_json_serializable_and_zero_safe() -> None:
    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)

    audit = audit_transactions(empty)

    assert audit["input_row_count"] == 0
    assert audit["input_column_count"] == len(REQUIRED_COLUMNS)
    assert audit["date_coverage"] == {"start": None, "end": None}
    assert audit["positive_purchase_eligible_row_count"] == 0
    assert all(
        value["count"] == 0 and value["percentage"] == 0.0
        for value in audit["missing_values"].values()
    )
    json.dumps(audit, allow_nan=False)
    result = clean_transactions(empty)
    assert result.cleaned_transactions.empty
    assert result.positive_purchase_transactions.empty


def test_invalid_input_and_duplicate_column_names_raise_clear_errors(
    sample_transactions: pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        audit_transactions([1, 2, 3])

    duplicated_columns = sample_transactions.copy()
    duplicated_columns.columns = [
        "InvoiceNo",
        "InvoiceNo",
        *REQUIRED_COLUMNS[2:],
    ]
    with pytest.raises(ValueError, match="duplicate column"):
        clean_transactions(duplicated_columns)


def test_positive_purchase_exclusion_categories_have_sequential_reconciliation(
    sample_transactions: pd.DataFrame,
) -> None:
    result = clean_transactions(sample_transactions)
    cleaning = result.audit["cleaning"]
    row_counts = cleaning["row_counts"]
    reconciliation = cleaning["positive_purchase_exclusion_reconciliation"]

    assert row_counts["cleaned_transactions"] - row_counts[
        "positive_purchase_transactions"
    ] == sum(reconciliation.values())
    assert cleaning["positive_purchase_exclusion_counts_overlap"] is True


def test_csv_and_json_exports_round_trip(tmp_path, sample_transactions) -> None:
    result = clean_transactions(sample_transactions)

    outputs = write_cleaning_outputs(result, tmp_path / "processed")

    assert set(outputs) == {
        "cleaned_transactions",
        "positive_purchase_transactions",
        "audit",
    }
    cleaned_reloaded = pd.read_csv(
        outputs["cleaned_transactions"], parse_dates=["InvoiceDate"]
    )
    purchases_reloaded = pd.read_csv(
        outputs["positive_purchase_transactions"],
        parse_dates=["InvoiceDate"],
        dtype={"InvoiceNo": "string"},
    )
    audit_reloaded = json.loads(outputs["audit"].read_text(encoding="utf-8"))

    assert len(cleaned_reloaded) == len(result.cleaned_transactions)
    assert pd.api.types.is_datetime64_any_dtype(cleaned_reloaded["InvoiceDate"])
    assert purchases_reloaded["InvoiceNo"].tolist() == ["1001", "1006"]
    assert audit_reloaded == result.audit
