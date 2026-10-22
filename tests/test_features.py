"""Tests for customer features, repeat-purchase labels, and data splits."""

import json

import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_customer_modeling_dataset,
    run_feature_pipeline,
    split_customer_dataset,
)
from src.preprocessing import REQUIRED_COLUMNS

OBSERVATION_START = "2011-04-01"
OBSERVATION_END = "2011-06-30"
OUTCOME_START = "2011-07-01"
OUTCOME_END = "2011-08-31"


@pytest.fixture
def phase2_transactions() -> pd.DataFrame:
    """Return Phase 2-style data with boundaries, returns, cancels and both labels."""
    rows = []
    for customer_id in range(1, 13):
        rows.append(
            [
                f"{customer_id:04d}A",
                f"P{customer_id}",
                "Observation item",
                2,
                "2011-04-01 10:00:00",
                5.0,
                customer_id,
                "UK",
            ]
        )
    rows.extend(
        [
            ["0001B", "P1", "Boundary item 1", 1, "2011-06-30 23:59:00", 3.0, 1, "UK"],
            ["0001B", "P2", "Boundary item 2", 2, "2011-06-30 23:59:00", 4.0, 1, "UK"],
            ["C0001", "P1", "Cancellation", 2, "2011-05-01", 10.0, 1, "UK"],
            ["0001R", "P1", "Return", -1, "2011-05-02", 2.0, 1, "UK"],
            ["0001Z", "P1", "Zero quantity", 0, "2011-05-03", 2.0, 1, "UK"],
            ["0001P", "P1", "Zero price", 1, "2011-05-04", 0.0, 1, "UK"],
            ["0001D", "P1", "Bad date", 1, "not-a-date", 2.0, 1, "UK"],
            ["0001C", "P1", "Missing customer", 1, "2011-05-05", 2.0, None, "UK"],
            ["0001X", "P1", "No observation", 1, "2011-07-01", 2.0, 99, "UK"],
            ["0001E", "P1", "First outcome day", 1, "2011-07-01 00:00:00", 2.0, 1, "UK"],
            ["0002E", "P1", "Last outcome day", 1, "2011-08-31 23:59:59", 2.0, 2, "UK"],
            ["0003E", "P1", "Future repeat", 1, "2011-07-15", 2.0, 3, "UK"],
            ["0004E", "P1", "Future repeat", 1, "2011-07-15", 2.0, 4, "UK"],
            ["0005E", "P1", "Future repeat", 1, "2011-07-15", 2.0, 5, "UK"],
            ["0006E", "P1", "Future repeat", 1, "2011-07-15", 2.0, 6, "UK"],
            ["C0007E", "P1", "Cancelled future", 1, "2011-07-20", 2.0, 7, "UK"],
            ["0008R", "P1", "Return future", -1, "2011-07-20", 2.0, 8, "UK"],
            ["0009Z", "P1", "Zero future", 1, "2011-07-20", 0.0, 9, "UK"],
            ["0010L", "P1", "Outside outcome", 1, "2011-09-01", 2.0, 10, "UK"],
        ]
    )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _build(transactions, output_directory=None):
    return build_customer_modeling_dataset(
        transactions,
        output_directory=output_directory,
        observation_start=OBSERVATION_START,
        observation_end=OBSERVATION_END,
        outcome_start=OUTCOME_START,
        outcome_end=OUTCOME_END,
    )


def test_rfm_features_use_distinct_invoices_and_line_revenue(
    phase2_transactions: pd.DataFrame,
) -> None:
    dataset, _, _ = _build(phase2_transactions)
    customer = dataset.set_index("CustomerID").loc[1]

    assert customer["recency_days"] == 1
    assert customer["frequency"] == 2
    assert customer["monetary"] == pytest.approx(10 + 3 + 8)
    assert customer["distinct_products"] == 2
    assert customer["active_purchase_days"] == 2
    assert customer["average_invoice_value"] == pytest.approx((10 + 11) / 2)
    assert customer["average_items_per_invoice"] == pytest.approx((2 + 3) / 2)
    assert customer["customer_tenure_days"] == 90


def test_labels_use_distinct_qualifying_invoices_and_inclusive_boundaries(
    phase2_transactions: pd.DataFrame,
) -> None:
    dataset, _, metadata = _build(phase2_transactions)
    labels = dataset.set_index("CustomerID")[TARGET_COLUMN]

    assert labels.loc[1] == 1
    assert labels.loc[2] == 1
    assert labels.loc[3] == 1
    assert labels.loc[6] == 1
    assert labels.loc[7] == 0
    assert labels.loc[8] == 0
    assert labels.loc[9] == 0
    assert labels.loc[10] == 0
    assert 99 not in labels.index
    assert metadata["windows"]["observation"]["start_inclusive"] == OBSERVATION_START
    assert metadata["windows"]["outcome"]["end_inclusive"] == OUTCOME_END
    assert labels.isin([0, 1]).all()
    assert labels.notna().all()


def test_invalid_transactions_and_missing_customer_ids_do_not_label_customers(
    phase2_transactions: pd.DataFrame,
) -> None:
    dataset, _, metadata = _build(phase2_transactions)

    assert len(dataset) == 12
    assert metadata["purchase_eligibility"][
        "missing_customer_ids_excluded_from_customer_analysis"
    ] == 1
    assert set(dataset["CustomerID"]) == set(range(1, 13))
    assert dataset["repeat_purchase"].sum() == 6


def test_outcome_changes_do_not_change_observation_features(
    phase2_transactions: pd.DataFrame,
) -> None:
    original, _, _ = _build(phase2_transactions)
    changed_transactions = phase2_transactions.copy()
    changed_mask = changed_transactions["InvoiceNo"].isin(
        ["0001E", "0002E", "0003E", "0004E", "0005E", "0006E"]
    )
    changed_transactions.loc[changed_mask, "Quantity"] = 1000
    changed_transactions.loc[changed_mask, "UnitPrice"] = 500.0
    changed_transactions.loc[changed_mask, "StockCode"] = "FUTURE-ONLY"
    changed, _, _ = _build(changed_transactions)

    pd.testing.assert_frame_equal(
        original[["CustomerID", *FEATURE_COLUMNS]].sort_values("CustomerID"),
        changed[["CustomerID", *FEATURE_COLUMNS]].sort_values("CustomerID"),
        check_dtype=False,
    )


def test_features_exclude_identifiers_target_and_future_columns(
    phase2_transactions: pd.DataFrame,
) -> None:
    dataset, splits, metadata = _build(phase2_transactions)

    assert metadata["features"]["predictive_feature_columns"] == list(FEATURE_COLUMNS)
    assert "CustomerID" not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    for split in splits.values():
        assert list(split.columns) == list(dataset.columns)
        assert "InvoiceNo" not in split.columns
        assert "InvoiceDate" not in split.columns
        assert split[TARGET_COLUMN].notna().all()


def test_date_boundaries_are_inclusive_for_observation_and_outcome(
    phase2_transactions: pd.DataFrame,
) -> None:
    dataset, _, _ = _build(phase2_transactions)
    customer_features = dataset.set_index("CustomerID")

    assert customer_features.loc[1, "frequency"] == 2
    assert customer_features.loc[1, TARGET_COLUMN] == 1
    assert customer_features.loc[2, TARGET_COLUMN] == 1


def test_incomplete_outcome_coverage_and_invalid_windows_fail_clearly(
    phase2_transactions: pd.DataFrame,
) -> None:
    incomplete = phase2_transactions.loc[
        phase2_transactions["InvoiceDate"] != "2011-09-01"
    ].copy()
    incomplete["InvoiceDate"] = pd.to_datetime(
        incomplete["InvoiceDate"], errors="coerce"
    )
    incomplete = incomplete.loc[incomplete["InvoiceDate"] < "2011-08-31"]
    with pytest.raises(ValueError, match="does not fully contain"):
        _build(incomplete)

    with pytest.raises(ValueError, match="must start after"):
        build_customer_modeling_dataset(
            phase2_transactions,
            observation_start=OBSERVATION_START,
            observation_end=OUTCOME_START,
            outcome_start=OUTCOME_START,
            outcome_end=OUTCOME_END,
        )


def test_deterministic_splits_have_no_customer_overlap_and_consistent_schema(
    phase2_transactions: pd.DataFrame,
) -> None:
    dataset, first, metadata = _build(phase2_transactions)
    _, second, _ = _build(phase2_transactions)

    for name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first[name], second[name])
        assert list(first[name].columns) == list(dataset.columns)
    customer_sets = {
        name: set(split["CustomerID"]) for name, split in first.items()
    }
    assert not customer_sets["train"] & customer_sets["validation"]
    assert not customer_sets["train"] & customer_sets["test"]
    assert not customer_sets["validation"] & customer_sets["test"]
    assert set.union(*customer_sets.values()) == set(dataset["CustomerID"])
    assert sum(len(split) for split in first.values()) == len(dataset)
    assert metadata["splits"]["stratification_feasible"] is True
    assert metadata["splits"]["ratios"] == {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }


def test_stratification_fails_with_actionable_error_for_small_class() -> None:
    rows = []
    for customer_id, target in ((1, 0), (2, 0), (3, 1), (4, 1), (5, 1)):
        row = {"CustomerID": customer_id, TARGET_COLUMN: target}
        row.update({feature: float(customer_id) for feature in FEATURE_COLUMNS})
        rows.append(row)
    small_class = pd.DataFrame(
        rows, columns=["CustomerID", *FEATURE_COLUMNS, TARGET_COLUMN]
    )

    with pytest.raises(ValueError, match="at least 3 customers"):
        split_customer_dataset(small_class)


def test_exports_and_metadata_are_readable_and_reproducible(
    phase2_transactions: pd.DataFrame, tmp_path
) -> None:
    dataset, splits, metadata = _build(phase2_transactions, tmp_path / "processed")

    expected = {
        "customer_features.csv",
        "repeat_purchase_labels.csv",
        "customer_modeling_dataset.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "modeling_metadata.json",
    }
    assert expected == {path.name for path in (tmp_path / "processed").iterdir()}
    saved_dataset = pd.read_csv(tmp_path / "processed" / "customer_modeling_dataset.csv")
    saved_train = pd.read_csv(tmp_path / "processed" / "train.csv")
    saved_metadata = json.loads(
        (tmp_path / "processed" / "modeling_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(saved_dataset) == len(dataset)
    assert len(saved_train) == len(splits["train"])
    assert saved_metadata["features"]["predictive_feature_columns"] == list(
        FEATURE_COLUMNS
    )
    assert saved_metadata["counts"]["eligible_customer_count"] == len(dataset)
    assert saved_metadata["artifacts"]["metadata"] == "modeling_metadata.json"
    json.dumps(metadata, allow_nan=False)


def test_phase2_csv_pipeline_loads_and_exports(tmp_path, phase2_transactions) -> None:
    source_path = tmp_path / "cleaned_transactions.csv"
    phase2_transactions.to_csv(source_path, index=False)

    dataset, splits, metadata = run_feature_pipeline(
        source_path,
        tmp_path / "outputs",
        observation_start=OBSERVATION_START,
        observation_end=OBSERVATION_END,
        outcome_start=OUTCOME_START,
        outcome_end=OUTCOME_END,
    )

    assert len(dataset) == 12
    assert sum(len(split) for split in splits.values()) == 12
    assert metadata["source"]["phase2_cleaning_reused"] is True


def test_missing_or_empty_inputs_raise_clear_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        run_feature_pipeline(tmp_path / "missing.csv")

    empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="no valid dates"):
        _build(empty)

    with pytest.raises(ValueError, match="missing required columns"):
        _build(pd.DataFrame({"InvoiceNo": ["1"]}))


def test_missing_invoice_identifier_fails_before_invoice_level_counting(
    phase2_transactions: pd.DataFrame,
) -> None:
    invalid_invoice = phase2_transactions.copy()
    invalid_invoice.loc[0, "InvoiceNo"] = None

    with pytest.raises(ValueError, match="missing or blank InvoiceNo"):
        _build(invalid_invoice)
