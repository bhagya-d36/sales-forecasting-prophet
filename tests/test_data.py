"""Tests for cleaning and aggregating the raw sales export."""

import pandas as pd
import pytest

from sales_forecasting import data


@pytest.fixture
def raw_sales():
    """A miniature version of the wide sales export, including its total row."""
    return pd.DataFrame(
        {
            "Category": ["School A", "School A", "Grand Total"],
            "Product": ["Shirt", "Shirt", ""],
            "Code": ["S-100", "S-104", ""],
            "Size": ["100", "104", ""],
            "Jan-22": [5.0, None, 5.0],
            "Feb-22": [3.0, 2.0, 5.0],
        }
    )


def test_clean_drops_the_grand_total_row(raw_sales):
    cleaned = data.clean_sales(raw_sales)
    assert "Grand Total" not in cleaned["Category"].values
    assert len(cleaned) == 2


def test_clean_treats_blank_months_as_zero_sales(raw_sales):
    cleaned = data.clean_sales(raw_sales)
    assert cleaned["Jan-22"].isnull().sum() == 0
    assert cleaned.loc[cleaned["Code"] == "S-104", "Jan-22"].iloc[0] == 0


def test_aggregate_produces_one_row_per_product_size_month(raw_sales):
    aggregated = data.aggregate_sales(data.clean_sales(raw_sales))
    assert list(aggregated.columns) == ["Product", "Size", "Month", "Sales"]
    assert len(aggregated) == 4  # 2 product-size lines x 2 months
    assert not aggregated.duplicated(["Product", "Size", "Month"]).any()


def test_aggregate_parses_months_to_month_start_dates(raw_sales):
    aggregated = data.aggregate_sales(data.clean_sales(raw_sales))
    assert set(aggregated["Month"]) == {pd.Timestamp("2022-01-01"), pd.Timestamp("2022-02-01")}


def test_aggregate_sums_sales_for_a_product_size_line(raw_sales):
    aggregated = data.aggregate_sales(data.clean_sales(raw_sales))
    row = aggregated[(aggregated["Size"] == "100") & (aggregated["Month"] == "2022-01-01")]
    assert row["Sales"].iloc[0] == 5.0
