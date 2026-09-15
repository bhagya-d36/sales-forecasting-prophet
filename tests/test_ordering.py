"""Tests for the lead-time demand and order quantity maths."""

import pandas as pd
import pytest

import config
from sales_forecasting import ordering


@pytest.fixture
def forecast():
    """Six forecast months with a deliberately wide interval in month two."""
    return pd.DataFrame(
        {
            "Month": pd.date_range("2025-06-01", periods=6, freq="MS"),
            "Forecasted_Sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Upper_Bound": [12.0, 26.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_lead_time_converts_weeks_to_months():
    assert ordering.lead_time_in_months(config.WEEKS_PER_MONTH) == pytest.approx(1.0)


def test_order_counts_whole_and_partial_lead_time_months(forecast):
    # 5 weeks is 1.15 months: all of June plus 15% of July.
    fraction = 5 / config.WEEKS_PER_MONTH - 1
    result = ordering.order_for_series(forecast, stock_on_hand=0, lead_time_weeks=5)
    assert result["Lead Time Sales"] == pytest.approx(10.0 + 20.0 * fraction)


def test_safety_stock_uses_the_same_partial_month_as_demand(forecast):
    # The buffer is Upper_Bound - Forecasted_Sales, weighted over the same span.
    fraction = 5 / config.WEEKS_PER_MONTH - 1
    result = ordering.order_for_series(forecast, stock_on_hand=0, lead_time_weeks=5)
    assert result["Safety Stock"] == pytest.approx(2.0 + 6.0 * fraction)


def test_order_quantity_subtracts_stock_on_hand(forecast):
    with_stock = ordering.order_for_series(forecast, stock_on_hand=5, lead_time_weeks=5)
    without_stock = ordering.order_for_series(forecast, stock_on_hand=0, lead_time_weeks=5)
    assert without_stock["Order Quantity"] - with_stock["Order Quantity"] == 5


def test_order_quantity_is_zero_when_stock_covers_demand(forecast):
    result = ordering.order_for_series(forecast, stock_on_hand=10_000, lead_time_weeks=5)
    assert result["Order Quantity"] == 0


def test_build_suggestions_skips_combinations_with_no_inventory(forecast):
    forecast_df = pd.concat(
        [
            forecast.assign(Product="Shirt", Size="100", Lower_Bound=0.0),
            forecast.assign(Product="Ghost", Size="999", Lower_Bound=0.0),
        ]
    )
    inventory = pd.DataFrame(
        {
            "Product": ["Shirt"],
            "Size": ["100"],
            "Stock On Hand": [5],
            "Lead Time (Weeks)": [5],
        }
    )
    orders = ordering.build_order_suggestions(forecast_df, inventory)
    assert list(orders["Product"]) == ["Shirt"]
