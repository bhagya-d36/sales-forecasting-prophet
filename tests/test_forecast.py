"""Tests for the forecast horizon, clipping and series preparation.

Only `test_fit_and_predict_returns_future_months` fits a real model; it is
marked slow so the fast tests can be run on their own with `-m "not slow"`.
"""

import pandas as pd
import pytest

import config
from sales_forecasting import forecast


def test_future_months_start_after_the_last_observed_month():
    """Regression test: a month-end frequency would repeat the last month."""
    months = forecast.future_months("2025-05-01", periods=6)
    assert list(months) == list(pd.date_range("2025-06-01", periods=6, freq="MS"))


def test_future_months_are_month_start_dates():
    months = forecast.future_months("2025-05-01", periods=6)
    assert all(month.day == 1 for month in months)


def test_future_months_roll_forward_from_a_month_end_date():
    months = forecast.future_months("2025-05-31", periods=2)
    assert list(months) == [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")]


def test_clip_at_zero_removes_negative_predictions():
    clipped = forecast.clip_at_zero(
        pd.DataFrame(
            {
                "ds": pd.date_range("2025-06-01", periods=2, freq="MS"),
                "yhat": [-3.0, 2.0],
                "yhat_lower": [-8.0, -1.0],
                "yhat_upper": [1.0, 5.0],
            }
        )
    )
    assert (clipped["yhat"] >= 0).all()
    assert (clipped["yhat_lower"] >= 0).all()
    assert clipped["yhat"].tolist() == [0.0, 2.0]


def test_prepare_series_filters_to_one_line_and_sorts_by_month():
    df = pd.DataFrame(
        {
            "Product": ["Shirt", "Shirt", "Hat"],
            "Size": ["100", "100", "100"],
            "Month": pd.to_datetime(["2022-02-01", "2022-01-01", "2022-01-01"]),
            "Sales": [2.0, 1.0, 9.0],
        }
    )
    series = forecast.prepare_series(df, "Shirt", "100")
    assert list(series.columns) == ["ds", "y"]
    assert series["y"].tolist() == [1.0, 2.0]


@pytest.mark.slow
def test_fit_and_predict_returns_future_months_after_the_history():
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2022-01-01", periods=36, freq="MS"),
            "y": [float(i % 12) for i in range(36)],
        }
    )
    result = forecast.fit_and_predict(history, periods=config.FORECAST_PERIODS)

    future = result[result["ds"] > history["ds"].max()]
    assert len(future) == config.FORECAST_PERIODS
    assert (future["yhat"] >= 0).all()
    assert future["ds"].iloc[0] == pd.Timestamp("2025-01-01")
