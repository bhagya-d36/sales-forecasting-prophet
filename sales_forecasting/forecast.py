"""Prophet forecasting for each product-size series.

One model is fitted per product-size combination. The series are short (41
months) and sparse, so `backtest` reports forecast error on a held-out tail
alongside a seasonal-naive baseline - an error figure on a mostly-zero series
means little on its own.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
from prophet import Prophet

import config

logger = logging.getLogger(__name__)

FORECAST_COLUMNS = [
    "Month",
    "Product",
    "Size",
    "Forecasted_Sales",
    "Lower_Bound",
    "Upper_Bound",
]
_RENAMES = {
    "ds": "Month",
    "yhat": "Forecasted_Sales",
    "yhat_lower": "Lower_Bound",
    "yhat_upper": "Upper_Bound",
}
_BOUND_COLUMNS = ["yhat", "yhat_lower", "yhat_upper"]


@contextmanager
def _quiet_prophet():
    """Silence the per-fit chatter from Prophet and cmdstanpy."""
    noisy = [logging.getLogger(name) for name in ("prophet", "cmdstanpy")]
    previous = [(log, log.level, log.disabled) for log in noisy]
    for log in noisy:
        log.setLevel(logging.CRITICAL)
        log.disabled = True
    # cmdstanpy writes progress straight to the console on some platforms.
    os.environ.setdefault("CMDSTANPY_LOG_LEVEL", "CRITICAL")
    try:
        yield
    finally:
        for log, level, disabled in previous:
            log.setLevel(level)
            log.disabled = disabled


def future_months(last_date, periods: int = config.FORECAST_PERIODS) -> pd.DatetimeIndex:
    """Month-start dates strictly after `last_date`.

    Prophet's own `make_future_dataframe(freq="ME")` would return month-*end*
    dates, the first of which still falls inside the last observed month - so
    a 6-period horizon would deliver 5 new months and a duplicate.
    """
    # MonthBegin(1) always rolls strictly forward, from a month-start or a month-end.
    start = pd.Timestamp(last_date).normalize() + pd.offsets.MonthBegin(1)
    return pd.date_range(start=start, periods=periods, freq=config.FREQ)


def prepare_series(df: pd.DataFrame, product: str, size) -> pd.DataFrame:
    """Pull one product-size series out of the aggregated frame, Prophet-shaped."""
    subset = df[(df["Product"] == product) & (df["Size"] == size)][["Month", "Sales"]]
    subset = subset.rename(columns={"Month": "ds", "Sales": "y"})
    return subset.sort_values("ds").reset_index(drop=True)


def clip_at_zero(forecast: pd.DataFrame) -> pd.DataFrame:
    """Floor the prediction and its interval at zero - sales are never negative."""
    if not config.CLIP_AT_ZERO:
        return forecast
    forecast = forecast.copy()
    for column in _BOUND_COLUMNS:
        if column in forecast:
            forecast[column] = forecast[column].clip(lower=0)
    return forecast


def fit_and_predict(
    series: pd.DataFrame, periods: int = config.FORECAST_PERIODS
) -> pd.DataFrame:
    """Fit Prophet on one series and predict `periods` months past its end.

    Returns history and future rows with ds/yhat/yhat_lower/yhat_upper.
    """
    with _quiet_prophet():
        model = Prophet(**config.PROPHET_KWARGS)
        model.fit(series)
        future = pd.DataFrame(
            {"ds": series["ds"].tolist() + list(future_months(series["ds"].max(), periods))}
        )
        forecast = model.predict(future)
    return clip_at_zero(forecast[["ds"] + _BOUND_COLUMNS])


def forecast_one(
    df: pd.DataFrame, product: str, size, periods: int = config.FORECAST_PERIODS
) -> pd.DataFrame:
    """Forecast a single product-size combination, future months only."""
    forecast = fit_and_predict(prepare_series(df, product, size), periods)
    future = forecast.tail(periods).rename(columns=_RENAMES)
    future["Product"] = product
    future["Size"] = size
    return future[FORECAST_COLUMNS]


def forecast_all(
    df: pd.DataFrame, periods: int = config.FORECAST_PERIODS
) -> pd.DataFrame:
    """Forecast every product-size combination in the frame."""
    combinations = df[["Product", "Size"]].drop_duplicates()
    total = len(combinations)
    logger.info("Forecasting %s product-size combinations %s months ahead", total, periods)

    forecasts = []
    for position, (_, row) in enumerate(combinations.iterrows(), start=1):
        forecasts.append(forecast_one(df, row["Product"], row["Size"], periods))
        if position % 50 == 0 or position == total:
            logger.info("  fitted %s/%s", position, total)

    return pd.concat(forecasts, ignore_index=True)


def seasonal_naive(series: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Baseline prediction: the same calendar month one year earlier."""
    lookup = series.set_index("ds")["y"]
    return np.array(
        [lookup.get(month - pd.DateOffset(years=1), 0.0) for month in holdout["ds"]]
    )


def backtest_one(
    series: pd.DataFrame, holdout_months: int = config.BACKTEST_MONTHS
) -> dict | None:
    """Score a forecast against the held-out tail of its own history.

    Returns None when the series is too short to hold anything out.
    """
    if len(series) <= holdout_months + 1:
        return None

    train = series.iloc[:-holdout_months]
    holdout = series.iloc[-holdout_months:]

    predicted = fit_and_predict(train, holdout_months).tail(holdout_months)
    errors = holdout["y"].to_numpy() - predicted["yhat"].to_numpy()
    baseline_errors = holdout["y"].to_numpy() - seasonal_naive(train, holdout)

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Baseline_MAE": float(np.mean(np.abs(baseline_errors))),
        "Holdout_Mean_Sales": float(holdout["y"].mean()),
    }


def backtest_all(
    df: pd.DataFrame, holdout_months: int = config.BACKTEST_MONTHS
) -> pd.DataFrame:
    """Backtest every product-size combination."""
    combinations = df[["Product", "Size"]].drop_duplicates()
    total = len(combinations)
    logger.info("Backtesting %s combinations on the last %s months", total, holdout_months)

    results = []
    for position, (_, row) in enumerate(combinations.iterrows(), start=1):
        series = prepare_series(df, row["Product"], row["Size"])
        scores = backtest_one(series, holdout_months)
        if scores is None:
            logger.warning(
                "Skipping %s (%s): only %s months of history",
                row["Product"],
                row["Size"],
                len(series),
            )
            continue
        results.append({"Product": row["Product"], "Size": row["Size"], **scores})
        if position % 50 == 0 or position == total:
            logger.info("  scored %s/%s", position, total)

    return pd.DataFrame(results)
