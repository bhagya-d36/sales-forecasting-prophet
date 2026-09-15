"""Command line entry point for the sales forecasting pipeline.

    python main.py clean      # raw sales      -> data/processed/sales_aggregated.csv
    python main.py forecast   # aggregated     -> data/processed/sales_forecast.csv
    python main.py order      # forecast+stock -> data/processed/order_suggestions.csv
    python main.py backtest   # aggregated     -> data/processed/backtest_metrics.csv
    python main.py all        # clean, forecast and order in one go
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import config
from sales_forecasting import data

logger = logging.getLogger("main")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level, format=config.LOG_FORMAT, stream=sys.stderr
    )


def _write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s rows to %s", len(df), path)


def run_clean(args: argparse.Namespace) -> pd.DataFrame:
    return data.build_aggregated_sales()


def run_forecast(args: argparse.Namespace) -> pd.DataFrame:
    from sales_forecasting import forecast

    aggregated = data.load_aggregated()
    forecast_df = forecast.forecast_all(aggregated, periods=args.periods)
    _write(forecast_df, config.FORECAST_CSV)
    return forecast_df


def run_order(args: argparse.Namespace) -> pd.DataFrame:
    from sales_forecasting import ordering

    forecast_df = pd.read_csv(config.FORECAST_CSV, parse_dates=["Month"])
    orders = ordering.build_order_suggestions(forecast_df, data.load_inventory())
    _write(orders, config.ORDERS_CSV)
    return orders


def run_backtest(args: argparse.Namespace) -> pd.DataFrame:
    from sales_forecasting import forecast

    metrics = forecast.backtest_all(data.load_aggregated(), args.holdout)
    _write(metrics, config.BACKTEST_CSV)
    logger.info(
        "Mean MAE %.2f against a seasonal-naive baseline of %.2f",
        metrics["MAE"].mean(),
        metrics["Baseline_MAE"].mean(),
    )
    return metrics


def run_all(args: argparse.Namespace) -> None:
    run_clean(args)
    run_forecast(args)
    run_order(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log-level", default=config.LOG_LEVEL, help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=config.FORECAST_PERIODS,
        help=f"Months to forecast (default: {config.FORECAST_PERIODS})",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=config.BACKTEST_MONTHS,
        help=f"Months held out when backtesting (default: {config.BACKTEST_MONTHS})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in [
        ("clean", run_clean, "Clean and aggregate the raw sales export"),
        ("forecast", run_forecast, "Forecast every product-size combination"),
        ("order", run_order, "Suggest order quantities from forecasts and stock"),
        ("backtest", run_backtest, "Measure forecast error on held-out months"),
        ("all", run_all, "Run clean, forecast and order in sequence"),
    ]:
        subparsers.add_parser(name, help=help_text).set_defaults(handler=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
