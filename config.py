"""Project-wide paths and parameters.

Every tunable value used by the pipeline lives here so the CLI, the Streamlit app
and the tests all read the same numbers.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Paths ---------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_SALES_CSV = RAW_DIR / "sales_data.csv"
RAW_INVENTORY_CSV = RAW_DIR / "inventory_data.csv"
AGGREGATED_CSV = PROCESSED_DIR / "sales_aggregated.csv"
FORECAST_CSV = PROCESSED_DIR / "sales_forecast.csv"
ORDERS_CSV = PROCESSED_DIR / "order_suggestions.csv"
BACKTEST_CSV = PROCESSED_DIR / "backtest_metrics.csv"

# --- Forecasting ---------------------------------------------------------
FORECAST_PERIODS = 6
# Month-start: the history is month-start, so the future must be too.
FREQ = "MS"
PROPHET_KWARGS = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
}
# Months held out from the end of the history when measuring forecast skill.
BACKTEST_MONTHS = 6
# Sales cannot be negative, but Prophet's linear trend can predict below zero.
CLIP_AT_ZERO = True

# --- Ordering ------------------------------------------------------------
WEEKS_PER_MONTH = 4.345

# --- Logging -------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
