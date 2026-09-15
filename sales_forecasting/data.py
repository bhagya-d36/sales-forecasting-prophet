"""Loading, cleaning and aggregating the raw sales and inventory files.

The raw sales file is wide: one row per product-size, one column per month
("Jan-22" ... "May-25"). Forecasting needs it long, so `aggregate_sales` melts
it into one row per Product / Size / Month.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

ID_COLUMNS = ["Category", "Product", "Code", "Size"]
MONTH_FORMAT = "%b-%y"
GRAND_TOTAL = "Grand Total"


def load_raw_sales(path: Path = config.RAW_SALES_CSV) -> pd.DataFrame:
    """Read the wide monthly sales export."""
    df = pd.read_csv(path)
    logger.info("Loaded %s rows of raw sales from %s", len(df), path.name)
    return df


def load_inventory(path: Path = config.RAW_INVENTORY_CSV) -> pd.DataFrame:
    """Read the inventory file (stock on hand and supplier lead time)."""
    df = pd.read_csv(path)
    logger.info("Loaded %s inventory rows from %s", len(df), path.name)
    return df


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the spreadsheet's summary row and treat blank months as zero sales."""
    before = len(df)
    df = df[df["Category"] != GRAND_TOTAL]
    if before - len(df):
        logger.info("Dropped %s '%s' row(s)", before - len(df), GRAND_TOTAL)

    missing = int(df.isnull().sum().sum())
    if missing:
        logger.info("Filling %s blank month cells with 0 sales", missing)
    return df.fillna(0)


def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide sales frame into monthly Product / Size / Sales rows."""
    long_df = df.melt(
        id_vars=ID_COLUMNS,
        var_name="Month",
        value_name="Sales",
    )
    long_df["Month"] = pd.to_datetime(long_df["Month"], format=MONTH_FORMAT)

    aggregated = (
        long_df.groupby(["Product", "Size", "Month"])["Sales"].sum().reset_index()
    )
    logger.info(
        "Aggregated to %s rows across %s product-size combinations",
        len(aggregated),
        len(aggregated[["Product", "Size"]].drop_duplicates()),
    )
    return aggregated


def load_aggregated(path: Path = config.AGGREGATED_CSV) -> pd.DataFrame:
    """Read a previously written aggregated file, with Month already parsed."""
    return pd.read_csv(path, parse_dates=["Month"])


def build_aggregated_sales(
    raw_path: Path = config.RAW_SALES_CSV,
    output_path: Path | None = config.AGGREGATED_CSV,
) -> pd.DataFrame:
    """Run the full clean-and-aggregate step, optionally writing the result."""
    aggregated = aggregate_sales(clean_sales(load_raw_sales(raw_path)))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_path, index=False)
        logger.info("Wrote %s", output_path)
    return aggregated
