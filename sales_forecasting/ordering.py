"""Turning forecasts plus current stock into order quantities.

For each product-size line: work out how much will sell during the supplier's
lead time, add a safety buffer taken from the width of the forecast interval,
then subtract what is already on the shelf.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

import config

logger = logging.getLogger(__name__)

ORDER_COLUMNS = [
    "Product",
    "Size",
    "Lead Time Sales",
    "Safety Stock",
    "Stock On Hand",
    "Order Quantity",
]


def lead_time_in_months(weeks: float) -> float:
    """Convert a supplier lead time from weeks to months."""
    return weeks / config.WEEKS_PER_MONTH


def _weighted_sum(values: pd.Series, lead_time_months: float) -> float:
    """Sum `values` over the whole months of the lead time, plus the part month.

    A 5-week lead time is 1.15 months: one whole month of demand plus 15% of
    the next month's.
    """
    full_months = math.floor(lead_time_months)
    fraction = lead_time_months - full_months

    total = float(values.iloc[:full_months].sum()) if full_months > 0 else 0.0
    if fraction > 0 and full_months < len(values):
        total += float(values.iloc[full_months]) * fraction
    return total


def order_for_series(
    forecast: pd.DataFrame, stock_on_hand: float, lead_time_weeks: float
) -> dict:
    """Order components for one product-size line.

    `forecast` must hold the future months for a single combination, with
    Forecasted_Sales / Upper_Bound columns.
    """
    forecast = forecast.sort_values("Month")
    lead_time_months = lead_time_in_months(lead_time_weeks)

    if lead_time_months > len(forecast):
        logger.warning(
            "Lead time of %.2f months exceeds the %s-month forecast horizon; "
            "demand beyond the horizon is not counted",
            lead_time_months,
            len(forecast),
        )

    sales = _weighted_sum(forecast["Forecasted_Sales"], lead_time_months)
    # The buffer is the upside of the forecast interval, weighted over the same
    # span as the demand it is protecting.
    buffer = forecast["Upper_Bound"] - forecast["Forecasted_Sales"]
    safety_stock = _weighted_sum(buffer, lead_time_months)

    return {
        "Lead Time Sales": sales,
        "Safety Stock": safety_stock,
        "Stock On Hand": stock_on_hand,
        "Order Quantity": round(max(0.0, sales + safety_stock - stock_on_hand)),
    }


def build_order_suggestions(
    forecast_df: pd.DataFrame, inventory_df: pd.DataFrame
) -> pd.DataFrame:
    """Join forecasts to inventory and size an order for every combination."""
    inventory = inventory_df.copy()
    inventory["Size"] = inventory["Size"].astype(str)
    forecast_df = forecast_df.copy()
    forecast_df["Size"] = forecast_df["Size"].astype(str)

    merged = forecast_df.merge(
        inventory[["Product", "Size", "Stock On Hand", "Lead Time (Weeks)"]],
        on=["Product", "Size"],
        how="left",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"][["Product", "Size"]].drop_duplicates()
    if len(unmatched):
        logger.warning(
            "%s product-size combination(s) have no inventory record and are "
            "skipped: %s",
            len(unmatched),
            ", ".join(f"{p} ({s})" for p, s in unmatched.itertuples(index=False)),
        )
        merged = merged[merged["_merge"] == "both"]

    suggestions = []
    for (product, size), group in merged.groupby(["Product", "Size"]):
        suggestions.append(
            {
                "Product": product,
                "Size": size,
                **order_for_series(
                    group,
                    stock_on_hand=group["Stock On Hand"].iloc[0],
                    lead_time_weeks=group["Lead Time (Weeks)"].iloc[0],
                ),
            }
        )

    orders = pd.DataFrame(suggestions, columns=ORDER_COLUMNS)
    logger.info(
        "Sized orders for %s combinations; %s need restocking (%s units total)",
        len(orders),
        int((orders["Order Quantity"] > 0).sum()),
        int(orders["Order Quantity"].sum()),
    )
    return orders
