"""Streamlit app: pick a product and size, see its forecast and suggested order.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from sales_forecasting import data, forecast, ordering, plots

st.set_page_config(page_title="Sales Forecasting", layout="wide")


@st.cache_data
def load_sales() -> pd.DataFrame:
    return data.load_aggregated()


@st.cache_data
def load_inventory() -> pd.DataFrame:
    inventory = data.load_inventory()
    inventory["Size"] = inventory["Size"].astype(str)
    return inventory


@st.cache_data(show_spinner="Fitting Prophet...")
def run_forecast(product: str, size: str, periods: int) -> pd.DataFrame:
    return forecast.forecast_one(load_sales(), product, size, periods)


@st.cache_data(show_spinner="Backtesting...")
def run_backtest(product: str, size: str, holdout: int) -> dict | None:
    series = forecast.prepare_series(load_sales(), product, size)
    return forecast.backtest_one(series, holdout)


sales = load_sales()
inventory = load_inventory()

st.title("Sales Forecasting")
st.caption(
    f"Prophet forecasts per product-size line, from "
    f"{sales['Month'].min():%b %Y} to {sales['Month'].max():%b %Y} of history."
)

with st.sidebar:
    st.header("Selection")
    product = st.selectbox("Product", sorted(sales["Product"].unique()))
    sizes = sorted(sales[sales["Product"] == product]["Size"].astype(str).unique())
    size = st.selectbox("Size", sizes)
    periods = st.slider("Months to forecast", 1, 12, config.FORECAST_PERIODS)
    holdout = st.slider("Backtest holdout (months)", 3, 12, config.BACKTEST_MONTHS)

forecast_df = run_forecast(product, size, periods)
history = forecast.prepare_series(sales, product, size)

st.subheader(f"{product} - size {size}")
st.pyplot(plots.forecast_figure(history, forecast_df, product, size))

left, right = st.columns([3, 2])

with left:
    st.markdown("**Forecast**")
    st.dataframe(
        forecast_df.assign(Month=forecast_df["Month"].dt.strftime("%b %Y"))[
            ["Month", "Forecasted_Sales", "Lower_Bound", "Upper_Bound"]
        ].round(2),
        hide_index=True,
        width="stretch",
    )

with right:
    st.markdown("**Suggested order**")
    stock_row = inventory[
        (inventory["Product"] == product) & (inventory["Size"] == size)
    ]
    if stock_row.empty:
        st.info("No inventory record for this combination, so no order can be sized.")
    else:
        suggestion = ordering.order_for_series(
            forecast_df,
            stock_on_hand=stock_row["Stock On Hand"].iloc[0],
            lead_time_weeks=stock_row["Lead Time (Weeks)"].iloc[0],
        )
        st.metric("Order quantity", suggestion["Order Quantity"])
        st.write(
            pd.Series(
                {
                    "Lead time (weeks)": stock_row["Lead Time (Weeks)"].iloc[0],
                    "Stock on hand": suggestion["Stock On Hand"],
                    "Demand over lead time": round(suggestion["Lead Time Sales"], 2),
                    "Safety stock": round(suggestion["Safety Stock"], 2),
                }
            ).to_frame("Value")
        )

    st.markdown("**Forecast error on held-out months**")
    scores = run_backtest(product, size, holdout)
    if scores is None:
        st.info("Not enough history to hold months out.")
    else:
        st.write(
            pd.Series(
                {
                    "MAE": round(scores["MAE"], 2),
                    "RMSE": round(scores["RMSE"], 2),
                    "Seasonal-naive MAE": round(scores["Baseline_MAE"], 2),
                    "Mean actual sales": round(scores["Holdout_Mean_Sales"], 2),
                }
            ).to_frame("Value")
        )
        if scores["MAE"] > scores["Baseline_MAE"]:
            st.warning(
                "Prophet is doing worse than repeating last year's same month "
                "for this series."
            )
