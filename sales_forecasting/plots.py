"""Forecast figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def forecast_figure(
    history: pd.DataFrame, forecast: pd.DataFrame, product: str, size
) -> Figure:
    """Plot observed sales against the forecast and its confidence band.

    `history` holds ds/y rows; `forecast` holds the future months in the
    Month / Forecasted_Sales / Lower_Bound / Upper_Bound shape.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        history["ds"], history["y"], marker="o", markersize=3,
        color="grey", label="Actual",
    )
    ax.plot(
        forecast["Month"], forecast["Forecasted_Sales"], marker="o",
        color="steelblue", label="Forecast",
    )
    ax.fill_between(
        forecast["Month"],
        forecast["Lower_Bound"],
        forecast["Upper_Bound"],
        color="steelblue",
        alpha=0.25,
        label="Confidence interval",
    )
    ax.set_title(f"Forecast - {product} ({size})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
