# Sales Forecasting Using Prophet

Six-month demand forecasts for school uniform product-size lines, and the order
quantities that follow from them. Forecasting uses [Prophet](https://facebook.github.io/prophet/)
(by Meta); ordering combines each forecast with current stock and the supplier's
lead time.

## Project layout

```
.
├── main.py                 # CLI entry point for the whole pipeline
├── config.py               # paths and parameters - one place to change them
├── app.py                  # Streamlit app: pick a product-size, see its forecast
├── sales_forecasting/      # the package
│   ├── data.py             # load, clean and aggregate the raw export
│   ├── forecast.py         # Prophet fitting, horizon and backtesting
│   ├── ordering.py         # lead-time demand, safety stock, order quantity
│   └── plots.py            # forecast figures
├── data/
│   ├── raw/                # sales_data.csv, inventory_data.csv (inputs)
│   └── processed/          # generated: aggregated, forecast, orders, metrics
├── notebooks/              # the original analysis notebooks
├── outputs/                # figures
└── tests/
```

## Pipeline

```
data/raw/sales_data.csv                 (wide: one column per month)
        │  clean  - drop the Grand Total row, blank months become 0 sales
        │  melt   - one row per Product / Size / Month
        ▼
data/processed/sales_aggregated.csv     (484 lines x 41 months)
        │  forecast - one Prophet model per product-size line
        ▼
data/processed/sales_forecast.csv       (6 months ahead, with intervals)
        │  order - join to data/raw/inventory_data.csv
        ▼
data/processed/order_suggestions.csv
```

## Setup

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # plus tests and the notebooks
```

Prophet pulls in `cmdstanpy`; on a fresh machine the first install takes a few
minutes.

## Running it

```bash
python main.py all          # clean, forecast and order in sequence
python main.py clean        # just rebuild sales_aggregated.csv
python main.py forecast     # fit 484 models (a few minutes)
python main.py order        # size the orders from an existing forecast
python main.py backtest     # measure forecast error on held-out months
```

Options: `--periods` (forecast horizon, default 6), `--holdout` (backtest
holdout, default 6), `--log-level`.

The interactive product/size explorer:

```bash
streamlit run app.py
```

Tests:

```bash
pytest                      # all
pytest -m "not slow"        # skip the one test that fits a real model
```

## How the order quantity is worked out

For each product-size line, with lead time converted from weeks to months
(`weeks / 4.345`):

| Step | Logic |
|------|-------|
| Split the lead time | `full_months = floor(lead_time_months)`, plus the leftover fraction |
| Demand over lead time | Forecast summed over the whole months, plus `fraction x` the next month |
| Safety stock | `Upper_Bound - Forecast` summed over that same span |
| Order quantity | `max(0, demand + safety stock - stock on hand)`, rounded |

The safety buffer comes from the width of Prophet's confidence interval: the
less certain the forecast, the more cover is ordered.

## Notebooks

| Notebook | Contents |
|----------|----------|
| `Data-Cleaning-and-EDA.ipynb` | Cleaning, aggregation, outlier check, EDA plots |
| `prophet_sales_forecast.ipynb` | Single product-size forecast with an ipywidgets picker |
| `sales_forecast_and_order_logic.ipynb` | Forecasts for all lines, plus the order logic |
| `Web_Scraping.ipynb` | School roll data from Education Counts NZ, as a possible future regressor |

## Sample outputs

![Output 1](outputs/output1.png)
![Output 2](outputs/output2.png)
![Output 3](outputs/output3.png)

## Context

This project was completed as part of an assessment. All rights reserved.
