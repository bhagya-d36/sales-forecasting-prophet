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

## Known limitations

- **Prophet loses to a naive baseline on most lines.** Holding out the last six
  months (Dec 2024 - May 2025) and comparing against seasonal-naive - simply
  repeating the same month a year earlier - Prophet wins on only **113 of 484
  lines (23%)**. Mean absolute error is **3.85 units against the baseline's
  1.89**, and the gap is widest on the busiest lines (7.84 vs 3.59 on the 203
  lines selling 2+ units a month). Reproduce with `python main.py backtest`.
  The forecasts remain a reasonable starting point for a buyer, but should not
  be trusted over a human read of last year's numbers without further work.
  This was invisible before, because the metrics were scored on the same rows
  the model was fitted to.
- **The series are sparse, which is the likely cause.** Across 41 months, the
  median product-size line has only 11 months with any sales at all, and one
  line has a single non-zero month. Prophet is fitted to every line regardless,
  so yearly seasonality on the thinnest lines is fitting noise. The obvious next
  step is a fallback - seasonal-naive or a simple average - for lines below a
  volume threshold, with Prophet reserved for the lines that earn it.
- **The backtest is a single origin.** One six-month holdout, not a rolling
  evaluation. Enough to show the problem above, not enough to tune a model on.
- **Order quantities are rounded, not rounded up.** A buyer who never wants to
  under-order would use `ceil` instead; this keeps the original behaviour.
- **106 history rows are negative** (returns or corrections). They are left in
  the history as-is; only the forecast is floored at zero.
- **The notebooks duplicate the logic.** They are kept as the original analysis
  narrative, with their stored outputs, and their paths updated for the new
  folders. Those stored outputs predate the fixes listed below, so re-running a
  notebook will not reproduce the numbers it currently displays. `main.py` is
  the source of truth.

## Fixes applied during the restructure

1. **Forecast dates were off by a month.** The history is month-*start*, but
   `make_future_dataframe(freq='ME')` returns month-*end* dates, the first of
   which still falls inside the last observed month. A "6-month forecast" was
   really 5 new months plus a duplicate. Now month-start throughout.
2. **Metrics were in-sample.** R²/RMSE/MAE were scored on the same rows the
   model was fitted to, which measures fit, not forecast skill. Replaced with a
   holdout backtest against a seasonal-naive baseline.
3. **Forecasts could go negative.** Prophet's linear trend predicts below zero
   on sparse lines, and those values fed straight into the order maths.
   Predictions and their intervals are now floored at zero.
4. **Safety stock ignored the part month.** Demand over the lead time counted
   the fractional month, but the safety buffer did not. Both now use the same
   span.

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
