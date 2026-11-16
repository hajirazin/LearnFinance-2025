# V1 close-only lookback shadow comparison

Compare the saved 60-session research refit (`current.pt`) with the saved H2 one-year checkpoint (`h2_year.pt`). Both predict the next five trading sessions using adjusted-close log returns. The one-year model takes the sessions within 365 calendar days, padded to 255 inputs as in H2. The files record both checkpoint hashes and the adjusted-close and volume snapshot hash. This is a scratch forecast comparison using the frozen 428-symbol research roster and simulated selections.

Each week, apply 15/20 rank stickiness to the stocks eligible for both models. Measure forecast error on the same 15 equal-weight names selected by the 60-session model. Also score the per-stock forecasts against the same observed five-session outcomes. Include a zero-return forecast and dividend-adjusted SPY as references. Own-basket returns are adjusted-close marks before costs.

## November 6–13, 2026 replay

This is a retrospective replay after the archived October 30 decision cutoff. Model inputs stop at the November 6 close. The adjusted-price snapshot was retrieved after the outcome, so the week does not count as a prospectively frozen forecast.

| Metric | 60 sessions | One year | Zero forecast |
| --- | ---: | ---: | ---: |
| Forecast for the same 15 names | +3.09% | +1.34% | 0.00% |
| Absolute error on the same 15 names | 2.19 pp | 0.44 pp | 0.90 pp |
| Individual-stock MAE over 422 outcomes | 4.23 pp | 3.57 pp | 3.96 pp |

The common basket gained **0.90%** and SPY gained **2.59%**. The one-year model's own 15 picks gained **9.54%** before costs. The one-year model had lower forecast error than both references in this single replay. [Settlement](data/v1_lookback_shadow/2026-11-06/settlement.json) · [Per-stock forecasts and outcomes](data/v1_lookback_shadow/2026-11-06/stock_outcomes.csv)

## November 13–20, 2026 prospective forecast

Both forecasts were frozen on **November 15 at 4:11 p.m. Singapore time**, before the next US open. There were 422 common candidates and the one-year arm used 251 observed sessions. On the same fixed basket, the 60-session model predicted **+2.79%** and the one-year model predicted **+1.51%**. The outcome is due at the November 20 New York close, **November 21 at 5:00 a.m. Singapore time**.

[Frozen manifest and basket names](data/v1_lookback_shadow/2026-11-13/forecast.json) · [Frozen forecasts for all 422 stocks](data/v1_lookback_shadow/2026-11-13/stock_forecasts.csv)

## Score the frozen forecast

After the November 20 close, run these commands from the repository root with the PatchTST scratch dependencies installed. The settlement command verifies the frozen forecast file and writes the observed results into a separate file for the November 13 decision.

```sh
export PYTHONPATH=brain_api:scratch/AlphaAllocationV2
python scratch/AlphaAllocationV1/fetch_v1_lookback_prices.py --start 2025-10-20 --through 2026-11-20
python scratch/AlphaAllocationV1/run_v1_lookback_shadow.py settle --date 2026-11-13 \
  --prices scratch/AlphaAllocationV1/data/v1_lookback_shadow/prices/2025-10-20_through_2026-11-20.parquet
```

For subsequent decisions, fetch a snapshot with at least 365 days of warmup before the decision, then use `run_v1_lookback_shadow.py freeze --date <trading-Friday> --prices <snapshot-path>` before the next session opens. Each target is the fifth trading session after the decision. The runner refuses to overwrite a frozen decision or to settle before the target close. Treat the one replay as descriptive and evaluate prospective weeks separately when estimating paired forecast improvements.
