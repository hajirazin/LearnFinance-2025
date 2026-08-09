# PatchTST era walk-forward (scratch)

Artifacts live only under `scratch/patchtst_era_walkforward_2026/` — **not** `data/models/`.

## Setup
- Symbols: AAPL, MSFT, GOOGL, JNJ
- Train end: **2025-12-31**
- Models: `close_only` (Phase A denorm close MSE) vs `multitask` (`outputs.loss`, all 5 OHLCV like 337a8af)
- Eval: every 2026 ISO week with 5 sessions available in data (through 2026-08-07) = **25 weeks × 4 symbols = 100 rows**

## Does week 2 see week 1 actuals?
**Yes.** Context = last 60 trading days of **actual** OHLCV log returns with `index < week_start`.  
So for week starting 2026-01-12, context ends 2026-01-09 (week 1 actuals included). Predictions are **not** fed forward — only market actuals.

## Results (MAE of weekly close return %)
| Model | MAE (pp) | Direction accuracy |
|-------|----------|--------------------|
| close_only | **3.88** | 0.45 |
| multitask | 5.46 | 0.41 |

Winner on MAE: **close_only**.

## Files
- `run_walkforward.py` — train + eval
- `models/close_only/`, `models/multitask/`
- `results/summary.json`, `weekly_rows.csv`, `weekly_rows.json`
- `run.log`
