# H26 and H27 seven-target PatchTST

Test whether supervising the close forecast alongside six other market series improves after-cost HRP returns over the saved V1 specification. The primary null for each arm is a nonpositive mean paired weekly net return difference versus V1. Use a one-sided circular block bootstrap with four-period blocks, 20,000 replicates and seed 42. Normalize longer holding periods to seven days. Apply Holm correction across H26 and H27 at 5%. Use H5's seven-input daily close-only forecast and H21's seven-input weekly close-loss model as additional reference arms.

Both new arms use the H5 encoder, 60 daily sessions, the same seven input channels, and the 356-stock fitting roster. A joint linear head forecasts all seven series. The close channel forecasts adjusted log returns. The other targets are the future observed 20-session momentum, 12–1 momentum, 20-session annualized volatility, 20-session relative volume, seven-calendar-day news sentiment, and log-one-plus reported volume. The underlying price, news, and coverage formulas come from `v1_hypothesis_data.py`. Verified empty news windows use neutral sentiment. Unobserved targets are excluded.

- **H26** forecasts one value per channel per session for the next five trading sessions. For ranking, undo close-target standardization and compound the five close returns.
- **H27** forecasts the Friday-close-to-next-Friday-close log return and the six auxiliary values as observed at that next Friday's close. Short holiday weeks enter fitting when both Fridays are trading days. For ranking, undo close-target standardization and convert the one log return to a simple return.

Minimize the equally weighted mean squared error across the seven channels and forecast steps after z-scoring targets using the 2021–2024 fitting rows only. Use seed 42, 256 rows per batch, a 60-epoch cap and eight-epoch patience. Both checkpoints use the same broader 2025 validation rows with complete seven-target labels and exact five-session Friday-to-Friday outcomes. Select by weekly Spearman rank correlation and use the standardized all-target validation MSE to break ties. H26 and H27 use each horizon's available fitting rows, so the comparison includes differences in training coverage.

Score the same 2026 candidates and dates as the H21–H23 replay. Save actual and predicted Friday ranks for all stocks. Measure Spearman and top-15 hits over matched Fridays with outcomes. For forecast error, restrict to Friday pairs with exactly five following sessions so the target horizons align across arms. H26 still scores a five-session forecast at shorter weeks, and H27 still scores a normal one-week forecast on decisions preceding a Friday market holiday. Replay all decisions with the same 15/20 sticky rule, carrying NAV from $10,000. Run equal weights and 252-session HRP on identical picks within each model. Fill at the next exchange open with archived adjusted opens for returns and unadjusted opens for IBKR Singapore Tiered fees. Apply 10 bps per traded dollar and compare with SPY adjusted open-to-open buy-and-hold.

Inputs are the frozen roster manifest from the August 23 universe audit, the scratch 2016–2025 OHLCV and news archives, the audited Friday parquet and flagged-return files, the 2026 holdout prices and news, and the saved H21–H23 stock scores. Training metadata records the base Git revision, checkpoint hashes, target scaling, and fit counts. The historical roster is survivorship-conditioned. This single-seed experiment reuses the 2026 period.

```sh
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/train_v1_multitarget.py --device mps
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/run_v1_multitarget.py --device mps
```

## Results

H26 fit 72,216 examples over 207 decision weeks. H27 fit 69,486 examples over 197 Friday pairs. Both used all 356 fitting stocks and shared 15,867 validation examples over 39 weeks. H26 selected epoch 8 with validation Spearman +0.0178. H27 selected epoch 38 with +0.0453.

The January 12–November 9, 2026 replay scored 406–421 candidates per decision across 40 holding periods. Rank metrics use 37 matched Friday pairs. Basket MAE uses 33 pairs with five trading sessions.

| Model | Mean Spearman | Mean actual top-15 hits | Basket MAE | Equal-weight net return | HRP net return | HRP max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| V1 | +0.0057 | 1.65 | 4.93 pp | +42.94% | +27.95% | −23.52% |
| H5 | −0.0228 | 0.81 | 3.24 pp | −0.12% | −2.22% | −16.74% |
| H21 | −0.0033 | 1.27 | 4.11 pp | +17.08% | +15.41% | −23.47% |
| H26 | +0.0027 | 0.49 | 1.94 pp | −3.94% | −9.00% | −16.19% |
| H27 | +0.0230 | 0.62 | 2.22 pp | +17.98% | +12.56% | −7.31% |

SPY returned +8.26%. Both Holm-adjusted p-values for higher HRP return than V1 were 1.00. H27 had the highest universe-wide Spearman correlation and the smallest HRP drawdown. V1 found more of the actual winners and finished with the highest net return under both weightings. H26 had the smallest basket forecast error and finished below H5 on both portfolio replays. H27 finished slightly ahead of H21 with equal weights and behind it with HRP. The 95% weekly HRP return difference intervals against H5 and H21 both included zero. Keep V1 on this evidence.

The table uses Metal for scoring. A CPU preflight of the same H26 checkpoint produced −3.63% with equal weights and −9.12% with HRP, illustrating the replay's sensitivity to the scoring backend.

[Summary](data/v1_hypotheses/loss_comparison/multitarget/summary.json) · [Portfolio chart](data/v1_hypotheses/loss_comparison/multitarget/portfolio_comparison.png) · [Per-stock forecasts and ranks](data/v1_hypotheses/loss_comparison/multitarget/stock_scores.csv) · [Weekly ranks and forecast errors](data/v1_hypotheses/loss_comparison/multitarget/rank_weekly.csv) · [Picks and weights](data/v1_hypotheses/loss_comparison/multitarget/picks_and_weights.csv)
