# V1 loss comparison

H21–H23 use H5's seven daily feature channels, 60-session input, 356-stock training cohort, and 2021–2024 fitting and 2025 validation dates. The model still emits five daily log-return values and compounds them to a scalar score. The 2026 replay scores the broader roster and uses the candidate intersection with the saved close-only V1 and H5 baselines. Checkpoints maximize validation weekly rank IC with daily validation MSE as the tie-break, as in H5. Adam's learning-rate scheduler tracks the selected training objective. All runs use seed 42 and the existing 60-epoch cap and eight-epoch patience.

| Arm | Objective | Batch |
|---|---|---|
| H21 | MSE between predicted and realized five-session simple returns, each compounded from daily logs | Shuffled stock-weeks, 256 per batch, as in H5 |
| H22 | Mean logistic pairwise loss within each week. Each strictly higher realized return should get a higher score. Use a 0.02 return scale in the score gap. | One full cross-section per optimizer step |
| H23 | Compute the realized return of a differentiable approximation to the equal-weight top 15. Subtract SPY's adjusted-close return over the same five sessions. For excess `e`, minimize `-e + max(0, -e)`. A shortfall costs twice as much as the reward for the same outperformance. | One full cross-section per optimizer step |

H23 uses a sigmoid inclusion probability with a temperature of 0.01 and solves for a threshold where the expected number of selected names is 15. It normalizes inclusion probabilities to portfolio weights and computes the gradient through the threshold. The training proxy uses close-to-close returns, and the evaluation uses hard sticky selection and simulated next-open execution. H22 and H23 scores are evaluated as rankings.

Use the 2026 dates from the V1 replay, 15/20 stickiness, equal weights, $10,000 starting NAV, and the same regulatory fees and 10 bps per traded dollar. Report stock and basket forecast errors against actual five-session returns for H21, rank correlations and actual Friday-to-Friday top-15 hits for H22, and net excess versus SPY for H23. Also report all models' after-cost performance. Primary one-sided tests compare H21 basket squared error and H22 sticky top-15 hits against V1, and H23 weekly net excess against SPY. Use the existing circular block bootstrap and Holm correction across these three comparisons at 5%. The reused 2026 period is exploratory.

Save dividend-adjusted 2021–2025 SPY daily closes from Yahoo Finance as `data/v1_hypotheses/loss_comparison/spy_training_closes.csv`, with columns `date` and `close`. Use the saved SPY adjusted opens from the ranking and market study for evaluation.

```sh
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/train_v1_loss_comparison.py --device mps
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/run_v1_loss_comparison.py --device mps
```

## Results

The replay used 406–421 common candidates per decision, 40 portfolio holding periods, and 37 exact Friday-to-Friday rank comparisons. All three losses improved net portfolio return over H5's daily MSE. The close-only V1 reference retained the highest return and the most realized top-15 hits. H21 improved basket forecast MAE against V1 from 4.70 to 4.01 percentage points. H5's basket MAE was 3.19 points. H22 found 1.41 realized winners per week versus 1.65 for V1 and 0.81 for H5. H23 finished 3.02 percentage points ahead of SPY, with the smallest drawdown among the five arms. It beat SPY in seven of 18 down periods and lagged by 0.57 percentage points on average during those periods. Holm-adjusted p-values for H21, H22, and H23 were 0.250, 0.812, and 0.769. [Summary](data/v1_hypotheses/loss_comparison/summary.json)

## H24 · Weekly MSE with SAC's IBKR fees

Keep H21's architecture, features, 356-stock training cohort, 60-session context, and data splits. Minimize `weekly return MSE + 0.02 × (IBKR fees / $10,000)`, with returns expressed as decimals. Fix the coefficient at 0.02 before looking at the replay. A 10-basis-point cost adds 0.00002 to the objective, comparable to the squared error of a 0.45-percentage-point miss.

Process one full cross-section at a time in chronological order. Start each fold in cash. Carry the prior hard 15/20 selection and drift its weights using adjusted closes up to the current decision. The forward pass charges SAC's IBKR Singapore Tiered fee for each buy, sell, and resize. Training estimates shares using adjusted closes, the available historical price series. The loss uses a straight-through gradient through the smooth top-15 approximation, keeping the hard basket in the forward pass. Previous selections are detached. Checkpoint selection still uses validation rank IC. Refit a zero-fee H21 control using the same chronological batches to separate the cost term from the training-order change.

Replay the saved V1 and H21, the chronological H21 control, and H24 with the same candidate universe and 15/20 rule. Use equal weights, $10,000 initial NAV, and the same 40 next-open holding periods. Apply SAC's production per-leg IBKR schedule to both sides at raw opening prices, plus the prior 10 bps per traded dollar for spread and slippage. Charge all actual trades in the replay, including small reweights. The SAC environment's 0.5% weight-delta filter is specific to that environment and is outside this rebalance policy. Report fees, traded notional, after-cost return, drawdown, and forecast accuracy. The primary one-sided 5% test compares H24 with the chronological control. The original H21 comparison is descriptive and its p-value is also saved.

```sh
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/train_v1_loss_comparison.py --arm h24 --device mps
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/train_v1_loss_comparison.py --arm h21_chronological --device mps
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/run_v1_h24.py --device mps
```

| Arm | Compounded net return | Max drawdown | IBKR fees | Fees plus spread and slippage | Names replaced per rebalance |
|---|---:|---:|---:|---:|---:|
| Close-only V1 | +42.94% | 30.41% | $240.94 | $605.68 | 4.31 |
| Original H21 | +17.08% | 26.23% | $309.54 | $848.32 | 8.03 |
| H21 with chronological batches | −27.99% | 32.80% | $319.30 | $749.54 | 8.79 |
| H24 | −10.90% | 28.72% | $297.40 | $695.55 | 7.51 |

H24 beat the matched chronological control by 17.08 percentage points cumulatively, with a mean weekly advantage of 0.63 points and one-sided `p=0.0461`. Its two-sided 95% weekly interval was −0.03 to +1.37 points. H24 trailed original H21 by 27.98 percentage points cumulatively. The replay reuses the 2026 period and a single seed, so the matched improvement is exploratory. [Summary](data/v1_hypotheses/loss_comparison/h24_ibkr/summary.json) · [Weekly portfolios](data/v1_hypotheses/loss_comparison/h24_ibkr/portfolio_weekly.csv)
