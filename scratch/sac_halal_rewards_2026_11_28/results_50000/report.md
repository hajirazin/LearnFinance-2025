# SAC halal reward experiment results

Completed November 28, 2026. Research results only; production was unchanged.

All 12 runs completed: four rewards, three matched seeds each, 50,000 steps per run. 360 training weeks and 104 held-out weeks, 2024-11-25 to 2026-11-23. Returns include the existing IBKR Singapore Tiered model at fixed USD 10,000 NAV. Sharpe assumes zero risk-free rate.

**The historical panel includes survivor-selected stocks and later-edited news. These are biased exploratory results, not an unbiased investment backtest.**

## Performance

SAC entries are medians across all three seeds. Benchmarks are independent buy-and-hold portfolios with an IBKR entry fee. No winning seed was selected.

| Strategy | CAGR | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| QQQ_buy_hold | 13.99% | 0.696 | 24.17% |
| halal_equal_weight_buy_hold | 41.32% | 1.244 | 27.73% |
| SAC current | 30.46% | 1.147 | 25.73% |
| SAC qqq_excess | 31.31% | 1.150 | 25.85% |
| SAC equal_weight_excess | 34.85% | 1.231 | 26.60% |
| SAC qqq_hybrid | 29.53% | 1.195 | 29.22% |

## Improvement against current reward

Paired differences compare the same seed under identical data and training budgets.

| Reward | Median paired CAGR change (pp) | Median paired Sharpe change | Seeds improving both | Seeds beating QQQ on both | Seeds beating equal weight on both |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | +0.00 | +0.000 | 0/3 | 3/3 | 0/3 |
| qqq_excess | +0.16 | -0.033 | 1/3 | 3/3 | 0/3 |
| equal_weight_excess | +4.39 | +0.099 | 2/3 | 3/3 | 0/3 |
| qqq_hybrid | -1.62 | +0.012 | 1/3 | 3/3 | 0/3 |

## Individual seeds

| Reward | Seed | CAGR | Sharpe | Average cash | Average weekly turnover | Fees over 104 weeks (USD, fixed NAV) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 42 | 31.15% | 1.183 | 7.88% | 33.21% | 411.14 |
| current | 123 | 30.46% | 1.133 | 4.89% | 27.86% | 403.45 |
| current | 2026 | 29.08% | 1.147 | 4.08% | 30.53% | 416.29 |
| qqq_excess | 42 | 31.31% | 1.150 | 3.73% | 31.16% | 425.35 |
| qqq_excess | 123 | 28.98% | 1.085 | 7.47% | 32.88% | 406.67 |
| qqq_excess | 2026 | 40.09% | 1.428 | 3.99% | 33.81% | 406.79 |
| equal_weight_excess | 42 | 30.48% | 1.185 | 4.48% | 31.11% | 422.85 |
| equal_weight_excess | 123 | 34.85% | 1.231 | 4.62% | 32.61% | 416.11 |
| equal_weight_excess | 2026 | 36.16% | 1.256 | 4.89% | 34.31% | 416.89 |
| qqq_hybrid | 42 | 29.53% | 1.195 | 16.62% | 33.51% | 391.08 |
| qqq_hybrid | 123 | 34.73% | 1.215 | 7.47% | 27.21% | 416.91 |
| qqq_hybrid | 2026 | 25.22% | 0.984 | 2.94% | 30.70% | 395.87 |

## Execution sensitivity

Evaluation only, without retraining. Every executed delta pays IBKR fees; deadbands freeze small trades. Slippage is charged per stock notional traded. Values are medians across three seeds. These are alternate execution assumptions, separate from the matched reward experiment.

| Reward | True deadband | Slippage (bps) | CAGR | Sharpe | Fees and slippage (USD, fixed NAV) |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 0% | 0 | 30.16% | 1.138 | 453.33 |
| current | 1% | 0 | 30.59% | 1.143 | 370.06 |
| current | 2% | 0 | 31.45% | 1.154 | 298.76 |
| current | 0% | 10 | 26.63% | 1.020 | 1056.57 |
| equal_weight_excess | 0% | 0 | 34.59% | 1.224 | 455.08 |
| equal_weight_excess | 1% | 0 | 35.79% | 1.251 | 372.90 |
| equal_weight_excess | 2% | 0 | 35.80% | 1.254 | 320.49 |
| equal_weight_excess | 0% | 10 | 30.34% | 1.107 | 1100.21 |
| qqq_excess | 0% | 0 | 31.08% | 1.144 | 454.09 |
| qqq_excess | 1% | 0 | 31.42% | 1.156 | 367.68 |
| qqq_excess | 2% | 0 | 31.27% | 1.158 | 316.40 |
| qqq_excess | 0% | 10 | 27.05% | 1.027 | 1103.78 |
| qqq_hybrid | 0% | 0 | 29.15% | 1.183 | 449.34 |
| qqq_hybrid | 1% | 0 | 29.95% | 1.207 | 344.20 |
| qqq_hybrid | 2% | 0 | 28.50% | 1.158 | 285.67 |
| qqq_hybrid | 0% | 10 | 25.12% | 1.051 | 1070.79 |

## Interpretation

Equal-weight-relative reward produced median CAGR 34.85% and median Sharpe 1.231. Relative to the current reward, those medians increased by 4.39 percentage points and 0.085. It improved both metrics in 2 of three paired seeds.

Every SAC reward variant beat QQQ on both metrics in every seed.

No SAC seed beat the equal-weight buy-and-hold basket on both metrics. The strong passive universe baseline remains a higher hurdle than QQQ.

No alternative reward improved both CAGR and Sharpe in all three paired seeds. Any median improvement should be treated as seed-sensitive evidence.

An action-independent benchmark subtraction preserves the ideal unnormalized objective ranking. Differences here reflect learning, normalization, and approximation. The hybrid changes the reward sign convention, not the ideal action ranking. See the experiment README for formulas, assumptions, and reproduction commands.

Integrity checks are recorded in verification.json: all input hashes unchanged, all traces aligned to the same 104 weeks, finite constrained weights, metrics independently recomputed, and 48 execution sensitivity evaluations present.

Total training time: 59.0 minutes. All results, models, and evidence remain in this scratch directory. No production model was promoted.

## Portfolio growth graphs

The four SAC curves use the actual median-CAGR seed for each reward. The tables
above summarize all three seeds; a plotted seed's Sharpe can differ from the
median Sharpe. All curves start at USD 10,000 over the same two-year window.

![SAC halal: four reward variants and two passive benchmarks](../../sac_reward_comparison_charts_2026_11_28/01_sac_halal.png)

[Vector SVG](../../sac_reward_comparison_charts_2026_11_28/01_sac_halal.svg) · [Plotted values](../../sac_reward_comparison_charts_2026_11_28/01_sac_halal.csv)

![SAC halal: original six curves plus 252-session HRP](../../sac_hrp_comparison_charts_2026_11_28/01_sac_halal_with_hrp.png)

[Vector SVG](../../sac_hrp_comparison_charts_2026_11_28/01_sac_halal_with_hrp.svg) · [Plotted values](../../sac_hrp_comparison_charts_2026_11_28/01_sac_halal_with_hrp.csv)

[Combined twelve-line graph](../../sac_reward_comparison_charts_2026_11_28/03_all_12_scenarios.png) ·
[HRP methodology](../../sac_hrp_comparison_charts_2026_11_28/README.md)
