# Allocator learning experiment

Research only. All code, data, checkpoints and reports stay in this directory.
No production source edits, API training jobs, promotions, orders or messages.

## Controlled allocation experiment

- Ten fixed selected stocks, 10% fixed cash, $10,000 historical NAV.
- An observed signal identifies one stock, uniformly changing its position.
  Its conditional next-week simple return is +4%; the other nine are -0.5%.
  Independent zero-mean 1% Gaussian return noise is added in training.
- Apply the production IBKR Singapore Tiered transaction costs and HHI reward
  penalty. Each contextual-bandit episode starts from the same equal-weight
  book. This isolates conditional allocation and is not a market backtest.
- Use the production allocation head, concentration transform, Dirichlet
  sampling/log density and PPO clipped update. Condition on the fixed count,
  selected set and cash, so only the allocation log probability is optimized.
- Three cases: allocation-head-only with an explicit encoded signal; full
  production set encoder plus allocation head with an explicit asset signal;
  the same set encoder with the signal hidden (negative control).
- Train seeds 42, 123 and 2026 for 10,000 sampled decisions each. Use fixed
  budgets, no best-seed selection. Evaluate 1,000 independently drawn context
  instances per seed using deterministic means and noiseless conditional
  returns, plus sampled actions. The context support has ten possible winner
  slots, so this tests new draws, not previously unseen state values. Identify
  future returns from the generator only after the action.
- A supervised deterministic-mean fit to a feasible target with the same head
  is a capacity positive control, explicitly not a PPO result. A scalar grid over winner
  weight establishes a feasible unequal-weight comparator under the reward.
- Practical success: every seed in the visible-signal case assigns at least
  30% of the stock sleeve to the signal winner on average, and the paired
  mean deterministic reward beats equal weight with a 95% confidence interval
  above zero. Report fraction of the feasible comparator's improvement.
- Report gradients, sampled-action versus deterministic-mean differences and
  all failed cases; do not interpret a negative control as proof of optimality.

After the controlled matrix, assess each frozen policy on the same 10,000 new
Gaussian-noise realizations (common random numbers across policies). This
separately checks future-outcome uncertainty instead of interpreting repeated
deterministic permutation contexts as independent market evidence. The wording
above clarifies the implementation; no training budgets or success thresholds
were changed in response to results.

## Historical extension, if the controlled diagnostic warrants it

The outer training-data cutoff is December 31, 2025. All training returns,
model selection, scalers and HMM fitting must end by that date. The 2026
chronological holdout is retrospective because earlier work inspected 2026.
Use only complete mandatory news coverage for the entire frozen roster.
The latest full-roster coverage will determine the test endpoint; do not use
zero-news substitutes for missing coverage. Keep every learned alternative
and equal-weight comparison outside production. Record actual gradient-fit
dates separately from the outer cutoff and disclose current-roster bias.

Historical implementation fixed before reading its 2026 outcomes: 312 training
weeks, final training return ending December 29, 2025; 36 test decisions January
5 through September 8, 2026, final return September 14. Fit all scalers and HMM
on the 312 training weeks, with HMM cutoff December 19. Use production Stage A
pretraining defaults and its internal pre-2026 validation, then production
Stage B seed 42 with the normal 10,000-decision minimum budget (native trainer
finishes its last episode). Evaluate the final-budget checkpoint without
2026-driven checkpoint selection. No current model weights initialize it.

For the 2026 allocation ablation, retain each baseline decision's selected
names, count and cash, replace only stock proportions with equal weight and
maintain separate drifted/costed books. Also evaluate equal weighting with its
own feedback loop. Compare paired weekly net logs using 4-, 8- and 13-week
moving-block bootstrap intervals. Report the actual selected-count distribution
and exclude K < 2 weeks from stock-dispersion evidence. This full real-data
retrain uses one seed; multi-seed capability evidence comes from the controlled
matrix and is not a claim of multi-seed market robustness.
