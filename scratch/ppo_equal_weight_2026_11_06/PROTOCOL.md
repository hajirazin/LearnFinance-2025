# PPO selected-stock equal-weight experiment

Written before fresh policy inference. Research only; no training, promotion,
orders, scheduled jobs, or production writes. All generated files stay here.

## Questions and fixed definitions

1. Does the currently promoted policy assign almost equal weights conditional
   on its selected stocks? Normalize each stock weight by total stock exposure,
   so changing cash cannot create apparent equality. Exclude K < 2 weeks.
2. Operational definition: every selected name has relative deviation
   `abs(K * equity_normalized_weight - 1) <= 10%` in a week. Report the fraction
   of weeks meeting this and all counterexamples; do not interpret a finite
   sample as proving "always". Also report thresholds 1, 2.5, 5, 15, and 20%.
3. Supporting practical-equivalence test: H0 mean weekly stock-weight CV >= 5%,
   H1 mean CV < 5%. Use a 95% one-sided moving-block-bootstrap upper bound,
   primary 4-week blocks; sensitivity 8 and 13 weeks. This reverses the user's
   verbal null so positive evidence for near-equality can be assessed. Exact
   equality and practical near-equality are different claims.
4. Replay the promoted policy's saved 69 test cutoff dates using fresh adjusted
   OHLCV and the existing mandatory DuckDB news evidence, train-fold scalers,
   frozen roster, and frozen HMM parameters. Record any price/data drift.
   This is a retrospective behavioral audit on a reused test set, not a new
   out-of-sample validation or a promotion evaluation.
5. Primary allocation ablation: preserve each primary policy decision's names,
   K, and cash; replace only stock proportions with equal weight. Maintain
   separate post-return books and transaction costs for the two strategies.
   Compare paired weekly economic net logs; use 4/8/13-week block intervals.
6. Secondary full feedback ablation: let an equal-stock-weight variant evolve
   its own book; differences can then include later selection/cash changes.
7. Diagnose rank/weight correlation and selection-logit rank IC against later
   realized returns. A high-ranked selection is the policy's preference, not
   evidence that it picked the objectively best future stocks.
8. Draw conditional Dirichlet stock weights for each selected set to distinguish
   stochastic training exploration from deterministic inference using means.

## Evidence protection

Hash the production source tree, model artifacts/current, and news DB before
and after. Load news through a read-only store subclass. No API calls or
production entrypoint executions. Public price downloads cache only here.
Store raw per-stock/per-week measurements, summaries, and a reproducible runner.
