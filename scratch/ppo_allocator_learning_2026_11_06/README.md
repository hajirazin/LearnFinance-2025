# Can PPO learn useful unequal allocations?

The controlled experiment answers **yes**. All six visible-signal PPO runs
learned a profitable unequal allocation under the actual cost and HHI reward.
The historical experiment below separately asks whether such tilts helped in
real 2026 decisions after a pre-2026-only retrain. No production source,
current pointer, broker state or news database was changed.

## Controlled experiment

Ten fixed selected stocks, 10% cash and $10,000 NAV. An observable feature
identifies one asset with conditional weekly return +4%, versus -0.5% for the
other nine, plus independent 1% Gaussian return noise during training. This is
a deliberately strong causal synthetic signal, not a claim about real news.
Each one-step episode starts at the same equal-weight book. Use production
broker costs, the HHI penalty, allocation head, concentration transform,
Dirichlet likelihood and clipped PPO update. Count, selected set and cash are
conditioned upon rather than optimized. Temporal embeddings are fixed at zero
in the set-encoder case, which learns from one explicit synthetic asset feature.
The conditioned tests use 128-decision one-step batches and 64-decision PPO
minibatches, with the normal learning rate and loss coefficients; they do not
test learning to extract the signal from raw price history.

Nine runs: three seeds, 10,000 sampled decisions per run. The set case uses
the actual production set encoder and allocation head. The head-only case
isolates the final linear head with a clear encoded feature. Hidden-signal
controls have no observable information identifying the favorable asset.

| Case | Seed | Mean winner share of stocks | Stock-weight CV | Feasible reward gain recovered |
| --- | ---: | ---: | ---: | ---: |
| head_signal | 42 | 36.62% | 88.74% | 72.7% |
| head_signal | 123 | 37.35% | 91.18% | 74.8% |
| head_signal | 2026 | 37.31% | 91.05% | 74.7% |
| set_signal | 42 | 40.54% | 101.79% | 83.7% |
| set_signal | 123 | 40.55% | 101.82% | 83.7% |
| set_signal | 2026 | 40.87% | 102.90% | 84.7% |
| set_hidden_signal | 42 | 10.00% | 0.00% | -0.0% |
| set_hidden_signal | 123 | 10.00% | 0.00% | -0.0% |
| set_hidden_signal | 2026 | 10.00% | 0.00% | -0.0% |

Equal weight assigns 10% to the winner. A symmetric scalar-grid comparator
under the same deterministic reward assigns 46.35%; its weights are feasible
under the concentration bounds. A separate supervised fit reproduces it,
showing representational capacity; that positive control is not a PPO result.
The grid comparator is not claimed to be a formal global optimizer over all
possible policies. The actual PPO objective averages sampled-action rewards,
so it need not select the deterministic comparator's exact mean weights.

Each frozen policy was additionally evaluated on 10,000 independent noisy
outcome realizations, with common random numbers across policies. Every
visible-signal policy's paired 95% interval is above equal weight. The hidden
controls have effectively zero deterministic advantage. New context instances
draw from the same ten possible winner slots; there is no claim of unseen
state-support generalization. See controlled_noisy_holdout.json for intervals.

This demonstrates useful unequal-weight learning in the conditioned problem.
It does not prove the unconstrained full PPO always discovers the best set,
cash and weights simultaneously, or that the real market supplies this signal.

## Historical experiment

Outer training-data cutoff: December 31, 2025. Train on 312
weeks from 2020-01-03 20:00:00+00:00 through actor cutoff
2025-12-19 20:00:00+00:00. The final training reward ends
2025-12-29 00:00:00. HMM fitting stops at 2025-12-19.
News and channel scalers use training weeks only. Encoder pretraining uses
its internal pre-2026 validation; the full PPO uses its fixed final-budget
checkpoint. No 2026 return chooses a checkpoint, seed or hyperparameter.

This is a new full seed-42 model from random initialization, with the normal
production architecture and current source defaults: not a continued fit of
the promoted model. Stage A stopped at epoch 13, with best internal validation
Rank IC about 0.02575. Stage B completed 10296
sampled weekly decisions; the native trainer finishes its last full episode
after the 10,000 minimum budget. Temporal freeze/unfreeze and PPO loss follow
the production trainer. These are repeated passes over 312 fixed historical
transitions, not thousands of additional independent market weeks.
Raw features are cached to remove repeated data
construction, and native/cached feature and closed-loop parity checks pass.

Test 36 decisions from 2026-01-05 00:00:00
through 2026-09-08 00:00:00, with returns through
2026-09-14 00:00:00. September 7 is the latest fully covered news
window for the entire roster; Labor Day moves its execution open to September
8. Later windows have coverage for only 26 of 432 names and are not substituted
with zero news. All test policies use the same prices and cost contract.

| Allocation | Test-period total return | Annualized CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| ppo | 34.95% | 54.18% | 1.990 | 9.25% |
| equal_fixed | 34.60% | 53.61% | 2.007 | 9.27% |
| equal_feedback | 34.60% | 53.61% | 2.007 | 9.27% |

`equal_fixed` retains each PPO decision's exact selected names, count and cash;
only the stock proportions are replaced. Both books carry their own drifted,
after-cost weights. `equal_feedback` evolves its own equal-weight book and
feeds that back to the policy, so later names or cash may also change.

Mean stock-weight CV: 5.17%, on 36 informative multi-stock weeks.
Count distribution: {'4': 33, '7': 3}. Mean cash:
48.11%. The allocation head's parameter L2 change from
Stage A to the final PPO is 0.223273; it was
actually updated rather than accidentally frozen.

PPO minus fixed-selection equal-weight CAGR:
+0.5732 percentage points.
Mean paired weekly net-log increment: +0.7163 basis
points, four-week-block 95% interval [-3.5915, +5.3750] basis
points per week. Eight- and 13-week-block sensitivity is in the summary.
The allocation increment is not distinguishable from zero in this sample.

## Interpretation and limits

The controlled result rules out a blanket inability of this allocation
parameterization and PPO update to learn advantageous unequal weights. It
does not identify why a particular real-data model chooses small tilts.
Weak or unstable conditional-return information, distributional exploration
and costs remain different explanations that these results do not isolate.
Do not reward unequal weights merely to force a different-looking portfolio.

The market run uses one full-training seed, not three. Its 36 test decisions
cannot establish market robustness. 2026 was already inspected in earlier
research, so this is a retrospective chronological holdout, not an untouched
new test set. The current 432-stock roster is applied historically and has
survivorship/selection bias. Fresh adjusted Yahoo data can differ from older
saved artifacts. The new run uses current source defaults, including the
1e-9 rebalance-weight epsilon; the old promoted artifact loads 0.005 for this
field. An old-model/new-model comparison therefore is not a pure cutoff-date
experiment. The allocation comparison inside the new model is controlled.

The production news database was opened read-only. CCO and KVUE have no price
history in the download and follow ordinary insufficient-history exclusion;
an SMCAY cache-lock failure was explicitly retried from the same provider and
recovered. No held or selected future prices are silently filled. Factual
news is never invented for missing coverage.

## Verification and reproduction

Five scratch behavior tests pass, including likelihood reconstruction,
gradient credit, actual-reward feasibility and production/cache book parity.
Independent arithmetic checks reproduce the reported CAGRs. SHA-256 checks
confirm 416 protected production source, Temporal, PPO model/current and
news database files are unchanged. Checkpoints, cached states and outputs live
only in this directory. No model is promoted or uploaded.

From the repository root, with GPU access for training:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python scratch/ppo_allocator_learning_2026_11_06/run_controlled.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python scratch/ppo_allocator_learning_2026_11_06/evaluate_controlled_noise.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python scratch/ppo_allocator_learning_2026_11_06/run_market_training.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python -m pytest scratch/ppo_allocator_learning_2026_11_06/test_controlled.py scratch/ppo_allocator_learning_2026_11_06/test_market_cache.py -q -p no:cacheprovider
```

Existing completed files/checkpoints are reused. The downloaded price evidence
and prepared dataset are frozen locally; refetching prices would be a new data
revision. PROTOCOL.md records definitions and its implementation clarifications.
summary.json contains both experiments; market_test_weekly.json and
market_stock_weights.csv contain the historical decisions and individual
weights. The training-only monitor never evaluates a 2026 context.

Git includes source, reports and compact result evidence. Generated price and
feature datasets, model checkpoints and provider session caches stay local and
are ignored. The two cache-parity tests explicitly skip on a clean checkout
until the local market dataset has been generated; the controlled tests do not
require that dataset.
