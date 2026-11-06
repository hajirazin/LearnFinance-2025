# PPO almost-equal-weight experiment

Run on November 6, 2026. The current trained PPO behaves like a stock selector
with nearly equal stock weights and a separate cash decision on this sample.
Its small weighting tilts did not demonstrate an incremental return benefit.
This does not establish that its selected stocks are objectively the best.

## Scope

- Artifact: `v2026-09-12-13a430da1914-seed42`, seed 42.
- Frozen universe: 432 names from the artifact.
- Sample: all 69 saved test actor cutoffs, May 9, 2025 through August 28, 2026.
- Data: newly downloaded adjusted Yahoo OHLCV; existing complete news queried
  through the production adapter with read-only DuckDB connections; artifact
  scalers and HMM parameters. No training or promotion.
- The primary counterfactual preserves PPO's exact selected names, count and
  cash every week, changing only their relative stock weights. Each arm carries
  its own after-return, after-cost portfolio into the next rebalance. Both use
  the artifact's IBKR Singapore Tiered cost settings and $10,000 capital base.
- A secondary equal-weight closed loop evolves its own state and selections.

## Findings

| Measurement | Result |
| --- | ---: |
| Natural selected count | 10 in 69/69 weeks |
| Average cash | 14.6474% |
| Cash range | 14.4774% to 14.7945% |
| Mean stock-weight CV | 3.9991% |
| 95% block-bootstrap interval for mean CV | 3.7929% to 4.1754% |
| 95% one-sided upper bound for mean CV | 4.1452% |
| All stock weights within 5% of equal share | 6/69 weeks |
| All stock weights within 10% of equal share | 61/69 weeks |
| All stock weights within 15% of equal share | 69/69 weeks |
| Largest individual relative departure from equal share | 14.2728% |
| Mean effective stock count | 9.9831 out of 10 |
| Average portfolio mass moved to reach equal weight | 1.4227% of NAV |

Weights are normalized by total stock exposure before calculating relative
deviations. With 10 stocks, a 10% relative departure means a stock's share of
the stock sleeve is between 9% and 11%; it does not mean a 10-percentage-point
portfolio-weight difference. CV is population standard deviation / mean across
selected stock weights; it is unaffected by the amount of cash.

The operational claim “always within 10%” is contradicted by eight weeks.
“Within 15% in all observed weeks” holds. No finite historical sample proves
future “always” behavior. The analyst-chosen 5% mean-CV equivalence margin is
supported: the one-sided 95% upper bound is below 5%, with 4-, 8- and 13-week
blocks. These are exploratory practical thresholds, not an objective universal
definition of almost equal and not a test of exact numerical equality.

The saved original evaluation independently reports mean CV 3.9831%, very
close to the fresh replay despite changed price evidence.

## Does unequal weighting add value?

| Strategy | Net CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| PPO original weights | 51.5554% | 1.5091 | 15.2365% |
| Equal weights, same exact names and cash | 51.4792% | 1.5186 | 15.0956% |
| Equal weights with its own feedback loop | 51.4792% | 1.5186 | 15.0956% |

The isolated weighting head adds 0.0763 percentage points of CAGR. The mean
paired weekly net-log advantage is 0.0968 basis points, with a 95% moving-block
interval of -2.8287 to +2.4301 basis points per week. It spans zero for all
three block lengths. This finds no reliable positive incremental weighting
benefit; it does not prove the two strategies have equivalent future returns.
The feedback ablation chooses the same names in all 69 weeks, with maximum
cash difference below 0.000073 percentage points.

## Ranking and training versus inference

The deterministic stock allocation is
`stock_mass * alpha_i / sum(alpha)`, where
`alpha_i = min(softplus(allocation_raw_i) + 1, 50)`.
The observed selected concentrations are similar: the mean weekly minimum is
2.1270, maximum 2.4249 and sum 22.7124. Their normalized means are consequently
near equal. This identifies the output mechanism, not the cause of the
training outcome.

Conditional on the same selected stocks, 1,000 Dirichlet draws per week have
average median CV 58.7565%, compared with 3.9991% for deterministic means.
Training samples meaningfully uneven weights; the deployed deterministic
policy uses their conditional mean. It would be incorrect to say every
training action was nearly equal weight.

Selection and allocation use distinct learned heads. Average within-selected
Spearman correlation between selection score and assigned weight is 0.3221,
so the allocation is not simply proportional to selection rank.

Selection-logit versus next-week realized-return rank IC averages 0.03394.
Its 95% interval is [0.00091, 0.07067] with four-week blocks, but includes zero
with eight-week [-0.00437, 0.06802] and 13-week [-0.00171, 0.06234] blocks.
Selection-only gross outperformance over the same-cash universe-mean return
also has an interval spanning zero. There is weak, non-robust evidence of rank
skill; these results cannot validate “selects the best future stocks.”

Forced top-K sensitivity on the same primary states:

| Forced K | Mean CV | Weeks every name within 10% | Largest departure |
| --- | ---: | ---: | ---: |
| 3 | 3.4123% | 97.10% | 11.7351% |
| 5 | 3.6693% | 94.20% | 11.5411% |
| 10 | 3.9991% | 88.41% | 14.2728% |
| 15 | 4.2320% | 81.16% | 16.5677% |

These forced-count probes are behavioral sensitivity checks, not separate
closed-loop performance backtests or evidence that every possible state or
every possible trained PPO behaves this way.

## Limits and evidence checks

This reuses historical test weeks and a frozen present-day roster, with the
same survivorship bias as the original experiment. It is not a new untouched
test set. Fresh Yahoo evidence differs from the saved price hashes; five
spot-checked frames all differ. Fresh CAGR 51.56% differs from the saved 57.66%,
so the new performance figures are not presented as an exact reproduction.
The news-adapter source hash matches the artifact. Two now-unavailable
downloads, CCO and KVUE, are explicitly recorded and treated by the normal
eligibility exclusion path; all policy-selected/traded prices are available,
and zero eligible names lack next-week prices in the rank measurements.

The current artifact's saved `allocation_head_diagnostics` references a full
PPO CAGR of 28.34% even though its primary saved evaluation is 57.66%.
The older one-off promotion script copies this diagnostic from the sibling
candidate without recomputing it. This experiment computes a new paired
comparison for the actual current policy and does not use that stale value.

SHA-256 checks confirm 416 production source, Temporal, model/current and news
database files remain unchanged. The scratch statistics tests pass (5/5),
exported results pass independent arithmetic checks and Ruff passes.

## Reproduce

From the repository root, replay the frozen downloaded evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python scratch/ppo_equal_weight_2026_11_06/run_experiment.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python scratch/ppo_equal_weight_2026_11_06/verify_results.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api brain_api/.venv/bin/python -m pytest scratch/ppo_equal_weight_2026_11_06/test_statistics_helpers.py -q -p no:cacheprovider
```

The original run used the same runner before its model-version lookup was
pinned to `download_manifest.json`; the production pointer and the downloaded
version both named the artifact above. Formatting changes do not alter the
experiment calculations. Running `fetch_prices.py` again intentionally fetches
fresh evidence and may produce different historical price bytes.

`PROTOCOL.md` contains definitions fixed before the fresh replay. `weekly.json`
contains weekly decisions, costs, return comparisons and diagnostics;
`stock_weights.csv` contains 690 individual weights and ranks. `summary.json`
contains bootstrap intervals and the saved-artifact audit. Input hashes,
download coverage, price-hash spot checks and protected-file hashes are saved
alongside them.

Git includes source, reports and compact result evidence. Downloaded prices,
model binaries and provider session caches stay local and are ignored.
