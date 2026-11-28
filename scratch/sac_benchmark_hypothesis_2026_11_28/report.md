# Does SPY reward teach SAC to reject weak stocks?

Analysis date: November 28, 2026. This uses the completed experiments and
adds retrospective behavior diagnostics, a supervised signal probe, and
hindsight opportunity ceilings. No SAC retraining or production change occurred.

## Reward mathematics

For the same exogenous historical market path, let `G(w)` be the existing
net-log-wealth reward including IBKR costs and the concentration term. Then:

```
R_EW(w)  = G(w) - log(1 + r_EW)
R_SPY(w) = G(w) - log(1 + r_SPY)
R_b(A) - R_b(B) = G(A) - G(B)
```

The benchmark term cancels when comparing two actions. The same cancellation
applies to the discounted sum over a fixed horizon because benchmark returns
are unaffected by SAC's actions. This result is for the raw objective; finite
function approximation, running reward normalization, clipping and entropy
learning can produce different learned policies.

Equal-weight subtraction contains no penalty for departing from equal weights.
It compares the allocator with a passive basket, rather than endorsing that
basket as the optimal investment. A losing basket can be beaten by cash if
the policy is allowed to choose cash. Comparing a +2% portfolio with a 0%
portfolio gives the same reward gap, log(1.02), whether the basket loses 4%
or SPY gains 1%.

## Action-space restriction

Current SAC emits tanh-bounded logits followed by masked softmax. With all 15
stocks eligible, it cannot assign a zero target to an eligible stock. Limits
of the current action mapping are:

| Quantity | Bound |
| --- | ---: |
| Minimum individual eligible stock target | 0.894% |
| Maximum CASH target | 33.003% |
| Minimum combined weight in any five stocks | 5.795% |
| Minimum combined weight in any six stocks | 7.510% |

These are targets; an existing holding can require no additional buy when its
current weight already exceeds the target. Masked ineligible stocks can have
zero targets, but eligibility masking is not discretionary stock rejection.

If all 15 stocks fall 10%, even perfect foresight cannot avoid a loss of about
6.70% before fees under the current action mapping. A policy permitting 100%
cash could avoid that gross loss. Changing the reward benchmark cannot remove
this action-space restriction.

## Observed stock-rejection behavior

Bad-stock labels below use the following week's realized outcomes only for
evaluation; those labels were never inputs to the original policy. Policies
are paired by seed. Weekly differences are averaged over seeds 42, 123, 2026,
then evaluated with a circular moving-block bootstrap: eight-week blocks,
2,000 draws, random seed 42. Intervals measure time-series uncertainty
conditional on the three trained seeds, without multiple-comparison correction.

| Filtered-universe regime | Weeks | SPY reward minus EW reward: mean weekly net return (pp) | Exploratory 95% interval (pp) |
| --- | ---: | ---: | ---: |
| All weeks | 104 | -0.006 | [-0.105, +0.083] |
| Exactly 5–6 stocks with negative returns | 17 | -0.300 | [-0.586, -0.014] |
| Majority of stocks with negative returns | 51 | +0.048 | [-0.081, +0.177] |
| Exactly 5–6 stocks underperform SPY | 24 | -0.140 | [-0.404, +0.149] |
| Majority of stocks underperform SPY | 56 | +0.135 | [+0.001, +0.259] |
| SPY down | 44 | -0.041 | [-0.230, +0.153] |

Results are mixed, rather than demonstrating universal defensive behavior.
In the 17 weeks with exactly 5–6 negative stocks, the SPY-reward policies
allocated only 0.13 percentage points less of their invested stock exposure
to those losers than EW-reward policies. Its interval was [-2.57, +2.40] pp,
so no clear loser-underweighting advantage was established.

The files include all four rewards, both universes and every seed, avoiding
selection of only a favorable trained policy. The SPY-versus-EW comparison
is directly available for halal_filtered; the earlier halal experiment used
QQQ, so its ETF comparison is QQQ-versus-EW, not a new SPY-trained model.

## AI predictive-information probe

A fixed histogram gradient boosting classifier was fit separately to each
universe and outcome label. Features were the six per-stock cross-sectional
SAC signal ranks: PatchTST, three momentums, news and volatility. Holdings and
the five global SAC features were excluded. The original 360 training weeks
were used for fitting and 104 held-out weeks for evaluation, keeping whole
weeks together. The model used depth 3, seven leaves, 100 iterations, learning
rate 0.05, L2 regularization 1, minimum 50 samples per leaf and seed 42.
Random internal early-stopping validation was disabled; no hyperparameter
search was performed.

| Universe | Future outcome predicted | Mean within-week AUC | PatchTST-rank AUC |
| --- | --- | ---: | ---: |
| halal | Negative next-week return | 0.501 | 0.500 |
| halal | Underperform SPY next week | 0.488 | 0.520 |
| halal_filtered | Negative next-week return | 0.506 | 0.517 |
| halal_filtered | Underperform SPY next week | 0.490 | 0.511 |

AUC 0.5 indicates chance discrimination. All four classifier Brier scores
were worse than the corresponding constant training-period class-rate
baseline. This probe did not establish a strong ability to identify future
losers. It does not prove the complete SAC state or other learning methods
lack predictive information, and binary classification does not test all
information about return magnitudes or risk.

## Linear-programming opportunity ceilings

Each test week was also optimized using perfect knowledge of its future
stock returns, zero fees and free rebalancing. One linear program enforces
the existing bounded-logit weight-ratio constraint and 2% minimum cash; a
second ceiling permits the full simplex with that cash minimum. These are
gross hindsight upper bounds, not executable strategies or forecasts.

| Universe | Bounded-actor ceiling below SPY gross return | Full-simplex ceiling below SPY gross return | Bounded-actor ceiling below zero |
| --- | ---: | ---: | ---: |
| halal | 0/104 | 0/104 | 8/104 |
| halal_filtered | 7/104 | 0/104 | 11/104 |

The zero-fee, perfect-foresight bounded allocator could not beat SPY in seven
filtered-universe weeks. The freedom to reject names and use cash matters
independently of benchmark choice. Hindsight success elsewhere does not
establish that an agent could predict and exploit those opportunities.

## Recommended next experiment

Use a factorial comparison in scratch:

| Reward baseline | Current bounded dense SAC | Correctly formulated selective SAC with zero targets and full cash |
| --- | --- | --- |
| Equal-weight basket | Existing control | Action-flexibility trial |
| SPY | Existing control | Benchmark-plus-flexibility trial |

Keep data, costs, training budget and seeds matched. Check future-loser
exposure, cash in weak-basket weeks, net return, downside losses and turnover
on locked chronological splits and prospective paper data. A selective SAC
policy requires correct stochastic density and entropy mathematics; replacing
softmax with a sparse transform alone is not sufficient. This is a research
experiment, not a new production promotion gate.

The frozen current-roster and later-edited-news biases remain. The test window
has already informed prior decisions, so these results are exploratory and
need a new confirmation period. Six synthetic numerical tests and Ruff checks
passed. No production reward, model artifact, current pointer or account changed.

Primary method references:

- [Chronological model evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Histogram gradient boosting classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [Risk-constrained RL with CVaR](https://arxiv.org/abs/1512.01629), an option if explicit downside constraints become the research objective.
