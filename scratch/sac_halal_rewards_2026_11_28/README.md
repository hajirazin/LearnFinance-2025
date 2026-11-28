# SAC halal reward comparison

Research run: November 28, 2026. Production code, model pointers, publication,
and trading workflows are outside this experiment.

## Current reward

For target stock weights `w`, stock returns `r`, and the IBKR cost fraction `c`:

```
gross_return = sum(w[i] * r[i])
HHI = sum(w[i] ** 2)  # risky assets only; excludes CASH
reward = reward_scale * (
    log(1 + gross_return - c)
    - 0.4 * max(HHI - 0.20, 0)
)
```

Production SAC defaults to `reward_scale=1`, followed by running reward
normalization in the trainer. Differential Sharpe is not part of this reward.
Sharpe is an evaluation metric. The environment updates holdings to their
after-cost, after-return drifted weights and includes those holdings in the
next observation. CASH earns zero and has a 2% target minimum.

All 13 stocks remain eligible in this panel. Under the existing `[-1, 1]`
logit bounds, stock HHI is at most approximately 0.16762 (two stock logits at
the upper bound and the remaining stock/CASH logits at the lower bound).
Consequently the 0.20 HHI threshold is never reached in these runs; the active
reward here is simply net log wealth before benchmark subtraction.

The fee engine is IBKR Singapore Tiered with a fixed USD 10,000 NAV anchor,
including per-order minimums, clearing, regulatory and pass-through components.
It exempts stock changes below 0.5 percentage points from commission calculation;
the environment still moves the corresponding holdings. This exemption is
different from an actual execution deadband.

The published IBKR schedule confirms the USD 0.35 tiered and USD 1 fixed order
minimums: [IBKR Singapore stock commissions](https://www.interactivebrokers.com.sg/en/pricing/commissions-stocks.php?re=amer).
The experiment freezes the repository's existing rate configuration; it does
not revise historical fee schedules.

## Matched experiment

The current November 2026 `halal` roster has 13 stocks:
AAPL, AMD, AVGO, GOOGL, LLY, META, MSFT, MU, NVDA, SAP, TSLA, TSM, XOM.

All variants reuse the production SAC v3 masked actor, critics, replay,
normalization, train-only HMM, train-only feature scaler, and training loop.
Each variant trains for 50,000 steps on each of seeds 42, 123, 2026. Only the
scalar reward changes; the concentration penalty remains identical.

| Variant | Reward adjustment to current reward |
| --- | --- |
| current | None |
| qqq_excess | Subtract `log(1 + QQQ weekly return)` |
| equal_weight_excess | Subtract `log(1 + equal-weight buy-and-hold weekly return)` |
| qqq_hybrid | Subtract `log(1 + max(QQQ weekly return, 0))` |

The equal-weight hurdle is an independently held basket starting at equal
weights at the beginning of each fold. It drifts with prices; it is not reset
to equal weights every Monday. Reward hurdles use adjusted benchmark returns
before entry fees; evaluation buy-and-hold benchmarks pay their one-time IBKR
entry fee. No benchmark return is added to the policy's input features.
Keeping *constant* equal weights would require rebalancing the price drift and
would incur delta-trade fees. The suggestion's claim that static equal target
weights automatically imply zero fees confuses constant targets with holding
constant share quantities.

There are 464 weekly transitions: 360 train and 104 held out, with identical
open-to-next-open dates for every strategy. The held-out window runs November
25, 2024 through November 23, 2026. Sharpe uses weekly arithmetic returns,
sample standard deviation, annualization by `sqrt(52)`, and zero risk-free rate.
CAGR compounds net returns and annualizes by 52 weeks. Drawdown includes the
initial capital point. No test-based seed or checkpoint selection is performed.

QQQ is an evaluation comparator, not a screened halal investment recommendation.

## Interpretation of the suggestions

Subtracting an exogenous benchmark adds the same constant to every action's
reward at a given historical week. For a fixed start and horizon:

```
J_relative(policy) = J_current(policy) - discounted_benchmark_sum
```

The ideal policy ranking is therefore unchanged. This includes the proposed
hybrid, whose hurdle is also independent of the policy. It changes the reward's
sign in down markets but does not add an action-dependent downside penalty.
Finite neural approximation, clipping, replay, reward normalization, and entropy
tuning can produce different learned policies, so this is an empirical learning
experiment. Subtraction alone does not establish stationarity or guarantee
lower variance. SAC's off-policy and entropy mechanics are described in the
[original SAC paper](https://proceedings.mlr.press/v80/haarnoja18b).

Existing drifted-weight observations already implement that part of the
suggestion. Existing open-to-open labels assume execution at the weekly first
session's open, rather than a Monday 09:00 premarket fill. The experiment does
not change live order types or model opening-auction fill uncertainty.

`execution_sensitivity.py` evaluates completed policies without retraining:
charge every executed delta; charge every delta with actual 1% or 2% deadbands;
or charge every delta plus 10 bps of stock traded notional. Frozen positions
stay frozen unless replenishing the 2% cash floor requires a sale. This is an
execution robustness check, separate from the reward-only comparison.

## Limits

- The November 2026 roster is projected backward: survivorship/selection bias.
- Existing annual PatchTST snapshots are used causally by model cutoff, but
  historical inputs follow the previously approved `allow_later_revisions`
  news policy. There are 1,018 later-revision input cells and the manifest
  explicitly flags look-ahead bias. This is not an unbiased investment backtest.
- There is one two-year evaluation window and three optimization seeds.
  Improvement is evidence for this experiment, not proof of generalization.
- Prices are Yahoo dividend/split-adjusted OHLCV, as in the current training
  path. Withholding taxes, actual fills, FX, and time-varying fee schedules are
  excluded. Adjusted price levels also inherit the production assumptions for
  converting notional to historical share counts.
- The 31 output slots represent 30 possible stocks plus CASH; 17 stock slots
  are padded and masked for this 13-stock roster. The tanh-bounded actor and
  softmax limit attainable concentration and cash allocations. With all 13
  stocks eligible, maximum cash is `exp(2) / (13 + exp(2))`, about 36.24%.
  Having a CASH slot does not allow this actor to move entirely into cash.
- All matched runs use one CPU thread and deterministic PyTorch operations.
  The hardware choice is fixed across rewards; algorithm settings retain the
  production defaults.

## Reproduce

From the repository root, with its existing virtual environment:

```sh
.venv/bin/python scratch/sac_halal_rewards_2026_11_28/download_prices.py
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/prepare_data.py
PYTHONPATH=brain_api .venv/bin/python -m pytest scratch/sac_halal_rewards_2026_11_28/test_reward_environment.py -q
PYTHONPATH=brain_api .venv/bin/python -u scratch/sac_halal_rewards_2026_11_28/run_experiment.py
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/execution_sensitivity.py
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/summarize.py
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/verify_results.py
```

The download is a real-provider experiment operation; tests are synthetic and
offline. Preparation reads the news database in read-only mode and refuses
snapshot downloads. Optimization makes no provider calls. Every candidate is
written under this directory and completed seeds can be reused on restart.
Frozen input hashes, snapshot identities, code revision, dates, and policy
limitations are recorded in `evidence/manifest.json`.

See `results_50000/report.md` for measured results once all 12 runs complete.

The completed run passed 55 focused offline checks (eight experiment checks
plus 47 existing SAC/fee/contract/seed-policy checks), Ruff lint and formatting,
and the final frozen-input/trace/artifact integrity verification. All 12 full
training runs and all 48 execution sensitivity evaluations completed.
