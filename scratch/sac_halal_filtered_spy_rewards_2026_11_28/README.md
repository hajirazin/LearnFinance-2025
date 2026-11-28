# SAC halal_filtered reward comparison against SPY

Research run: November 28, 2026. This repeats the earlier halal/QQQ experiment
with `halal_filtered` as the stock universe and SPY as the ETF benchmark.
The canonical repository name is `halal_filtered`.

## Protocol

- Freeze the November 2026 cached 15-stock roster: ASX, ATLKY, BDRFY, BURL,
  DLR, FFIV, FICO, ILMN, LEN, LII, MSTR, NVO, NWS, NWSA, SAR.
- Reuse the production SAC v3 actor, critics, training loop, feature scaler,
  train-only HMM, forecasts from existing annual PatchTST snapshots, and
  IBKR Singapore Tiered cost engine at fixed USD 10,000 capital.
- Train current net-log reward, SPY excess-log reward, equal-weight excess-log
  reward, and the SPY hybrid (benchmark return floored at zero).
- Run 50,000 steps for every reward on seeds 42, 123, 2026: 12 independent
  optimization runs, serialized on one CPU thread.
- Keep the same 360 chronological training transitions and 104 held-out
  transitions. Test opens run from November 25, 2024 to November 23, 2026.
- Evaluate every policy deterministically over all 104 weeks without annual
  resets. CAGR and zero-risk-free Sharpe use 52 weekly periods per year.
- Compare with SPY and equal-weight buy-and-hold of the frozen 15 stocks.
  Those benchmarks pay an IBKR entry fee.
- Repeat the four execution-only checks per trained model: charge every delta,
  charge every delta with a true 1% or 2% deadband, and charge every delta plus
  10 bps slippage on stock traded notional. These are 48 evaluations without
  additional training.

Only the reward changes between the matched training variants. Drifted weights,
costs, cash floor, inputs, dates, architecture, and hyperparameters stay fixed.
The model has 13–15 eligible stocks during training and all 15 during testing;
no week is dropped and no missing forecast is replaced by a synthetic value.

## Reward definitions

Let `R` be the current production net-log-wealth reward including its existing
concentration term. Let `b` be actual realized SPY weekly return and `e` be
the return of the independent equal-weight buy-and-hold basket.

| Variant | Training reward |
| --- | --- |
| current | `R` |
| spy_excess | `R - log(1 + b)` |
| equal_weight_excess | `R - log(1 + e)` |
| spy_hybrid | `R - log(1 + max(b, 0))` |

The next week's realized benchmark is used only for historical training reward;
it is never supplied to the policy as a forecast or an input feature. Each
exogenous hurdle preserves ideal unnormalized policy ranking. Neural learning,
reward normalization, clipping, and entropy mechanics can still produce
different finite-training outcomes.

The production cost engine exempts deltas below 0.5 percentage points from
fees while the environment still changes those holdings. This behavior is
preserved in the reward comparison; the execution sensitivity checks explicitly
charge every executed delta and implement actual no-trade bands.

## Limits

This is the current frozen roster applied backward, not a historical monthly
walk-forward reconstruction of `halal_filtered` screening. Selection and
survivorship bias therefore remain. Historical PatchTST inputs also follow the
existing approved `allow_later_revisions` news policy: 41 cells use later
article revisions, recorded as look-ahead bias in `evidence/manifest.json`.
There is one two-year test window and three training seeds. Prices are Yahoo
dividend/split-adjusted OHLCV, including the existing production assumption of
using adjusted price levels for historical share-based fees. Taxes, FX, actual
fills, and time-varying historical fee rates are excluded.

Production code, production artifacts, current pointers, publication, and
trading jobs are outside this experiment. The previous experiment's evidence,
models, and results are retained.

## Reproduce

The shared runner remains in the original experiment folder. Every stage gets
the explicit scenario JSON; this selects both input universe and output folder.
Running shared scripts without `--scenario` retains the original halal/QQQ
defaults. Commands below run from the repository root.

```sh
.venv/bin/python scratch/sac_halal_rewards_2026_11_28/download_prices.py --scenario scratch/sac_halal_filtered_spy_rewards_2026_11_28/scenario.json
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/prepare_data.py --scenario scratch/sac_halal_filtered_spy_rewards_2026_11_28/scenario.json
PYTHONPATH=brain_api .venv/bin/python -u scratch/sac_halal_rewards_2026_11_28/run_experiment.py --scenario scratch/sac_halal_filtered_spy_rewards_2026_11_28/scenario.json
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/execution_sensitivity.py --scenario scratch/sac_halal_filtered_spy_rewards_2026_11_28/scenario.json
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/summarize.py --scenario scratch/sac_halal_filtered_spy_rewards_2026_11_28/scenario.json
PYTHONPATH=brain_api .venv/bin/python scratch/sac_halal_rewards_2026_11_28/verify_results.py --scenario scratch/sac_halal_filtered_spy_rewards_2026_11_28/scenario.json
```

Preparation uses read-only news connections and existing local snapshots;
training forbids network access. Data downloads are explicit experiment
operations, separate from offline synthetic tests. Completed seeds are reused
only when their frozen input hash and configuration match.

See `results_50000/report.md` for the completed comparison.

## One-shot background completion

`run_all.py` runs the prepared experiment through training, all 48 execution
checks, summary generation, and final verification without further interaction.
Its optional `--existing-pid PID` adopts the already-running trainer by waiting
for it to exit, then reuses completed seeds. A controller lock prevents duplicate
one-shot controllers. Frozen input hashes are checked before work begins.

`pipeline.log` records stage progress, `run.log` records training progress, and
`status.json` records the current phase or any terminal error. Successful
completion ends with `ALL DONE` in `pipeline.log` and `phase: complete` in
`status.json`. No production artifact is promoted.
