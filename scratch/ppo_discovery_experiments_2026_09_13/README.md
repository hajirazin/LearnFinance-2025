# PPO discovery research experiments (2026-09-13)

Research only. This directory is not part of `POST /train/ppo-discovery/full`,
Temporal monthly training, or `POST /train/ppo-discovery/promote`.

Production trains seed 42, evaluates the candidate once on the test split, and
promotes only when test CAGR > 12% and (if an incumbent exists) both test CAGR
and test Sharpe are strictly greater than the incumbent.

A winning scratch run does not write `current`. It licenses a later production
code change, after which a normal train plus those two numeric gates apply.

## What lives here

- Extra seeds `(42, 123, 2026)` via `train_ppo_discovery_seeds`
- Ablation retrains and eval-only ablations (`ablations.py`)
- Matched-K closed loops (`matched_k.py`)
- Locked-random baseline (`baselines.py`)
- Ablation comparison diagnostics (`diagnostics.py`)

## Run

From the repo root, with `brain_api` on `PYTHONPATH`:

```
python scratch/ppo_discovery_experiments_2026_09_13/run_experiments.py \
  --candidate-dir /path/to/ppo_discovery_halal_new/vDATE-HASH
```

Writes JSON under `results/`. Never calls `promote_version`.
