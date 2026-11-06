"""Summarize both experiments and verify that production evidence is unchanged."""

import json

import numpy as np
import pandas as pd
import torch

from research_support import ROOT, protected_hashes, write_json


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    controlled = load("controlled_summary.json")
    noisy = load("controlled_noisy_holdout.json")
    market = load("market_test_summary.json")
    weeks = load("market_test_weekly.json")
    manifest = load("market_data_manifest.json")
    assert len(weeks) == 36
    visible = [r for r in controlled["runs"] if r["signal_visible"]]
    hidden = [r for r in controlled["runs"] if not r["signal_visible"]]
    assert len(visible) == 6 and len(hidden) == 3
    assert all(r["final"]["passes"] for r in visible)
    assert all(r["ci95"][0] > 0 for r in noisy if r["case"] != "set_hidden_signal")
    assert all(
        abs(r["mean_paired_reward_advantage"]) < 1e-6
        for r in noisy
        if r["case"] == "set_hidden_signal"
    )
    for arm in ["ppo", "equal_fixed", "equal_feedback"]:
        logs = np.array([w[arm]["net_log"] for w in weeks])
        recomputed = float(np.expm1(logs.mean() * 52))
        assert abs(recomputed - market["performance"][arm]["cagr"]) < 1e-12
        market["performance"][arm]["period_total_return"] = float(np.expm1(logs.sum()))
    cvs = [r["cv"] for r in weeks if r["cv"] is not None]
    stocks = []
    for row in weeks:
        assert abs(sum(row["weights"].values()) - 1) < 1e-6
        stock_mass = 1 - row["cash"]
        for rank, symbol in enumerate(row["selection_order"], 1):
            stocks.append(
                {
                    "decision_open": row["decision_open"],
                    "symbol": symbol,
                    "selection_rank": rank,
                    "portfolio_weight": row["weights"][symbol],
                    "equity_weight": row["weights"][symbol] / stock_mass,
                    "relative_deviation": row["k"] * row["weights"][symbol] / stock_mass
                    - 1,
                }
            )
    pd.DataFrame(stocks).to_csv(ROOT / "market_stock_weights.csv", index=False)
    final = torch.load(
        ROOT / "market_seed42_final.pt", map_location="cpu", weights_only=True
    )
    initial = torch.load(
        ROOT / "market_pretrained.pt", map_location="cpu", weights_only=True
    )
    parameter_change = {
        prefix: float(
            torch.sqrt(
                sum(
                    (final[k] - initial[k]).float().square().sum()
                    for k in final
                    if k.startswith(prefix)
                )
            )
        )
        for prefix in [
            "allocation_head.",
            "selection_head.",
            "cash_head.",
            "temporal.",
            "set_encoder.",
        ]
    }
    before = load("protected_before.json")
    after = protected_hashes()
    write_json("protected_after.json", after)
    changed = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
    assert not changed, f"Protected files changed: {changed}"
    result = {
        "controlled": controlled,
        "controlled_noisy_holdout": noisy,
        "market": market,
        "market_weight_dispersion": {
            "mean_cv": None if not cvs else float(np.mean(cvs)),
            "max_cv": None if not cvs else float(np.max(cvs)),
            "informative_weeks": len(cvs),
        },
        "parameter_l2_changes": parameter_change,
        "protected_files_unchanged": True,
        "protected_file_count": len(before),
        "real_market_training_seeds": [42],
        "outer_training_cutoff": "2025-12-31",
        "retrospective_holdout": True,
    }
    write_json("summary.json", result)
    rows = []
    for r in controlled["runs"]:
        f = r["final"]
        rows.append(
            f"| {r['case']} | {r['seed']} | {100 * f['mean_winner_equity_weight']:.2f}% | {100 * f['mean_cv']:.2f}% | {100 * f['fraction_feasible_gain']:.1f}% |"
        )
    market_rows = []
    for name in ["ppo", "equal_fixed", "equal_feedback"]:
        m = market["performance"][name]
        market_rows.append(
            f"| {name} | {100 * m['period_total_return']:.2f}% | {100 * m['cagr']:.2f}% | {m['sharpe']:.3f} | {100 * m['max_drawdown']:.2f}% |"
        )
    difference = market["paired_weekly_net_log_difference"]["4"]
    lo, hi = difference["ci95"]
    conclusion = (
        "The allocation increment is not distinguishable from zero in this sample."
        if lo <= 0 <= hi
        else "The four-week bootstrap interval excludes zero; this remains a single-seed retrospective result."
    )
    cv_text = (
        "not measurable because no multi-stock decisions occurred"
        if not cvs
        else f"{100 * np.mean(cvs):.2f}%"
    )
    text = f"""# Can PPO learn useful unequal allocations?

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
{chr(10).join(rows)}

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

Outer training-data cutoff: December 31, 2025. Train on {manifest["train_weeks"]}
weeks from {manifest["train_first_actor"]} through actor cutoff
{manifest["train_last_actor"]}. The final training reward ends
{manifest["last_training_outcome"]}. HMM fitting stops at {manifest["hmm_cutoff"]}.
News and channel scalers use training weeks only. Encoder pretraining uses
its internal pre-2026 validation; the full PPO uses its fixed final-budget
checkpoint. No 2026 return chooses a checkpoint, seed or hyperparameter.

This is a new full seed-42 model from random initialization, with the normal
production architecture and current source defaults: not a continued fit of
the promoted model. Stage A stopped at epoch 13, with best internal validation
Rank IC about 0.02575. Stage B completed {int(market["training"]["timesteps"])}
sampled weekly decisions; the native trainer finishes its last full episode
after the 10,000 minimum budget. Temporal freeze/unfreeze and PPO loss follow
the production trainer. These are repeated passes over 312 fixed historical
transitions, not thousands of additional independent market weeks.
Raw features are cached to remove repeated data
construction, and native/cached feature and closed-loop parity checks pass.

Test {manifest["test_weeks"]} decisions from {manifest["test_first_decision_open"]}
through {manifest["test_last_decision_open"]}, with returns through
{manifest["test_last_outcome"]}. September 7 is the latest fully covered news
window for the entire roster; Labor Day moves its execution open to September
8. Later windows have coverage for only 26 of 432 names and are not substituted
with zero news. All test policies use the same prices and cost contract.

| Allocation | Test-period total return | Annualized CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(market_rows)}

`equal_fixed` retains each PPO decision's exact selected names, count and cash;
only the stock proportions are replaced. Both books carry their own drifted,
after-cost weights. `equal_feedback` evolves its own equal-weight book and
feeds that back to the policy, so later names or cash may also change.

Mean stock-weight CV: {cv_text}, on {len(cvs)} informative multi-stock weeks.
Count distribution: {market["k_counts"]}. Mean cash:
{100 * market["mean_cash"]:.2f}%. The allocation head's parameter L2 change from
Stage A to the final PPO is {parameter_change["allocation_head."]:.6f}; it was
actually updated rather than accidentally frozen.

PPO minus fixed-selection equal-weight CAGR:
{100 * market["cagr_difference_vs_equal"]:+.4f} percentage points.
Mean paired weekly net-log increment: {difference["mean"] * 10000:+.4f} basis
points, four-week-block 95% interval [{lo * 10000:+.4f}, {hi * 10000:+.4f}] basis
points per week. Eight- and 13-week-block sensitivity is in the summary.
{conclusion}

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
confirm {len(before)} protected production source, Temporal, PPO model/current and
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
"""
    (ROOT / "README.md").write_text(text)
    print(
        json.dumps(
            {
                "controlled_visible_passes": sum(r["final"]["passes"] for r in visible),
                "market": market["performance"],
                "cagr_increment": market["cagr_difference_vs_equal"],
                "cv": result["market_weight_dispersion"],
                "production_unchanged": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
