"""Evaluate loser avoidance and stock-signal predictability without SAC retrains."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from diagnostic_math import (
    action_bounds,
    bad_stock_share,
    bounded_oracle_return,
    conditional_bootstrap,
    reward_gap,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from brain_api.core.portfolio_rl.state import cross_sectional_rank

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SEEDS = (42, 123, 2026)
EXPERIMENTS = (
    ("halal", "QQQ", "sac_halal_rewards_2026_11_28"),
    ("halal_filtered", "SPY", "sac_halal_filtered_spy_rewards_2026_11_28"),
)


def load_panel(folder):
    panel = pickle.loads((folder / "evidence/panel.pkl").read_bytes())
    data = panel["data"]
    start = data.n_weeks - 104
    return panel, data, start


def classifier_diagnostic(panel, data, start, universe):
    raw = np.concatenate((data.patchtst_forecasts[:, :, None], data.signals), axis=2)
    features = np.zeros_like(raw)
    for week in range(data.n_weeks):
        valid = data.asset_masks[week]
        for feature in range(raw.shape[2]):
            features[week, valid, feature] = cross_sectional_rank(
                raw[week, valid, feature]
            )
    valid_train = data.asset_masks[:start].reshape(-1)
    valid_test = data.asset_masks[start:].reshape(-1)
    train_x = features[:start].reshape(-1, 6)[valid_train]
    test_x = features[start:].reshape(-1, 6)[valid_test]
    spy = np.asarray(panel["spy_returns"])
    output = []
    for label, labels in (
        ("negative_next_week", data.symbol_returns < 0),
        ("underperform_SPY_next_week", data.symbol_returns < spy[:, None]),
    ):
        train_y = labels[:start].reshape(-1)[valid_train]
        test_y = labels[start:].reshape(-1)[valid_test]
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            max_leaf_nodes=7,
            learning_rate=0.05,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        with threadpool_limits(limits=1):
            model.fit(train_x, train_y)
            probabilities = model.predict_proba(test_x)[:, 1]
        forecast_scores = -features[start:, :, 0].reshape(-1)[valid_test]
        week_ids = np.repeat(np.arange(104), data.n_stocks)[valid_test]
        weekly_auc, forecast_auc = [], []
        for week in range(104):
            indices = week_ids == week
            if len(np.unique(test_y[indices])) == 2:
                weekly_auc.append(
                    roc_auc_score(test_y[indices], probabilities[indices])
                )
                forecast_auc.append(
                    roc_auc_score(test_y[indices], forecast_scores[indices])
                )
        output.append(
            {
                "universe": universe,
                "label": label,
                "train_samples": len(train_y),
                "test_samples": len(test_y),
                "test_auc": float(roc_auc_score(test_y, probabilities)),
                "mean_within_week_auc": float(np.mean(weekly_auc)),
                "patchtst_rank_within_week_auc": float(np.mean(forecast_auc)),
                "weeks_with_both_classes": len(weekly_auc),
                "brier": float(brier_score_loss(test_y, probabilities)),
                "constant_train_rate_brier": float(
                    brier_score_loss(test_y, np.full(len(test_y), train_y.mean()))
                ),
            }
        )
        symbols = np.tile(data.symbol_order, 104)[valid_test]
        pd.DataFrame(
            {
                "week": week_ids,
                "symbol": symbols,
                "bad_label": test_y,
                "predicted_bad_probability": probabilities,
            }
        ).to_csv(ROOT / f"{universe}_{label}_predictions.csv", index=False)
    return output


def analyze_experiment(universe, benchmark, directory):
    folder = REPO / "scratch" / directory
    panel, data, start = load_panel(folder)
    actual = data.symbol_returns[start:]
    losing = actual < 0
    underperforming = actual < np.asarray(panel["spy_returns"])[start:, None]
    negative_count = losing.sum(axis=1)
    relative_count = underperforming.sum(axis=1)
    regimes = {
        "all_weeks": np.ones(104, dtype=bool),
        "5_or_6_negative_stocks": (negative_count >= 5) & (negative_count <= 6),
        "majority_negative_stocks": negative_count > data.n_stocks / 2,
        "5_or_6_underperform_SPY": (relative_count >= 5) & (relative_count <= 6),
        "majority_underperform_SPY": relative_count > data.n_stocks / 2,
        "SPY_down": np.asarray(panel["spy_returns"])[start:] < 0,
    }
    rows, paired = [], []
    traces = {}
    for variant in (
        "current",
        f"{benchmark.lower()}_excess",
        "equal_weight_excess",
        f"{benchmark.lower()}_hybrid",
    ):
        for seed in SEEDS:
            trace = pd.read_csv(
                folder / f"results_50000/{variant}/seed_{seed}/test_returns.csv"
            )
            if list(trace.session) != panel["sessions"][start:-1]:
                raise ValueError("Trace dates do not match the frozen panel")
            weights = trace[
                [f"weight_{symbol}" for symbol in data.symbol_order]
            ].to_numpy()
            bad_share, equal_share = bad_stock_share(weights, losing)
            relative_share, equal_relative = bad_stock_share(weights, underperforming)
            trace["negative_weight_gap"] = 100 * (bad_share - equal_share)
            trace["relative_weight_gap"] = 100 * (relative_share - equal_relative)
            traces[variant, seed] = trace
            for regime, mask in regimes.items():
                if not mask.any():
                    continue
                rows.append(
                    {
                        "universe": universe,
                        "variant": variant,
                        "seed": seed,
                        "regime": regime,
                        "weeks": int(mask.sum()),
                        "mean_weekly_net_return_pct": float(
                            trace.net_return[mask].mean() * 100
                        ),
                        "mean_cash_pct": float(trace.cash[mask].mean() * 100),
                        "negative_stock_weight_gap_pp": float(
                            trace.negative_weight_gap[mask].mean()
                        ),
                        "underperform_SPY_weight_gap_pp": float(
                            trace.relative_weight_gap[mask].mean()
                        ),
                    }
                )
    comparison = f"{benchmark.lower()}_excess"
    for regime, mask in regimes.items():
        for metric, scale in (
            ("net_return", 100),
            ("negative_weight_gap", 1),
            ("relative_weight_gap", 1),
        ):
            deltas = np.mean(
                [
                    scale
                    * (
                        traces[comparison, seed][metric].to_numpy()
                        - traces["equal_weight_excess", seed][metric].to_numpy()
                    )
                    for seed in SEEDS
                ],
                axis=0,
            )
            paired.append(
                {
                    "universe": universe,
                    "comparison": f"{benchmark}_reward_minus_EW_reward",
                    "regime": regime,
                    "metric": metric,
                    **conditional_bootstrap(deltas, mask),
                }
            )
    classifiers = classifier_diagnostic(panel, data, start, universe)
    oracle = np.array([bounded_oracle_return(row) for row in actual])
    unrestricted = np.maximum(0, 0.98 * actual.max(axis=1))
    pd.DataFrame(
        {
            "session": panel["sessions"][start:-1],
            "SPY_gross_return": np.asarray(panel["spy_returns"])[start:],
            "bounded_actor_hindsight_gross_ceiling": oracle,
            "full_simplex_hindsight_gross_ceiling": unrestricted,
        }
    ).to_csv(ROOT / f"{universe}_opportunity_ceiling.csv", index=False)
    return rows, paired, classifiers


def main():
    rows, paired, classifiers = [], [], []
    for experiment in EXPERIMENTS:
        new_rows, new_paired, new_classifiers = analyze_experiment(*experiment)
        rows.extend(new_rows)
        paired.extend(new_paired)
        classifiers.extend(new_classifiers)
    pd.DataFrame(rows).to_csv(ROOT / "regime_diagnostics.csv", index=False)
    pd.DataFrame(paired).to_csv(ROOT / "paired_block_bootstrap.csv", index=False)
    pd.DataFrame(classifiers).to_csv(ROOT / "signal_classification.csv", index=False)
    result = {
        "action_bounds_15_stocks": {str(k): action_bounds(15, k) for k in (1, 5, 6)},
        "reward_gap_example": {
            "EW_minus_4pct": reward_gap(0.02, 0, -0.04),
            "SPY_plus_1pct": reward_gap(0.02, 0, 0.01),
        },
        "classifiers": classifiers,
        "paired_comparisons": paired,
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(pd.DataFrame(classifiers).to_string(index=False))
    print(
        pd.DataFrame(paired)
        .query(
            "universe == 'halal_filtered' and regime in ['all_weeks', '5_or_6_negative_stocks', 'majority_negative_stocks']"
        )
        .to_string(index=False)
    )
    print(json.dumps(result["action_bounds_15_stocks"], indent=2))


if __name__ == "__main__":
    main()
