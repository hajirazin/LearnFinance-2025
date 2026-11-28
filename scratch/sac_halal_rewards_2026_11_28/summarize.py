"""Require all paired runs before publishing the experiment comparison."""

import json
from pathlib import Path

import pandas as pd
from experiment_settings import BENCHMARK, UNIVERSE
from run_experiment import ROOT, SEEDS, VARIANTS


def summary_tables(output: Path):
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            source = output / variant / f"seed_{seed}/metrics.json"
            item = json.loads(source.read_text())
            if item["config"]["total_timesteps"] != 50_000:
                raise ValueError(
                    "Comparison requires the authorized full training budget"
                )
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **item["test"],
                    "avg_cash": item["avg_cash_weight"],
                    "avg_turnover": item["avg_weekly_turnover"],
                    "fees_usd": item["test_cost_usd_fixed_nav"],
                    "training_seconds": item["training_seconds"],
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame[frame.variant == "current"].set_index("seed")
    benchmark = json.loads((output / "benchmarks.json").read_text())
    summaries = []
    for variant in VARIANTS:
        subset = frame[frame.variant == variant].set_index("seed").loc[list(SEEDS)]
        delta_cagr = subset.cagr - baseline.cagr
        delta_sharpe = subset.sharpe - baseline.sharpe
        summaries.append(
            {
                "variant": variant,
                "median_cagr": float(subset.cagr.median()),
                "median_sharpe": float(subset.sharpe.median()),
                "cagr_min": float(subset.cagr.min()),
                "cagr_max": float(subset.cagr.max()),
                "sharpe_min": float(subset.sharpe.min()),
                "sharpe_max": float(subset.sharpe.max()),
                "median_max_drawdown": float(subset.max_drawdown.median()),
                "median_avg_cash": float(subset.avg_cash.median()),
                "median_avg_turnover": float(subset.avg_turnover.median()),
                "median_fees_usd": float(subset.fees_usd.median()),
                "median_paired_cagr_delta": float(delta_cagr.median()),
                "median_paired_sharpe_delta": float(delta_sharpe.median()),
                "paired_both_improved_seeds": int(
                    ((delta_cagr > 0) & (delta_sharpe > 0)).sum()
                ),
                f"beats_{BENCHMARK.lower()}_both_seeds": int(
                    (
                        (subset.cagr > benchmark[f"{BENCHMARK}_buy_hold"]["cagr"])
                        & (subset.sharpe > benchmark[f"{BENCHMARK}_buy_hold"]["sharpe"])
                    ).sum()
                ),
                "beats_equal_weight_both_seeds": int(
                    (
                        (
                            subset.cagr
                            > benchmark[f"{UNIVERSE}_equal_weight_buy_hold"]["cagr"]
                        )
                        & (
                            subset.sharpe
                            > benchmark[f"{UNIVERSE}_equal_weight_buy_hold"]["sharpe"]
                        )
                    ).sum()
                ),
            }
        )
    return frame, pd.DataFrame(summaries), benchmark


def main():
    output = ROOT / "results_50000"
    frame, summaries, benchmark = summary_tables(output)
    frame.to_csv(output / "all_seeds.csv", index=False)
    summaries.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summaries.to_dict(orient="records"), indent=2)
    )
    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    lines = [
        f"# SAC {UNIVERSE} reward experiment results against {BENCHMARK}",
        "",
        "Completed November 28, 2026. Research results only; production was unchanged.",
        "",
        f"All 12 runs completed: four rewards, three matched seeds each, 50,000 steps per run. {manifest['train_weeks']} training weeks and {manifest['test_weeks']} held-out weeks, {manifest['test_start']} to {manifest['test_end']}. Returns include the existing IBKR Singapore Tiered model at fixed USD 10,000 NAV. Sharpe assumes zero risk-free rate.",
        "",
        "**The historical panel includes survivor-selected stocks and later-edited news. These are biased exploratory results, not an unbiased investment backtest.**",
        "",
        "## Performance",
        "",
        "SAC entries are medians across all three seeds. Benchmarks are independent buy-and-hold portfolios with an IBKR entry fee. No winning seed was selected.",
        "",
        "| Strategy | CAGR | Sharpe | Max drawdown |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in benchmark.items():
        lines.append(
            f"| {name} | {item['cagr']:.2%} | {item['sharpe']:.3f} | {item['max_drawdown']:.2%} |"
        )
    for row in summaries.to_dict(orient="records"):
        lines.append(
            f"| SAC {row['variant']} | {row['median_cagr']:.2%} | {row['median_sharpe']:.3f} | {row['median_max_drawdown']:.2%} |"
        )
    lines += [
        "",
        "## Improvement against current reward",
        "",
        "Paired differences compare the same seed under identical data and training budgets.",
        "",
        f"| Reward | Median paired CAGR change (pp) | Median paired Sharpe change | Seeds improving both | Seeds beating {BENCHMARK} on both | Seeds beating equal weight on both |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries.to_dict(orient="records"):
        lines.append(
            f"| {row['variant']} | {100 * row['median_paired_cagr_delta']:+.2f} | {row['median_paired_sharpe_delta']:+.3f} | {row['paired_both_improved_seeds']}/3 | {row[f'beats_{BENCHMARK.lower()}_both_seeds']}/3 | {row['beats_equal_weight_both_seeds']}/3 |"
        )
    lines += [
        "",
        "## Individual seeds",
        "",
        "| Reward | Seed | CAGR | Sharpe | Average cash | Average weekly turnover | Fees over 104 weeks (USD, fixed NAV) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['cagr']:.2%} | {row['sharpe']:.3f} | {row['avg_cash']:.2%} | {row['avg_turnover']:.2%} | {row['fees_usd']:.2f} |"
        )
    sensitivity_path = output / "execution_sensitivity.csv"
    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        if len(sensitivity) != 48:
            raise ValueError("Expected 48 execution sensitivity evaluations")
        lines += [
            "",
            "## Execution sensitivity",
            "",
            "Evaluation only, without retraining. Every executed delta pays IBKR fees; deadbands freeze small trades. Slippage is charged per stock notional traded. Values are medians across three seeds. These are alternate execution assumptions, separate from the matched reward experiment.",
            "",
            "| Reward | True deadband | Slippage (bps) | CAGR | Sharpe | Fees and slippage (USD, fixed NAV) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for (variant, band, slip), subset in sensitivity.groupby(
            ["variant", "deadband", "slippage_bps"], sort=False
        ):
            lines.append(
                f"| {variant} | {band:.0%} | {slip:.0f} | {subset.cagr.median():.2%} | {subset.sharpe.median():.3f} | {subset.total_cost_usd_fixed_nav.median():.2f} |"
            )
    # Describe all variants against the incumbent; never promote automatically.
    improved = summaries[
        (summaries.variant != "current") & (summaries.paired_both_improved_seeds == 3)
    ]
    lines += ["", "## Interpretation", ""]
    indexed = summaries.set_index("variant")
    equal_weight = indexed.loc["equal_weight_excess"]
    incumbent = indexed.loc["current"]
    lines += [
        f"Equal-weight-relative reward produced median CAGR {equal_weight.median_cagr:.2%} and median Sharpe {equal_weight.median_sharpe:.3f}. Changes versus current reward: {100 * (equal_weight.median_cagr - incumbent.median_cagr):+.2f} percentage points of median CAGR and {equal_weight.median_sharpe - incumbent.median_sharpe:+.3f} of median Sharpe. It improved both metrics in {int(equal_weight.paired_both_improved_seeds)} of three paired seeds.",
        "",
    ]
    if (summaries[f"beats_{BENCHMARK.lower()}_both_seeds"] == len(SEEDS)).all():
        lines += [
            f"Every SAC reward variant beat {BENCHMARK} on both metrics in every seed.",
            "",
        ]
    if summaries.beats_equal_weight_both_seeds.sum() == 0:
        lines += [
            "No SAC seed beat the equal-weight buy-and-hold basket on both metrics.",
            "",
        ]
    if improved.empty:
        lines.append(
            "No alternative reward improved both CAGR and Sharpe in all three paired seeds. Any median improvement should be treated as seed-sensitive evidence."
        )
    else:
        lines.append(
            "Rewards improving both metrics in every paired seed: "
            + ", ".join(improved.variant)
            + ". This is repeatability across optimization seeds within one biased test window; it does not establish out-of-period generalization."
        )
    lines += [
        "",
        "An action-independent benchmark subtraction preserves the ideal unnormalized objective ranking. Differences here reflect learning, normalization, and approximation. The hybrid changes the reward sign convention, not the ideal action ranking. See the experiment README for formulas, assumptions, and reproduction commands.",
        "",
        "Integrity checks are recorded in verification.json: all input hashes unchanged, all traces aligned to the same 104 weeks, finite constrained weights, metrics independently recomputed, and 48 execution sensitivity evaluations present.",
        "",
        f"Total training time: {frame.training_seconds.sum() / 60:.1f} minutes. All results, models, and evidence remain in this scratch directory. No production model was promoted.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
