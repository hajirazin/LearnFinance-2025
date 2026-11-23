"""Extend the feature-return study to OHLCV and compare seven versus ten inputs."""

import hashlib
import importlib.metadata
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_feature_relationships import (
    LABELS,
    PERIODS,
    SEVEN,
    SOURCE,
    ablation_estimates,
    fit_ablation,
    load_panel,
    markdown_table,
    period_panel,
    summarize_relationships,
)
from build_friday_dataset import OHLCV_FEATURES, plt, weekly_spearman
from feature_relationship_plots import draw_figures
from feature_relationships import (
    close_cost_diagnostic,
    portfolio_observations,
    weekly_relationships,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/feature_relationships_ohlcv"
FIGURES = ROOT / "figures/feature_relationships_ohlcv"
REPORT = ROOT / "V1_FEATURE_RELATIONSHIPS_OHLCV.md"
TEN = tuple(dict.fromkeys((*SEVEN, *OHLCV_FEATURES)))
NAMES = {**LABELS, **OHLCV_FEATURES}
ARMS = {
    "all_ten": TEN,
    "all_seven": SEVEN,
    **{f"without_{f}": tuple(x for x in TEN if x != f) for f in TEN},
}


def compare_bundle(scored, costs):
    """Summarize identical paired weeks for the seven- and ten-feature arms."""
    rows = []
    for horizon, observations in scored.groupby("horizon"):
        for metric in ("mae", "ic", "basket_return"):
            matched = observations.pivot(index="friday", columns="arm", values=metric)[
                ["all_seven", "all_ten"]
            ].dropna()
            rows.append(
                {
                    "horizon": horizon,
                    "metric": metric,
                    "weeks": len(matched),
                    "seven": matched.all_seven.mean(),
                    "ten": matched.all_ten.mean(),
                    "ten_minus_seven": (matched.all_ten - matched.all_seven).mean(),
                }
            )
    matched = costs.pivot(index="friday", columns="arm", values="net_return")[
        ["all_seven", "all_ten"]
    ].dropna()
    rows.append(
        {
            "horizon": "weekly",
            "metric": "net_10bp_diagnostic",
            "weeks": len(matched),
            "seven": matched.all_seven.mean(),
            "ten": matched.all_ten.mean(),
            "ten_minus_seven": (matched.all_ten - matched.all_seven).mean(),
        }
    )
    return pd.DataFrame(rows)


def price_overlap(panel):
    rows = []
    for period, (first, last) in PERIODS.items():
        subset = period_panel(panel, first, last)
        for left, right in combinations(("open", "high", "low", "close"), 2):
            observations = weekly_spearman(subset, left, right)
            rows.append(
                {
                    "period": period,
                    "left": left,
                    "right": right,
                    "mean_weekly_spearman": observations.spearman.mean(),
                    "weeks": len(observations),
                }
            )
    return pd.DataFrame(rows)


def draw_comparison(bundle, effects):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, metric in zip(axes, ("mae", "basket_return"), strict=True):
        rows = (
            effects.loc[
                (effects.feature == "added_ohlcv")
                & (effects.block == 4)
                & (effects.metric == metric)
            ]
            .set_index("horizon")
            .reindex(["daily", "weekly"])
        )
        ax.hlines(
            range(2), rows.ci_low * 100, rows.ci_high * 100, color="#0c7c86", lw=3
        )
        ax.scatter(rows["mean"] * 100, range(2), color="#0c7c86", s=50)
        ax.axvline(0, color="#999999", lw=1)
        ax.set_yticks(range(2), ["Next session", "Next Friday"])
        ax.invert_yaxis()
        ax.set(
            title="Stock MAE improvement"
            if metric == "mae"
            else "Top-15 gross return improvement",
            xlabel="Percentage points; positive favors ten inputs",
        )
    fig.suptitle(
        "Does adding open, high, and low help?\n2025 paired estimates with 95% intervals",
        fontsize=16,
    )
    fig.savefig(FIGURES / "seven_vs_ten.png", dpi=160)
    plt.close(fig)


def write_report(summary, estimates, effects, bundle, overlap):
    primary = estimates.loc[estimates.block.eq(4)]
    table = []
    for feature in TEN:
        row = {"Feature": NAMES[feature]}
        for horizon in ("daily", "weekly"):
            result = primary.loc[
                (primary.feature == feature)
                & (primary.horizon == horizon)
                & (primary.period == "validation")
                & (primary.metric == "ic")
            ].iloc[0]
            row[f"2025 {horizon} IC [95% CI]"] = (
                f"{result['mean']:+.3f} [{result.ci_low:+.3f}, {result.ci_high:+.3f}]"
            )
            row[f"{horizon} Holm p"] = f"{result.p_holm:.3f}"
        table.append(row)
    comparison = []
    for item in bundle.itertuples(index=False):
        if item.metric == "ic":
            unit, factor = "IC", 1
        else:
            unit, factor = "pp", 100
        comparison.append(
            {
                "Horizon": item.horizon,
                "Metric": item.metric,
                "Matched weeks": item.weeks,
                "Seven (pp except IC)": f"{item.seven * factor:.4f}",
                "Ten (pp except IC)": f"{item.ten * factor:.4f}",
                "Ten minus seven": f"{item.ten_minus_seven * factor:+.4f} {unit}",
            }
        )
    inference = []
    for item in effects.loc[
        effects.block.eq(4) & effects.feature.eq("added_ohlcv")
    ].itertuples(index=False):
        inference.append(
            {
                "Horizon": item.horizon,
                "Metric": item.metric,
                "Weeks": int(item.weeks),
                "Improvement from adding OHL, pp [95% CI]": f"{item.mean * 100:+.4f} [{item.ci_low * 100:+.4f}, {item.ci_high * 100:+.4f}]",
                "Holm p": f"{item.p_holm:.4f}",
            }
        )
    relation_hits = int(primary.p_holm.lt(0.05).sum())
    ablation_hits = int(effects.loc[effects.block.eq(4), "p_holm"].lt(0.05).sum())
    overlap_close = overlap.loc[
        overlap.period.eq("validation") & overlap.right.eq("close")
    ]
    text = f"## Results\n\nThere are **ten distinct features**, since close and volume were already included in the original seven. The comparison covers {summary['coverage']['stocks']} archived stocks and {summary['coverage']['common_feature_rows']:,} complete stock-Friday rows, with {summary['invalid_ohlc_rows']} inconsistent OHLC rows excluded at the feature cutoff.\n\n"
    text += f"At the prespecified four-week block length, {relation_hits} of 120 relationship tests and {ablation_hits} of 44 incremental-value tests pass Holm at 5%. Inspect the effects and intervals below when judging practical value. All results remain exploratory.\n\n"
    weekly_effect = effects.loc[
        effects.block.eq(4)
        & effects.feature.eq("added_ohlcv")
        & effects.horizon.eq("weekly")
        & effects.metric.eq("basket_return")
    ].iloc[0]
    text += f"### Conclusion\n\nAdding open, high, and low changes mean weekly gross basket return by {weekly_effect['mean'] * 100:+.4f} pp, with a 95% interval of [{weekly_effect.ci_low * 100:+.4f}, {weekly_effect.ci_high * 100:+.4f}] pp. "
    if weekly_effect.ci_low <= 0 <= weekly_effect.ci_high:
        text += (
            "This interval leaves the direction of the incremental return uncertain. "
        )
    text += "Raw OHLC levels carry strongly overlapping cross-sectional information. A feature set for deployment must earn its place through the executable V1 comparison with the incumbent and SPY.\n\n"
    text += (
        "### Individual feature relationships\n\n"
        + markdown_table(pd.DataFrame(table))
        + "\n\n![Ten features versus future daily and weekly returns](figures/feature_relationships_ohlcv/information_coefficients.png)\n\n"
    )
    text += (
        "### Seven versus ten inputs on matched weeks\n\n"
        + markdown_table(pd.DataFrame(comparison))
        + "\n\nMAE is lower when forecasts are more accurate. Rank IC and basket return increase when the scored ordering performs better. Gross and cost-adjusted basket means use the intersection of complete baskets for the two models.\n\n"
    )
    text += (
        markdown_table(pd.DataFrame(inference))
        + "\n\nPositive improvement favors ten inputs. For MAE this is `seven MAE - ten MAE`. For basket return it is `ten return - seven return`.\n\n![Adding OHLC prices to the original seven inputs](figures/feature_relationships_ohlcv/seven_vs_ten.png)\n\n"
    )
    text += (
        "### OHLC price overlap\n\n"
        + markdown_table(
            overlap_close[["left", "right", "mean_weekly_spearman", "weeks"]].round(6)
        )
        + "\n\nThese are contemporaneous feature-to-feature correlations. Strong overlap limits the distinct information in raw nominal OHLC levels. The seven-versus-ten bundle comparison tests their combined incremental value.\n\n"
    )
    text += "### Chart gallery\n\n![Feature groups and subsequent returns](figures/feature_relationships_ohlcv/feature_groups.png)\n\n![Yearly stability](figures/feature_relationships_ohlcv/yearly_ic.png)\n\n![Leave-one-feature-out comparisons](figures/feature_relationships_ohlcv/ablation.png)\n\n"
    text += "### Reproducibility and limitations\n\nRun `PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/analyze_ohlcv_relationships.py` from the repository root. The [manifest](data/feature_relationships_ohlcv/summary.json) records source and code hashes, environment versions, common-row coverage, and the protocol hash. [Relationship estimates](data/feature_relationships_ohlcv/relationship_estimates.csv), [incremental-value estimates](data/feature_relationships_ohlcv/ablation_estimates.csv), and [paired seven-versus-ten metrics](data/feature_relationships_ohlcv/seven_vs_ten.csv) contain the numeric results. The data directory also stores forecasts, monthly folds, standardized coefficients, selected stocks, weekly observations, group means, and omitted basket outcomes.\n\n"
    text += "Historical feature selection and the frozen roster limit generalization. Daily means the next exchange session after Friday. Snapshot Ridge measures linear predictive value, while PatchTST uses a sequence. The Friday-close basket returns and the 10-bp cost sensitivity assume hypothetical entry at the feature cutoff and omit IBKR order minima and realistic execution. This experiment informs candidate selection. Executable V1 performance versus SPY requires the separate V1 protocol.\n"
    REPORT.write_text(REPORT.read_text().split("## Results\n", 1)[0] + text)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel, close, coverage = load_panel(TEN)
    prices = panel[["open", "high", "low", "close"]]
    good = (
        prices.gt(0).all(axis=1)
        & prices.low.le(prices[["open", "close"]].min(axis=1) + 1e-8)
        & prices.high.ge(prices[["open", "close"]].max(axis=1) - 1e-8)
    )
    invalid = int((~good).sum())
    panel = panel.loc[good]
    coverage["common_feature_rows"] = len(panel)
    weekly, groups = [], []
    for period, (first, last) in PERIODS.items():
        observations, group_returns = weekly_relationships(
            period_panel(panel, first, last), TEN
        )
        weekly.append(observations.assign(period=period))
        groups.append(group_returns.assign(period=period))
    weekly, groups = pd.concat(weekly), pd.concat(groups)
    estimates, group_estimates = summarize_relationships(weekly, groups)
    predictions, folds, coefficients = fit_ablation(panel, ARMS)
    observed, selected = [], []
    for horizon in ("daily", "weekly"):
        rows, picks = portfolio_observations(
            predictions.loc[predictions.horizon.eq(horizon)], ARMS, horizon
        )
        observed.append(rows)
        selected.append(picks)
    observed, picks = pd.concat(observed), pd.concat(selected)
    effects = ablation_estimates(
        observed, features=TEN, full="all_ten", extra={"added_ohlcv": "all_seven"}
    )
    costs = close_cost_diagnostic(picks, close)
    bundle = compare_bundle(observed, costs)
    overlap = price_overlap(panel)
    frames = {
        "relationships_weekly": weekly,
        "groups_weekly": groups,
        "relationship_estimates": estimates,
        "group_estimates": group_estimates,
        "folds": folds,
        "coefficients": coefficients,
        "ablation_weekly": observed,
        "picks": picks,
        "ablation_estimates": effects,
        "close_cost_diagnostic": costs,
        "seven_vs_ten": bundle,
        "price_overlap": overlap,
        "missing_basket_outcomes": observed.loc[observed.basket_return.isna()],
    }
    for name, frame in frames.items():
        frame.to_csv(OUTPUT / f"{name}.csv", index=False)
    predictions.to_parquet(OUTPUT / "stock_forecasts.parquet")
    files = [
        Path(__file__),
        ROOT / "analyze_feature_relationships.py",
        ROOT / "feature_relationships.py",
        ROOT / "feature_relationship_plots.py",
        ROOT / "test_feature_relationships.py",
        SOURCE / "friday_dataset.parquet",
        SOURCE / "alpaca_sip_evidence.json",
        SOURCE.parent / "daily_ohlcv_2016_2025.csv",
        SOURCE.parents[1] / "build_friday_dataset.py",
        SOURCE.parents[1] / "verify_friday_prices.py",
        SOURCE.parents[1] / "compare_feature_sets.py",
        SOURCE.parents[1] / "audit_data.py",
    ]
    summary = {
        "classification": "exploratory OHLCV feature study",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "features": list(TEN),
        "arms": ARMS,
        "coverage": coverage,
        "invalid_ohlc_rows": invalid,
        "protocol_sha256": hashlib.sha256(
            REPORT.read_text().split("## Results\n", 1)[0].encode()
        ).hexdigest(),
        "sha256": {
            str(path.relative_to(ROOT.parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "duckdb",
                "exchange-calendars",
                "matplotlib",
            )
        },
        "bootstrap": {
            "reps": 10_000,
            "seed": 42,
            "blocks": [4, 8, 13],
            "calendar": "W-FRI",
            "interval": "pointwise percentile 95%",
            "p_value": "two-sided centered bootstrap",
            "alpha": 0.05,
            "relationship_family": 120,
            "ablation_family": 44,
        },
        "seven_vs_ten": bundle.to_dict("records"),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )
    draw_figures(estimates, group_estimates, weekly, effects, FIGURES, NAMES, TEN)
    draw_comparison(bundle, effects)
    write_report(summary, estimates, effects, bundle, overlap)
    print(f"Saved report to {REPORT}", flush=True)


if __name__ == "__main__":
    main()
