"""Measure seven features against future daily and weekly close returns."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from build_friday_dataset import (
    ACTION_DATES,
    FEATURES,
    audit_candidates,
    join_friday_ohlcv,
    load_prices,
    verify_prices,
)
from build_friday_dataset import (
    OUTPUT as SOURCE,
)
from compare_feature_sets import SEVEN
from feature_relationship_plots import draw_figures
from feature_relationships import (
    bootstrap_mean,
    calendar_matrix,
    close_cost_diagnostic,
    forward_daily,
    holm,
    portfolio_observations,
    training_rows,
    weekly_relationships,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/feature_relationships"
FIGURES = ROOT / "figures/feature_relationships"
REPORT = ROOT / "V1_FEATURE_RELATIONSHIPS.md"
LABELS = {**FEATURES, "close": "Adjusted close", "volume": "Volume"}
PERIODS = {"development": (2016, 2024), "validation": (2025, 2025)}
ARMS = {
    "all_seven": SEVEN,
    **{f"without_{f}": tuple(x for x in SEVEN if x != f) for f in SEVEN},
}


def load_panel(features=SEVEN):
    """Reuse the existing audited panel and price evidence without network calls."""
    archived = pd.read_parquet(SOURCE / "friday_dataset.parquet")
    panel = join_friday_ohlcv(archived.set_index(["friday", "symbol"]).sort_index())
    calendar, close, _ = load_prices()
    fridays = pd.DatetimeIndex(panel.index.unique("friday"))
    _, audit, daily_audit = audit_candidates(
        close, pd.date_range("2016-01-01", "2025-12-31", freq="W-FRI")
    )
    adjusted, _ = verify_prices(audit, daily_audit, close)
    breaks = daily_audit.index[daily_audit.status.str.startswith("unresolved")]
    breaks = breaks.union(
        pd.MultiIndex.from_tuples(
            [(pd.Timestamp(day), symbol) for symbol, day in ACTION_DATES]
        )
    )
    daily, next_session = forward_daily(adjusted, breaks)
    daily.index.name, daily.columns.name = "friday", "symbol"
    panel["daily"] = daily.reindex(fridays).stack().reindex(panel.index)
    panel["daily_end"] = panel.index.get_level_values("friday").map(next_session)
    panel["weekly"] = panel.forward_return
    panel["weekly_end"] = panel.next_friday
    for horizon in ("daily", "weekly"):
        panel.loc[panel[f"{horizon}_end"].gt("2025-12-31"), horizon] = np.nan
    finite = np.isfinite(panel[list(features)]).all(axis=1)
    valid_news = panel.news_status.isin(["complete", "verified_empty"])
    eligible = finite & valid_news & panel.price_usable_at_cutoff
    if not panel.index.is_unique or not all(
        calendar.is_session(day) for day in fridays
    ):
        raise ValueError("Invalid stock-Friday keys or session grid")
    coverage = {
        "archive_rows": len(panel),
        "stocks": panel.index.unique("symbol").size,
        "common_feature_rows": int(eligible.sum()),
        "news_status": panel.news_status.value_counts().to_dict(),
        "feature_missing_rows": panel[list(features)].isna().sum().to_dict(),
        "common_rows_by_year": panel.loc[eligible]
        .groupby(panel.loc[eligible].index.get_level_values("friday").year)
        .size()
        .to_dict(),
    }
    return panel.loc[eligible].copy(), adjusted, coverage


def period_panel(panel, first, last):
    dates = panel.index.get_level_values("friday")
    result = panel.loc[(dates.year >= first) & (dates.year <= last)].copy()
    for horizon in ("daily", "weekly"):
        result.loc[result[f"{horizon}_end"].dt.year.gt(last), horizon] = np.nan
    return result


def summarize_relationships(weekly, group_returns):
    estimates, group_estimates = [], []
    for (period, feature, horizon), rows in weekly.groupby(
        ["period", "feature", "horizon"], sort=False
    ):
        matrix = calendar_matrix(
            rows.set_index("friday")[["ic", "spread", "curvature"]]
        )
        for block in (4, 8, 13):
            result = bootstrap_mean(matrix, block=block)
            result["metric"] = ["ic", "spread", "curvature"]
            estimates.append(
                result.assign(
                    period=period, feature=feature, horizon=horizon, block=block
                )
            )
        groups = group_returns.loc[
            (group_returns.period == period)
            & (group_returns.feature == feature)
            & (group_returns.horizon == horizon)
        ]
        result = bootstrap_mean(
            calendar_matrix(
                groups.pivot(index="friday", columns="group", values="return")
            )
        )
        result["group"] = range(1, 6)
        group_estimates.append(
            result.assign(period=period, feature=feature, horizon=horizon)
        )
    estimates = pd.concat(estimates, ignore_index=True)
    primary = estimates.block.eq(4)
    estimates.loc[primary, "p_holm"] = holm(estimates.loc[primary, "p_value"])
    return estimates, pd.concat(group_estimates, ignore_index=True)


def fit_ablation(panel, arms=ARMS):
    """Fit chronological monthly snapshot regressions on identical stock rows."""
    dates = panel.index.unique("friday")
    evaluation = dates[dates.year == 2025]
    predictions, folds, coefficients = [], [], []
    for horizon in ("daily", "weekly"):
        for month in evaluation.to_period("M").unique():
            month_dates = evaluation[evaluation.to_period("M") == month]
            refit = month_dates.min()
            train = training_rows(panel, refit, horizon)
            score = panel.loc[panel.index.get_level_values("friday").isin(month_dates)]
            if len(train) < 1000 or score.groupby(level="friday").size().min() < 15:
                raise ValueError("Insufficient training or scoring coverage")
            if not train[f"{horizon}_end"].max() < refit:
                raise ValueError("Training label crosses the refit cutoff")
            forecast = score[[horizon]].copy()
            forecast["training_mean"] = train[horizon].mean()
            for arm, features in arms.items():
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(train[list(features)], train[horizon])
                forecast[arm] = model.predict(score[list(features)])
                for feature, coefficient in zip(features, model[-1].coef_, strict=True):
                    coefficients.append(
                        {
                            "refit": refit,
                            "horizon": horizon,
                            "arm": arm,
                            "feature": feature,
                            "standardized_coefficient": coefficient,
                        }
                    )
            forecast["horizon"] = horizon
            predictions.append(forecast)
            folds.append(
                {
                    "refit": refit,
                    "horizon": horizon,
                    "training_rows": len(train),
                    "training_first": train.index.get_level_values("friday").min(),
                    "last_training_label": train[f"{horizon}_end"].max(),
                    "scored_rows": len(score),
                }
            )
            print(
                f"Fitted {horizon}, {month}, {len(train):,} training rows", flush=True
            )
    return pd.concat(predictions), pd.DataFrame(folds), pd.DataFrame(coefficients)


def ablation_estimates(weekly, *, features=SEVEN, full="all_seven", extra=None):
    controls = {f: f"without_{f}" for f in features}
    controls.update(extra or {})
    comparisons = []
    for horizon, rows in weekly.groupby("horizon"):
        for metric in ("mae", "basket_return"):
            table = rows.pivot(index="friday", columns="arm", values=metric)
            for feature, reduced in controls.items():
                values = (
                    table[reduced] - table[full]
                    if metric == "mae"
                    else table[full] - table[reduced]
                )
                matrix = calendar_matrix(values)
                for block in (4, 8, 13):
                    estimate = bootstrap_mean(matrix, block=block).iloc[0].to_dict()
                    comparisons.append(
                        {
                            **estimate,
                            "horizon": horizon,
                            "feature": feature,
                            "metric": metric,
                            "block": block,
                        }
                    )
    result = pd.DataFrame(comparisons)
    primary = result.block.eq(4)
    result.loc[primary, "p_holm"] = holm(result.loc[primary, "p_value"])
    return result


def markdown_table(frame):
    rows = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    rows += [
        "| " + " | ".join(map(str, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def write_report(summary, estimates, ablations):
    primary = estimates.loc[estimates.block.eq(4)]
    table = pd.DataFrame(index=SEVEN)
    for horizon in ("daily", "weekly"):
        rows = (
            primary.loc[
                (primary.period == "validation")
                & (primary.horizon == horizon)
                & (primary.metric == "ic")
            ]
            .set_index("feature")
            .reindex(SEVEN)
        )
        table[f"2025 {horizon} IC [95% CI]"] = [
            f"{r['mean']:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
            for _, r in rows.iterrows()
        ]
        table[f"{horizon} Holm p"] = rows.p_holm.map(lambda x: f"{x:.3f}")
    table.insert(0, "Feature", [LABELS[f] for f in table.index])
    delta = ablations.loc[
        ablations.block.eq(4) & ablations.horizon.eq("weekly")
    ].set_index(["feature", "metric"])
    incremental = []
    for feature in SEVEN:
        row = {"Feature": LABELS[feature]}
        for metric, label in (
            ("mae", "MAE improvement, pp [95% CI]"),
            ("basket_return", "Top-15 gross improvement, pp [95% CI]"),
        ):
            r = delta.loc[(feature, metric)]
            row[label] = (
                f"{100 * r['mean']:+.3f} [{100 * r.ci_low:+.3f}, {100 * r.ci_high:+.3f}]"
            )
            row["MAE Holm p" if metric == "mae" else "Basket Holm p"] = (
                f"{r.p_holm:.3f}"
            )
        incremental.append(row)
    significant = primary.loc[
        primary.p_holm.lt(0.05),
        ["period", "horizon", "feature", "metric", "mean", "p_holm"],
    ]
    counts = primary.groupby(["period", "horizon", "metric"]).weeks.agg(["min", "max"])
    text = "## Results\n\n"
    text += f"The archive contains {summary['coverage']['stocks']} stocks and {summary['coverage']['archive_rows']:,} rows. Complete seven-feature rows total {summary['coverage']['common_feature_rows']:,}. Inference uses {len(primary)} relationship tests in one Holm family and 28 ablation tests in a separate family. All reported intervals are pointwise and exploratory.\n\n"
    text += "The 12–1 momentum warmup means the complete-feature analysis actually starts in 2017. Development covers 2017–2024. Validation covers 2025, with 50 next-session observations and 47 next-Friday observations per feature. These daily outcomes are sampled after Friday decisions. They cover the following exchange session, usually Monday.\n\n"
    text += "None of the 84 relationship tests or 28 incremental-value tests passed the primary Holm-adjusted 5% threshold. All seven 2025 weekly IC intervals and all seven paired weekly basket-improvement intervals include zero. The evidence leaves each feature's contribution uncertain. Retain the list as research candidates and evaluate it against the incumbent in the causal V1 protocol before choosing a production change.\n\n"
    text += (
        "### Future-return rank relationships\n\n"
        + markdown_table(table.reset_index(drop=True))
        + "\n\n"
    )
    text += "![Daily and weekly information coefficients](figures/feature_relationships/information_coefficients.png)\n\n"
    text += (
        "### Incremental value in weekly-return Ridge\n\nPositive values favor keeping the feature. MAE and basket returns answer different questions. All models use the same candidate rows, with training-only scaling and completed-label purging.\n\n"
        + markdown_table(pd.DataFrame(incremental))
        + "\n\n"
    )
    text += "![Leave-one-feature-out estimates](figures/feature_relationships/ablation.png)\n\n"
    performance = []
    for horizon in ("daily", "weekly"):
        full = next(
            row
            for row in summary["model_metrics"]
            if row["horizon"] == horizon and row["arm"] == "all_seven"
        )
        baseline = next(row for row in summary["baseline"] if row["horizon"] == horizon)
        performance.append(
            {
                "Outcome": horizon,
                "Seven-feature stock MAE, pp": f"{full['mae'] * 100:.4f}",
                "Training-mean stock MAE, pp": f"{baseline['training_mean_mae'] * 100:.4f}",
                "Seven-feature rank IC": f"{full['ic']:+.4f}",
            }
        )
    text += (
        "### Forecast context\n\n"
        + markdown_table(pd.DataFrame(performance))
        + "\n\nThese MAEs are averages of weekly stock MAEs on the same scored observations. The training-mean predictor provides a simple forecast reference.\n\n"
    )
    cost_rows = []
    for item in summary["cost_comparisons"]:
        cost_rows.append(
            {
                "Feature": LABELS[item["feature"]],
                "Paired weeks": item["weeks"],
                "10-bp diagnostic improvement, pp": f"{item['mean_net_difference'] * 100:+.3f}",
                "Turnover difference, full minus reduced": f"{item['mean_turnover_difference']:+.3f}",
            }
        )
    text += (
        "### Proportional-cost sensitivity\n\n"
        + markdown_table(pd.DataFrame(cost_rows))
        + "\n\nThis table pairs the complete observations for each comparison. It applies 10 bp per unit of traded notional to hypothetical Friday-close rebalances and tracks weight drift between decisions. It provides a descriptive sensitivity check under those assumptions.\n\n"
    )
    text += "### Nonlinear patterns and temporal stability\n\nThe percentile-group chart displays each feature's shape. The low-versus-high spread and the `Q1 - 2*Q3 + Q5` curvature contrast are dependence-aware comparisons of group means. Empty groups caused by tied values stay missing. Tests with fewer than twelve observations have no inferential conclusion.\n\n"
    text += "![Feature groups and future returns](figures/feature_relationships/feature_groups.png)\n\n![Yearly rank information coefficients](figures/feature_relationships/yearly_ic.png)\n\n"
    text += (
        "### Multiplicity-adjusted relationships\n\n"
        + (
            markdown_table(significant.round(6))
            if len(significant)
            else "No relationship test passed Holm at 5%."
        )
        + "\n\n"
    )
    text += (
        "### Coverage\n\n"
        + markdown_table(
            counts.reset_index().rename(
                columns={"min": "minimum weeks", "max": "maximum weeks"}
            )
        )
        + "\n\n"
    )
    text += "### Reproducibility and limits\n\nRun from the repository root with `PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/analyze_feature_relationships.py`. Install pandas, numpy, scipy, scikit-learn, duckdb, exchange-calendars, matplotlib, requests, and pyarrow. The [manifest and model metrics](data/feature_relationships/summary.json) record source hashes, environment versions, code revision, coverage, fit boundaries, and the analysis settings. The [relationship estimates](data/feature_relationships/relationship_estimates.csv) and [ablation estimates](data/feature_relationships/ablation_estimates.csv) include 4-, 8-, and 13-week blocks. [Weekly observations](data/feature_relationships/relationships_weekly.csv), [group returns](data/feature_relationships/groups_weekly.csv), [stock forecasts](data/feature_relationships/stock_forecasts.parquet), [basket picks](data/feature_relationships/picks.csv), [folds](data/feature_relationships/folds.csv), [omitted basket outcomes](data/feature_relationships/missing_basket_outcomes.csv), and [cost sensitivity](data/feature_relationships/close_cost_diagnostic.csv) support independent review.\n\n"
    text += "The cohort and feature choices reflect historical exploration. The daily label shares the original price evidence but has only the archived large-daily-move and action masks, while weekly labels also use the weekly audit. Friday-close basket entry and flat proportional costs are diagnostic assumptions. The cost diagnostic flags incomplete price marks and records a reset at the next usable decision. It excludes IBKR order minima, executable timing, HRP, and a matched SPY portfolio comparison. Candidate removal, deployment, and claims of SPY outperformance require the V1 validation protocol.\n"
    protocol = REPORT.read_text().split("## Results\n", 1)[0]
    REPORT.write_text(protocol + text)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel, close, coverage = load_panel()
    weekly_frames, group_frames = [], []
    for period, (first, last) in PERIODS.items():
        weekly, groups = weekly_relationships(period_panel(panel, first, last), SEVEN)
        weekly_frames.append(weekly.assign(period=period))
        group_frames.append(groups.assign(period=period))
    weekly, groups = pd.concat(weekly_frames), pd.concat(group_frames)
    weekly.to_csv(OUTPUT / "relationships_weekly.csv", index=False)
    groups.to_csv(OUTPUT / "groups_weekly.csv", index=False)
    estimates, group_estimates = summarize_relationships(weekly, groups)
    estimates.to_csv(OUTPUT / "relationship_estimates.csv", index=False)
    group_estimates.to_csv(OUTPUT / "group_estimates.csv", index=False)
    predictions, folds, coefficients = fit_ablation(panel)
    predictions.to_parquet(OUTPUT / "stock_forecasts.parquet")
    folds.to_csv(OUTPUT / "folds.csv", index=False)
    coefficients.to_csv(OUTPUT / "coefficients.csv", index=False)
    scored, selected = [], []
    baselines = []
    for horizon in ("daily", "weekly"):
        forecasts = predictions.loc[predictions.horizon.eq(horizon)]
        rows, picks = portfolio_observations(forecasts, ARMS, horizon)
        scored.append(rows)
        selected.append(picks)
        error = (forecasts.training_mean - forecasts[horizon]).abs()
        baselines.append(
            {
                "horizon": horizon,
                "training_mean_mae": error.groupby(level="friday").mean().mean(),
            }
        )
    scored, picks = pd.concat(scored), pd.concat(selected)
    scored.to_csv(OUTPUT / "ablation_weekly.csv", index=False)
    scored.loc[scored.basket_return.isna()].to_csv(
        OUTPUT / "missing_basket_outcomes.csv", index=False
    )
    picks.to_csv(OUTPUT / "picks.csv", index=False)
    ablations = ablation_estimates(scored)
    ablations.to_csv(OUTPUT / "ablation_estimates.csv", index=False)
    costs = close_cost_diagnostic(picks, close)
    costs.to_csv(OUTPUT / "close_cost_diagnostic.csv", index=False)
    net = costs.pivot(index="friday", columns="arm", values="net_return")
    turnover = costs.pivot(index="friday", columns="arm", values="turnover")
    cost_comparisons = []
    for feature in SEVEN:
        arm = f"without_{feature}"
        difference = (net.all_seven - net[arm]).dropna()
        cost_comparisons.append(
            {
                "feature": feature,
                "weeks": len(difference),
                "mean_net_difference": difference.mean(),
                "mean_turnover_difference": (turnover.all_seven - turnover[arm])
                .reindex(difference.index)
                .mean(),
            }
        )
    sources = [
        SOURCE / "friday_dataset.parquet",
        SOURCE / "alpaca_sip_evidence.json",
        SOURCE / "flagged_daily_returns.csv",
        SOURCE.parent / "daily_ohlcv_2016_2025.csv",
    ]
    code = [
        Path(__file__),
        ROOT / "feature_relationships.py",
        ROOT / "test_feature_relationships.py",
        ROOT / "feature_relationship_plots.py",
        SOURCE.parents[1] / "build_friday_dataset.py",
        SOURCE.parents[1] / "compare_feature_sets.py",
        SOURCE.parents[1] / "audit_data.py",
        SOURCE.parents[1] / "verify_friday_prices.py",
    ]
    summary = {
        "classification": "exploratory historical feature study",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "features": list(SEVEN),
        "protocol_sha256": hashlib.sha256(
            REPORT.read_text().split("## Results\n", 1)[0].encode()
        ).hexdigest(),
        "coverage": coverage,
        "bootstrap": {
            "seed": 42,
            "reps": 10_000,
            "blocks": [4, 8, 13],
            "calendar": "W-FRI",
            "p_value": "two-sided centered bootstrap",
            "interval": "pointwise percentile 95%",
            "alpha": 0.05,
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
        "source_sha256": {
            str(path.relative_to(ROOT.parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sources
        },
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code
        },
        "model_metrics": scored.groupby(["horizon", "arm"])[
            ["mae", "ic", "basket_return"]
        ]
        .mean()
        .reset_index()
        .to_dict("records"),
        "coverage_by_model": scored.groupby(["horizon", "arm"])[
            ["mae", "ic", "basket_return"]
        ]
        .count()
        .reset_index()
        .to_dict("records"),
        "baseline": baselines,
        "cost_comparisons": cost_comparisons,
        "cost_diagnostic": costs.groupby("arm")[["turnover", "net_return"]]
        .mean()
        .reset_index()
        .to_dict("records"),
    }
    # Built-in conversion handles numpy counts without losing numeric types.
    (OUTPUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )
    draw_figures(estimates, group_estimates, weekly, ablations, FIGURES, LABELS, SEVEN)
    write_report(summary, estimates, ablations)
    print(f"Saved report to {REPORT}", flush=True)


if __name__ == "__main__":
    main()
