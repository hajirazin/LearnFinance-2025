"""Compare the frozen MAD screen with matched controls and market benchmarks."""

import json
from itertools import pairwise
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from compare_patchtst import sha256
from compare_v1_forecast_accuracy import holm, line, paired_interval, plt, set_style
from mad_experiment_data import OUTPUT, ROOT, SHARED, load_arrays
from report_v1_xgboost_comparison import selected_baskets
from sklearn.metrics import r2_score
from train_v1_xgboost_comparison import completed, load_split, write_json
from v1_hypothesis_data import session_data

FIGURES = ROOT / "figures/rolling_mad"
MAD = "MAD PatchTST"
RAW = "Raw PatchTST"
V1 = "V1-style refit, same cohort"
V1_FULL = "V1-style refit, broad universe"
PRIOR = "Prior seven-feature, broad universe"
MODELS = (MAD, RAW, V1, V1_FULL, PRIOR)
REFERENCES = (RAW, V1, "SPY", "QQQ")


def table(frame):
    shown = frame.copy()
    for col in shown.select_dtypes(include="number"):
        shown[col] = shown[col].map(
            lambda x: "—" if pd.isna(x) else f"{x:.4f}".rstrip("0").rstrip(".")
        )
    return "\n".join(
        [
            "| " + " | ".join(shown.columns) + " |",
            "| " + " | ".join(["---"] * len(shown.columns)) + " |",
            *[
                "| " + " | ".join(map(str, row)) + " |"
                for row in shown.itertuples(index=False, name=None)
            ],
        ]
    )


def load_evidence():
    _, _, full_index = load_split("test")
    _, _, cohort_index = load_arrays("test", "raw")
    old = pd.read_csv(
        ROOT.parent / "AlphaAllocationV1/data/v1_hypotheses/scores.csv",
        parse_dates=["friday"],
    ).set_index(["friday", "symbol"])
    if not completed("patchtst_42"):
        raise ValueError("Original seven-feature reference is incomplete")
    full = pd.DataFrame(
        {
            V1_FULL: old.current.reindex(full_index),
            PRIOR: np.load(SHARED / "patchtst_42_predictions.npy"),
        },
        index=full_index,
    )
    cohort = pd.DataFrame(index=cohort_index)
    for key, name in (("mad", MAD), ("raw", RAW)):
        metadata = json.loads((OUTPUT / f"{key}.json").read_text())
        if metadata["specification"]["dataset_sha256"] != sha256(
            OUTPUT / "dataset.json"
        ):
            raise ValueError(f"Dataset mismatch for {key}")
        for filename, digest in metadata["artifacts"].items():
            if sha256(OUTPUT / filename) != digest:
                raise ValueError(f"Modified model artifact: {filename}")
        cohort[name] = np.load(OUTPUT / f"{key}_predictions.npy")
    cohort[V1] = full[V1_FULL].reindex(cohort_index)
    if (
        not np.isfinite(cohort.to_numpy()).all()
        or not np.isfinite(full.to_numpy()).all()
    ):
        raise ValueError("Missing archived model forecasts")
    with duckdb.connect() as db:
        evidence = db.read_parquet(
            str(
                ROOT.parent
                / "AlphaAllocationV1/data/forecast_accuracy/stock_forecasts.parquet"
            )
        ).df()
    evidence.friday = evidence.friday.dt.as_unit("ns")
    actual = evidence.set_index(["friday", "symbol"]).actual.dropna().sort_index()
    panel = cohort.join(actual, how="inner", validate="one_to_one")
    panel[PRIOR] = full[PRIOR].reindex(panel.index)
    panel["Zero"] = 0.0
    if not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Invalid paired stock outcomes")
    choices = selected_baskets(cohort, arms=(MAD, RAW, V1))
    choices.update(selected_baskets(full, arms=(V1_FULL, PRIOR)))
    return cohort, full, panel, actual, choices


def market_returns(dates):
    frame = pd.read_csv(OUTPUT / "benchmarks.csv", parse_dates=["date"]).pivot(
        index="date", columns="symbol", values="close"
    )
    ends = dates + pd.Timedelta(days=7)
    returns = pd.DataFrame(
        frame.reindex(ends).to_numpy() / frame.reindex(dates).to_numpy() - 1,
        index=dates,
        columns=frame.columns,
    )
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Incomplete Friday-close benchmark evidence")
    return returns


def weekly_results(cohort, full, actual, choices):
    dates = actual.index.unique("friday").sort_values()
    benchmark = market_returns(dates)
    records = []
    for day in dates:
        labels = actual.xs(day, level="friday")
        for name in MODELS:
            names = choices[day, name]
            observed = labels.reindex(names)
            source = cohort if name in cohort else full
            predicted = source.xs(day, level="friday").loc[names, name]
            records.append(
                {
                    "friday": day,
                    "model": name,
                    "actual": observed.mean() if observed.notna().all() else np.nan,
                    "predicted": predicted.mean(),
                    "missing_outcomes": int(observed.isna().sum()),
                    "symbols": ",".join(names),
                }
            )
    weekly = pd.DataFrame(records)
    pivot = weekly.pivot(index="friday", columns="model", values="actual")
    common = pivot.notna().all(axis=1)
    returns = pivot.loc[common].join(benchmark)
    omissions = [str(x.date()) for x in pivot.index[~common]]
    if len(returns) < 20:
        raise ValueError("Too few common observed weeks")
    return weekly, returns, omissions


def statistics(returns, panel, choices):
    rows, forecasts, tests = [], [], []
    for name in returns:
        values = returns[name]
        wealth = pd.Series(np.r_[1.0, np.cumprod(1 + values)])
        rows.append(
            {
                "Model": name,
                "Weeks": len(values),
                "Mean weekly return (%)": 100 * values.mean(),
                "Excess SPY (pp)": 100 * (values - returns.SPY).mean(),
                "Excess QQQ (pp)": 100 * (values - returns.QQQ).mean(),
                "Weeks above SPY": int(values.gt(returns.SPY).sum()),
                "Weeks above QQQ": int(values.gt(returns.QQQ).sum()),
                "Sample compounded (%)": 100 * (wealth.iloc[-1] - 1),
                "Sample max drawdown (%)": 100 * (wealth / wealth.cummax() - 1).min(),
            }
        )
    common_panel = panel.loc[panel.index.get_level_values("friday").isin(returns.index)]
    for name in (MAD, RAW, V1, PRIOR, "Zero"):
        error = 100 * (common_panel[name] - common_panel.actual)
        mse = error.pow(2).groupby(level="friday").mean()
        mae = error.abs().groupby(level="friday").mean()
        ic = common_panel.groupby(level="friday")[[name, "actual"]].apply(
            lambda g, feature=name: (
                g[feature].rank().corr(g.actual.rank())
                if g[feature].nunique() > 1
                else np.nan
            )
        )
        forecasts.append(
            {
                "Model": name,
                "Stock RMSE (pp)": np.sqrt(mse.mean()),
                "Stock MAE (pp)": mae.mean(),
                "Stock R²": r2_score(common_panel.actual, common_panel[name]),
                "Mean rank IC": ic.mean(),
            }
        )
    for block in (2, 4, 8):
        group = []
        for reference in REFERENCES:
            delta = 100 * (returns[MAD] - returns[reference])
            mean, low, high, p_lower = paired_interval(delta, block=block)
            p_upper = paired_interval(-delta, block=block)[-1]
            group.append(
                {
                    "Reference": reference,
                    "Block": block,
                    "MAD minus reference (pp)": mean,
                    "95% low": low,
                    "95% high": high,
                    "Two-sided p": min(1, 2 * min(p_lower, p_upper)),
                }
            )
        group = pd.DataFrame(group)
        group["Holm p"] = holm(group["Two-sided p"]).reindex(group.index)
        tests.append(group)
    turnover = []
    for name in MODELS:
        baskets = [
            set(choices[day, name]) for day in sorted({key[0] for key in choices})
        ]
        turnover.append(
            {
                "Model": name,
                "Mean names replaced": np.mean(
                    [len(new - old) for old, new in pairwise(baskets)]
                ),
                "Replacement fraction": np.mean(
                    [len(new - old) / 15 for old, new in pairwise(baskets)]
                ),
            }
        )
    return (
        pd.DataFrame(rows),
        pd.DataFrame(forecasts),
        pd.concat(tests, ignore_index=True),
        pd.DataFrame(turnover),
    )


def charts(returns, forecasts, adjustments):
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    colors = {
        MAD: "#126f73",
        RAW: "#bf5968",
        V1: "#7860aa",
        "SPY": "#c37224",
        "QQQ": "#357aba",
    }
    fig, axes = plt.subplots(3, 1, figsize=(15, 15), layout="constrained")
    for name, color in colors.items():
        line(axes[0], 100 * (1 + returns[name]).cumprod(), name, color, scale=1)
    axes[0].set_title(
        f"Compounded over {len(returns)} sampled weeks · omitted weeks excluded · gross",
        loc="left",
    )
    axes[0].set_ylabel("Index (initial = 100)")
    axes[0].legend(fontsize=9, ncol=3)
    for name in ("SPY", "QQQ"):
        line(axes[1], returns[MAD] - returns[name], f"MAD minus {name}", colors[name])
    axes[1].axhline(0, color="#6c7c8e", linewidth=0.8)
    axes[1].set_title("Weekly gross excess of the MAD-selected basket", loc="left")
    axes[1].set_ylabel("Percentage points")
    axes[1].legend()
    labels = ["MAD", "Raw", "V1 refit", "Prior seven", "Zero"]
    axes[2].bar(
        labels,
        forecasts["Stock RMSE (pp)"],
        color=["#126f73", "#bf5968", "#7860aa", "#357aba", "#8b99aa"],
    )
    for i, value in enumerate(forecasts["Stock RMSE (pp)"]):
        axes[2].text(i, value + 0.02, f"{value:.4f}", ha="center")
    axes[2].set_ylim(0, forecasts["Stock RMSE (pp)"].max() * 1.12)
    axes[2].set_title(
        "Individual-stock weekly forecast RMSE · identical observed stock-weeks",
        loc="left",
    )
    axes[2].set_ylabel("Percentage points")
    fig.suptitle(
        "Rolling-MAD price inputs · PatchTST · seed 42\nEqual-weight sticky 15 · V1 is a research refit",
        fontsize=21,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    counts = adjustments.groupby(pd.to_datetime(adjustments.session).dt.year).size()
    counts.plot.bar(ax=axes[0], color="#126f73")
    axes[0].set_title("Clipped daily input returns by year · all retained securities")
    axes[0].set_ylabel("Clipped observations")
    axes[0].set_xlabel("Year")
    axes[1].scatter(
        100 * np.expm1(adjustments.raw_log_return),
        100 * np.expm1(adjustments.clipped_log_return),
        s=8,
        alpha=0.35,
        color="#126f73",
    )
    axes[1].set_xlabel("Observed daily return (%)")
    axes[1].set_ylabel("Clipped model-input return (%)")
    axes[1].set_title("Clipped observations only · labels remain observed")
    fig.savefig(FIGURES / "clipping_audit.png", dpi=150)
    plt.close(fig)


def calendar_sensitivity(cohort, full, choices):
    """Include four-session holiday weeks without changing fits or selections."""
    _, _, close, _, logs, _ = session_data()
    records, sessions_per_week = [], {}
    for day in cohort.index.unique("friday").sort_values():
        end = day + pd.Timedelta(days=7)
        if end not in close.index:
            continue
        names = full.xs(day, level="friday").index
        path = logs.loc[(logs.index > day) & (logs.index <= end), names]
        values = (close.loc[end, names] / close.loc[day, names] - 1).where(
            path.notna().all()
        )
        sessions_per_week[day] = len(path)
        records.extend(
            {"friday": day, "symbol": name, "actual": value}
            for name, value in values.dropna().items()
        )
    actual = pd.DataFrame(records).set_index(["friday", "symbol"]).actual
    panel = cohort.join(actual, how="inner", validate="one_to_one")
    panel[PRIOR] = full[PRIOR].reindex(panel.index)
    panel["Zero"] = 0.0
    weekly, returns, omissions = weekly_results(cohort, full, actual, choices)
    metrics, forecasts, tests, turnover = statistics(returns, panel, choices)
    directory = OUTPUT / "calendar_friday"
    directory.mkdir(exist_ok=True)
    weekly.to_csv(directory / "weekly_results.csv", index=False)
    returns.rename_axis("friday").to_csv(directory / "paired_returns.csv")
    metrics.to_csv(directory / "metrics.csv", index=False)
    forecasts.to_csv(directory / "forecast_metrics.csv", index=False)
    tests.to_csv(directory / "paired_tests.csv", index=False)
    turnover.to_csv(directory / "turnover.csv", index=False)
    pd.Series(sessions_per_week, name="sessions").rename_axis("friday").to_csv(
        directory / "session_counts.csv"
    )
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), layout="constrained")
    colors = {
        MAD: "#126f73",
        RAW: "#bf5968",
        V1: "#7860aa",
        "SPY": "#c37224",
        "QQQ": "#357aba",
    }
    for name, color in colors.items():
        line(axes[0], 100 * (1 + returns[name]).cumprod(), name, color, scale=1)
    axes[0].set_title(
        "Compounded only over evaluated Friday pairs · gaps excluded · gross",
        loc="left",
    )
    axes[0].set_ylabel("Index (initial = 100)")
    axes[0].legend(fontsize=9, ncol=3)
    for name in ("SPY", "QQQ"):
        line(axes[1], returns[MAD] - returns[name], f"MAD minus {name}", colors[name])
    axes[1].axhline(0, color="#6c7c8e", linewidth=0.8)
    axes[1].set_ylabel("Excess return (pp)")
    axes[1].set_title("MAD-selected 15 versus realized market returns", loc="left")
    axes[1].legend()
    fig.suptitle(
        f"Calendar Friday-close sensitivity · {len(returns)} matched weeks\nIncludes four-session holiday weeks · unchanged predictions and holdings",
        fontsize=20,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "calendar_friday.png", dpi=150)
    plt.close(fig)
    holiday_dates = [
        str(day.date()) for day, count in sessions_per_week.items() if count != 5
    ]
    summary = {
        "weeks": len(returns),
        "four_session_decisions": holiday_dates,
        "omitted_incomplete_baskets": omissions,
        "metrics": metrics.to_dict("records"),
        "tests": tests.to_dict("records"),
    }
    write_json(directory / "summary.json", summary)
    section = f"""
### Calendar Friday-close sensitivity

The strict five-session archive omits holiday-shortened holding periods even when both Fridays trade. As an additional descriptive check, reuse all fitted forecasts and frozen selections and calculate observed Friday-close to next-Friday-close returns directly from the unmodified prices. Preserve the existing unresolved-return masks. The decision to add this check was recorded after fitting and before reading the primary test outcomes; it does not redefine the primary sample or choose a model.

This yields {len(returns)} matched weeks. The four-session decision dates are {", ".join(holiday_dates)}. Additional incomplete-basket omissions: {omissions or "none"}. Weeks with a closed Friday endpoint remain excluded. The model still targets five trading sessions, so its forecast-error comparison on four-session weeks has a horizon mismatch; no retraining or target correction was performed here.

{table(metrics)}

![Calendar Friday-close comparison including holiday-shortened weeks](figures/rolling_mad/calendar_friday.png)

[Calendar-week returns](data/rolling_mad/calendar_friday/paired_returns.csv) · [Session counts](data/rolling_mad/calendar_friday/session_counts.csv) · [Sensitivity statistics](data/rolling_mad/calendar_friday/paired_tests.csv)
"""
    return summary, section


def main():
    manifest = json.loads((OUTPUT / "dataset.json").read_text())
    cohort, full, panel, actual, choices = load_evidence()
    weekly, returns, omissions = weekly_results(cohort, full, actual, choices)
    metrics, forecasts, tests, turnover = statistics(returns, panel, choices)
    adjustments = pd.read_csv(OUTPUT / "clipped_returns.csv", parse_dates=["session"])
    weekly.to_csv(OUTPUT / "weekly_results.csv", index=False)
    returns.rename_axis("friday").to_csv(OUTPUT / "paired_returns.csv")
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    forecasts.to_csv(OUTPUT / "forecast_metrics.csv", index=False)
    tests.to_csv(OUTPUT / "paired_tests.csv", index=False)
    turnover.to_csv(OUTPUT / "turnover.csv", index=False)
    pd.DataFrame(
        [
            {"friday": day, "model": name, "symbols": ",".join(names)}
            for (day, name), names in choices.items()
        ]
    ).to_csv(OUTPUT / "selections.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(panel.reset_index()).write_parquet(
            str(OUTPUT / "stock_forecasts.parquet")
        )
    charts(returns, forecasts, adjustments)
    calendar, calendar_section = calendar_sensitivity(cohort, full, choices)
    summary = {
        "weeks": len(returns),
        "omitted_incomplete_baskets": omissions,
        "common_stock_outcomes": len(panel),
        "decision_fridays": cohort.index.unique("friday").size,
        "metrics": metrics.to_dict("records"),
        "forecast_metrics": forecasts.replace({np.nan: None}).to_dict("records"),
        "tests": tests.to_dict("records"),
        "cohort_counts": manifest["cohort_counts"],
        "clipped_observations": len(adjustments),
        "clipped_fraction": len(adjustments) / manifest["finite_daily_returns"],
        "calendar_friday_sensitivity": calendar,
        "source_sha256": {
            p.name: sha256(p)
            for p in [
                Path(__file__),
                OUTPUT / "dataset.json",
                OUTPUT / "benchmarks.csv",
                OUTPUT / "raw.json",
                OUTPUT / "mad.json",
                SHARED / "patchtst_42.json",
                ROOT.parent / "AlphaAllocationV1/data/v1_hypotheses/scores.csv",
            ]
        },
    }
    write_json(OUTPUT / "summary.json", summary)
    path = ROOT / "ROLLING_MAD_EXPERIMENT.md"
    protocol = path.read_text().split("\n## Results\n")[0]
    report = f"""
## Results

The paired comparison used {len(returns)} observed weeks and {len(panel):,} common stock outcomes. Selections carried through all {cohort.index.unique("friday").size} decision Fridays. Additional incomplete-basket omissions: {omissions or "none"}.

The retained roster has 364 securities. Per-window eligibility produced {manifest["splits"]["train"]["rows"]:,} training examples across {manifest["splits"]["train"]["stocks"]} stocks, {manifest["splits"]["validation"]["rows"]:,} validation examples across {manifest["splits"]["validation"]["stocks"]} stocks, and {manifest["splits"]["test"]["rows"]:,} test forecasts across {manifest["splits"]["test"]["stocks"]} stocks. Each arm used the identical rows.

{table(metrics)}

All returns are gross. The compounded and drawdown columns use only the matched observed weeks and do not represent a continuously invested, executable account. V1 means the archived research refit; deployed production performance is not measured here.

{table(forecasts)}

Primary four-observed-week paired tests, with a two-sided Holm-adjusted family at 5%. The 95% intervals are pointwise, not simultaneous family intervals:

{table(tests.loc[tests.Block.eq(4)])}

{table(turnover)}

![Matched model and benchmark comparisons](figures/rolling_mad/comparison.png)

The rule clipped {len(adjustments):,} of {manifest["finite_daily_returns"]:,} finite daily input returns ({100 * summary["clipped_fraction"]:.3f}%) across the archived history. This includes genuine extremes and is not a count of proven bad quotes. Volume and sentiment inputs, raw labels, eligibility, and source files were preserved.

![Price-input clipping audit](figures/rolling_mad/clipping_audit.png)

[Universe exclusions and reasons](data/rolling_mad/universe_audit.csv) · [Ticker identity evidence](data/rolling_mad/identity_mismatches.csv) · [Retained roster](data/rolling_mad/retained_symbols.json) · [Clipping audit](data/rolling_mad/clipped_returns.csv) · [Model and raw close paths](data/rolling_mad/model_prices.parquet) · [Weekly comparisons](data/rolling_mad/paired_returns.csv) · [Stock forecasts and actuals](data/rolling_mad/stock_forecasts.parquet) · [All block sensitivities](data/rolling_mad/paired_tests.csv) · [Summary and provenance](data/rolling_mad/summary.json)

This single-seed result is exploratory on a reused historical sample. The listing-age proxy, frozen roster, incomplete independent identity audit, original per-window normalization, and gross Friday-close convention limit any production interpretation. No model was promoted.

{calendar_section}

### Decision

This run does not support adopting the fixed rolling-MAD rule. Its 33-week advantage over the matched raw control was only +0.070 pp per week, with a 95% interval of −0.905 to +1.097 pp and Holm p = 1.00. Including the four holiday-shortened Friday-to-Friday periods reversed that difference to −0.236 pp per week: the MAD basket compounded to −9.20%, versus −0.23% for raw, −2.36% for SPY, and −3.94% for QQQ on the same 37 sampled periods. These are point estimates, not proof that MAD is universally harmful.

Stock RMSE improved by only about 0.00037 pp versus raw on the original matched sample; both remained worse than the zero-return baseline. The stronger data-quality finding is the 13 verified ticker identity mismatches. They were excluded from the new fits, while the older broad-universe models remain frozen references with that limitation. Capping price extremes does not repair wrong security identities or establish that all other features are clean.
"""
    path.write_text(protocol + "\n" + report.lstrip())
    print(metrics.to_string(index=False))
    print(forecasts.to_string(index=False))
    print(tests.loc[tests.Block.eq(4)].to_string(index=False))


if __name__ == "__main__":
    main()
