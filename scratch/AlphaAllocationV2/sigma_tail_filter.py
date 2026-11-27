"""Freeze a top-decile pooled-sigma exclusion and describe the retained cohort."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/friday/friday_dataset.parquet"
OUTPUT = ROOT / "data/sigma_top_decile"
FIGURES = ROOT / "figures/sigma_top_decile"
REPORT = ROOT / "THREE_SIGMA_TOP_DECILE.md"
CUTOFF = pd.Timestamp("2024-12-31")


def rank_crossings(returns, lower, upper, *, minimum=30):
    """Measure pooled crossings per observed return and retain ticker diagnostics."""
    count = returns.count()
    mean, std = returns.mean(), returns.std(ddof=1)
    below, above = returns.lt(lower).sum(), returns.gt(upper).sum()
    individual = returns.lt(mean - 3 * std).sum() + returns.gt(mean + 3 * std).sum()
    ranking = pd.DataFrame(
        {
            "observed_returns": count,
            "below": below,
            "above": above,
            "crossings": below + above,
            "weekly_mean": mean,
            "weekly_std": std,
        }
    )
    ranking["crossing_rate"] = ranking.crossings.div(count).where(count.gt(0))
    ranking["per_stock_sigma_crossing_rate"] = individual.div(count).where(
        count.ge(2) & std.gt(0)
    )
    ranking["eligible"] = count.ge(minimum) & std.gt(0)
    return ranking.rename_axis("symbol")


def select_exclusions(ranking, *, fraction=0.1):
    if not 0 < fraction < 1:
        raise ValueError("The exclusion fraction must be between zero and one")
    count = math.ceil(len(ranking) * fraction)
    eligible = (
        ranking.loc[ranking.eligible]
        .sort_index()
        .sort_values(["crossing_rate", "crossings"], ascending=False, kind="stable")
    )
    if len(eligible) < count:
        raise ValueError(
            "Insufficient eligible stocks to exclude the requested fraction"
        )
    return eligible.index[:count].tolist()


def fit_filter(panel, *, cutoff=CUTOFF, minimum=30):
    """Use completed training outcomes only for both thresholds and exclusions."""
    train = panel.loc[panel.next_friday.le(cutoff), "forward_return"].unstack("symbol")
    values = train.to_numpy().ravel()
    values = values[np.isfinite(values)]
    std = values.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Positive finite return dispersion is required")
    mean = values.mean()
    lower, upper = mean - 3 * std, mean + 3 * std
    ranking = rank_crossings(train, lower, upper, minimum=minimum)
    excluded = select_exclusions(ranking)
    ranking["excluded"] = ranking.index.isin(excluded)
    ranking = ranking.sort_index().sort_values(
        ["crossing_rate", "crossings"], ascending=False, kind="stable"
    )
    ranking.insert(0, "crossing_rate_rank", np.arange(1, len(ranking) + 1))
    return {
        "cutoff": cutoff,
        "mean": mean,
        "std": std,
        "lower": lower,
        "upper": upper,
        "training_rows": len(values),
        "excluded": excluded,
        "ranking": ranking,
    }


def filtered_panel(panel, excluded):
    """Preserve observations and recompute outcome ranks for the retained symbols."""
    result = panel.loc[~panel.index.get_level_values("symbol").isin(excluded)].copy()
    groups = result.groupby(level="friday").forward_return
    result["realized_return_rank"] = groups.rank(method="average", ascending=False)
    result["realized_rank_universe_size"] = groups.transform("count")
    result["universe_selection_as_of"] = CUTOFF
    return result


def period_returns(panel, start, end):
    dates = panel.index.get_level_values("friday")
    selected = panel.loc[(dates >= start) & panel.next_friday.le(end), "forward_return"]
    return selected.unstack("symbol")


def measure_returns(frame, lower, upper):
    values = frame.to_numpy().ravel()
    values = values[np.isfinite(values)]
    below, above = int((values < lower).sum()), int((values > upper).sum())
    return {
        "stocks": len(frame.columns),
        "observed_returns": len(values),
        "mean_return": values.mean(),
        "return_std": values.std(ddof=1),
        "below": below,
        "above": above,
        "crossings": below + above,
        "crossing_rate": (below + above) / len(values),
        "negative_return_fraction": (values < 0).mean(),
    }


def opportunity_cost(frame, excluded):
    observations = []
    for friday, row in frame.iterrows():
        actual = row.dropna().sort_index().sort_values(ascending=False, kind="stable")
        if len(actual) >= 15:
            observations.append(
                {
                    "friday": friday,
                    "excluded_from_realized_top15": int(
                        actual.head(15).index.isin(excluded).sum()
                    ),
                }
            )
    return pd.DataFrame(observations)


def draw_figures(fitted, periods, metrics):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig = plt.figure(figsize=(13, 13), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 1))
    ax = fig.add_subplot(grid[:, 0])
    ranking = fitted["ranking"].loc[fitted["excluded"]]
    y = np.arange(len(ranking))
    ax.barh(
        y,
        ranking.below / ranking.observed_returns * 100,
        color="#b8443c",
        label="Below pooled −3σ",
    )
    ax.barh(
        y,
        ranking.above / ranking.observed_returns * 100,
        left=ranking.below / ranking.observed_returns * 100,
        color="#22866c",
        label="Above pooled +3σ",
    )
    ax.set_yticks(y, ranking.index)
    ax.invert_yaxis()
    ax.set(
        title="Excluded 36 · ranked using 2016–2024",
        xlabel="Training returns crossing frozen thresholds (%)",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_subplot(grid[0, 1])
    for i, cohort in enumerate(("baseline", "retained")):
        item = metrics["evaluation"][cohort]
        ax.bar(
            np.arange(2) + (i - 0.5) * 0.3,
            [
                100 * item["below"] / item["observed_returns"],
                100 * item["above"] / item["observed_returns"],
            ],
            width=0.3,
            label=f"{item['stocks']} stocks",
        )
    ax.set_xticks(range(2), ["Below frozen −3σ", "Above frozen +3σ"])
    ax.set(
        title="2025 tail rates at the same thresholds", ylabel="Observed returns (%)"
    )
    ax.legend()
    ax = fig.add_subplot(grid[1, 1])
    full = periods["evaluation"]["baseline"].to_numpy().ravel()
    full = full[np.isfinite(full)] * 100
    bins = np.linspace(full.min(), full.max(), 101)
    for cohort, color in (("baseline", "#71879b"), ("retained", "#0c7c86")):
        values = periods["evaluation"][cohort].to_numpy().ravel()
        values = values[np.isfinite(values)] * 100
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            color=color,
            lw=1.5,
            label=f"{metrics['evaluation'][cohort]['stocks']} stocks",
        )
    ax.axvline(fitted["lower"] * 100, ls="--", color="#b8443c", lw=1)
    ax.axvline(fitted["upper"] * 100, ls="--", color="#b8443c", lw=1)
    ax.set_yscale("log")
    ax.set(
        title="2025 weekly return distribution",
        xlabel="Friday-to-Friday return (%)",
        ylabel="Share per bin (%, log scale)",
    )
    ax.legend()
    fig.suptitle(
        "V2 pooled three-sigma research · 356 → 320 stocks\nThresholds and exclusions frozen at December 2024",
        fontsize=17,
    )
    fig.savefig(FIGURES / "overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, name, title in zip(
        axes[0],
        ("training", "evaluation"),
        ("2016–2024 training", "2025 evaluation"),
        strict=True,
    ):
        arrays = [frame.to_numpy().ravel() for frame in periods[name].values()]
        arrays = [values[np.isfinite(values)] * 100 for values in arrays]
        ax.boxplot(
            arrays,
            tick_labels=["356 baseline", "320 retained"],
            flierprops={"markersize": 2, "alpha": 0.2},
        )
        ax.axhline(fitted["lower"] * 100, ls="--", color="#b8443c", lw=1)
        ax.axhline(fitted["upper"] * 100, ls="--", color="#b8443c", lw=1)
        ax.set(title=title, ylabel="Weekly return (%) · all observed extremes shown")
    ax = axes[1, 0]
    for cohort, color in (("baseline", "#71879b"), ("retained", "#0c7c86")):
        frame = periods["evaluation"][cohort]
        values = frame.to_numpy().ravel()
        values = values[np.isfinite(values)]
        normalized = (values - values.mean()) / values.std(ddof=1)
        theoretical, observed = stats.probplot(normalized, dist="norm", fit=False)
        ax.plot(
            theoretical,
            observed,
            ".",
            ms=2,
            color=color,
            alpha=0.5,
            label=cohort.title(),
        )
    ax.plot([-4, 4], [-4, 4], "--", color="#b8443c", lw=1)
    ax.set(
        title="2025 normal Q–Q · each distribution standardized",
        xlabel="Normal quantile",
        ylabel="Observed standardized return",
    )
    ax.legend()
    ax = axes[1, 1]
    for cohort, color in (("baseline", "#71879b"), ("retained", "#0c7c86")):
        frame = periods["full_archive"][cohort]
        counts = frame.count(axis=1).reindex(
            pd.date_range("2016-01-01", "2025-12-31", freq="W-FRI")
        )
        ax.plot(
            counts.index, counts.where(counts.gt(0)), color=color, label=cohort.title()
        )
    ax.axvline(CUTOFF, color="#b8443c", ls="--", lw=1)
    ax.set(
        title="Observed return availability · retrospective cohorts",
        ylabel="Stocks with an observed weekly outcome",
    )
    ax.legend()
    fig.suptitle("Distribution and coverage after the 36-stock exclusion", fontsize=17)
    fig.savefig(FIGURES / "distributions.png", dpi=160)
    plt.close(fig)


def table(frame):
    header = (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join("---" for _ in frame.columns)
        + " |\n"
    )
    return header + "\n".join(
        "| " + " | ".join(map(str, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    )


def write_report(fitted, metrics, opportunity):
    ranking = fitted["ranking"].loc[fitted["excluded"]].reset_index()
    shown = pd.DataFrame(
        {
            "Stock": ranking.symbol,
            "Observed training returns": ranking.observed_returns,
            "Below −3σ": ranking.below,
            "Above +3σ": ranking.above,
            "Crossing rate": ranking.crossing_rate.map(lambda x: f"{100 * x:.2f}%"),
        }
    )
    rows = []
    for period, cohorts in metrics.items():
        for cohort, item in cohorts.items():
            rows.append(
                {
                    "Period": period,
                    "Cohort": cohort,
                    "Stocks": item["stocks"],
                    "Observed returns": item["observed_returns"],
                    "Tail crossings": item["crossings"],
                    "Crossing rate": f"{100 * item['crossing_rate']:.3f}%",
                    "Weekly return σ": f"{100 * item['return_std']:.3f}%",
                }
            )
    text = f"## Results\n\nThe fitted pooled mean is {fitted['mean'] * 100:.4f}% and sample standard deviation is {fitted['std'] * 100:.4f}%. The frozen thresholds are **{fitted['lower'] * 100:.4f}%** and **{fitted['upper'] * 100:.4f}%**. The fit uses {fitted['training_rows']:,} audited training outcomes. Exactly 36 of 356 stocks are excluded, leaving 320.\n\n"
    text += table(pd.DataFrame(rows)) + "\n\n"
    change = (
        metrics["evaluation"]["retained"]["crossing_rate"]
        - metrics["evaluation"]["baseline"]["crossing_rate"]
    )
    text += f"The 2025 tail-rate difference, retained minus baseline, is {change * 100:+.3f} percentage points at the same thresholds. The hindsight opportunity-cost diagnostic finds an average of {opportunity.excluded_from_realized_top15.mean():.2f} excluded stocks among each observed week's realized top 15 across {len(opportunity)} evaluation weeks. This describes the trade-off between reducing extremes and retaining potential winners. It leaves the performance of a trained top-15 strategy unresolved.\n\n"
    text += (
        "![Excluded names and 2025 tail comparison](figures/sigma_top_decile/overview.png)\n\n### Excluded stocks\n\n"
        + table(shown)
        + "\n\n"
    )
    text += "![Box plots, normal Q–Q, and observed coverage](figures/sigma_top_decile/distributions.png)\n\n"
    text += "### Reproduce and use the cohort\n\nRun `python scratch/AlphaAllocationV2/sigma_tail_filter.py` with numpy, pandas, DuckDB, matplotlib, and scipy installed. The [summary](data/sigma_top_decile/summary.json) records the cutoff, methods, hashes, and results. [All stock rankings](data/sigma_top_decile/stock_rankings.csv), [excluded names](data/sigma_top_decile/excluded_stocks.csv), [retained names](data/sigma_top_decile/retained_symbols.json), and [weekly opportunity counts](data/sigma_top_decile/opportunity_cost_2025.csv) support review. The [separate retained Friday panel](data/sigma_top_decile/friday_dataset_320.parquet) preserves all source fields and observations for retained names, recomputes outcome ranks, and adds `universe_selection_as_of`. Its selection cutoff must be respected when interpreting earlier rows.\n\n"
    text += "Stocks are ranked by pooled-threshold crossing rate. The exported within-stock sigma rates use each ticker's own mean and standard deviation and answer a different question. This study uses audited labels, so its counts differ from the original raw-return box-plot experiment. Tail flags describe large observed moves. The fixed exclusion is an exploratory universe policy whose prediction and execution performance still needs evaluation.\n"
    REPORT.write_text(REPORT.read_text().split("## Results\n", 1)[0] + text)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        panel = (
            db.read_parquet(str(SOURCE))
            .df()
            .set_index(["friday", "symbol"])
            .sort_index()
        )
    if panel.index.unique("symbol").size != 356 or not panel.index.is_unique:
        raise ValueError("Expected the unique 356-stock Friday panel")
    if (
        np.isinf(panel.forward_return.to_numpy()).any()
        or panel.forward_return.le(-1).any()
    ):
        raise ValueError("Invalid simple return in the audited input")
    if (
        panel.label_status.str.startswith("unresolved") & panel.forward_return.notna()
    ).any():
        raise ValueError("Unresolved source labels must remain missing")
    fitted = fit_filter(panel)
    retained = filtered_panel(panel, fitted["excluded"])
    if retained.index.unique("symbol").size != 320:
        raise ValueError("The frozen exclusion must retain exactly 320 stocks")
    periods, metrics = {}, {}
    for name, start, end in (
        ("training", "2016-01-01", "2024-12-31"),
        ("evaluation", "2025-01-01", "2025-12-31"),
        ("full_archive", "2016-01-01", "2025-12-31"),
    ):
        baseline = period_returns(panel, start, end)
        after = baseline.drop(columns=fitted["excluded"])
        periods[name] = {"baseline": baseline, "retained": after}
        metrics[name] = {
            key: measure_returns(value, fitted["lower"], fitted["upper"])
            for key, value in periods[name].items()
        }
    ranking = fitted["ranking"]
    ranking.to_csv(OUTPUT / "stock_rankings.csv")
    ranking.loc[fitted["excluded"]].to_csv(OUTPUT / "excluded_stocks.csv")
    (OUTPUT / "retained_symbols.json").write_text(
        json.dumps(
            {
                "selection_cutoff": CUTOFF.date().isoformat(),
                "symbols": retained.index.unique("symbol").sort_values().tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    for name, cohorts in periods.items():
        for cohort, values in cohorts.items():
            values.to_csv(OUTPUT / f"{name}_{cohort}_returns.csv")
    opportunity = opportunity_cost(
        periods["evaluation"]["baseline"], fitted["excluded"]
    )
    opportunity.to_csv(OUTPUT / "opportunity_cost_2025.csv", index=False)
    with duckdb.connect() as db:
        db.from_df(retained.reset_index()).write_parquet(
            str(OUTPUT / "friday_dataset_320.parquet")
        )
    summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "classification": "exploratory frozen universe exclusion",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            REPORT.read_text().split("## Results\n", 1)[0].encode()
        ).hexdigest(),
        "selection_cutoff": CUTOFF.date().isoformat(),
        "fraction": 0.1,
        "method": "pooled mean and sample std, rank strict crossings / observed returns; minimum 30 outcomes; positive ticker variance",
        "mean": fitted["mean"],
        "std": fitted["std"],
        "lower": fitted["lower"],
        "upper": fitted["upper"],
        "excluded": fitted["excluded"],
        "retained_stocks": 320,
        "training_rows": fitted["training_rows"],
        "metrics": metrics,
        "mean_realized_top15_excluded": opportunity.excluded_from_realized_top15.mean(),
        "retained_panel_rows": len(retained),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": duckdb.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )
    draw_figures(fitted, periods, metrics)
    write_report(fitted, metrics, opportunity)
    print(
        json.dumps(
            {"excluded": fitted["excluded"], "metrics": metrics["evaluation"]}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
