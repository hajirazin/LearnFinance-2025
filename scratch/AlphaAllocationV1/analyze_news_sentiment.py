"""Describe the news archive and measure conditional close-return associations."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_feature_relationships import (
    SOURCE,
    load_panel,
    markdown_table,
    period_panel,
)
from news_sentiment_analysis import (
    THRESHOLD_KEYS,
    THRESHOLD_NAMES,
    THRESHOLDS,
    classify_articles,
    conditional_weekly_returns,
    distribution_row,
    load_articles,
    summarize_conditional,
)
from news_sentiment_plots import draw_distributions, draw_effects, draw_quality

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/news_sentiment"
FIGURES = ROOT / "figures/news_sentiment"
REPORT = ROOT / "V1_NEWS_SENTIMENT.md"
NEWS_DB = SOURCE.parent / "news/news_v1.duckdb"
PERIODS = {"all": (2016, 2025), "development": (2016, 2024), "validation": (2025, 2025)}


def outcome_table(estimates, period, horizon):
    selected = estimates.loc[
        estimates.period.eq(period)
        & estimates.horizon.eq(horizon)
        & estimates.block.eq(4)
    ]
    rows = []
    for threshold in THRESHOLDS:
        subset = selected.loc[selected.threshold.eq(threshold)].set_index("metric")
        effect, negative, control = (
            subset.loc[key] for key in ("difference", "negative_mean", "control_mean")
        )
        rows.append(
            {
                "News condition": THRESHOLD_NAMES[threshold],
                "Weeks": int(effect.weeks),
                "Negative stock-weeks": int(effect.negative_stock_weeks),
                "Negative group return, %": f"{negative['mean'] * 100:+.3f}",
                "Neutral/positive return, %": f"{control['mean'] * 100:+.3f}",
                "Difference, pp [95% CI]": f"{effect['mean'] * 100:+.3f} [{effect.ci_low * 100:+.3f}, {effect.ci_high * 100:+.3f}]",
                "Holm p": f"{effect.p_holm:.3f}",
            }
        )
    return pd.DataFrame(rows)


def write_report(summary, distributions, articles, estimates):
    quality = summary["article_quality"]
    valid = articles.loc[articles.quality_valid]
    classes = valid.predicted_label.value_counts()
    counts = []
    for label in ("positive", "neutral", "negative"):
        n = int(classes.get(label, 0))
        counts.append(
            {
                "Winning FinBERT class": label,
                "Unique articles": f"{n:,}",
                "Share": f"{100 * n / len(valid):.2f}%",
            }
        )
    table = []
    for item in distributions.to_dict("records"):
        row = {"Denominator": item["unit"], "Observations": f"{item['observations']:,}"}
        for key, label in zip(
            THRESHOLD_KEYS,
            ["Score < 0", "Score < −0.001 (−0.1%)", "Score < −0.10", "Score < −0.50"],
            strict=True,
        ):
            row[label] = f"{item[key]:,} ({item[key + '_pct']:.2f}%)"
        table.append(row)
    neutral_negative = quality["negative_score_but_neutral_class"]
    all_negative = quality["negative_signed_score"]
    fraction = 100 * neutral_negative / all_negative if all_negative else np.nan
    primary = (
        estimates.block.eq(4)
        & estimates.metric.eq("difference")
        & estimates.horizon.isin(["daily", "weekly"])
    )
    hits = estimates.loc[primary & estimates.p_holm.lt(0.05)]
    text = "## Results\n\n"
    text += f"The inventory contains {summary['article_inventory']['unique_articles']:,} unique articles across {summary['article_inventory']['stocks_with_articles']} stocks with articles, from {summary['article_inventory']['created_start']} through {summary['article_inventory']['created_end']}. It has {summary['article_inventory']['latest_article_stock_mappings']:,} latest article-stock mappings. The return analysis uses the existing 356-stock cohort, with {summary['return_panel_rows']:,} price-usable rows with known sentiment.\n\n"
    text += f"The pinned news archive has coverage records for {summary['article_inventory']['symbols_with_coverage_records']} requested securities. Those with coverage but no article records are {', '.join(summary['article_inventory']['covered_symbols_without_articles'])}. Verified no-news observations remain usable neutral inputs.\n\n"
    same = estimates.loc[
        estimates.period.eq("all")
        & estimates.horizon.eq("same_week")
        & estimates.threshold.eq(-0.1)
        & estimates.metric.eq("difference")
        & estimates.block.eq(4)
    ].iloc[0]
    later = estimates.loc[
        estimates.period.eq("validation")
        & estimates.horizon.eq("weekly")
        & estimates.threshold.eq(-0.1)
        & estimates.metric.eq("difference")
        & estimates.block.eq(4)
    ].iloc[0]
    text += f"### Interpretation\n\nAt a weekly score below −0.10, the negative-news group differs from neutral/positive stocks by {same['mean'] * 100:+.3f} pp during the news week. Its subsequent weekly difference in the 2025 comparison is {later['mean'] * 100:+.4f} pp, with a 95% interval of [{later.ci_low * 100:+.4f}, {later.ci_high * 100:+.4f}] pp. This describes a strong contemporaneous association and uncertain subsequent predictive value. Price changes can generate news, and unmeasured company characteristics can affect both sentiment and returns.\n\n"
    text += "### Score units and the meaning of negative\n\nA sentiment score is a probability difference, `P(positive) - P(negative)`. The literal −0.1% threshold is −0.001. The −0.10 score threshold is also displayed to make the unit ambiguity explicit. Thresholds are nested, so their percentages should be read separately.\n\n"
    text += markdown_table(pd.DataFrame(table)) + "\n\n"
    text += markdown_table(pd.DataFrame(counts)) + "\n\n"
    text += f"Among the {all_negative:,} unique articles with a negative signed score, {neutral_negative:,} ({fraction:.2f}%) have neutral as FinBERT's most probable class. A small imbalance between negative and positive probabilities can coexist with a much larger neutral probability. The archive contains {quality['negative_winning_class_probability_at_least_0_8']:,} articles with a winning negative probability of at least 0.80.\n\n"
    text += "![Article and weekly sentiment distributions](figures/news_sentiment/news_distribution.png)\n\n"
    text += "### Technical quality and coverage\n\n"
    checks = pd.DataFrame(
        [
            {"Check": key.replace("_", " "), "Count": value}
            for key, value in quality.items()
        ]
    )
    text += markdown_table(checks) + "\n\n"
    text += (
        markdown_table(
            pd.DataFrame(
                [
                    {"Friday news status": key, "Stock-weeks": value}
                    for key, value in summary["friday_news_status"].items()
                ]
            )
        )
        + "\n\n"
    )
    text += "Probability arithmetic checks and source metadata describe technical quality. Blank source metadata is recorded as missing. The model's confidence is its largest predicted class probability. Human sentiment accuracy, probability calibration, and company-specific article relevance remain unmeasured. The inventory uses the latest stored article revision before archive end. The return analysis uses the original Friday aggregates, with revisions available only after the cutoff excluded or marked unknown.\n\n"
    text += "![Coverage, confidence, and technical quality](figures/news_sentiment/news_quality.png)\n\n"
    text += "### Does negative sentiment relate to subsequent close returns?\n\nThe comparison is between negative-score stocks and the same-Friday nonnegative group, which includes verified no-news zero scores. Results average matched weeks equally. A week needs at least five observed stock returns in each group. The historical archive gives associations. Causal effects of news remain unresolved.\n\n"
    text += (
        "#### Following Friday close, 2016–2025\n\n"
        + markdown_table(outcome_table(estimates, "all", "weekly"))
        + "\n\n"
    )
    text += (
        "#### Next exchange-session close, 2016–2025\n\n"
        + markdown_table(outcome_table(estimates, "all", "daily"))
        + "\n\n"
    )
    text += (
        "#### Following Friday close, 2025 exploratory validation\n\n"
        + markdown_table(outcome_table(estimates, "validation", "weekly"))
        + "\n\n"
    )
    text += f"There are {len(hits)} Holm-adjusted results below 5% among the 24 forward-return comparisons in the prespecified family. Pointwise intervals and effect sizes remain the primary description of uncertainty.\n\n"
    if len(hits):
        text += (
            markdown_table(
                hits[
                    [
                        "period",
                        "horizon",
                        "threshold",
                        "mean",
                        "ci_low",
                        "ci_high",
                        "p_holm",
                    ]
                ].round(6)
            )
            + "\n\n"
        )
    text += "![News-week and subsequent close-return differences](figures/news_sentiment/negative_news_returns.png)\n\n"
    text += "The first panel compares returns during the week from which the news was collected. News can react to price changes within that window. The other panels compare outcomes after the Friday cutoff and are the relevant prediction diagnostics. Sector, size, volatility, and provider coverage differences can affect these associations. An unconditional sentiment correlation can conceal a conditional tail relationship, so the threshold results are reported independently.\n\n"
    text += "### Reproducibility\n\nRun `PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/analyze_news_sentiment.py` from the repository root. The [summary](data/news_sentiment/summary.json) records source/code hashes, score units, denominators, quality checks, coverage, and versions. The [distribution counts](data/news_sentiment/distribution_counts.csv), [weekly group observations](data/news_sentiment/conditional_weekly_returns.csv), and [conditional estimates](data/news_sentiment/conditional_estimates.csv) preserve the numeric results. Alternative controls, loss frequencies, minimum-group exclusions, and 4-, 8-, and 13-week blocks are retained in those files. The [article inventory](data/news_sentiment/articles.parquet) contains identifiers and scoring metadata without article text. No trading rule or production change is selected by this exploratory analysis.\n"
    REPORT.write_text(REPORT.read_text().split("## Results\n", 1)[0] + text)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    archived = (
        pd.read_parquet(SOURCE / "friday_dataset.parquet")
        .set_index(["friday", "symbol"])
        .sort_index()
    )
    articles, inventory, archive_coverage = load_articles(
        NEWS_DB, set(archived.index.unique("symbol"))
    )
    articles, quality = classify_articles(articles)
    valid_articles = articles.loc[articles.quality_valid]
    known = archived.news_status.isin(["complete", "verified_empty"])
    with_articles = archived.news_status.eq("complete")
    empty = archived.news_status.eq("verified_empty")
    if (
        not archived.loc[empty, "news_sentiment"].eq(0).all()
        or not archived.loc[empty, "news_article_count"].eq(0).all()
    ):
        raise ValueError("Verified no-news windows must be neutral zeros")
    distributions = pd.DataFrame(
        [
            distribution_row(
                valid_articles.sentiment_score, "Unique articles · archive"
            ),
            distribution_row(
                valid_articles.loc[valid_articles.cohort_article, "sentiment_score"],
                "Unique articles · 356-stock cohort",
            ),
            distribution_row(
                archived.loc[known, "news_sentiment"], "Stock-weeks · including no news"
            ),
            distribution_row(
                archived.loc[with_articles, "news_sentiment"],
                "Stock-weeks · with articles",
            ),
        ]
    )
    panel, _, coverage = load_panel(("news_sentiment",))
    previous = archived["forward_return"].copy()
    previous.index = pd.MultiIndex.from_arrays(
        [
            previous.index.get_level_values("friday") + pd.Timedelta(days=7),
            previous.index.get_level_values("symbol"),
        ],
        names=["friday", "symbol"],
    )
    panel["same_week"] = previous.reindex(panel.index)
    weekly = []
    for period, (first, last) in PERIODS.items():
        subset = period_panel(panel, first, last)
        subset.loc[
            subset.index.get_level_values("friday") - pd.Timedelta(days=7)
            < pd.Timestamp(first, 1, 1),
            "same_week",
        ] = np.nan
        rows = conditional_weekly_returns(subset)
        weekly.append(rows.assign(period=period))
    weekly = pd.concat(weekly, ignore_index=True)
    estimates = summarize_conditional(weekly)
    distributions.to_csv(OUTPUT / "distribution_counts.csv", index=False)
    weekly.to_csv(OUTPUT / "conditional_weekly_returns.csv", index=False)
    estimates.to_csv(OUTPUT / "conditional_estimates.csv", index=False)
    weekly.loc[weekly.difference.isna()].to_csv(
        OUTPUT / "minimum_group_exclusions.csv", index=False
    )
    archive_coverage.to_csv(OUTPUT / "archive_coverage.csv", index=False)
    articles.to_parquet(OUTPUT / "articles.parquet", index=False)
    source_files = [
        NEWS_DB,
        SOURCE / "friday_dataset.parquet",
        SOURCE / "alpaca_sip_evidence.json",
        SOURCE.parent / "daily_ohlcv_2016_2025.csv",
    ]
    code_files = [
        Path(__file__),
        ROOT / "news_sentiment_analysis.py",
        ROOT / "news_sentiment_plots.py",
        ROOT / "test_news_sentiment_analysis.py",
        ROOT / "feature_relationships.py",
        ROOT / "analyze_feature_relationships.py",
        SOURCE.parents[1] / "build_friday_dataset.py",
        ROOT.parents[1] / "brain_api/brain_api/news/sentiment.py",
    ]
    summary = {
        "classification": "exploratory news distribution and conditional-return study",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "score_formula": "P(positive) - P(negative)",
        "thresholds": list(THRESHOLDS),
        "article_inventory": inventory,
        "article_quality": quality,
        "distributions": distributions.to_dict("records"),
        "friday_news_status": archived.news_status.value_counts().to_dict(),
        "return_panel_rows": len(panel),
        "return_panel_coverage": coverage,
        "bootstrap": {
            "seed": 42,
            "reps": 10_000,
            "blocks": [4, 8, 13],
            "calendar": "W-FRI",
            "interval": "pointwise percentile 95%",
            "p_value": "two-sided centered bootstrap",
            "forward_holm_family": 24,
            "alpha": 0.05,
            "minimum_stocks_per_group": 5,
        },
        "protocol_sha256": hashlib.sha256(
            REPORT.read_text().split("## Results\n", 1)[0].encode()
        ).hexdigest(),
        "sha256": {
            str(path.relative_to(ROOT.parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in [*source_files, *code_files]
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "duckdb",
                "exchange-calendars",
                "matplotlib",
                "pyarrow",
            )
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
    draw_distributions(valid_articles, archived, distributions, FIGURES)
    draw_quality(valid_articles, archived, quality, FIGURES)
    draw_effects(estimates, FIGURES)
    write_report(summary, distributions, articles, estimates)
    print(f"Saved report to {REPORT}", flush=True)


if __name__ == "__main__":
    main()
