"""Article quality, sentiment denominators, and within-week return comparisons."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from brain_api.news.models import (
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
)
from feature_relationships import bootstrap_mean, calendar_matrix, holm

THRESHOLDS = (0.0, -0.001, -0.1, -0.5)
THRESHOLD_NAMES = {
    0.0: "Score < 0",
    -0.001: "Score < −0.001 (−0.1%)",
    -0.1: "Score < −0.10",
    -0.5: "Score < −0.50",
}
THRESHOLD_KEYS = (
    "below_zero",
    "below_literal_minus_0_1_percent",
    "below_score_minus_0_1",
    "below_score_minus_0_5",
)


def load_articles(path: Path, cohort):
    """Read a descriptive latest-version inventory, deduplicated across symbols."""
    identity = [
        NEWS_PROVIDER,
        NEWS_SCHEMA_VERSION,
        NEWS_SENTIMENT_MODEL,
        NEWS_SENTIMENT_REVISION,
    ]
    where_identity = "provider = ? AND schema_version = ? AND sentiment_model = ? AND sentiment_model_revision = ?"
    with duckdb.connect(str(path), read_only=True) as db:
        db.execute("SET TimeZone='UTC'")
        rows = db.execute(
            """SELECT * FROM news_events
               WHERE provider = ? AND schema_version = ?
                 AND sentiment_model = ? AND sentiment_model_revision = ?
                 AND created_at >= TIMESTAMPTZ '2016-01-01 00:00:00 America/New_York'
                 AND created_at < TIMESTAMPTZ '2026-01-01 00:00:00 America/New_York'
                 AND updated_at < TIMESTAMPTZ '2026-01-01 00:00:00 America/New_York'
               ORDER BY updated_at DESC, symbol ASC""",
            [
                NEWS_PROVIDER,
                NEWS_SCHEMA_VERSION,
                NEWS_SENTIMENT_MODEL,
                NEWS_SENTIMENT_REVISION,
            ],
        ).df()
        coverage = db.execute(
            f"SELECT status,count(*) AS windows,count(DISTINCT symbol) AS stocks,sum(future_revision_excluded_count) AS future_revision_excluded FROM news_coverage WHERE {where_identity} GROUP BY status",
            identity,
        ).df()
        covered = set(
            db.execute(
                f"SELECT DISTINCT symbol FROM news_coverage WHERE {where_identity}",
                identity,
            ).fetchnumpy()["symbol"]
        )
    keys = ["provider", "provider_article_id"]
    rows["cohort_article"] = (
        rows.symbol.isin(cohort).groupby([rows[k] for k in keys]).transform("any")
    )
    mappings = rows.drop_duplicates([*keys, "symbol"])
    unique = rows.drop_duplicates(keys).copy()
    inventory = {
        "event_rows": len(rows),
        "latest_article_stock_mappings": len(mappings),
        "unique_articles": len(unique),
        "stocks_with_articles": rows.symbol.nunique(),
        "symbols_with_coverage_records": len(covered),
        "covered_symbols_without_articles": sorted(covered - set(rows.symbol)),
        "cohort_articles": int(unique.cohort_article.sum()),
        "created_start": rows.created_at.min().isoformat(),
        "created_end": rows.created_at.max().isoformat(),
        "article_stock_revision_rows_removed": len(rows) - len(mappings),
    }
    return unique, inventory, coverage


def classify_articles(rows):
    """Check stored probability arithmetic without claiming human-label accuracy."""
    result = rows.copy()
    probabilities = result[["p_positive", "p_negative", "p_neutral"]].to_numpy(
        dtype=float
    )
    finite = np.isfinite(probabilities).all(axis=1)
    valid_range = finite & ((probabilities >= 0) & (probabilities <= 1)).all(axis=1)
    valid_sum = np.abs(probabilities.sum(axis=1) - 1) <= 1e-3
    score_ok = (
        np.isfinite(result.sentiment_score)
        & (result.sentiment_score - (result.p_positive - result.p_negative))
        .abs()
        .le(1e-8)
        & result.sentiment_score.between(-1, 1)
    )
    confidence_ok = np.isfinite(result.confidence) & np.abs(
        result.confidence - probabilities.max(axis=1)
    ).le(1e-8)
    hash_ok = result.scored_text_sha256.fillna("").str.fullmatch("[0-9a-f]{64}")
    result["quality_valid"] = (
        valid_range & valid_sum & score_ok & confidence_ok & hash_ok
    )
    result["predicted_label"] = np.array(["positive", "negative", "neutral"])[
        np.argmax(probabilities, axis=1)
    ]
    valid = result.quality_valid
    quality = {
        "unique_articles_checked": len(result),
        "valid_records": int(valid.sum()),
        "invalid_probability_range": int((~valid_range).sum()),
        "invalid_probability_sum": int((~valid_sum).sum()),
        "invalid_score_formula": int((~score_ok).sum()),
        "invalid_confidence_formula": int((~confidence_ok).sum()),
        "invalid_text_hash": int((~hash_ok).sum()),
        "blank_source": int(result.source.fillna("").str.strip().eq("").sum()),
        "probability_ties": int(
            (
                (probabilities == probabilities.max(axis=1)[:, None]).sum(axis=1) > 1
            ).sum()
        ),
        "negative_score_but_neutral_class": int(
            (
                valid
                & result.sentiment_score.lt(0)
                & result.predicted_label.eq("neutral")
            ).sum()
        ),
        "negative_signed_score": int((valid & result.sentiment_score.lt(0)).sum()),
        "negative_winning_class": int(
            (valid & result.predicted_label.eq("negative")).sum()
        ),
        "negative_winning_class_probability_at_least_0_8": int(
            (
                valid
                & result.predicted_label.eq("negative")
                & result.p_negative.ge(0.8)
            ).sum()
        ),
    }
    return result, quality


def distribution_row(values, name):
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    result = {
        "unit": name,
        "observations": len(series),
        "exact_zero": int(series.eq(0).sum()),
    }
    for threshold, key in zip(THRESHOLDS, THRESHOLD_KEYS, strict=True):
        count = int(series.lt(threshold).sum())
        result[key] = count
        result[f"{key}_pct"] = 100 * count / len(series) if len(series) else np.nan
    return result


def conditional_weekly_returns(panel, *, minimum=5):
    """Compare observed groups on each decision date, retaining sparse counts."""
    rows = []
    for friday, week in panel.groupby(level="friday", sort=True):
        for horizon in ("same_week", "daily", "weekly"):
            outcome = week[horizon]
            observed = outcome.notna()
            nonnegative = week.news_sentiment.ge(0) & observed
            with_articles = nonnegative & week.news_status.eq("complete")
            no_news = observed & week.news_status.eq("verified_empty")
            control = outcome[nonnegative]
            for threshold in THRESHOLDS:
                selected = outcome[week.news_sentiment.lt(threshold) & observed]
                usable = len(selected) >= minimum and len(control) >= minimum
                article_control = outcome[with_articles]
                rows.append(
                    {
                        "friday": friday,
                        "horizon": horizon,
                        "threshold": threshold,
                        "negative_count": len(selected),
                        "control_count": len(control),
                        "nonnegative_article_count": len(article_control),
                        "no_news_count": int(no_news.sum()),
                        "negative_mean": selected.mean(),
                        "control_mean": control.mean(),
                        "nonnegative_articles_mean": article_control.mean(),
                        "no_news_mean": outcome[no_news].mean(),
                        "negative_loss_fraction": selected.lt(0).mean()
                        if len(selected)
                        else np.nan,
                        "control_loss_fraction": control.lt(0).mean()
                        if len(control)
                        else np.nan,
                        "difference": selected.mean() - control.mean()
                        if usable
                        else np.nan,
                        "difference_vs_articles": selected.mean()
                        - article_control.mean()
                        if usable and len(article_control) >= minimum
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_conditional(weekly):
    estimates = []
    metrics = [
        "difference",
        "negative_mean",
        "control_mean",
        "negative_loss_fraction",
        "control_loss_fraction",
        "no_news_mean",
        "difference_vs_articles",
    ]
    for (period, horizon, threshold), rows in weekly.groupby(
        ["period", "horizon", "threshold"], sort=False
    ):
        rows = rows.set_index("friday").sort_index()
        paired = rows.difference.notna()
        values = rows[metrics].where(paired, np.nan)
        for block in (4, 8, 13):
            result = bootstrap_mean(calendar_matrix(values), block=block)
            result["metric"] = metrics
            result["negative_stock_weeks"] = int(
                rows.loc[paired, "negative_count"].sum()
            )
            result["control_stock_weeks"] = int(rows.loc[paired, "control_count"].sum())
            estimates.append(
                result.assign(
                    period=period, horizon=horizon, threshold=threshold, block=block
                )
            )
    result = pd.concat(estimates, ignore_index=True)
    primary = (
        result.block.eq(4)
        & result.metric.eq("difference")
        & result.horizon.isin(["daily", "weekly"])
    )
    result.loc[primary, "p_holm"] = holm(result.loc[primary, "p_value"])
    return result
