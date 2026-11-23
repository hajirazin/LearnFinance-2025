"""Check sentiment units, article classes, and conditional return comparisons."""

import numpy as np
import pandas as pd
import pytest
from news_sentiment_analysis import (
    classify_articles,
    conditional_weekly_returns,
    distribution_row,
)


def test_literal_percent_cutoff_and_strict_inequality():
    result = distribution_row(
        pd.Series([-0.5, -0.1, -0.001, -0.0001, 0, 0.1]), "example"
    )
    assert result["below_zero"] == 4
    assert result["below_literal_minus_0_1_percent"] == 2
    assert result["below_score_minus_0_1"] == 1
    assert result["below_score_minus_0_5"] == 0
    assert result["observations"] == 6


def test_neutral_article_can_have_negative_signed_score():
    rows = pd.DataFrame(
        {
            "p_positive": [0.01],
            "p_negative": [0.02],
            "p_neutral": [0.97],
            "sentiment_score": [-0.01],
            "confidence": [0.97],
            "scored_text_sha256": ["a" * 64],
            "source": [""],
        }
    )
    actual, quality = classify_articles(rows)
    assert actual.iloc[0].predicted_label == "neutral"
    assert actual.iloc[0].quality_valid
    assert quality["negative_score_but_neutral_class"] == 1
    assert quality["blank_source"] == 1


def test_invalid_probability_record_is_flagged():
    rows = pd.DataFrame(
        {
            "p_positive": [0.3],
            "p_negative": [0.4],
            "p_neutral": [0.5],
            "sentiment_score": [-0.1],
            "confidence": [0.5],
            "scored_text_sha256": ["a" * 64],
            "source": ["benzinga"],
        }
    )
    actual, quality = classify_articles(rows)
    assert not actual.iloc[0].quality_valid
    assert quality["invalid_probability_sum"] == 1


def test_conditional_control_includes_no_news_as_neutral():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-03")], ["A", "B", "C", "D"]], names=["friday", "symbol"]
    )
    panel = pd.DataFrame(
        {
            "news_sentiment": [-0.5, -0.5, 0.2, 0.0],
            "news_status": ["complete", "complete", "complete", "verified_empty"],
            "daily": [-0.02, 0.0, 0.02, 0.04],
            "weekly": [-0.02, 0.0, 0.02, 0.04],
            "same_week": [-0.02, 0.0, 0.02, 0.04],
        },
        index=index,
    )
    rows = conditional_weekly_returns(panel, minimum=2)
    chosen = rows.loc[rows.threshold.eq(-0.1) & rows.horizon.eq("weekly")].iloc[0]
    assert chosen.negative_count == 2
    assert chosen.control_count == 2
    assert chosen.control_mean == pytest.approx(0.03)
    assert chosen.difference == pytest.approx(-0.04)
    assert chosen.no_news_mean == pytest.approx(0.04)


def test_small_negative_group_keeps_counts_and_suppresses_inference():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-03")], range(6)], names=["friday", "symbol"]
    )
    panel = pd.DataFrame(
        {
            "news_sentiment": [-0.8, 0, 0, 0, 0, 0],
            "news_status": ["complete"] * 6,
            "daily": np.arange(6) / 100,
            "weekly": np.arange(6) / 100,
            "same_week": np.arange(6) / 100,
        },
        index=index,
    )
    chosen = (
        conditional_weekly_returns(panel)
        .query("threshold == -0.5 and horizon == 'weekly'")
        .iloc[0]
    )
    assert chosen.negative_count == 1
    assert np.isnan(chosen.difference)


def test_article_inventory_deduplicates_stocks_and_uses_available_revision(tmp_path):
    import duckdb
    import news_sentiment_analysis as news

    path = tmp_path / "news.duckdb"
    identity = [
        news.NEWS_PROVIDER,
        news.NEWS_SCHEMA_VERSION,
        news.NEWS_SENTIMENT_MODEL,
        news.NEWS_SENTIMENT_REVISION,
    ]
    with duckdb.connect(str(path)) as db:
        db.execute(
            "CREATE TABLE news_events (provider VARCHAR, schema_version INTEGER, sentiment_model VARCHAR, sentiment_model_revision VARCHAR, provider_article_id VARCHAR, symbol VARCHAR, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ, sentiment_score DOUBLE)"
        )
        for article, symbol, updated, score in [
            ("1", "A", "2016-01-04T16:00:00Z", -0.1),
            ("1", "A", "2016-01-05T16:00:00Z", -0.8),
            ("1", "B", "2016-01-05T16:00:00Z", -0.8),
            ("1", "A", "2026-01-02T16:00:00Z", 0.9),
            ("2", "B", "2016-01-04T16:00:00Z", 0.2),
        ]:
            db.execute(
                "INSERT INTO news_events VALUES (?,?,?,?,?,?,?,?,?)",
                [*identity, article, symbol, "2016-01-04T15:00:00Z", updated, score],
            )
        db.execute(
            "CREATE TABLE news_coverage (provider VARCHAR, schema_version INTEGER, sentiment_model VARCHAR, sentiment_model_revision VARCHAR, symbol VARCHAR, status VARCHAR, future_revision_excluded_count INTEGER)"
        )
        for symbol, status in [
            ("A", "complete"),
            ("B", "complete"),
            ("C", "verified_empty"),
        ]:
            db.execute(
                "INSERT INTO news_coverage VALUES (?,?,?,?,?,?,?)",
                [*identity, symbol, status, 0],
            )
    articles, inventory, _ = news.load_articles(path, {"A"})
    assert len(articles) == 2
    assert articles.set_index("provider_article_id").loc[
        "1", "sentiment_score"
    ] == pytest.approx(-0.8)
    assert inventory["article_stock_revision_rows_removed"] == 1
    assert inventory["cohort_articles"] == 1
    assert inventory["symbols_with_coverage_records"] == 3
    assert inventory["covered_symbols_without_articles"] == ["C"]
