from __future__ import annotations

import pytest

from brain_api.news.errors import SentimentScoringError
from brain_api.news.models import NEWS_SENTIMENT_REVISION
from brain_api.news.sentiment import (
    StrictFinBERTScorer,
    assemble_scored_text,
    normalize_scored_text,
    try_assemble_scored_text,
)


def test_normalize_and_assemble() -> None:
    assert normalize_scored_text("  <b>Hello</b>   world ") == "Hello world"
    text = assemble_scored_text("Apple beats", "Revenue up")
    assert text == "Apple beats Revenue up"
    assert try_assemble_scored_text("Apple beats", "Revenue up") == text


def test_empty_text_raises() -> None:
    with pytest.raises(SentimentScoringError, match="empty text"):
        assemble_scored_text("   ", "")


def test_try_assemble_returns_none_for_empty_normalized_text() -> None:
    assert try_assemble_scored_text("   ", "") is None
    assert try_assemble_scored_text("", "") is None
    assert try_assemble_scored_text("<p></p>", "") is None
    assert try_assemble_scored_text("&nbsp;", "   ") is None
    assert try_assemble_scored_text("Apple beats", "") == "Apple beats"


def test_revision_constant() -> None:
    assert NEWS_SENTIMENT_REVISION == "4556d13015211d73dccd3fdd39d39232506f3e43"


def test_pipeline_exception_is_scoring_error() -> None:
    def boom(_texts):
        raise RuntimeError("oom")

    scorer = StrictFinBERTScorer(pipeline=boom, batch_size=32)
    with pytest.raises(SentimentScoringError, match="FinBERT scoring failed"):
        scorer.score_texts(["Apple beats estimates"])


def test_batch_size_is_used() -> None:
    seen: list[int] = []

    def pipe(texts):
        seen.append(len(texts))
        return [
            [
                {"label": "positive", "score": 0.8},
                {"label": "negative", "score": 0.1},
                {"label": "neutral", "score": 0.1},
            ]
            for _ in texts
        ]

    scorer = StrictFinBERTScorer(pipeline=pipe, batch_size=2)
    results = scorer.score_texts(["a", "b", "c"])
    assert seen == [2, 1]
    assert results[0].sentiment_score == pytest.approx(0.7)
    assert results[0].confidence == pytest.approx(0.8)


def test_unknown_label_is_scoring_error() -> None:
    def pipe(texts):
        return [[{"label": "bullish", "score": 1.0}] for _ in texts]

    scorer = StrictFinBERTScorer(pipeline=pipe)
    with pytest.raises(SentimentScoringError, match="unknown label"):
        scorer.score_texts(["Apple beats estimates"])


def test_missing_label_is_scoring_error() -> None:
    def pipe(texts):
        return [
            [
                {"label": "positive", "score": 0.6},
                {"label": "negative", "score": 0.4},
            ]
            for _ in texts
        ]

    scorer = StrictFinBERTScorer(pipeline=pipe)
    with pytest.raises(SentimentScoringError, match="missing labels"):
        scorer.score_texts(["Apple beats estimates"])


def test_probabilities_must_sum_to_one() -> None:
    def pipe(texts):
        return [
            [
                {"label": "positive", "score": 0.9},
                {"label": "negative", "score": 0.9},
                {"label": "neutral", "score": 0.9},
            ]
            for _ in texts
        ]

    scorer = StrictFinBERTScorer(pipeline=pipe)
    with pytest.raises(SentimentScoringError, match="sum"):
        scorer.score_texts(["Apple beats estimates"])
