"""Concurrency coverage for the shared FinBERT scorer."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from brain_api.core.finbert import FINBERT_MODEL, FinBERTScorer


def test_concurrent_lazy_load_builds_one_pipeline():
    FinBERTScorer.reset()
    scorer = FinBERTScorer(use_gpu=False)
    model = MagicMock()
    tokenizer = MagicMock()
    sentiment_pipeline = MagicMock()

    with (
        patch(
            "brain_api.core.finbert.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as load_tokenizer,
        patch(
            "brain_api.core.finbert.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ) as load_model,
        patch(
            "brain_api.core.finbert.pipeline", return_value=sentiment_pipeline
        ) as build,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        list(executor.map(lambda _: scorer._ensure_loaded(), range(2)))

    load_tokenizer.assert_called_once_with(FINBERT_MODEL)
    load_model.assert_called_once_with(FINBERT_MODEL)
    build.assert_called_once()
    assert scorer._pipeline is sentiment_pipeline
    FinBERTScorer.reset()
