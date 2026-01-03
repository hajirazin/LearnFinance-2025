"""FinBERT sentiment scoring with batching and GPU support.

This module provides the single source of truth for FinBERT sentiment analysis.
Used by both brain_api (real-time news) and news_sentiment_etl (historical ETL).
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from shared.ml.device import get_device

# FinBERT model configuration
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512


@dataclass
class FinBERTResult:
    """FinBERT sentiment classification result.

    Attributes:
        label: Winning label ("positive", "negative", "neutral")
        p_pos: Probability of positive sentiment
        p_neg: Probability of negative sentiment
        p_neu: Probability of neutral sentiment
        score: Article score (p_pos - p_neg), range [-1, 1]
        confidence: Max probability (confidence of prediction)
    """

    label: str
    p_pos: float
    p_neg: float
    p_neu: float
    score: float
    confidence: float

    @property
    def article_score(self) -> float:
        """Alias for score (backwards compatibility with brain_api)."""
        return self.score

    def passes_threshold(self, threshold: float = 0.1) -> bool:
        """Check if sentiment is clear enough (bounded filter).

        Args:
            threshold: Minimum |p_pos - p_neg| to pass

        Returns:
            True if sentiment is above threshold
        """
        return abs(self.p_pos - self.p_neg) >= threshold


class SentimentCache(Protocol):
    """Protocol for sentiment caching backends."""

    def get_batch(self, hashes: list[str]) -> dict[str, FinBERTResult | None]:
        """Get cached scores for article hashes."""
        ...

    def put_batch(self, entries: list[tuple[str, FinBERTResult]]) -> None:
        """Store scores in cache."""
        ...


class FinBERTScorer:
    """FinBERT sentiment scorer with batching, GPU support, and optional caching.

    Implements singleton pattern to avoid loading the model multiple times.
    Supports optional caching for ETL workloads.

    Example (simple scoring):
        scorer = FinBERTScorer()
        result = scorer.score("Apple stock rises on strong earnings")
        print(result.label, result.score)

    Example (with caching for ETL):
        cache = MySQLiteCache(...)
        scorer = FinBERTScorer(cache=cache)
        results, hits, new = scorer.score_batch_with_stats(texts)
    """

    _instance: "FinBERTScorer | None" = None
    _pipeline = None
    _device: str = "cpu"
    _cache: SentimentCache | None = None
    _use_gpu: bool | None = None

    def __new__(
        cls, use_gpu: bool | None = None, cache: SentimentCache | None = None
    ) -> "FinBERTScorer":
        """Create or return singleton instance.

        Args:
            use_gpu: Force GPU usage. None = auto-detect.
            cache: Optional cache for storing/retrieving scores.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._use_gpu = use_gpu
            cls._instance._cache = cache
        elif cache is not None and cls._instance._cache is None:
            # Update cache if provided on subsequent call
            cls._instance._cache = cache
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._pipeline = None

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self._use_gpu is False:
            return "cpu"
        return get_device()

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._pipeline is not None:
            return

        self._device = self._detect_device()

        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

        # Move model to device
        if self._device != "cpu":
            model = model.to(self._device)

        # Configure device for pipeline
        if self._device == "cuda":
            device_param = 0
        elif self._device == "cpu":
            device_param = -1
        else:
            device_param = self._device  # "mps"

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            device=device_param,
        )

    @property
    def device(self) -> str:
        """Return the device being used."""
        self._ensure_loaded()
        return self._device

    def _parse_scores(self, scores: list[dict]) -> FinBERTResult:
        """Parse raw pipeline scores into FinBERTResult."""
        p_pos = 0.0
        p_neg = 0.0
        p_neu = 0.0

        for item in scores:
            label = item["label"].lower()
            prob = item["score"]
            if label == "positive":
                p_pos = prob
            elif label == "negative":
                p_neg = prob
            elif label == "neutral":
                p_neu = prob

        # Determine winning label
        if p_pos >= p_neg and p_pos >= p_neu:
            label = "positive"
        elif p_neg >= p_pos and p_neg >= p_neu:
            label = "negative"
        else:
            label = "neutral"

        score_val = p_pos - p_neg
        confidence = max(p_pos, p_neg, p_neu)

        return FinBERTResult(
            label=label,
            p_pos=round(p_pos, 4),
            p_neg=round(p_neg, 4),
            p_neu=round(p_neu, 4),
            score=round(score_val, 4),
            confidence=round(confidence, 4),
        )

    def _neutral_result(self) -> FinBERTResult:
        """Return a neutral result for error cases."""
        return FinBERTResult(
            label="neutral",
            p_pos=0.33,
            p_neg=0.33,
            p_neu=0.34,
            score=0.0,
            confidence=0.34,
        )

    def score(self, text: str) -> FinBERTResult:
        """Score a single text.

        Args:
            text: Text to analyze

        Returns:
            FinBERTResult with probabilities and score
        """
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[FinBERTResult]:
        """Score a batch of texts (simple interface).

        This is the simple interface that doesn't report cache stats.
        For ETL workloads that need cache stats, use score_batch_with_stats().

        Args:
            texts: List of texts to analyze

        Returns:
            List of FinBERTResult objects
        """
        if not texts:
            return []

        self._ensure_loaded()

        try:
            batch_results = self._pipeline(texts)
        except Exception:
            return [self._neutral_result() for _ in texts]

        return [self._parse_scores(scores) for scores in batch_results]

    def score_batch_with_stats(
        self, texts: list[str], hash_fn: callable = None
    ) -> tuple[list[FinBERTResult], int, int]:
        """Score a batch of texts with caching support.

        This interface supports caching and reports cache statistics.
        Requires a cache to be set and a hash function provided.

        Args:
            texts: List of texts to analyze
            hash_fn: Function to compute article hash from text

        Returns:
            Tuple of (list of FinBERTResult objects, cache_hits, new_scores)
        """
        if not texts:
            return [], 0, 0

        # If no cache or hash function, fall back to simple batch
        if self._cache is None or hash_fn is None:
            return self.score_batch(texts), 0, len(texts)

        # Compute hashes for all texts
        hashes = [hash_fn(text) for text in texts]

        # Check cache for existing scores
        results: list[FinBERTResult | None] = [None] * len(texts)
        texts_to_score: list[tuple[int, str, str]] = []  # (index, text, hash)

        cached = self._cache.get_batch(hashes)
        for i, (text, h) in enumerate(zip(texts, hashes)):
            if cached.get(h) is not None:
                results[i] = cached[h]
            else:
                texts_to_score.append((i, text, h))

        cache_hits = len(texts) - len(texts_to_score)
        new_scores = len(texts_to_score)

        # Score remaining texts with FinBERT
        if texts_to_score:
            self._ensure_loaded()

            texts_only = [t[1] for t in texts_to_score]
            try:
                batch_results = self._pipeline(texts_only)
            except Exception:
                batch_results = [
                    [
                        {"label": "neutral", "score": 0.34},
                        {"label": "positive", "score": 0.33},
                        {"label": "negative", "score": 0.33},
                    ]
                    for _ in texts_only
                ]

            # Process results and save to cache
            new_cache_entries: list[tuple[str, FinBERTResult]] = []

            for (orig_idx, _, article_hash), scores in zip(texts_to_score, batch_results):
                sentiment_score = self._parse_scores(scores)
                results[orig_idx] = sentiment_score
                new_cache_entries.append((article_hash, sentiment_score))

            # Batch insert into cache
            if new_cache_entries:
                self._cache.put_batch(new_cache_entries)

        return [r for r in results if r is not None], cache_hits, new_scores

