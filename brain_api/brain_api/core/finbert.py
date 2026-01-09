"""Unified FinBERT sentiment scorer with optional caching and GPU support.

This module consolidates the FinBERT implementations from:
- brain_api/core/news_sentiment.py (real-time scoring)
- news_sentiment_etl/core/sentiment.py (batch ETL with caching)

Usage:
    # Simple scoring (no cache)
    scorer = FinBERTScorer()
    result = scorer.score("Apple reports record earnings")

    # With caching (for ETL)
    from brain_api.core.sentiment_cache import SentimentCache
    cache = SentimentCache(Path("data/cache"))
    scorer = FinBERTScorer(cache=cache)
    results, hits, new = scorer.score_batch_with_stats(texts)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

if TYPE_CHECKING:
    from brain_api.core.sentiment_cache import SentimentCache

# FinBERT model configuration
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text for cache key.

    Args:
        text: Text to hash

    Returns:
        16-character hex hash
    """
    return hashlib.md5(text.encode()).hexdigest()[:16]


@dataclass
class SentimentScore:
    """Unified FinBERT sentiment result.

    Combines fields from both previous implementations:
    - FinBERTResult (brain_api): label, p_pos, p_neg, p_neu, article_score
    - SentimentScore (ETL): label, p_pos, p_neg, p_neu, score, confidence

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

    def passes_threshold(self, threshold: float = 0.1) -> bool:
        """Check if sentiment is clear enough (bounded filter).

        Args:
            threshold: Minimum |p_pos - p_neg| to pass

        Returns:
            True if sentiment is above threshold
        """
        return abs(self.p_pos - self.p_neg) >= threshold

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "label": self.label,
            "p_pos": self.p_pos,
            "p_neg": self.p_neg,
            "p_neu": self.p_neu,
            "score": self.score,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SentimentScore:
        """Create from dictionary."""
        return cls(
            label=data["label"],
            p_pos=data["p_pos"],
            p_neg=data["p_neg"],
            p_neu=data["p_neu"],
            score=data["score"],
            confidence=data["confidence"],
        )


class FinBERTScorer:
    """Unified FinBERT sentiment scorer with optional caching and GPU auto-detect.

    Features:
    - Singleton pattern to avoid loading model multiple times
    - Auto-detects best device (CUDA > MPS > CPU)
    - Optional SQLite caching for ETL workloads
    - Both simple and stats-returning batch interfaces

    Usage:
        # Simple (real-time)
        scorer = FinBERTScorer()
        results = scorer.score_batch(texts)

        # With caching (ETL)
        scorer = FinBERTScorer(cache=my_cache)
        results, hits, new = scorer.score_batch_with_stats(texts)
    """

    _instance: FinBERTScorer | None = None
    _pipeline = None
    _device: str = "cpu"
    _cache: SentimentCache | None = None
    _use_gpu: bool | None = None

    def __new__(
        cls,
        cache: SentimentCache | None = None,
        use_gpu: bool | None = None,
    ) -> FinBERTScorer:
        """Create or return singleton instance.

        Args:
            cache: Optional SentimentCache for storing/retrieving scores.
            use_gpu: Force GPU usage. None = auto-detect.
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
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._pipeline = None
        cls._device = "cpu"
        cls._cache = None

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self._use_gpu is False:
            return "cpu"

        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            return "mps"

        # Check for CUDA
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

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

        # Determine device parameter for pipeline
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
            top_k=None,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            device=device_param,
        )

    @property
    def device(self) -> str:
        """Return the device being used."""
        self._ensure_loaded()
        return self._device

    @property
    def cache(self) -> SentimentCache | None:
        """Return the cache instance if set."""
        return self._cache

    def score(self, text: str) -> SentimentScore:
        """Score the sentiment of a single text.

        Args:
            text: Text to analyze (typically article title or title + summary)

        Returns:
            SentimentScore with label, probabilities, and score
        """
        results = self.score_batch([text])
        return results[0]

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Score sentiment for a batch of texts (simple interface).

        This is the primary interface for real-time use cases where
        cache statistics are not needed.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScore objects
        """
        results, _, _ = self.score_batch_with_stats(texts)
        return results

    def score_batch_with_stats(
        self, texts: list[str]
    ) -> tuple[list[SentimentScore], int, int]:
        """Score a batch of texts, returning cache statistics.

        This interface is used by ETL workloads that need to track
        cache hit rates for progress reporting.

        Args:
            texts: List of texts to analyze

        Returns:
            Tuple of (list of SentimentScore objects, cache_hits, new_scores)
        """
        if not texts:
            return [], 0, 0

        # Compute hashes for all texts
        hashes = [compute_text_hash(text) for text in texts]

        # Check cache for existing scores
        results: list[SentimentScore | None] = [None] * len(texts)
        texts_to_score: list[tuple[int, str, str]] = []  # (index, text, hash)

        if self._cache is not None:
            cached = self._cache.get_batch(hashes)
            for i, (text, h) in enumerate(zip(texts, hashes, strict=False)):
                if cached.get(h) is not None:
                    results[i] = cached[h]
                else:
                    texts_to_score.append((i, text, h))
        else:
            texts_to_score = [
                (i, text, h)
                for i, (text, h) in enumerate(zip(texts, hashes, strict=False))
            ]

        cache_hits = len(texts) - len(texts_to_score)
        new_scores = len(texts_to_score)

        # Score remaining texts with FinBERT
        if texts_to_score:
            self._ensure_loaded()

            texts_only = [t[1] for t in texts_to_score]
            try:
                batch_results = self._pipeline(texts_only)
            except Exception:
                # Return neutral on error
                batch_results = [
                    [
                        {"label": "neutral", "score": 0.34},
                        {"label": "positive", "score": 0.33},
                        {"label": "negative", "score": 0.33},
                    ]
                    for _ in texts_only
                ]

            # Process results and save to cache
            new_cache_entries: list[tuple[str, SentimentScore]] = []

            for (orig_idx, _, text_hash), scores in zip(
                texts_to_score, batch_results, strict=False
            ):
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

                sentiment_score = SentimentScore(
                    label=label,
                    p_pos=round(p_pos, 4),
                    p_neg=round(p_neg, 4),
                    p_neu=round(p_neu, 4),
                    score=round(score_val, 4),
                    confidence=round(confidence, 4),
                )

                results[orig_idx] = sentiment_score
                new_cache_entries.append((text_hash, sentiment_score))

            # Batch insert into cache
            if self._cache is not None and new_cache_entries:
                self._cache.put_batch(new_cache_entries)

        # All results should be filled now
        return [r for r in results if r is not None], cache_hits, new_scores
