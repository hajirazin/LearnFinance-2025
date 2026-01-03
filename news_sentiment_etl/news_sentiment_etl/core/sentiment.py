"""FinBERT sentiment scoring with batching and GPU support.

Uses shared FinBERTScorer implementation with ETL-specific caching support.
"""

from dataclasses import dataclass
from typing import Callable

from shared.ml.finbert import FinBERTResult as SharedFinBERTResult
from shared.ml.finbert import FinBERTScorer as SharedFinBERTScorer

from news_sentiment_etl.core.cache import SentimentCache, compute_article_hash


@dataclass
class SentimentScore:
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

    def passes_threshold(self, threshold: float = 0.1) -> bool:
        """Check if sentiment is clear enough (bounded filter).

        Args:
            threshold: Minimum |p_pos - p_neg| to pass

        Returns:
            True if sentiment is above threshold
        """
        return abs(self.p_pos - self.p_neg) >= threshold


def _shared_to_local_score(shared: SharedFinBERTResult) -> SentimentScore:
    """Convert shared FinBERTResult to local SentimentScore format."""
    return SentimentScore(
        label=shared.label,
        p_pos=shared.p_pos,
        p_neg=shared.p_neg,
        p_neu=shared.p_neu,
        score=shared.score,
        confidence=shared.confidence,
    )


class CacheAdapter:
    """Adapter to make SentimentCache compatible with shared scorer protocol."""

    def __init__(self, cache: SentimentCache):
        self._cache = cache

    def get_batch(self, hashes: list[str]) -> dict[str, SharedFinBERTResult | None]:
        """Get cached scores, converting from SentimentScore to SharedFinBERTResult."""
        results = self._cache.get_batch(hashes)
        converted: dict[str, SharedFinBERTResult | None] = {}
        for h, score in results.items():
            if score is None:
                converted[h] = None
            else:
                converted[h] = SharedFinBERTResult(
                    label=score.label,
                    p_pos=score.p_pos,
                    p_neg=score.p_neg,
                    p_neu=score.p_neu,
                    score=score.score,
                    confidence=score.confidence,
                )
        return converted

    def put_batch(self, entries: list[tuple[str, SharedFinBERTResult]]) -> None:
        """Store scores, converting from SharedFinBERTResult to SentimentScore."""
        converted = [
            (h, SentimentScore(
                label=r.label,
                p_pos=r.p_pos,
                p_neg=r.p_neg,
                p_neu=r.p_neu,
                score=r.score,
                confidence=r.confidence,
            ))
            for h, r in entries
        ]
        self._cache.put_batch(converted)


class FinBERTScorer:
    """FinBERT sentiment scorer with batching, GPU support, and caching.

    This is a wrapper around the shared FinBERTScorer that:
    - Converts results to local SentimentScore format
    - Provides ETL-specific caching support
    """

    _instance: "FinBERTScorer | None" = None
    _shared_scorer: SharedFinBERTScorer | None = None
    _cache: SentimentCache | None = None
    _cache_adapter: CacheAdapter | None = None

    def __new__(
        cls, use_gpu: bool | None = None, cache: SentimentCache | None = None
    ) -> "FinBERTScorer":
        """Create or return singleton instance.

        Args:
            use_gpu: Force GPU usage. None = auto-detect.
            cache: SentimentCache for storing/retrieving scores.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = cache
            cls._instance._cache_adapter = CacheAdapter(cache) if cache else None
            # Initialize shared scorer with cache adapter
            cls._instance._shared_scorer = SharedFinBERTScorer(
                use_gpu=use_gpu,
                cache=cls._instance._cache_adapter,
            )
        elif cache is not None and cls._instance._cache is None:
            # Update cache if provided on subsequent call
            cls._instance._cache = cache
            cls._instance._cache_adapter = CacheAdapter(cache)
            # Reset shared scorer to use new cache
            SharedFinBERTScorer.reset_instance()
            cls._instance._shared_scorer = SharedFinBERTScorer(
                use_gpu=use_gpu,
                cache=cls._instance._cache_adapter,
            )
        return cls._instance

    @property
    def device(self) -> str:
        """Return the device being used."""
        return self._shared_scorer.device

    def score(self, text: str) -> SentimentScore:
        """Score a single text.

        Args:
            text: Text to analyze

        Returns:
            SentimentScore with probabilities and score
        """
        shared_result = self._shared_scorer.score(text)
        return _shared_to_local_score(shared_result)

    def score_batch(
        self, texts: list[str], hash_fn: Callable[[str], str] | None = None
    ) -> tuple[list[SentimentScore], int, int]:
        """Score a batch of texts, using cache when available.

        Args:
            texts: List of texts to analyze
            hash_fn: Function to compute article hash from text. Defaults to
                compute_article_hash if cache is available.

        Returns:
            Tuple of (list of SentimentScore objects, cache_hits, new_scores)
        """
        if not texts:
            return [], 0, 0

        # Use compute_article_hash by default when cache is available
        if hash_fn is None and self._cache is not None:
            hash_fn = compute_article_hash

        # Use the shared scorer with stats if hash function provided
        shared_results, cache_hits, new_scores = self._shared_scorer.score_batch_with_stats(
            texts, hash_fn=hash_fn
        )

        return [_shared_to_local_score(r) for r in shared_results], cache_hits, new_scores

    def score_batch_simple(self, texts: list[str]) -> list[SentimentScore]:
        """Score a batch of texts (simple interface, ignores cache stats).

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScore objects
        """
        shared_results = self._shared_scorer.score_batch(texts)
        return [_shared_to_local_score(r) for r in shared_results]
