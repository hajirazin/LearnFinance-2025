"""FinBERT sentiment scoring with batching and GPU support."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from news_sentiment_etl.core.cache import SentimentCache, compute_article_hash
from news_sentiment_etl.core.config import FINBERT_MAX_LENGTH, FINBERT_MODEL


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


class FinBERTScorer:
    """FinBERT sentiment scorer with batching, GPU support, and caching.

    Uses SQLite cache to avoid recomputing sentiment for articles
    that have already been scored in previous runs.
    """

    _instance: "FinBERTScorer | None" = None
    _pipeline = None
    _device: str = "cpu"
    _cache: SentimentCache | None = None

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
            cls._instance._use_gpu = use_gpu
            cls._instance._cache = cache
        elif cache is not None and cls._instance._cache is None:
            # Update cache if provided on subsequent call
            cls._instance._cache = cache
        return cls._instance

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

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            device=0 if self._device == "cuda" else -1 if self._device == "cpu" else self._device,
        )

    @property
    def device(self) -> str:
        """Return the device being used."""
        self._ensure_loaded()
        return self._device

    def score(self, text: str) -> SentimentScore:
        """Score a single text.

        Args:
            text: Text to analyze

        Returns:
            SentimentScore with probabilities and score
        """
        return self.score_batch([text])[0]

    def score_batch(
        self, texts: list[str]
    ) -> tuple[list[SentimentScore], int, int]:
        """Score a batch of texts, using cache when available.

        Args:
            texts: List of texts to analyze

        Returns:
            Tuple of (list of SentimentScore objects, cache_hits, new_scores)
        """
        if not texts:
            return [], 0, 0

        # Compute hashes for all texts
        hashes = [compute_article_hash(text) for text in texts]

        # Check cache for existing scores
        results: list[SentimentScore | None] = [None] * len(texts)
        texts_to_score: list[tuple[int, str, str]] = []  # (index, text, hash)

        if self._cache is not None:
            cached = self._cache.get_batch(hashes)
            for i, (text, h) in enumerate(zip(texts, hashes)):
                if cached.get(h) is not None:
                    results[i] = cached[h]
                else:
                    texts_to_score.append((i, text, h))
        else:
            texts_to_score = [(i, text, h) for i, (text, h) in enumerate(zip(texts, hashes))]

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

            for (orig_idx, _, article_hash), scores in zip(texts_to_score, batch_results):
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
                new_cache_entries.append((article_hash, sentiment_score))

            # Batch insert into cache
            if self._cache is not None and new_cache_entries:
                self._cache.put_batch(new_cache_entries)

        # All results should be filled now
        return [r for r in results if r is not None], cache_hits, new_scores

    def score_batch_simple(self, texts: list[str]) -> list[SentimentScore]:
        """Score a batch of texts (simple interface, ignores cache stats).

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScore objects
        """
        scores, _, _ = self.score_batch(texts)
        return scores

