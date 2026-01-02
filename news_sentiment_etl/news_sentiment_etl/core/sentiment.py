"""FinBERT sentiment scoring with batching and GPU support."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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
    """FinBERT sentiment scorer with batching and GPU support.

    Singleton pattern to avoid loading the model multiple times.
    """

    _instance: "FinBERTScorer | None" = None
    _pipeline = None
    _device: str = "cpu"

    def __new__(cls, use_gpu: bool | None = None) -> "FinBERTScorer":
        """Create or return singleton instance.

        Args:
            use_gpu: Force GPU usage. None = auto-detect.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._use_gpu = use_gpu
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

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Score a batch of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScore objects
        """
        self._ensure_loaded()

        if not texts:
            return []

        try:
            batch_results = self._pipeline(texts)
        except Exception:
            # Return neutral on error
            return [
                SentimentScore(
                    label="neutral",
                    p_pos=0.33,
                    p_neg=0.33,
                    p_neu=0.34,
                    score=0.0,
                    confidence=0.34,
                )
                for _ in texts
            ]

        results = []
        for scores in batch_results:
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

            score = p_pos - p_neg
            confidence = max(p_pos, p_neg, p_neu)

            results.append(
                SentimentScore(
                    label=label,
                    p_pos=round(p_pos, 4),
                    p_neg=round(p_neg, 4),
                    p_neu=round(p_neu, 4),
                    score=round(score, 4),
                    confidence=round(confidence, 4),
                )
            )

        return results

