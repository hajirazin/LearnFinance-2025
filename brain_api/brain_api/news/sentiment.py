"""Strict pinned FinBERT scorer. Fail closed. No silent-neutral fallback."""

from __future__ import annotations

import hashlib
import html
import logging
import math
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

from brain_api.news.errors import SentimentScoringError
from brain_api.news.models import (
    FINBERT_BATCH_SIZE,
    FINBERT_MAX_LENGTH,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    SCORING_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def normalize_scored_text(text: str) -> str:
    unescaped = html.unescape(text or "")
    stripped = _TAG_RE.sub(" ", unescaped)
    normalized = unicodedata.normalize("NFKC", stripped)
    return _WS_RE.sub(" ", normalized).strip()


def assemble_scored_text(headline: str, summary: str) -> str:
    head = normalize_scored_text(headline)
    summ = normalize_scored_text(summary)
    if not head and not summ:
        raise SentimentScoringError("empty text after normalization")
    if summ:
        return f"{head} {summ}".strip()
    return head


def scored_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoredSentiment:
    sentiment_score: float
    p_positive: float
    p_negative: float
    p_neutral: float
    confidence: float
    scored_text_sha256: str
    scored_text: str


class StrictFinBERTScorer:
    """Pinned ProsusAI/finbert. Exceptions propagate. Cache only successes."""

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        batch_size: int = FINBERT_BATCH_SIZE,
    ) -> None:
        self._injected_pipeline = pipeline
        self._pipeline: Any | None = pipeline
        self._load_lock = threading.Lock()
        self.batch_size = batch_size
        self.model = NEWS_SENTIMENT_MODEL
        self.revision = NEWS_SENTIMENT_REVISION
        self.scoring_schema_version = SCORING_SCHEMA_VERSION

    def _ensure_loaded(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from transformers import pipeline as hf_pipeline

            if torch.cuda.is_available():
                device_name = "cuda"
                device_param: int | str = 0
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device_name = "mps"
                device_param = "mps"
            else:
                device_name = "cpu"
                device_param = -1
            tokenizer = AutoTokenizer.from_pretrained(
                NEWS_SENTIMENT_MODEL, revision=NEWS_SENTIMENT_REVISION
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                NEWS_SENTIMENT_MODEL, revision=NEWS_SENTIMENT_REVISION
            )
            model.eval()
            for module in model.modules():
                if hasattr(module, "dropout"):
                    module.dropout = torch.nn.Dropout(p=0.0)
            if device_name != "cpu":
                model = model.to(device_name)
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=FINBERT_MAX_LENGTH,
                batch_size=self.batch_size,
                device=device_param,
            )
            logger.info(
                "FinBERT loaded model=%s revision=%s device=%s batch_size=%s",
                NEWS_SENTIMENT_MODEL,
                NEWS_SENTIMENT_REVISION,
                device_name,
                self.batch_size,
            )
            return self._pipeline

    def score_texts(self, texts: list[str]) -> list[ScoredSentiment]:
        if not texts:
            return []
        pipe = self._ensure_loaded()
        results: list[ScoredSentiment] = []
        scored_new = 0
        started = time.perf_counter()
        try:
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset : offset + self.batch_size]
                raw = pipe(batch)
                if len(raw) != len(batch):
                    raise SentimentScoringError("FinBERT batch size mismatch")
                for text, item in zip(batch, raw, strict=True):
                    results.append(self._parse_item(text, item))
                scored_new += len(batch)
                if scored_new % 100 == 0:
                    logger.info(
                        "FinBERT scored=%s cache_hits=0 elapsed_ms=%.0f",
                        scored_new,
                        (time.perf_counter() - started) * 1000,
                    )
        except SentimentScoringError:
            raise
        except Exception as exc:
            raise SentimentScoringError("FinBERT scoring failed") from exc
        return results

    @staticmethod
    def _parse_item(text: str, item: object) -> ScoredSentiment:
        rows = item if isinstance(item, list) else [item]
        required = {"positive", "negative", "neutral"}
        probs: dict[str, float] = {}
        try:
            for row in rows:
                label = str(row["label"]).lower()
                score = float(row["score"])
                if label not in required:
                    raise SentimentScoringError(
                        f"FinBERT returned unknown label {label!r}"
                    )
                if label in probs:
                    raise SentimentScoringError(
                        f"FinBERT returned duplicate label {label!r}"
                    )
                if not math.isfinite(score) or score < 0.0 or score > 1.0:
                    raise SentimentScoringError(
                        "FinBERT probability is not a finite value in [0, 1]"
                    )
                probs[label] = score
        except SentimentScoringError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SentimentScoringError("FinBERT returned an unusable payload") from exc
        missing = required - probs.keys()
        if missing:
            raise SentimentScoringError(f"FinBERT missing labels {sorted(missing)}")
        p_pos = probs["positive"]
        p_neg = probs["negative"]
        p_neu = probs["neutral"]
        total = p_pos + p_neg + p_neu
        if not math.isfinite(total) or abs(total - 1.0) > 1e-3:
            raise SentimentScoringError(f"FinBERT probabilities sum to {total}, not 1")
        return ScoredSentiment(
            sentiment_score=p_pos - p_neg,
            p_positive=p_pos,
            p_negative=p_neg,
            p_neutral=p_neu,
            confidence=max(p_pos, p_neg, p_neu),
            scored_text_sha256=scored_text_sha256(text),
            scored_text=text,
        )
