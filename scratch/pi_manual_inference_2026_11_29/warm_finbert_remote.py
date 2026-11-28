"""Warm the deployed scorer's pinned cache without writing news or models."""

import json
import logging
import time

import torch

from brain_api.news.models import NEWS_SENTIMENT_MODEL, NEWS_SENTIMENT_REVISION
from brain_api.news.sentiment import StrictFinBERTScorer

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(1)
start = time.monotonic()
scorer = StrictFinBERTScorer(batch_size=1)
scores = scorer.score_texts(
    ["Readiness probe: the company reported its quarterly results."]
)
assert len(scores) == 1
print(
    json.dumps(
        {
            "status": "ready",
            "model": NEWS_SENTIMENT_MODEL,
            "revision": NEWS_SENTIMENT_REVISION,
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }
    ),
    flush=True,
)
