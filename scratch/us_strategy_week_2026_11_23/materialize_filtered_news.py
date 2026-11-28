"""Fetch one canonical Monday news window into a scratch-only store."""

import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from brain_api.news.models import (
    FINBERT_MAX_LENGTH,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    NewsWindow,
)
from brain_api.news.sentiment import StrictFinBERTScorer
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore
from brain_api.routes.news.endpoints import _to_result

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
load_dotenv(REPO / "brain_api/.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
torch.set_num_threads(1)
smoke = ROOT.parent / "pi_manual_inference_2026_11_29"
symbols = json.loads((smoke / "active_halal_filtered.json").read_text())["response"][
    "symbols"
]
window_data = json.loads((smoke / "decision_window.json").read_text())["response"]
window = NewsWindow(
    datetime.fromisoformat(window_data["start_exclusive"]),
    datetime.fromisoformat(window_data["end_inclusive"]),
)
tokenizer = AutoTokenizer.from_pretrained(
    NEWS_SENTIMENT_MODEL, revision=NEWS_SENTIMENT_REVISION, local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    NEWS_SENTIMENT_MODEL, revision=NEWS_SENTIMENT_REVISION, local_files_only=True
)
model.eval()
for module in model.modules():
    if hasattr(module, "dropout"):
        module.dropout = torch.nn.Dropout(p=0.0)
pipe = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    truncation=True,
    max_length=FINBERT_MAX_LENGTH,
    batch_size=8,
    device=-1,
)
scorer = StrictFinBERTScorer(pipeline=pipe, batch_size=8)
service = NewsService(NewsStore(ROOT / "scratch_news"), scorer=scorer)
coverage, events = service.materialize(symbols, window)
result = _to_result(window, coverage, events).model_dump(mode="json")
(ROOT / "filtered_news_nov23.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "symbols": len(symbols),
            "events": len(events),
            "coverage": result["coverage"],
        },
        indent=2,
    ),
    flush=True,
)
