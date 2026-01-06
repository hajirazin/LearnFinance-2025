"""ETL pipeline for news sentiment processing.

This package contains the batch ETL pipeline moved from news_sentiment_etl.
It processes historical news articles from HuggingFace datasets using FinBERT
and outputs daily sentiment scores per symbol.
"""

from brain_api.etl.aggregation import DailySentiment
from brain_api.etl.config import ETLConfig
from brain_api.etl.pipeline import run_pipeline

__all__ = [
    "DailySentiment",
    "ETLConfig",
    "run_pipeline",
]

