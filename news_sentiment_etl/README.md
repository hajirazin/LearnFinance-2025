# News Sentiment ETL

ETL pipeline to process the HuggingFace `Brianferrell787/financial-news-multisource` dataset (57M+ news articles) into daily aggregated sentiment scores per stock symbol.

## Features

- **Streaming Processing**: Handles 57M+ articles without loading everything into memory
- **FinBERT Sentiment Analysis**: Uses ProsusAI/finbert for financial-specific sentiment
- **Confidence-Weighted Aggregation**: Research-backed method for combining multiple sentiments
- **Bounded Filtering**: Filters out ambiguous articles for cleaner signal
- **Halal Universe Filtering**: Optional filtering to halal ETF holdings (SPUS, HLAL, SPTE)
- **GPU Support**: Auto-detects CUDA/MPS for faster inference
- **Checkpointing**: Resume from interruption
- **Cloud-Ready**: Deployable as Google Cloud Function

## Installation

```bash
cd news_sentiment_etl
pip install -e .
```

Or with uv:
```bash
cd news_sentiment_etl
uv pip install -e .
```

## Usage

### CLI

```bash
# Basic run (with halal filtering)
news-sentiment-etl --output-dir data/output

# Test with limited articles
news-sentiment-etl --max-articles 1000

# Disable halal filtering (include all symbols)
news-sentiment-etl --no-halal-filter

# Force CPU
news-sentiment-etl --cpu

# Custom batch size and threshold
news-sentiment-etl --batch-size 512 --threshold 0.15
```

### Python API

```python
from pathlib import Path
from news_sentiment_etl.core.config import ETLConfig
from news_sentiment_etl.main import run_pipeline

config = ETLConfig(
    output_dir=Path("data/output"),
    batch_size=256,
    sentiment_threshold=0.1,
    filter_to_halal=True,
    max_articles=10000,  # For testing
)

stats = run_pipeline(config)
print(f"Processed {stats['articles']['total_processed']} articles")
```

### Google Cloud Function

Deploy with `handler` as entrypoint:

```bash
gcloud functions deploy news-sentiment-etl \
    --runtime python311 \
    --trigger-http \
    --entry-point handler \
    --memory 4096MB \
    --timeout 540s
```

## Output Schema

The output Parquet file contains:

| Column | Type | Description |
|--------|------|-------------|
| date | date | Trading date (YYYY-MM-DD) |
| symbol | string | Stock ticker |
| sentiment_score | float | Aggregated score [-1, 1] |
| article_count | int | Number of articles used |
| avg_confidence | float | Average FinBERT confidence |
| p_pos_avg | float | Average positive probability |
| p_neg_avg | float | Average negative probability |
| total_articles | int | Articles before filtering |

## Aggregation Method

Uses **confidence-weighted averaging with bounded filtering** (research-backed):

1. **Bounded Filter**: Exclude articles where `|p_pos - p_neg| < 0.1` (too ambiguous)
2. **Confidence Weight**: Weight = `max(p_pos, p_neg, p_neu)`
3. **Aggregate**: `daily_sentiment = Σ(weight × score) / Σ(weight)`

This approach is supported by [VU Business Analytics research](https://vu-business-analytics.github.io/internship-office/papers/paper-be.pdf).

## Requirements

- Python 3.11+
- ~8GB RAM for streaming processing
- GPU recommended for faster inference (but CPU works)
- HuggingFace account (dataset requires access request)

## Dataset Access

The dataset requires access approval. Request access at:
https://huggingface.co/datasets/Brianferrell787/financial-news-multisource

Then login:
```bash
huggingface-cli login
```

