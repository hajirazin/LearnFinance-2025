# LearnFinance Shared

Shared utilities for LearnFinance-2025. Contains code used by both `brain_api` and `news_sentiment_etl`.

## Modules

### universe
- `get_halal_universe()` - Fetch halal stock universe from ETF holdings (SPUS, HLAL, SPTE)

### ml
- `get_device()` - Detect best available device (MPS, CUDA, CPU)
- `FinBERTScorer` - Singleton FinBERT sentiment scorer with batching and caching

## Installation

From the project root:

```bash
# For brain_api
cd brain_api
uv pip install -e ../shared

# For news_sentiment_etl
cd news_sentiment_etl
uv pip install -e ../shared
```

## Usage

```python
from shared.universe import get_halal_universe
from shared.ml import get_device, FinBERTScorer

# Get halal stocks
universe = get_halal_universe()
symbols = [stock["symbol"] for stock in universe["stocks"]]

# Detect device
device = get_device()  # Returns "mps", "cuda", or "cpu"

# Score sentiment
scorer = FinBERTScorer()
result = scorer.score("Apple stock rises on strong earnings")
print(result.label, result.score)
```

