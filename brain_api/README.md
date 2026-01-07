# Brain API

FastAPI service for LearnFinance-2025. This is the Python "brain" that will handle ML inference, training, and evidence synthesis.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
cd brain_api
uv sync --all-extras

# Copy environment variables template and fill in your values
cp .env.example .env
```

## Run locally

```bash
uv run uvicorn brain_api.main:app --reload --port 8000
```

The API will be available at http://localhost:8000

### Docker networking (n8n integration)

If n8n runs in Docker while Brain API runs on your host machine, n8n must call `http://host.docker.internal:8000` (not `localhost`).

To ensure Brain API is reachable from Docker containers, bind to all interfaces:

```bash
uv run uvicorn brain_api.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Hello world |
| GET | `/health` | Generic health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/universe/halal` | Get halal stock universe |
| POST | `/train/lstm` | Train LSTM model (Sunday cron) |
| POST | `/inference/lstm` | LSTM weekly return predictions (Monday run). Returns predictions sorted highest gain → highest loss, with insufficient-history symbols at the end. |
| POST | `/signals/news` | **Current** news sentiment for inference. Uses **yfinance + FinBERT**. Returns recency-weighted sentiment scores per symbol. |
| POST | `/signals/news/historical` | **Historical** news sentiment for training. Uses **parquet file** (no rate limits). Takes date range, returns daily sentiment for all (date, symbol) combos. Missing data returns neutral (0.0). |
| POST | `/signals/fundamentals` | **Current** fundamentals for inference. Uses **yfinance** (no rate limits). |
| POST | `/signals/fundamentals/historical` | **Historical** fundamentals for training. Uses **Alpha Vantage** (25/day limit, cached). Takes date range, returns n symbols × m quarters. |

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. The `.env` file is auto-loaded on startup via `python-dotenv`.

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | None |
| `HF_MODEL_REPO` | HuggingFace model repository (e.g., `username/learnfinance-models`) | None |
| `HF_NEWS_SENTIMENT_REPO` | HuggingFace news sentiment dataset repository | None |
| `HF_TWITTER_SENTIMENT_REPO` | HuggingFace twitter sentiment dataset repository | None |
| `STORAGE_BACKEND` | Storage backend: `local` or `hf` | `local` |
| `ALPACA_API_KEY` | Alpaca News API key (for sentiment backfill) | None |
| `ALPACA_API_SECRET` | Alpaca News API secret | None |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key (for historical fundamentals) | None |
| `LSTM_TRAIN_LOOKBACK_YEARS` | Years of historical data for training | `15` |
| `LSTM_TRAIN_WINDOW_END_DATE` | Override training window end date (YYYY-MM-DD) | Today |

### Data Sources

| Endpoint | Data Source | Rate Limit | Cache |
|----------|-------------|------------|-------|
| `/signals/news` | yfinance + FinBERT | None | Run-based (JSON files) |
| `/signals/news/historical` | `data/output/daily_sentiment.parquet` | None | N/A (reads from file) |
| `/signals/fundamentals` | yfinance | None | Not needed |
| `/signals/fundamentals/historical` | Alpha Vantage | 25/day (free tier) | Yes (SQLite + JSON) |

The fundamentals historical endpoint response includes `api_status`:
```json
{"api_status": {"calls_today": 8, "daily_limit": 25, "remaining": 17}}
```

## API documentation

Once running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Run tests

```bash
uv run pytest
```



