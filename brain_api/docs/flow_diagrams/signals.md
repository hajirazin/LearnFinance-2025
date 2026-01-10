# Signals Endpoints

## Overview

The signals endpoints provide access to external market signals: news sentiment and fundamental financial ratios. These signals are inputs to the ML models.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/signals/news` | Get current news sentiment |
| POST | `/signals/news/historical` | Get historical news sentiment |
| POST | `/signals/fundamentals` | Get current fundamentals |
| POST | `/signals/fundamentals/historical` | Get historical fundamentals |

---

## POST /signals/news

**Get Current News Sentiment**

Fetches recent news articles and scores them with FinBERT for real-time inference.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /signals/news] --> B[Parse Request]
    B --> C[Determine as_of_date]
    B --> D[Generate run_id if not provided]
    
    C & D --> E{Check Cache}
    E -->|Hit| F[Return Cached Result]
    E -->|Miss| G[Fetch News from yfinance]
    
    G --> H[For Each Symbol]
    H --> I[Get Recent Articles]
    I --> J[Score with FinBERT]
    J --> K[Compute Aggregate Score<br/>Recency-weighted]
    
    K --> L[Persist Raw Articles]
    L --> M[Persist Features]
    M --> N[Return Top K Articles]
```

### FinBERT Scoring

```mermaid
flowchart LR
    subgraph "Per Article"
        A[Article Text] --> B[FinBERT Model]
        B --> C[p_positive]
        B --> D[p_negative]
        B --> E[p_neutral]
        C & D --> F["score = p_pos - p_neg"]
    end
    
    subgraph "Aggregation"
        F --> G[Apply Recency Weights]
        G --> H[Weighted Average]
        H --> I[Symbol Score -1 to +1]
    end
```

### Request Schema

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "run_id": "paper:2026-01-11",
  "attempt": 1,
  "as_of_date": "2026-01-11",
  "max_articles_per_symbol": 10,
  "return_top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `symbols` | array | required | Stock symbols to analyze |
| `run_id` | string | null | Run identifier for caching |
| `attempt` | int | 1 | Attempt number for idempotency |
| `as_of_date` | string | today | Reference date |
| `max_articles_per_symbol` | int | 10 | Max articles to fetch |
| `return_top_k` | int | 5 | Articles to return in response |

### Response Schema

```json
{
  "as_of_date": "2026-01-11",
  "per_symbol": [
    {
      "symbol": "AAPL",
      "aggregate_score": 0.25,
      "article_count": 8,
      "top_articles": [
        {
          "title": "Apple Reports Strong Quarter",
          "published": "2026-01-10T14:30:00Z",
          "sentiment_score": 0.65,
          "source": "Reuters"
        }
      ]
    }
  ]
}
```

---

## POST /signals/news/historical

**Get Historical News Sentiment**

Returns pre-computed daily sentiment from the ETL pipeline. Used for training data.

### Flow Diagram

```mermaid
flowchart LR
    A[POST /signals/news/historical] --> B[Read daily_sentiment.parquet]
    B --> C[Filter by Date Range]
    C --> D[Filter by Symbols]
    D --> E{Missing Data?}
    E -->|Yes| F[Fill with Neutral 0.0]
    E -->|No| G[Return Data]
    F --> G
```

### Request Schema

```json
{
  "symbols": ["AAPL", "MSFT"],
  "start_date": "2020-01-01",
  "end_date": "2025-12-31"
}
```

### Response Schema

```json
{
  "start_date": "2020-01-01",
  "end_date": "2025-12-31",
  "data": [
    {
      "symbol": "AAPL",
      "date": "2020-01-02",
      "sentiment_score": 0.15
    }
  ]
}
```

---

## POST /signals/fundamentals

**Get Current Fundamentals**

Fetches the most recent fundamental ratios from yfinance for real-time inference.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /signals/fundamentals] --> B[For Each Symbol]
    B --> C[Call yfinance ticker.info]
    
    C --> D{Data Available?}
    D -->|Yes| E[Extract Ratios]
    D -->|No| F[Return Error for Symbol]
    
    E --> G[gross_margin]
    E --> H[operating_margin]
    E --> I[net_margin]
    E --> J[current_ratio]
    E --> K[debt_to_equity]
    
    G & H & I & J & K --> L[Build Response]
    F --> L
    L --> M[Return Results]
```

### Fundamental Ratios

```mermaid
flowchart TB
    subgraph "Profitability"
        A[Gross Margin] --> D[Revenue - COGS / Revenue]
        B[Operating Margin] --> E[Operating Income / Revenue]
        C[Net Margin] --> F[Net Income / Revenue]
    end
    
    subgraph "Liquidity & Leverage"
        G[Current Ratio] --> H[Current Assets / Current Liabilities]
        I[Debt to Equity] --> J[Total Debt / Shareholder Equity]
    end
```

### Request Schema

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"]
}
```

### Response Schema

```json
{
  "as_of_date": "2026-01-11",
  "per_symbol": [
    {
      "symbol": "AAPL",
      "ratios": {
        "gross_margin": 0.43,
        "operating_margin": 0.30,
        "net_margin": 0.25,
        "current_ratio": 1.04,
        "debt_to_equity": 1.87
      },
      "error": null
    }
  ]
}
```

---

## POST /signals/fundamentals/historical

**Get Historical Fundamentals**

Fetches quarterly fundamental data from Alpha Vantage for training data.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /signals/fundamentals/historical] --> B[For Each Symbol]
    
    B --> C[Fetch from Alpha Vantage]
    C --> D[Check Cache]
    D -->|Hit| E[Use Cached Data]
    D -->|Miss| F[API Call: INCOME_STATEMENT]
    F --> G[API Call: BALANCE_SHEET]
    
    E & G --> H[Parse Quarterly Statements]
    H --> I[Filter by Date Range]
    I --> J[Compute Ratios for Each Quarter]
    
    J --> K[Collect All Ratios]
    K --> L[Return with API Status]
```

### Request Schema

```json
{
  "symbols": ["AAPL", "MSFT"],
  "start_date": "2020-01-01",
  "end_date": "2025-12-31",
  "force_refresh": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `symbols` | array | Stock symbols |
| `start_date` | string | Start of date range |
| `end_date` | string | End of date range |
| `force_refresh` | bool | Bypass cache and refresh from API |

### Response Schema

```json
{
  "start_date": "2020-01-01",
  "end_date": "2025-12-31",
  "api_status": {
    "calls_today": 25,
    "calls_remaining": 475,
    "daily_limit": 500
  },
  "data": [
    {
      "symbol": "AAPL",
      "as_of_date": "2024-09-30",
      "gross_margin": 0.46,
      "operating_margin": 0.31,
      "net_margin": 0.24,
      "current_ratio": 0.99,
      "debt_to_equity": 1.72
    }
  ]
}
```

---

## Signal Data Flow

```mermaid
flowchart TB
    subgraph "Real-Time (Inference)"
        A[yfinance News] --> B[FinBERT]
        B --> C[News Sentiment]
        D[yfinance Info] --> E[Fundamentals]
    end
    
    subgraph "Historical (Training)"
        F[HuggingFace Dataset] --> G[FinBERT ETL]
        G --> H[daily_sentiment.parquet]
        I[Alpha Vantage] --> J[Quarterly Statements]
        J --> K[Fundamentals Cache]
    end
    
    subgraph "Model Input"
        C --> L[State Vector]
        E --> L
        H --> M[Training Data]
        K --> M
    end
```

---

## Data Sources Summary

| Signal | Real-Time Source | Historical Source |
|--------|-----------------|-------------------|
| News Sentiment | yfinance + FinBERT | HuggingFace + FinBERT ETL |
| Fundamentals | yfinance ticker.info | Alpha Vantage API |
