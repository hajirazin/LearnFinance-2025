# Inference Endpoints

## Overview

The inference endpoints run trained models to generate predictions. There are two categories:
- **Forecasters**: LSTM and PatchTST predict weekly stock returns
- **Allocators**: PPO and SAC determine portfolio weights

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/infer/lstm` | LSTM weekly return prediction |
| POST | `/infer/patchtst` | PatchTST weekly return prediction |
| POST | `/infer/ppo_lstm` | PPO+LSTM portfolio allocation |
| POST | `/infer/ppo_patchtst` | PPO+PatchTST portfolio allocation |
| POST | `/infer/sac_lstm` | SAC+LSTM portfolio allocation |
| POST | `/infer/sac_patchtst` | SAC+PatchTST portfolio allocation |

---

## Forecaster Endpoints

### POST /infer/lstm

**LSTM Weekly Return Prediction**

Predicts weekly returns using a pure-price LSTM model trained on OHLCV data.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /infer/lstm] --> B[Parse Request]
    B --> C[Get as_of_date]
    C --> D[Compute Week Boundaries<br/>Holiday-aware]
    
    D --> E[Load Model Artifacts]
    E --> F{Local Available?}
    F -->|Yes| G[Use Local Model]
    F -->|No| H[Download from HuggingFace]
    
    G & H --> I[Fetch Price History<br/>60 trading days]
    I --> J[Build Feature Sequences<br/>per Symbol]
    
    J --> K[Run Inference]
    K --> L[Sort by Predicted Return]
    L --> M[Return Predictions]
```

### LSTM Architecture

```mermaid
flowchart LR
    subgraph "Input"
        A[60-day OHLCV<br/>5 channels]
    end
    
    subgraph "Model"
        B[StandardScaler] --> C[LSTM Layer 1<br/>64 units]
        C --> D[Dropout]
        D --> E[LSTM Layer 2<br/>32 units]
        E --> F[Dense<br/>1 unit]
    end
    
    subgraph "Output"
        G[Weekly Return %]
    end
    
    A --> B
    F --> G
```

### Request Schema

```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "as_of_date": "2026-01-11"
}
```

### Response Schema

```json
{
  "predictions": [
    {
      "symbol": "AAPL",
      "predicted_weekly_return_pct": 2.34,
      "has_enough_history": true,
      "history_days_used": 60,
      "data_end_date": "2026-01-10"
    }
  ],
  "model_version": "v2026.01.05-abc123",
  "as_of_date": "2026-01-11",
  "target_week_start": "2026-01-13",
  "target_week_end": "2026-01-17"
}
```

---

### POST /infer/patchtst

**PatchTST OHLCV Prediction**

Predicts weekly returns using 5-channel OHLCV log returns (open, high, low, close, volume).

### Flow Diagram

```mermaid
flowchart TD
    A[POST /infer/patchtst] --> B[Parse Request]
    B --> C[Compute Week Boundaries]
    C --> D[Load Model Artifacts]
    
    D --> E[Fetch Data Sources]
    
    subgraph "Data Sources"
        E --> F[Price History<br/>yfinance]
        E --> G[News Sentiment<br/>yfinance + FinBERT]
        E --> H[Fundamentals<br/>yfinance]
    end
    
    F & G & H --> I[Build Multi-Channel Features<br/>11 channels]
    I --> J[Run Inference]
    J --> K[Return Predictions]
```

### PatchTST Input Channels

```mermaid
flowchart TB
    subgraph "OHLCV (5 channels)"
        A[Open Log Return]
        B[High Log Return]
        C[Low Log Return]
        D[Close Log Return]
        E[Volume Log Return]
    end
    
    subgraph "Sentiment (1 channel)"
        F[News Sentiment Score]
    end
    
    subgraph "Fundamentals (5 channels)"
        G[Gross Margin]
        H[Operating Margin]
        I[Net Margin]
        J[Current Ratio]
        K[Debt to Equity]
    end
    
    A & B & C & D & E & F & G & H & I & J & K --> L[11-Channel Input]
```

### Response Schema

```json
{
  "predictions": [...],
  "model_version": "v2026.01.05-def456",
  "as_of_date": "2026-01-11",
  "target_week_start": "2026-01-13",
  "target_week_end": "2026-01-17",
  "signals_used": ["ohlcv", "news_sentiment", "fundamentals"]
}
```

---

## Allocator Endpoints

### POST /infer/ppo_lstm

**PPO + LSTM Portfolio Allocation**

Uses PPO policy with LSTM forecasts to determine portfolio weights.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /infer/ppo_lstm] --> B[Parse Portfolio Snapshot]
    B --> C[Compute Current Weights]
    C --> D[Compute Week Boundaries]
    
    D --> E[Load PPO Model]
    E --> F[Build Current Signals<br/>News + Fundamentals]
    F --> G[Generate LSTM Forecasts]
    
    G --> H[Build State Vector]
    H --> I[Run PPO Policy]
    I --> J[Get Target Weights]
    J --> K[Compute Turnover]
    K --> L[Return Allocation]
```

### State Vector Composition

```mermaid
flowchart LR
    subgraph "Per-Stock Features"
        A[Current Weight]
        B[LSTM Forecast]
        C[News Sentiment]
        D[Gross Margin]
        E[Operating Margin]
        F[Net Margin]
        G[Current Ratio]
        H[Debt to Equity]
        I[Fundamental Age]
    end
    
    A & B & C & D & E & F & G & H & I --> J[Flatten to State Vector]
    J --> K[Normalize with Scaler]
    K --> L[PPO Policy Input]
```

### Request Schema

```json
{
  "portfolio": {
    "cash": 50000.00,
    "positions": [
      {"symbol": "AAPL", "shares": 100, "market_value": 15000.00},
      {"symbol": "MSFT", "shares": 50, "market_value": 20000.00}
    ]
  },
  "as_of_date": "2026-01-11"
}
```

### Response Schema

```json
{
  "target_weights": {
    "AAPL": 0.12,
    "MSFT": 0.15,
    "GOOGL": 0.10,
    "CASH": 0.63
  },
  "turnover": 0.18,
  "target_week_start": "2026-01-13",
  "target_week_end": "2026-01-17",
  "model_version": "v2026.01.05-ppo123",
  "weight_changes": [
    {"symbol": "AAPL", "current_weight": 0.18, "target_weight": 0.12, "change": -0.06},
    {"symbol": "MSFT", "current_weight": 0.24, "target_weight": 0.15, "change": -0.09}
  ]
}
```

---

### POST /infer/ppo_patchtst

**PPO + PatchTST Portfolio Allocation**

Same as PPO+LSTM but uses PatchTST forecasts instead.

```mermaid
flowchart LR
    A[Portfolio Snapshot] --> B[Current Weights]
    B --> C[Signals + PatchTST Forecasts]
    C --> D[PPO Policy]
    D --> E[Target Weights]
```

---

### POST /infer/sac_lstm

**SAC + LSTM Portfolio Allocation**

Uses Soft Actor-Critic policy with LSTM forecasts.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /infer/sac_lstm] --> B[Parse Portfolio]
    B --> C[Compute Current Weights Vector]
    C --> D[Load SAC Actor Model]
    
    D --> E[Build Signals]
    E --> F[Generate LSTM Forecasts]
    F --> G[Build State]
    
    G --> H[SAC Actor Network]
    H --> I[Sample Action<br/>or Mean Action]
    I --> J[Softmax → Weights]
    J --> K[Compute Turnover]
    K --> L[Return Allocation]
```

### SAC vs PPO

```mermaid
flowchart TB
    subgraph "PPO (On-Policy)"
        A[Collect Experience] --> B[Update Policy]
        B --> C[Discard Experience]
        C --> A
    end
    
    subgraph "SAC (Off-Policy)"
        D[Collect Experience] --> E[Store in Replay Buffer]
        E --> F[Sample Batch]
        F --> G[Update Actor & Critic]
        G --> D
    end
```

---

### POST /infer/sac_patchtst

**SAC + PatchTST Portfolio Allocation**

Same as SAC+LSTM but uses PatchTST forecasts.

---

## Model Comparison

| Endpoint | Forecaster | Allocator | Input Channels |
|----------|------------|-----------|----------------|
| `/infer/lstm` | LSTM | - | 5 (OHLCV) |
| `/infer/patchtst` | PatchTST | - | 11 (OHLCV + Sentiment + Fundamentals) |
| `/infer/ppo_lstm` | LSTM | PPO | 9 per stock |
| `/infer/ppo_patchtst` | PatchTST | PPO | 9 per stock |
| `/infer/sac_lstm` | LSTM | SAC | 9 per stock |
| `/infer/sac_patchtst` | PatchTST | SAC | 9 per stock |

---

## Week Boundary Computation

All inference endpoints use holiday-aware week boundary computation:

```mermaid
flowchart LR
    A[as_of_date] --> B{Is Trading Day?}
    B -->|Yes| C[Find Next Monday]
    B -->|No| D[Roll Forward]
    D --> B
    C --> E[target_week_start = Next Monday]
    E --> F[target_week_end = Following Friday]
    F --> G{Check Holidays}
    G -->|Holiday| H[Adjust Dates]
    G -->|No Holiday| I[Return Boundaries]
    H --> I
```

---

## Error Responses

| Status | Condition |
|--------|-----------|
| 503 | No trained model available |
| 400 | Invalid request parameters |
