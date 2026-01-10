# Universe Endpoints

## Overview

The universe endpoints provide access to the stock universe used for trading. Currently supports the halal-compliant stock universe derived from Shariah-compliant ETFs.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/universe/halal` | Get halal stock universe |

---

## GET /universe/halal

**Get Halal Stock Universe**

Returns the deduplicated union of top holdings from halal ETFs (SPUS, HLAL, SPTE). This defines the investable universe for the portfolio allocation system.

### Flow Diagram

```mermaid
flowchart TD
    A[Client Request] --> B["/universe/halal"]
    B --> C[get_halal_universe]
    
    subgraph Data Sources
        D[SPUS ETF Holdings]
        E[HLAL ETF Holdings]
        F[SPTE ETF Holdings]
    end
    
    D --> G[Merge & Deduplicate]
    E --> G
    F --> G
    
    G --> H[Compute Max Weight per Symbol]
    H --> I[Track Source ETFs]
    I --> J[Return Universe Data]
    
    C --> D
    C --> E
    C --> F
```

### Response Schema

```json
{
  "stocks": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "max_weight": 8.5,
      "etfs": ["SPUS", "HLAL", "SPTE"]
    },
    {
      "symbol": "MSFT",
      "name": "Microsoft Corporation",
      "max_weight": 7.2,
      "etfs": ["SPUS", "HLAL"]
    }
    // ... more stocks
  ],
  "total_count": 45,
  "source_etfs": ["SPUS", "HLAL", "SPTE"]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `stocks` | array | List of stocks in the universe |
| `stocks[].symbol` | string | Stock ticker symbol |
| `stocks[].name` | string | Company name |
| `stocks[].max_weight` | float | Maximum weight across source ETFs (%) |
| `stocks[].etfs` | array | List of ETFs holding this stock |
| `total_count` | int | Total number of unique stocks |
| `source_etfs` | array | ETFs used to build the universe |

---

## High-Level Architecture

```mermaid
flowchart TB
    subgraph "Halal Universe Construction"
        A[SPUS ETF<br/>SP Funds S&P 500] --> D[Union]
        B[HLAL ETF<br/>Wahed FTSE USA] --> D
        C[SPTE ETF<br/>SP Funds S&P 500 ESG] --> D
        
        D --> E[Deduplicate by Symbol]
        E --> F[Calculate Max Weight]
        F --> G[~45 Unique Stocks]
    end
    
    subgraph "Usage"
        G --> H[HRP Allocation]
        G --> I[PPO/SAC Training]
        G --> J[Inference Endpoints]
    end
```

---

## Usage

The halal universe is used throughout the system:
- **Allocation**: HRP endpoint uses this as the investment universe
- **Training**: RL models (PPO, SAC) train on top 15 symbols from this universe
- **Inference**: LSTM/PatchTST predictions cover symbols in this universe
- **Signals**: News sentiment and fundamentals are fetched for these symbols
