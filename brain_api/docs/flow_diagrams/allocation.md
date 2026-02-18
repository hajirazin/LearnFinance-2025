# Allocation Endpoints

## Overview

The allocation endpoints compute portfolio weights using optimization algorithms. Currently supports Hierarchical Risk Parity (HRP) allocation.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/allocation/hrp` | Compute HRP portfolio allocation |

---

## POST /allocation/hrp

**Compute HRP Portfolio Allocation**

Computes portfolio weights using Hierarchical Risk Parity (López de Prado, 2016). HRP uses hierarchical clustering to group similar assets and recursive bisection to allocate weights by inverse variance.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /allocation/hrp] --> B[Parse Request]
    
    B --> C[Get Halal Universe<br/>~14 symbols]
    B --> D[Determine as_of_date]
    
    C --> E[Fetch Price History<br/>from yfinance]
    D --> E
    
    E --> F{All Symbols<br/>Have Data?}
    F -->|Some Missing| G[Track Excluded Symbols]
    F -->|All Present| H[Build Returns Matrix]
    G --> H
    
    H --> I[Compute Correlation Matrix]
    I --> J[Convert to Distance Matrix]
    J --> K[Hierarchical Clustering]
    K --> L[Recursive Bisection]
    L --> M[Inverse Variance Weighting]
    
    M --> N[Normalize to 100%]
    N --> O[Sort by Weight Descending]
    O --> P[Return Allocation]
```

### Algorithm Details

```mermaid
flowchart LR
    subgraph "HRP Algorithm"
        A[Daily Returns] --> B[Correlation Matrix]
        B --> C["Distance = √(0.5 × (1 - corr))"]
        C --> D[Ward's Clustering]
        D --> E[Dendrogram]
        E --> F[Quasi-Diagonalization]
        F --> G[Recursive Bisection]
        G --> H[Inverse Variance Weights]
    end
```

### Request Schema

```json
{
  "lookback_days": 252,
  "as_of_date": "2026-01-10"
}
```

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `lookback_days` | int | 252 | 60-504 | Trading days for return calculation |
| `as_of_date` | string | today | - | Reference date (YYYY-MM-DD) |

### Response Schema

```json
{
  "percentage_weights": {
    "AAPL": 8.23,
    "MSFT": 7.15,
    "GOOGL": 6.42,
    "...": "..."
  },
  "symbols_used": 42,
  "symbols_excluded": ["XYZ", "ABC"],
  "lookback_days": 252,
  "as_of_date": "2026-01-10"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `percentage_weights` | dict | Symbol → weight (%) sorted descending |
| `symbols_used` | int | Number of symbols in allocation |
| `symbols_excluded` | array | Symbols with insufficient data |
| `lookback_days` | int | Trading days used |
| `as_of_date` | string | Reference date used |

### Error Responses

| Status | Condition |
|--------|-----------|
| 400 | No symbols have sufficient data |

---

## Why HRP?

```mermaid
flowchart TB
    subgraph "Traditional MVO Problems"
        A[Mean-Variance Optimization]
        A --> B[Sensitive to Estimation Errors]
        A --> C[Concentrated Portfolios]
        A --> D[Requires Return Forecasts]
    end
    
    subgraph "HRP Advantages"
        E[Hierarchical Risk Parity]
        E --> F[Uses Only Covariance]
        E --> G[Naturally Diversified]
        E --> H[Stable Allocations]
        E --> I[No Return Forecasts Needed]
    end
```

---

## Usage Example

```python
import httpx

response = httpx.post(
    "http://localhost:8000/allocation/hrp",
    json={
        "lookback_days": 252,
        "as_of_date": "2026-01-10"
    }
)

allocation = response.json()
print(f"Top holdings: {list(allocation['percentage_weights'].items())[:5]}")
```
