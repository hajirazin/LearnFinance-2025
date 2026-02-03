# Plan 04: Weekly Context Endpoint

## Goal

Create `GET /run/weekly-context` endpoint that returns the top 20 halal symbols, as_of_date, and run_id for the weekly workflow.

## TODOs

- [ ] Create `brain_api/brain_api/routes/run.py` with GET /run/weekly-context endpoint
- [ ] Wire router in `main.py`
- [ ] Write tests for the endpoint in `tests/test_run.py`
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Endpoint Specification

### GET /run/weekly-context

**Response**:
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "DIS", "ADBE", "CRM", "NFLX", "COST", "PEP"],
  "as_of_date": "2026-02-03",
  "run_id": "paper:2026-02-03",
  "stock_count": 20
}
```

## Implementation Details

### Symbols Selection

1. Call existing `get_halal_universe()` function from `brain_api.universe`
2. Extract the `stocks` list from the response
3. Take the first 20 symbols (already sorted by weight/liquidity)

### as_of_date Calculation

- Get current date in Asia/Kolkata timezone
- Format as `YYYY-MM-DD`

### run_id Format

- Format: `paper:{as_of_date}`
- Example: `paper:2026-02-03`

### Code Structure

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from brain_api.universe import get_halal_universe

def get_weekly_context():
    universe = get_halal_universe()
    stocks = universe["stocks"][:20]
    symbols = [s["symbol"] for s in stocks]
    
    as_of_date = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    run_id = f"paper:{as_of_date}"
    
    return {
        "symbols": symbols,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "stock_count": len(symbols)
    }
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `brain_api/brain_api/routes/run.py` | Create |
| `brain_api/brain_api/main.py` | Add router |
| `brain_api/tests/test_run.py` | Create |

## Test Requirements

1. Test returns exactly 20 symbols
2. Test as_of_date format is YYYY-MM-DD
3. Test run_id format is paper:YYYY-MM-DD
4. Test stock_count equals 20
5. Test symbols are from halal universe (mock get_halal_universe)
