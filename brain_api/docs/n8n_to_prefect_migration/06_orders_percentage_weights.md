# Plan 06: Orders Percentage Weights Extension

## Goal

Extend `POST /orders/generate` endpoint to accept `percentage_weights` in addition to `target_weights`. When `percentage_weights` is provided, convert to decimal weights internally.

## TODOs

- [ ] Extend `GenerateOrdersRequest` model to accept optional `percentage_weights` field
- [ ] Add conversion logic: if `percentage_weights` provided, convert to `target_weights` by dividing by 100
- [ ] Update endpoint logic to use the converted weights
- [ ] Write tests for the new functionality in `tests/test_orders.py` (add to existing file)
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Endpoint Specification

### POST /orders/generate (Extended)

**Request Option 1 (existing - decimal weights)**:
```json
{
  "target_weights": {"AAPL": 0.105, "MSFT": 0.082, "CASH": 0.05},
  "portfolio": {
    "cash": 10000.0,
    "positions": [{"symbol": "AAPL", "qty": 5.0, "market_value": 875.50}]
  },
  "run_id": "paper:2026-02-03",
  "attempt": 1,
  "algorithm": "ppo"
}
```

**Request Option 2 (new - percentage weights for HRP)**:
```json
{
  "percentage_weights": {"AAPL": 10.5, "MSFT": 8.2, "CASH": 5.0},
  "portfolio": {
    "cash": 10000.0,
    "positions": [{"symbol": "AAPL", "qty": 5.0, "market_value": 875.50}]
  },
  "run_id": "paper:2026-02-03",
  "attempt": 1,
  "algorithm": "hrp"
}
```

**Response**: Same as existing endpoint.

## Implementation Details

### Conversion Logic

```python
if request.percentage_weights:
    target_weights = {
        symbol: weight / 100
        for symbol, weight in request.percentage_weights.items()
    }
elif request.target_weights:
    target_weights = request.target_weights
else:
    raise HTTPException(422, "Either target_weights or percentage_weights required")
```

### Model Changes

```python
class GenerateOrdersRequest(BaseModel):
    target_weights: dict[str, float] | None = Field(
        None,
        description="Target allocation weights as decimals (0.105 = 10.5%)"
    )
    percentage_weights: dict[str, float] | None = Field(
        None,
        description="Target allocation weights as percentages (10.5 = 10.5%)"
    )
    portfolio: PortfolioModel
    run_id: str
    attempt: int
    algorithm: str
    
    @model_validator(mode="after")
    def validate_weights(self):
        if not self.target_weights and not self.percentage_weights:
            raise ValueError("Either target_weights or percentage_weights required")
        if self.target_weights and self.percentage_weights:
            raise ValueError("Provide either target_weights or percentage_weights, not both")
        return self
```

## Files to Modify

| File | Action |
|------|--------|
| `brain_api/brain_api/routes/orders.py` | Extend model and logic |
| `brain_api/tests/test_orders.py` | Add tests |

## Test Requirements

1. Test with target_weights (existing behavior unchanged)
2. Test with percentage_weights converts correctly (10.5 → 0.105)
3. Test with both target_weights and percentage_weights returns 422
4. Test with neither target_weights nor percentage_weights returns 422
5. Test percentage_weights with CASH key
