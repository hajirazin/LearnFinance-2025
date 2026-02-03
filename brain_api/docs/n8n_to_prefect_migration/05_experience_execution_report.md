# Plan 05: Experience Build Execution Report Endpoint

## Goal

Extend `routes/experience.py` with `POST /experience/build-execution-report` endpoint that matches intended orders with executed orders by `client_order_id` and returns an execution report.

## TODOs

- [ ] Add POST /experience/build-execution-report endpoint to `brain_api/brain_api/routes/experience.py`
- [ ] Add Pydantic models for request/response
- [ ] Write tests for the endpoint in `tests/test_experience.py` (add to existing file)
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Endpoint Specification

### POST /experience/build-execution-report

**Request**:
```json
{
  "run_id": "paper:2026-02-03",
  "model_type": "ppo",
  "intended_orders": [
    {
      "symbol": "AAPL",
      "qty": 5,
      "side": "buy",
      "client_order_id": "paper:2026-02-03:attempt-1:AAPL:BUY"
    },
    {
      "symbol": "MSFT",
      "qty": 3,
      "side": "buy",
      "client_order_id": "paper:2026-02-03:attempt-1:MSFT:BUY"
    }
  ],
  "executed_orders": [
    {
      "client_order_id": "paper:2026-02-03:attempt-1:AAPL:BUY",
      "status": "filled",
      "filled_qty": "5",
      "filled_avg_price": "175.25"
    },
    {
      "client_order_id": "paper:2026-02-03:attempt-1:MSFT:BUY",
      "status": "partially_filled",
      "filled_qty": "2",
      "filled_avg_price": "420.50"
    }
  ]
}
```

**Response**:
```json
{
  "run_id": "paper:2026-02-03",
  "model_type": "ppo",
  "execution_report": [
    {
      "symbol": "AAPL",
      "side": "buy",
      "intended_qty": 5,
      "filled_qty": 5,
      "filled_avg_price": 175.25,
      "status": "filled",
      "client_order_id": "paper:2026-02-03:attempt-1:AAPL:BUY"
    },
    {
      "symbol": "MSFT",
      "side": "buy",
      "intended_qty": 3,
      "filled_qty": 2,
      "filled_avg_price": 420.50,
      "status": "partially_filled",
      "client_order_id": "paper:2026-02-03:attempt-1:MSFT:BUY"
    }
  ]
}
```

## Implementation Details

### Matching Logic

For each intended order:
1. Find the executed order with matching `client_order_id`
2. If found, extract `status`, `filled_qty`, `filled_avg_price`
3. If not found, set `status: "not_found"`, `filled_qty: 0`, `filled_avg_price: null`

### Code Structure

```python
def build_execution_report(intended_orders, executed_orders):
    executed_map = {o["client_order_id"]: o for o in executed_orders}
    
    report = []
    for intended in intended_orders:
        executed = executed_map.get(intended["client_order_id"])
        report.append({
            "symbol": intended["symbol"],
            "side": intended["side"],
            "intended_qty": intended["qty"],
            "filled_qty": float(executed["filled_qty"]) if executed else 0,
            "filled_avg_price": float(executed["filled_avg_price"]) if executed and executed.get("filled_avg_price") else None,
            "status": executed["status"] if executed else "not_found",
            "client_order_id": intended["client_order_id"]
        })
    
    return report
```

## Files to Modify

| File | Action |
|------|--------|
| `brain_api/brain_api/routes/experience.py` | Add endpoint and models |
| `brain_api/tests/test_experience.py` | Add tests |

## Test Requirements

1. Test all intended orders matched with executed orders
2. Test some intended orders not found in executed (status: "not_found")
3. Test executed orders with partial fills
4. Test empty intended_orders returns empty report
5. Test invalid request returns 422
