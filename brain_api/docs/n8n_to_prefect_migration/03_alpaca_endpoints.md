# Plan 03: Alpaca Endpoints

## Goal

Create three Alpaca endpoints in brain_api that handle all Alpaca API interactions:
1. `GET /alpaca/portfolio` - Get account, positions, and open orders count
2. `POST /alpaca/submit-orders` - Submit an array of orders
3. `GET /alpaca/order-history` - Get order history

## TODOs

- [ ] Create `brain_api/brain_api/routes/alpaca.py` with all three endpoints
- [ ] Add env vars to `.env.example`: `ALPACA_PPO_KEY`, `ALPACA_PPO_SECRET`, `ALPACA_SAC_KEY`, `ALPACA_SAC_SECRET`, `ALPACA_HRP_KEY`, `ALPACA_HRP_SECRET`
- [ ] Wire router in `main.py`
- [ ] Write tests for all endpoints in `tests/test_alpaca.py`
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Endpoint Specifications

### GET /alpaca/portfolio

**Query Parameters**:
- `account`: Required. One of `ppo`, `sac`, `hrp`

**Response**:
```json
{
  "cash": 10000.0,
  "positions": [
    {"symbol": "AAPL", "qty": 5.0, "market_value": 875.50}
  ],
  "open_orders_count": 0
}
```

**Implementation**:
1. Get Alpaca credentials for the specified account from env vars
2. Call Alpaca API: `GET /v2/account` → extract `cash`
3. Call Alpaca API: `GET /v2/positions` → normalize to `{symbol, qty, market_value}`
4. Call Alpaca API: `GET /v2/orders?status=open&limit=1` → count open orders
5. Return combined response

### POST /alpaca/submit-orders

**Request**:
```json
{
  "account": "ppo",
  "orders": [
    {
      "symbol": "AAPL",
      "qty": 5,
      "side": "buy",
      "type": "limit",
      "time_in_force": "day",
      "limit_price": 175.50,
      "client_order_id": "paper:2026-02-03:attempt-1:AAPL:BUY"
    }
  ]
}
```

**Response**:
```json
{
  "account": "ppo",
  "orders_submitted": 1,
  "orders_failed": 0,
  "skipped": false,
  "results": [
    {"id": "abc123", "status": "accepted", "symbol": "AAPL", "client_order_id": "paper:2026-02-03:attempt-1:AAPL:BUY"}
  ]
}
```

**Implementation**:
1. Get Alpaca credentials for the specified account
2. For each order in the array, call Alpaca API: `POST /v2/orders`
3. Collect results (success or error per order)
4. Return summary with counts and individual results

### GET /alpaca/order-history

**Query Parameters**:
- `account`: Required. One of `ppo`, `sac`, `hrp`
- `after`: Required. ISO date string (e.g., `2026-02-03`)

**Response**:
```json
[
  {
    "id": "abc123",
    "client_order_id": "paper:2026-02-03:attempt-1:AAPL:BUY",
    "status": "filled",
    "filled_qty": "5",
    "filled_avg_price": "175.25",
    "symbol": "AAPL",
    "side": "buy"
  }
]
```

**Implementation**:
1. Get Alpaca credentials for the specified account
2. Call Alpaca API: `GET /v2/orders?status=all&after={after}&limit=100`
3. Return the order list

## Alpaca API Details

- Base URL: `https://paper-api.alpaca.markets` (paper trading)
- Headers:
  - `APCA-API-KEY-ID`: API key
  - `APCA-API-SECRET-KEY`: API secret

## Files to Create/Modify

| File | Action |
|------|--------|
| `brain_api/brain_api/routes/alpaca.py` | Create |
| `brain_api/brain_api/main.py` | Add router |
| `brain_api/.env.example` | Add 6 Alpaca env vars |
| `brain_api/tests/test_alpaca.py` | Create |

## Test Requirements

1. Test GET /alpaca/portfolio returns normalized portfolio (mock Alpaca API)
2. Test GET /alpaca/portfolio with different accounts (ppo, sac, hrp)
3. Test GET /alpaca/portfolio with invalid account returns 422
4. Test POST /alpaca/submit-orders submits all orders (mock Alpaca API)
5. Test POST /alpaca/submit-orders handles partial failures
6. Test POST /alpaca/submit-orders with empty orders array
7. Test GET /alpaca/order-history returns order list (mock Alpaca API)
8. Test GET /alpaca/order-history with invalid account returns 422
9. Test Alpaca API failure returns 503
