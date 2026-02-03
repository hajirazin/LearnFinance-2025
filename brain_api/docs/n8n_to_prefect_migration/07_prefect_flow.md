# Plan 07: Prefect Weekly Forecast Email Flow

## Goal

Create `prefect/flows/weekly_forecast_email.py` with 27 Prefect tasks that mirror the n8n workflow graph. Each task calls exactly one brain_api endpoint.

## TODOs

- [ ] Create `prefect/flows/weekly_forecast_email.py` with all 27 tasks
- [ ] Add Pydantic models in `prefect/flows/models/` for request/response types
- [ ] Update `prefect/README.md` with new flow documentation
- [ ] Write tests for the flow in `prefect/tests/test_weekly_forecast_email.py`
- [ ] Run ruff and fix all lint issues
- [ ] Run all tests and fix any failures

## Flow Structure

### Schedule

- Cron: Monday 18:00 IST
- Timezone: Asia/Kolkata

### Tasks (27 total)

| # | Task Name | Endpoint | Parallel Group |
|---|-----------|----------|----------------|
| 1 | `get_weekly_context` | GET /run/weekly-context | Phase 0 |
| 2 | `get_ppo_portfolio` | GET /alpaca/portfolio?account=ppo | Phase 0 |
| 3 | `get_sac_portfolio` | GET /alpaca/portfolio?account=sac | Phase 0 |
| 4 | `get_hrp_portfolio` | GET /alpaca/portfolio?account=hrp | Phase 0 |
| 5 | `get_fundamentals` | POST /signals/fundamentals | Phase 1 |
| 6 | `get_news_sentiment` | POST /signals/news | Phase 1 |
| 7 | `get_lstm_forecast` | POST /inference/lstm | Phase 1 |
| 8 | `get_patchtst_forecast` | POST /inference/patchtst | Phase 1 |
| 9 | `infer_ppo` | POST /inference/ppo | Phase 2 |
| 10 | `infer_sac` | POST /inference/sac | Phase 2 |
| 11 | `allocate_hrp` | POST /allocation/hrp | Phase 2 |
| 12 | `generate_orders_ppo` | POST /orders/generate | Phase 3 |
| 13 | `generate_orders_sac` | POST /orders/generate | Phase 3 |
| 14 | `generate_orders_hrp` | POST /orders/generate | Phase 3 |
| 15 | `store_experience_ppo` | POST /experience/store | Phase 3 |
| 16 | `store_experience_sac` | POST /experience/store | Phase 3 |
| 17 | `submit_orders_ppo` | POST /alpaca/submit-orders | Phase 4 |
| 18 | `submit_orders_sac` | POST /alpaca/submit-orders | Phase 4 |
| 19 | `submit_orders_hrp` | POST /alpaca/submit-orders | Phase 4 |
| 20 | `get_order_history_ppo` | GET /alpaca/order-history | Phase 5 |
| 21 | `get_order_history_sac` | GET /alpaca/order-history | Phase 5 |
| 22 | `build_execution_report_ppo` | POST /experience/build-execution-report | Phase 5 |
| 23 | `build_execution_report_sac` | POST /experience/build-execution-report | Phase 5 |
| 24 | `update_execution_ppo` | POST /experience/update-execution | Phase 5 |
| 25 | `update_execution_sac` | POST /experience/update-execution | Phase 5 |
| 26 | `generate_summary` | POST /summary/generate | Phase 6 |
| 27 | `send_weekly_email` | POST /email/send-weekly-report | Phase 6 |

### Flow Logic

```python
@flow(name="Weekly Forecast Email")
def weekly_forecast_email_flow():
    # Phase 0: Context + Portfolios (parallel)
    context = get_weekly_context()
    ppo_portfolio = get_ppo_portfolio()
    sac_portfolio = get_sac_portfolio()
    hrp_portfolio = get_hrp_portfolio()
    
    # Determine skip flags
    run_ppo = ppo_portfolio.open_orders_count == 0
    run_sac = sac_portfolio.open_orders_count == 0
    run_hrp = hrp_portfolio.open_orders_count == 0
    
    # Phase 1: Signals + Forecasts (parallel)
    fundamentals = get_fundamentals(context.symbols)
    news = get_news_sentiment(context.symbols, context.as_of_date, context.run_id)
    lstm = get_lstm_forecast(context.symbols, context.as_of_date)
    patchtst = get_patchtst_forecast(context.symbols, context.as_of_date)
    
    # Phase 2: Allocators (parallel, skip if flagged)
    ppo_alloc = infer_ppo(ppo_portfolio, context.as_of_date) if run_ppo else SKIPPED_RESPONSE
    sac_alloc = infer_sac(sac_portfolio, context.as_of_date) if run_sac else SKIPPED_RESPONSE
    hrp_alloc = allocate_hrp(context.as_of_date) if run_hrp else SKIPPED_RESPONSE
    
    # Phase 3: Orders (per algorithm, skip if flagged)
    if run_ppo:
        ppo_orders = generate_orders_ppo(ppo_alloc, ppo_portfolio, context)
        store_experience_ppo(context, ppo_portfolio, ppo_alloc, fundamentals, news, lstm, patchtst)
    if run_sac:
        sac_orders = generate_orders_sac(sac_alloc, sac_portfolio, context)
        store_experience_sac(context, sac_portfolio, sac_alloc, fundamentals, news, lstm, patchtst)
    if run_hrp:
        hrp_orders = generate_orders_hrp(hrp_alloc, hrp_portfolio, context)
    
    # Phase 4: Submit orders (per algorithm, skip if flagged)
    ppo_submit = submit_orders_ppo(ppo_orders) if run_ppo else SKIPPED_SUBMIT
    sac_submit = submit_orders_sac(sac_orders) if run_sac else SKIPPED_SUBMIT
    hrp_submit = submit_orders_hrp(hrp_orders) if run_hrp else SKIPPED_SUBMIT
    
    # Phase 5: Execution reports (PPO and SAC only)
    if run_ppo:
        ppo_history = get_order_history_ppo(context.as_of_date)
        ppo_report = build_execution_report_ppo(context.run_id, ppo_orders, ppo_history)
        update_execution_ppo(ppo_report)
    if run_sac:
        sac_history = get_order_history_sac(context.as_of_date)
        sac_report = build_execution_report_sac(context.run_id, sac_orders, sac_history)
        update_execution_sac(sac_report)
    
    # Phase 6: Summary + Email
    summary = generate_summary(lstm, patchtst, news, fundamentals, hrp_alloc, sac_alloc, ppo_alloc, 
                               ppo_submit, sac_submit, hrp_submit, run_ppo, run_sac, run_hrp)
    send_weekly_email(summary, ppo_submit, sac_submit, hrp_submit, context)
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BRAIN_API_URL` | Base URL for brain_api (default: `http://localhost:8000`) |

### Timeouts

| Component | Timeout |
|-----------|---------|
| Flow total | 2 hours |
| HTTP read | 5 minutes |
| HTTP connect | 30 seconds |

## Files to Create/Modify

| File | Action |
|------|--------|
| `prefect/flows/weekly_forecast_email.py` | Create |
| `prefect/flows/models/forecast_email.py` | Create (Pydantic models) |
| `prefect/flows/models/__init__.py` | Update exports |
| `prefect/README.md` | Add documentation |
| `prefect/tests/test_weekly_forecast_email.py` | Create |

## Test Requirements

1. Test flow runs with all algorithms enabled (mock brain_api)
2. Test flow skips PPO when open_orders_count > 0
3. Test flow skips SAC when open_orders_count > 0
4. Test flow skips HRP when open_orders_count > 0
5. Test Phase 1 tasks run in parallel
6. Test Phase 2 tasks run in parallel
7. Test email is sent with correct data
