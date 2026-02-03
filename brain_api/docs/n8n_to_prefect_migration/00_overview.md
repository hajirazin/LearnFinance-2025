# n8n to Prefect Migration — Overview

## Summary

Migrate the weekly forecast email workflow from n8n to Prefect. The workflow graph stays in Prefect (same shape as n8n). All external integrations (Alpaca, OpenAI, Gmail) become brain_api endpoints.

## Principle

- **One Prefect task = one brain_api call**
- **Workflow graph stays in Prefect** (parallel branches, merge points, conditional skips)
- **All logic and integrations in brain_api** (Alpaca creds, OpenAI key, Gmail creds, Jinja2 templates)
- **Prefect does only**: schedule, call brain_api endpoints, pass results between tasks, check `open_orders_count > 0` to skip algorithms

## Sub-Plans

| Plan | Description | Status |
|------|-------------|--------|
| [01_openai_summary_endpoint.md](./01_openai_summary_endpoint.md) | POST /summary/generate with Jinja2 prompt | Pending |
| [02_gmail_email_endpoint.md](./02_gmail_email_endpoint.md) | POST /email/send-weekly-report with Jinja2 template | Pending |
| [03_alpaca_endpoints.md](./03_alpaca_endpoints.md) | GET /alpaca/portfolio, POST /alpaca/submit-orders, GET /alpaca/order-history | Pending |
| [04_weekly_context_endpoint.md](./04_weekly_context_endpoint.md) | GET /run/weekly-context | Pending |
| [05_experience_execution_report.md](./05_experience_execution_report.md) | POST /experience/build-execution-report | Pending |
| [06_orders_percentage_weights.md](./06_orders_percentage_weights.md) | Extend POST /orders/generate for percentage_weights | Pending |
| [07_prefect_flow.md](./07_prefect_flow.md) | Create flows/weekly_forecast_email.py with 27 tasks | Pending |

## New brain_api Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run/weekly-context` | GET | Top 20 symbols + as_of_date + run_id |
| `/alpaca/portfolio` | GET | Account + positions + open_orders_count (normalized) |
| `/alpaca/submit-orders` | POST | Submit order array to Alpaca |
| `/alpaca/order-history` | GET | Get order history from Alpaca |
| `/experience/build-execution-report` | POST | Match intended vs executed orders |
| `/summary/generate` | POST | Jinja2 prompt template → OpenAI → summary |
| `/email/send-weekly-report` | POST | Jinja2 email template → Gmail SMTP |

## Credentials in brain_api

| Credential | Env var | Used by |
|------------|---------|---------|
| Alpaca PPO | `ALPACA_PPO_KEY`, `ALPACA_PPO_SECRET` | `/alpaca/*` |
| Alpaca SAC | `ALPACA_SAC_KEY`, `ALPACA_SAC_SECRET` | `/alpaca/*` |
| Alpaca HRP | `ALPACA_HRP_KEY`, `ALPACA_HRP_SECRET` | `/alpaca/*` |
| OpenAI | `OPENAI_API_KEY` | `/summary/generate` |
| Gmail | `GMAIL_USER`, `GMAIL_APP_PASSWORD` | `/email/send-weekly-report` |

## Prefect Task Count: 27 tasks

| Category | Count | Tasks |
|----------|-------|-------|
| Context | 1 | `get_weekly_context` |
| Alpaca portfolios | 3 | `get_ppo_portfolio`, `get_sac_portfolio`, `get_hrp_portfolio` |
| Phase 1 | 4 | `get_fundamentals`, `get_news_sentiment`, `get_lstm_forecast`, `get_patchtst_forecast` |
| Phase 2 allocators | 3 | `infer_sac`, `infer_ppo`, `allocate_hrp` |
| Generate orders | 3 | `generate_orders_ppo`, `generate_orders_sac`, `generate_orders_hrp` |
| Store experience | 2 | `store_experience_ppo`, `store_experience_sac` |
| Submit orders | 3 | `submit_orders_ppo`, `submit_orders_sac`, `submit_orders_hrp` |
| Order history | 2 | `get_order_history_ppo`, `get_order_history_sac` |
| Build execution report | 2 | `build_execution_report_ppo`, `build_execution_report_sac` |
| Update execution | 2 | `update_execution_ppo`, `update_execution_sac` |
| Summary | 1 | `generate_summary` |
| Email | 1 | `send_weekly_email` |
