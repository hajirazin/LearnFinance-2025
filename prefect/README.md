# Prefect Workflows

Prefect orchestration for LearnFinance-2025 model training and inference pipelines.

## Overview

This package contains Prefect flows that orchestrate:

### Weekly Training Flow (`weekly_training.py`)

Runs every Sunday at 11:00 AM UTC to train all models:

1. **Get Halal Universe** - Fetch the list of halal stock symbols
2. **Refresh Training Data** - Fill sentiment gaps and refresh stale fundamentals
3. **Train LSTM** - Pure price forecaster model
4. **Train PatchTST** - OHLCV forecaster model
5. **Train PPO** - Reinforcement learning allocator
6. **Train SAC** - Reinforcement learning allocator
7. **Generate Training Summary** - LLM-powered analysis of training results (OpenAI/OLLAMA)

### Weekly Forecast Email Flow (`weekly_forecast_email.py`)

Runs every Monday at 18:00 IST (12:30 UTC) to execute the inference pipeline:

1. **Phase 0: Get Universe + Portfolios** (parallel)
   - Fetch halal stock universe
   - Get PPO, SAC, HRP Alpaca portfolio states

2. **Phase 1: Get Signals + Forecasts** (parallel)
   - Fetch fundamentals and news sentiment
   - Run LSTM and PatchTST inference

3. **Phase 2: Run Allocators** (parallel, conditional)
   - Run PPO and SAC inference
   - Run HRP allocation
   - *Skipped if algorithm has open orders*

4. **Phase 3: Generate Orders + Store Experience** (parallel)
   - Generate orders for each algorithm
   - Store RL experience for PPO and SAC

5. **Phase 4: Submit Orders** (parallel)
   - Submit orders to Alpaca for each algorithm

6. **Phase 5: Update Execution** (parallel)
   - Fetch order history from Alpaca
   - Match intended orders with executed orders
   - Update RL experience records

7. **Phase 6: Generate Summary + Send Email** (sequential)
   - Generate LLM summary of allocations and forecasts
   - Send weekly report email

## Prerequisites

- Python 3.11+
- Running `brain_api` server (see main README)
- Prefect 3.x installed

## Installation

```bash
cd prefect
pip install -e ".[dev]"
```

Or with uv:

```bash
cd prefect
uv pip install -e ".[dev]"
```

## Configuration

Set the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BRAIN_API_URL` | Base URL for brain_api | `http://localhost:8000` |
| `PREFECT_API_URL` | Prefect Cloud/Server URL (optional) | - |

## Running Locally

### Test the training flow manually

```bash
# Make sure brain_api is running first
cd brain_api
uv run uvicorn brain_api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run the training flow
cd prefect
python -m flows.weekly_training
```

### Test the forecast email flow manually

```bash
# Make sure brain_api is running first
cd brain_api
uv run uvicorn brain_api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run the forecast email flow once
cd prefect
python flows/weekly_forecast_email.py --test
```

### Run with Prefect UI

```bash
# Start Prefect server (if not using Prefect Cloud)
prefect server start

# In another terminal, serve a flow
cd prefect
python -m flows.weekly_training          # Training flow
python flows/weekly_forecast_email.py    # Forecast email flow (runs with cron)
```

## Deployment

### Deploy to Prefect Cloud/Server

Create deployments with cron schedules:

```bash
cd prefect

# Deploy training flow (Every Sunday 11 AM UTC)
prefect deploy flows/weekly_training.py:weekly_training_flow \
    --name "weekly-training" \
    --cron "0 11 * * 0" \
    --timezone "UTC"

# Deploy forecast email flow (Every Monday 18:00 IST)
prefect deploy flows/weekly_forecast_email.py:weekly_forecast_email_flow \
    --name "weekly-forecast-email" \
    --cron "0 18 * * 1" \
    --timezone "Asia/Kolkata"
```

Or create a `prefect.yaml` for declarative deployment:

```yaml
# prefect.yaml
deployments:
  - name: weekly-training
    entrypoint: flows/weekly_training.py:weekly_training_flow
    schedule:
      cron: "0 11 * * 0"
      timezone: "UTC"
    parameters: {}

  - name: weekly-forecast-email
    entrypoint: flows/weekly_forecast_email.py:weekly_forecast_email_flow
    schedule:
      cron: "0 18 * * 1"
      timezone: "Asia/Kolkata"
    parameters: {}
```

Then deploy:

```bash
prefect deploy --all
```

### Start a worker

```bash
prefect worker start --pool default-agent-pool
```

## Flow Details

### Weekly Training Flow

#### Schedule

- **Cron**: `0 11 * * 0` (Every Sunday at 11:00 AM UTC)
- **Timezone**: UTC

#### Timeouts

| Component | Timeout |
|-----------|---------|
| Flow total | 4 hours |
| HTTP read | 1 hour |
| HTTP connect | 30 seconds |

#### Retries

| Task | Retries | Retry Delay |
|------|---------|-------------|
| Get Halal Universe | 2 | 30s |
| Refresh Training Data | 1 | 60s |
| Train LSTM | 1 | 120s |
| Train PatchTST | 1 | 120s |
| Train PPO | 1 | 120s |
| Train SAC | 1 | 120s |
| Generate Training Summary | 1 | 30s |

### Weekly Forecast Email Flow

#### Schedule

- **Cron**: `0 18 * * 1` (Every Monday at 18:00 IST)
- **Timezone**: Asia/Kolkata

#### Timeouts

| Component | Timeout |
|-----------|---------|
| Flow total | 2 hours |
| HTTP read | 5 minutes |
| HTTP connect | 30 seconds |

#### Retries

| Task | Retries | Retry Delay |
|------|---------|-------------|
| Get Halal Universe | 2 | 30s |
| Get Portfolio (PPO/SAC/HRP) | 2 | 30s |
| Get Fundamentals | 1 | 30s |
| Get News Sentiment | 1 | 30s |
| Get LSTM Forecast | 1 | 60s |
| Get PatchTST Forecast | 1 | 60s |
| Infer PPO/SAC | 1 | 60s |
| Allocate HRP | 1 | 30s |
| Generate Orders | 1 | 30s |
| Submit Orders | 2 | 30s |
| Store/Update Experience | 1 | 30s |
| Generate Summary | 1 | 30s |
| Send Email | 2 | 30s |

#### Skip Logic

Algorithms are skipped when they have open orders from a previous run:

- If PPO account has `open_orders_count > 0`:
  - Skip `infer_ppo`, `generate_orders_ppo`, `store_experience_ppo`, `update_execution_ppo`
  - Use `SkippedAllocation` placeholder in email

- If SAC account has `open_orders_count > 0`:
  - Skip `infer_sac`, `generate_orders_sac`, `store_experience_sac`, `update_execution_sac`
  - Use `SkippedAllocation` placeholder in email

- If HRP account has `open_orders_count > 0`:
  - Skip `allocate_hrp`, `generate_orders_hrp`
  - Use `SkippedAllocation` placeholder in email

The email is always sent, indicating which algorithms were skipped.

#### brain_api Endpoints Called

| Endpoint | Purpose |
|----------|---------|
| `GET /universe/halal` | Fetch halal stock universe |
| `GET /alpaca/portfolio?account=ppo\|sac\|hrp` | Get portfolio state |
| `POST /signals/fundamentals` | Get fundamental ratios |
| `POST /signals/news` | Get news sentiment |
| `POST /inference/lstm` | LSTM price forecasts |
| `POST /inference/patchtst` | PatchTST OHLCV forecasts |
| `POST /inference/ppo` | PPO target weights |
| `POST /inference/sac` | SAC target weights |
| `POST /allocation/hrp` | HRP percentage weights |
| `POST /orders/generate` | Generate orders from weights |
| `POST /alpaca/submit-orders` | Submit orders to Alpaca |
| `GET /alpaca/order-history` | Fetch executed orders |
| `POST /experience/store` | Store RL experience |
| `POST /experience/update-execution` | Update with execution results |
| `POST /llm/weekly-summary` | Generate LLM summary |
| `POST /email/weekly-report` | Send email report |

#### HRP Weight Conversion

HRP returns `percentage_weights` (e.g., `{"AAPL": 20.0}` for 20%), which are converted to decimal `target_weights` (e.g., `{"AAPL": 0.20}`) before calling `/orders/generate`.

## Monitoring

View flow runs in the Prefect UI:

- **Local**: http://localhost:4200
- **Cloud**: https://app.prefect.cloud

## Troubleshooting

### brain_api connection errors

Ensure `brain_api` is running and `BRAIN_API_URL` is set correctly:

```bash
curl http://localhost:8000/health
```

### Training timeout

If training takes longer than expected, increase the timeout in `weekly_training.py`:

```python
DEFAULT_TIMEOUT = httpx.Timeout(
    read=7200.0,  # 2 hours
    ...
)
```

### Prefect server connection

If using Prefect Cloud, ensure you're logged in:

```bash
prefect cloud login
```
