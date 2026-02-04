# Prefect Training Workflows

Prefect orchestration for LearnFinance-2025 model training pipeline.

## Overview

This package contains Prefect flows that orchestrate the weekly training pipeline for all models:

1. **Get Halal Universe** - Fetch the list of halal stock symbols
2. **Refresh Training Data** - Fill sentiment gaps and refresh stale fundamentals
3. **Train LSTM** - Pure price forecaster model
4. **Train PatchTST** - Multi-signal forecaster model
5. **Train PPO** - Reinforcement learning allocator
6. **Train SAC** - Reinforcement learning allocator
7. **Generate Training Summary** - LLM-powered analysis of training results (OpenAI/OLLAMA)

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

### Test the flow manually

```bash
# Make sure brain_api is running first
cd brain_api
uv run uvicorn brain_api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, run the flow
cd prefect
python -m flows.weekly_training
```

### Run with Prefect UI

```bash
# Start Prefect server (if not using Prefect Cloud)
prefect server start

# In another terminal
cd prefect
python -m flows.weekly_training
```

## Deployment

### Deploy to Prefect Cloud/Server

Create a deployment with the weekly schedule:

```bash
cd prefect

# Create deployment with cron schedule (Every Sunday 11 AM UTC)
prefect deploy flows/weekly_training.py:weekly_training_flow \
    --name "weekly-training" \
    --cron "0 11 * * 0" \
    --timezone "UTC"
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

### Schedule

- **Cron**: `0 11 * * 0` (Every Sunday at 11:00 AM UTC)
- **Timezone**: UTC

### Timeouts

| Component | Timeout |
|-----------|---------|
| Flow total | 4 hours |
| HTTP read | 1 hour |
| HTTP connect | 30 seconds |

### Retries

| Task | Retries | Retry Delay |
|------|---------|-------------|
| Get Halal Universe | 2 | 30s |
| Refresh Training Data | 1 | 60s |
| Train LSTM | 1 | 120s |
| Train PatchTST | 1 | 120s |
| Train PPO | 1 | 120s |
| Train SAC | 1 | 120s |
| Generate Training Summary | 1 | 30s |

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
