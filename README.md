# LearnFinance-2025

A **learning-focused** weekly paper-trading portfolio system for **halal Nasdaq-500 stocks**. The goal is to **compare multiple approaches side-by-side** — not to pick a single "best" method.

## What it does

Every Monday **6:00 PM IST** (pre US open), the system orchestrates:

1. **Universe & Signals**: Fetch halal universe, collect signals (news sentiment, fundamentals)
2. **Price Forecasting**: Run LSTM (price-only) and PatchTST (OHLCV 5-channel)
3. **Portfolio Allocation**: Run multiple allocators for comparison:
   - **HRP** (Hierarchical Risk Parity) — math baseline
   - **PPO + LSTM** / **PPO + PatchTST** — on-policy RL agents
   - **SAC + LSTM** / **SAC + PatchTST** — off-policy RL agents
4. **LLM Summary**: OpenAI/GPT synthesizes all signals into market insights
5. **Email**: Send comparison tables with all approaches for learning

The email shows **all allocations side-by-side** so you can learn which approach performs best over time.

### What it does NOT do

- It does **not** execute live trades (you can later add that with additional guardrails).
- It is **not** financial advice.

## Architecture

```mermaid
flowchart LR
  prefect[Prefect_orchestrator] -->|schedule_Mon_18_IST| runApi[brain_api_FastAPI]

  subgraph brain[python_brain]
    runApi --> universe[universe_service]
    runApi --> signals[signal_service]
    runApi --> forecasters[forecasters_LSTM_PatchTST]
    runApi --> allocators[allocators_HRP_PPO_SAC]
  end

  signals --> raw[raw_evidence_store]
  forecasters --> db[run_db_Postgres]
  allocators --> db

  runApi -->|LLM_summary| openai[OpenAI_GPT]
  runApi -->|send_comparison_email| email[Gmail_SMTP]
  runApi -->|submit_limit_orders| alpaca[alpaca_paper_api]
```

**Architecture overview:**

- **Prefect** for scheduling/orchestration (triggers brain_api endpoints)
- **brain_api** handles all integrations: Alpaca trading, OpenAI/LLM summaries, Gmail SMTP
- A Python "AI brain" for price forecasting, allocation, and signal collection

## Model hierarchy

This repo compares multiple approaches at each stage:

### Price Forecasters (predict weekly returns)

| Model | Input | Status |
|-------|-------|--------|
| LSTM | OHLCV only (pure price) | ✅ Active |
| PatchTST | OHLCV + All signals | ✅ Active |

### Portfolio Allocators (decide weights)

| Model | Input | Status |
|-------|-------|--------|
| HRP | Covariance matrix | ✅ Active |
| PPO + LSTM | State vector + LSTM forecasts | ✅ Active |
| PPO + PatchTST | State vector + PatchTST forecasts | ✅ Active |
| SAC + LSTM | State vector + LSTM forecasts | ✅ Active |
| SAC + PatchTST | State vector + PatchTST forecasts | ✅ Active |

### Signals

| Signal | Status | Endpoint |
|--------|--------|----------|
| News sentiment (FinBERT) | ✅ Active | `/signals/news` |
| News sentiment (historical) | ✅ Active | `/signals/news/historical` |
| Fundamentals (5 ratios) | ✅ Active | `/signals/fundamentals` |
| Fundamentals (historical) | ✅ Active | `/signals/fundamentals/historical` |
| Twitter/Social sentiment | 🔜 To build | — |

### Signal state vector (for RL and PatchTST)

| Feature | Source |
|---------|--------|
| LSTM predicted return | `/inference/lstm` |
| News sentiment score | `/signals/news` |
| Gross margin | `/signals/fundamentals` |
| Operating margin | `/signals/fundamentals` |
| Net margin | `/signals/fundamentals` |
| Current ratio | `/signals/fundamentals` |
| Debt to equity | `/signals/fundamentals` |
| Current portfolio weight | Portfolio state |
| Cash available | Portfolio state |

**Key distinction:**
- **LSTM** = pure price forecaster (close returns only)
- **PatchTST** = OHLCV forecaster (5-channel: open, high, low, close, volume log returns)
- **PPO/SAC** = RL allocators (receive signals + dual forecaster output)

## Prerequisites

- **Docker & Docker Compose** (for Postgres)
- **Python 3.11+** with `uv` package manager
- **Prefect 3.x** (for workflow orchestration)
- Gmail app password (for email notifications)
- Alpaca paper trading accounts (for order execution)

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d
```

This starts:
- brain-api at http://localhost:8000
- Prefect server at http://localhost:4200
- Prefect training worker (Sunday cron)
- Prefect email worker (Monday cron)

### 2. Start Brain API

```bash
cd brain_api
cp .env.example .env
# Edit .env with your credentials (see below)
uv sync --extra dev
uv run uvicorn brain_api.main:app --reload --host 0.0.0.0 --port 8000
```

API available at http://localhost:8000 (docs at `/docs`)

### 3. Configure credentials in brain_api/.env

The brain_api handles all external integrations. Configure these environment variables:

**Gmail (for email notifications):**
```bash
GMAIL_USER=your-email@gmail.com
GMAIL_APP_PASSWORD=your-app-password
```

To get a Gmail app password:
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-Step Verification if not already enabled
3. Go to **App passwords** → Generate a new app password for "Mail"
4. Copy the 16-character password (no spaces)

**Alpaca Paper Trading (for order execution):**
```bash
# PPO account
ALPACA_PPO_KEY=your-ppo-api-key
ALPACA_PPO_SECRET=your-ppo-api-secret

# SAC account
ALPACA_SAC_KEY=your-sac-api-key
ALPACA_SAC_SECRET=your-sac-api-secret

# HRP account
ALPACA_HRP_KEY=your-hrp-api-key
ALPACA_HRP_SECRET=your-hrp-api-secret
```

Create 3 paper trading accounts at [Alpaca](https://alpaca.markets/) and get API keys from each dashboard.

| Account | Algorithm | Description |
|---------|-----------|-------------|
| PPO | PPO + LSTM | On-policy RL with pure price forecasts |
| SAC | SAC + PatchTST | Off-policy RL with OHLCV forecasts |
| HRP | HRP | Risk parity baseline |

**OpenAI (for LLM summaries):**
```bash
OPENAI_API_KEY=your-openai-api-key
```

### 4. Install Prefect

```bash
cd prefect
uv sync --extra dev
```

### 5. Test the workflow

```bash
# Make sure brain_api is running, then in another terminal:
cd prefect
python flows/weekly_forecast_email.py --test
```

This runs a single execution of the weekly forecast email flow.

## Weekly workflow setup

The Prefect flow runs every Monday at 18:00 IST. See [prefect/README.md](prefect/README.md) for full details.

### Deploy the workflow

```bash
cd prefect

# Option 1: Serve locally (for development)
python flows/weekly_forecast_email.py

# Option 2: Deploy to Prefect Cloud/Server
prefect deploy flows/weekly_forecast_email.py:weekly_forecast_email_flow \
    --name "weekly-forecast-email" \
    --cron "0 18 * * 1" \
    --timezone "Asia/Kolkata"

# Start a worker to execute flows
prefect worker start --pool default-agent-pool
```

### Workflow flow

```mermaid
sequenceDiagram
  participant Prefect as Prefect
  participant Brain as brain_api
  participant Alpaca as alpaca_paper
  participant DB as run_db
  participant Email as Gmail_SMTP

  Prefect->>Brain: GET /universe/halal
  Prefect->>Brain: GET /alpaca/portfolio (PPO, SAC, HRP)
  Brain->>Alpaca: Fetch positions and cash
  Prefect->>Brain: POST /signals/fundamentals, /signals/news
  Prefect->>Brain: POST /inference/lstm, /inference/patchtst
  Prefect->>Brain: POST /inference/ppo, /inference/sac, /allocation/hrp
  Prefect->>Brain: POST /orders/generate (for each algorithm)
  Prefect->>Brain: POST /alpaca/submit-orders
  Brain->>Alpaca: Submit limit orders
  Prefect->>Brain: POST /llm/weekly-summary
  Prefect->>Brain: POST /email/weekly-report
  Brain->>Email: Send via SMTP
```

### 7-Phase execution architecture

The Prefect flow executes in 7 phases with parallel tasks where possible:

```mermaid
flowchart TD
    Trigger[Monday_18_IST] --> Phase0

    subgraph Phase0[Phase 0 - Universe and Portfolios]
        GetUniverse[GET Universe]
        GetPPO[GET PPO Portfolio]
        GetSAC[GET SAC Portfolio]
        GetHRP[GET HRP Portfolio]
    end

    Phase0 --> Phase1

    subgraph Phase1[Phase 1 - Signals and Forecasts]
        Fundamentals[POST Fundamentals]
        NewsSentiment[POST News Sentiment]
        LSTMForecast[POST LSTM Forecast]
        PatchTSTForecast[POST PatchTST Forecast]
    end

    Phase1 --> Phase2

    subgraph Phase2[Phase 2 - Allocators]
        PPO[POST PPO Inference]
        SAC[POST SAC Inference]
        HRP[POST HRP Allocation]
    end

    Phase2 --> Phase3

    subgraph Phase3[Phase 3 - Generate Orders]
        OrdersPPO[Generate PPO Orders]
        OrdersSAC[Generate SAC Orders]
        OrdersHRP[Generate HRP Orders]
    end

    Phase3 --> Phase4

    subgraph Phase4[Phase 4 - Submit Orders]
        SubmitPPO[Submit PPO to Alpaca]
        SubmitSAC[Submit SAC to Alpaca]
        SubmitHRP[Submit HRP to Alpaca]
    end

    Phase4 --> Phase5

    subgraph Phase5[Phase 5 - Update Execution]
        HistoryPPO[Get PPO Order History]
        HistorySAC[Get SAC Order History]
    end

    Phase5 --> Phase6

    subgraph Phase6[Phase 6 - Summary and Email]
        Summary[POST LLM Summary]
        Summary --> SendEmail[POST Send Email]
    end
```

**Skip logic:** Algorithms are skipped if they have open orders from a previous run (prevents duplicate submissions).

### Environment variables

```bash
# Brain API URL (for Prefect to call)
BRAIN_API_URL=http://localhost:8000

# Timezone (IST for Monday 6 PM runs)
TZ=Asia/Kolkata
```

## Key design decisions

### Paper auto-submit, live manual

- **Paper orders are auto-submitted** each Monday.
- Live trading is intentionally out of scope until safety, monitoring, and backtesting maturity is higher.

### Run identity & rerun behavior

- **Run date** is the Monday date in IST, e.g. `2025-12-29`.
- **Run ID**: `paper:YYYY-MM-DD` (example: `paper:2025-12-29`).
- **Attempt**: integer starting at `1`.

**Rerun is read-only** if the latest attempt has any order that is not canceled/expired/rejected.

If you manually cancel all active paper orders in Alpaca, the next run can create **attempt=2** and submit new orders.

### Order idempotency (no accidental duplicates)

Every order uses a deterministic `client_order_id`:

- `paper:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>`
- Example: `paper:2025-12-29:attempt-1:AAPL:BUY`

On submit:

- If an order with the same `client_order_id` was already submitted, reruns **do not** submit again.
- We also query Alpaca by `client_order_id` as a secondary guardrail.

### Screening stage (runtime control)

We start from all Nasdaq-500 stocks and apply the **halal filter across the full set**. From the resulting halal universe, we only run expensive pipelines for:

- **Always**: your current holdings
- **Plus**: a Top-15 candidate set chosen via cheap deterministic filters + ranking

This keeps the system reliable and cost-bounded.

### Limit orders + fractional sizing

- Default order type: **limit orders**
- Sizing: **fractional shares when supported**
- Limit pricing uses a configurable buffer from last price/quote

### Safety caps (recommended defaults)

Even for paper, enforce hard limits (config):

- Max turnover (% of portfolio value traded)
- Max number of orders
- Max position size (% of portfolio)
- Minimum cash buffer
- Blocklist/allowlist overrides

## Data storage

We store three kinds of data:

- **Run DB (local Postgres via Docker)**:
  - runs (run_id, attempt, timestamps, config_hash, status)
  - universe & screening decisions
  - signals/features (as-of timestamps)
  - trade plan + explanations
  - orders (client_order_id, alpaca_order_id, status)
- **Raw evidence store (filesystem)**:
  - `data/raw/<run_id>/<attempt>/<source>/<symbol>.json`
- **Derived feature snapshots**:
  - `data/features/<run_id>/<attempt>/...`

## API overview

### Inference endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /inference/lstm` | LSTM price predictions (OHLCV only) |
| `POST /inference/patchtst` | PatchTST price predictions (OHLCV) |
| `POST /inference/ppo_lstm` | PPO allocation using LSTM forecasts |
| `POST /inference/ppo_patchtst` | PPO allocation using PatchTST forecasts |
| `POST /inference/sac_lstm` | SAC allocation using LSTM forecasts |
| `POST /inference/sac_patchtst` | SAC allocation using PatchTST forecasts |
| `POST /allocation/hrp` | HRP risk-parity allocation |

### Order generation endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /orders/generate` | Convert allocation weights to limit orders |

### Signal endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /signals/news` | News sentiment (FinBERT, real-time) |
| `POST /signals/news/historical` | News sentiment (historical) |
| `POST /signals/fundamentals` | Financial ratios (5 metrics) |
| `POST /signals/fundamentals/historical` | Historical fundamentals |

### Training endpoints

| Endpoint | Purpose | Trigger |
|----------|---------|---------|
| `POST /train/lstm` | Full LSTM retrain | Monthly (manual) |
| `POST /train/patchtst` | Full PatchTST retrain | Monthly (manual) |
| `POST /train/ppo_lstm/full` | Full PPO+LSTM retrain | Monthly (manual) |
| `POST /train/ppo_lstm/finetune` | PPO+LSTM fine-tune | Weekly (cron) |
| `POST /train/ppo_patchtst/full` | Full PPO+PatchTST retrain | Monthly (manual) |
| `POST /train/ppo_patchtst/finetune` | PPO+PatchTST fine-tune | Weekly (cron) |
| `POST /train/sac_lstm/full` | Full SAC+LSTM retrain | Monthly (manual) |
| `POST /train/sac_lstm/finetune` | SAC+LSTM fine-tune | Weekly (cron) |
| `POST /train/sac_patchtst/full` | Full SAC+PatchTST retrain | Monthly (manual) |
| `POST /train/sac_patchtst/finetune` | SAC+PatchTST fine-tune | Weekly (cron) |

### LLM endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /llm/weekly-summary` | Generate AI summary of weekly forecasts and allocations |
| `POST /llm/training-summary` | Generate AI summary of training results (OpenAI/OLLAMA) |

### Email endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /email/weekly-report` | Send weekly portfolio analysis email via Gmail SMTP |
| `POST /email/training-summary` | Send training summary email |

### Alpaca endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /alpaca/portfolio` | Get account positions, cash, and open orders count |
| `POST /alpaca/submit-orders` | Submit orders to Alpaca paper trading |
| `GET /alpaca/order-history` | Get order execution history |

### Other endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /universe/halal` | Get halal stock universe |
| `POST /etl/news-sentiment` | ETL pipeline for news sentiment |
| `POST /etl/sentiment-gaps` | Gap detection and backfill |
| `POST /etl/refresh-training-data` | Refresh training data (sentiment gaps + fundamentals) |
| `POST /experience/store` | Store RL experience |
| `POST /experience/label` | Label experience with rewards |
| `GET /experience/list` | List stored experiences |
| `GET /health`, `/health/live`, `/health/ready` | Health checks |

### Request/response examples

**LSTM inference:**

```json
// POST /inference/lstm
// Request
{ "symbols": ["AAPL", "MSFT"], "as_of_date": "2025-12-29" }

// Response
{
  "predictions": [
    {
      "symbol": "AAPL",
      "predicted_weekly_return_pct": 2.5,
      "direction": "UP",
      "has_enough_history": true
    }
  ],
  "model_version": "v2026-01-09-a4fecab1bdcc",
  "as_of_date": "2025-12-29"
}
```

**PPO+LSTM inference:**

```json
// POST /inference/ppo_lstm
// Request
{ "portfolio": { "positions": [...], "cash": 10000 }, "as_of_date": "2025-12-29" }

// Response
{ "allocation": { "AAPL": 0.15, "MSFT": 0.10, "CASH": 0.05 }, "turnover": 0.12 }
```

## Model lifecycle

Monday inference runs **do not retrain** models. Training happens separately.

### Training schedule

| When | What | Trigger |
|------|------|---------|
| Monthly (Saturday) | Full retrain all models | Manual |
| Weekly (Sunday) | Fine-tune PPO + SAC variants | Cron (Prefect) |
| Monday 6 PM IST | Inference only (all models) | Cron (Prefect) |

### Training workflow

```mermaid
flowchart LR
  trigger[Sunday_cron] --> train[Train_new_model]
  train --> version[Write_versioned_artifact]
  version --> eval[Evaluate_vs_baseline_and_prior]
  eval -->|better| promote[Promote_to_current]
  eval -->|worse| keep[Keep_prior_current]
  promote --> done[Done]
  keep --> done
```

### Model artifacts

Models produce **versioned artifact files** stored under `data/models/`:

```
data/models/
├── lstm/
│   ├── v2026-01-09-a4fecab1bdcc/   # versioned artifact (date + config hash)
│   │   ├── weights.pt              # model weights (PyTorch)
│   │   ├── feature_scaler.pkl      # feature scaler/normalizer
│   │   ├── config.json             # hyperparams, feature schema
│   │   └── metadata.json           # training date, data window, metrics
│   ├── snapshot-2025-12-31/        # point-in-time snapshots
│   └── current                     # text file with active version string
└── patchtst/
    └── (same structure)
```

**What's in a model artifact:**

| File | Purpose |
|------|---------|
| `weights.pt` | Trained parameters |
| `feature_scaler.pkl` | Preprocessing transforms fitted on training data |
| `config.json` | Hyperparameters, feature list, model architecture |
| `metadata.json` | Training timestamp, data window, git commit, eval metrics |

### How inference loads models

1. Read `data/models/lstm/current` to get the active version string
2. Load artifacts from `data/models/lstm/<version>/`

This means you can:

- **Roll back** by changing the `current` pointer
- **A/B test** by loading a different version
- **Audit** by inspecting exactly which version was used

### RL experience collection

After each Monday run, store the experience tuple:

- State: features/signals at decision time
- Action: portfolio weights chosen
- Reward: computed later (next-week return minus turnover cost)

Save to: `data/experience/<run_id>.json`

### Fine-tune guardrails

- **Evaluation gate**: new policy must beat prior + a baseline
- **Rollback**: keep last known-good version; promotion is atomic pointer swap
- **Drift detection**: if performance degrades 4 weeks in a row, consider full retrain

## Cloud migration

The API is designed so each endpoint can become a standalone **Google Cloud Function** or be backed by **HuggingFace Hub** for model storage.

### Storage abstraction

```
┌─────────────────────────────────────────────────────────────┐
│  FastAPI endpoint (local)  OR  Cloud Function (later)       │
│  • Validates request                                        │
│  • Calls core function                                      │
│  • Returns JSON response                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Core function (pure Python, no framework dependency)       │
│  • lstm_inference(features, model_path) → predictions       │
│  • ppo_inference(state, policy_path) → allocation           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Storage abstraction                                        │
│  • LocalStorage: reads/writes data/models/...               │
│  • HuggingFaceStorage: reads/writes to HuggingFace Hub      │
└─────────────────────────────────────────────────────────────┘
```

### Migration steps

1. Extract endpoint handler → standalone `main.py` with `def handler(request):`
2. Swap `LocalStorage` → `HuggingFaceStorage` via environment variable
3. Deploy: `gcloud functions deploy <name> --runtime python311 --trigger-http`
4. Update `BRAIN_API_URL` in Prefect to call Cloud Function URL instead of local FastAPI

## Code structure

```
brain_api/
├── routes/
│   ├── inference/
│   │   ├── lstm.py
│   │   ├── patchtst.py
│   │   ├── ppo_lstm.py
│   │   ├── ppo_patchtst.py
│   │   ├── sac_lstm.py
│   │   └── sac_patchtst.py
│   ├── training/
│   │   └── (same pattern)
│   ├── signals/
│   │   └── endpoints.py
│   ├── pipelines/
│   ├── allocation.py
│   ├── experience.py
│   ├── etl.py
│   └── universe.py
├── core/
│   ├── lstm/
│   ├── patchtst/
│   ├── ppo_lstm/
│   ├── ppo_patchtst/
│   ├── sac_lstm/
│   ├── sac_patchtst/
│   ├── hrp.py
│   └── ...
├── storage/
│   ├── base.py              # abstract Storage class
│   ├── local.py             # LocalStorage(base_path="data/")
│   ├── huggingface.py       # HuggingFaceStorage (swap via env var)
│   ├── lstm/
│   ├── patchtst/
│   └── ...
└── ...
```

## Repo docs

- `README.md`: overview + architecture + setup
- `CLAUDE.md`: working agreement for contributors/AI (coding rules, invariants, testing policy)

## License

See `LICENSE`.
