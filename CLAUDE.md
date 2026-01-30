# CLAUDE.md

This file is the **working agreement** for humans + AI assistants contributing to this repo.

## Project intent (north star)

Build a **learning-focused** weekly paper-trading portfolio system for halal Nasdaq-500 stocks that **compares multiple approaches side-by-side**:

- **Safe-by-default** (paper auto-submit only; reruns cannot duplicate orders)
- **Audit-friendly** (every run reproducible and explainable)
- **Learning-focused** (compare LSTM vs PatchTST, PPO vs SAC, all vs HRP baseline)
- **Cloud-ready** (local-first design that can migrate to Cloud Functions / HuggingFace Hub)

The goal is to learn which approaches work best, not to pick a single method upfront.

## Architecture boundaries

- **n8n** is the outer orchestrator:
  - schedule trigger (Monday 6 PM IST)
  - calling brain_api endpoints
  - calling OpenAI/LLM for summary
  - sending comparison email via Gmail
  - submitting orders to Alpaca (3 paper accounts: PPO, SAC, HRP)
  - status tracking + notifications
- **Python brain** owns:
  - universe build + screening
  - signal collection (news, fundamentals)
  - price forecasting (LSTM pure-price, PatchTST multi-signal)
  - portfolio allocation (HRP math baseline, PPO variants, SAC variants)
  - order generation (convert weights to limit orders with idempotent IDs)
  - explanation generation
  - persistence of run artifacts

Avoid putting "business logic" inside n8n nodes beyond simple orchestration.

## Code structure

```
brain_api/
├── routes/
│   ├── inference/           # One file per model
│   │   ├── lstm.py
│   │   ├── patchtst.py
│   │   ├── ppo.py
│   │   └── sac.py
│   ├── training/            # Same pattern as inference
│   │   └── ...
│   ├── signals/
│   │   └── endpoints.py
│   ├── pipelines/
│   ├── allocation.py        # HRP
│   ├── experience.py        # RL experience management
│   ├── etl.py               # ETL pipelines
│   └── universe.py
├── core/                    # Pure functions, no FastAPI dependency
│   ├── lstm/
│   ├── patchtst/
│   ├── ppo/                 # PPO allocator (dual forecasts: LSTM + PatchTST)
│   ├── sac/                 # SAC allocator (dual forecasts: LSTM + PatchTST)
│   ├── hrp.py
│   └── ...
├── storage/
│   ├── base.py              # Abstract Storage class
│   ├── local.py             # LocalStorage
│   ├── huggingface.py       # HuggingFaceStorage (swap via env var)
│   ├── lstm/
│   ├── patchtst/
│   └── ...
└── ...
```

## API design rules

### Endpoints

**Inference** (called by Monday run via n8n):

| Endpoint | Purpose |
|----------|---------|
| `POST /inference/lstm` | Price predictions (OHLCV only) |
| `POST /inference/patchtst` | Price predictions (multi-signal) |
| `POST /inference/ppo` | PPO allocation using dual forecasts (LSTM + PatchTST) |
| `POST /inference/sac` | SAC allocation using dual forecasts (LSTM + PatchTST) |
| `POST /allocation/hrp` | HRP risk-parity allocation |

**Orders** (called by Monday run via n8n after allocations):

| Endpoint | Purpose |
|----------|---------|
| `POST /orders/generate` | Convert allocation weights to limit orders |

**Signals** (called by Monday run via n8n):

| Endpoint | Purpose |
|----------|---------|
| `POST /signals/news` | News sentiment (FinBERT, real-time) |
| `POST /signals/news/historical` | News sentiment (historical) |
| `POST /signals/fundamentals` | Financial ratios (5 metrics) |
| `POST /signals/fundamentals/historical` | Historical fundamentals |

**Training** (called by Saturday/Sunday cron or manual):

| Endpoint | Purpose |
|----------|---------|
| `POST /train/lstm` | Full LSTM retrain |
| `POST /train/patchtst` | Full PatchTST retrain |
| `POST /train/ppo/full` | Full PPO retrain (dual forecasts) |
| `POST /train/ppo/finetune` | PPO fine-tune on experience buffer |
| `POST /train/sac/full` | Full SAC retrain (dual forecasts) |
| `POST /train/sac/finetune` | SAC fine-tune on experience buffer |

**Other**:

| Endpoint | Purpose |
|----------|---------|
| `GET /universe/halal` | Halal stock universe |
| `POST /etl/news-sentiment` | ETL pipeline for news sentiment |
| `POST /etl/sentiment-gaps` | Gap detection and backfill |
| `POST /experience/store` | Store RL experience |
| `POST /experience/label` | Label experience with rewards |
| `GET /experience/list` | List stored experiences |

### Design rules (do not violate)

1. **Stateless**: load model from storage on each request; no in-memory state across requests
2. **Storage abstraction**: use `storage.load_model(path)` that works for local or HuggingFace
3. **JSON in, JSON out**: core functions must not depend on FastAPI request objects
4. **Idempotent training**: version ID = `v{date}-{config_hash}`, so re-runs produce same version
5. **Thin endpoints**: FastAPI route handlers only validate + call core functions + return response

### Cloud Function migration

When migrating an endpoint to GCP:

1. Extract core function call into `main.py` with `def handler(request):`
2. Set `STORAGE_BACKEND=huggingface` environment variable
3. Deploy: `gcloud functions deploy <name> --runtime python311 --trigger-http`
4. Update caller (n8n) to use Cloud Function URL

## Model hierarchy

### Price Forecasters

| Model | Input | Output |
|-------|-------|--------|
| LSTM | OHLCV only (pure price) | Weekly return prediction |
| PatchTST | OHLCV + All signals | Weekly return prediction |

### Portfolio Allocators

| Model | Input | Output |
|-------|-------|--------|
| HRP | Covariance matrix | Allocation weights |
| PPO | State vector + dual forecasts (LSTM + PatchTST) | Allocation weights |
| SAC | State vector + dual forecasts (LSTM + PatchTST) | Allocation weights |

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
- **LSTM** = pure price forecaster (OHLCV only, does NOT receive signals)
- **PatchTST** = multi-signal forecaster (receives all signals + OHLCV)
- **PPO/SAC** = RL allocators (receive all signals + forecaster output)

## Data storage rules

Store three classes of data:

- **Structured DB** (local Postgres via Docker)
  - runs, screening decisions, signals, decisions, orders
- **Raw evidence snapshots** (filesystem)
  - `data/raw/<run_id>/<attempt>/<source>/<symbol>.json`
- **Feature snapshots**
  - `data/features/<run_id>/<attempt>/...`

Every persisted record must include:

- `run_id`, `attempt`
- an `as_of` timestamp for time-sensitive signals

### Model storage

Models are stored under `data/models/{lstm,patchtst}/<version>/`:

```
data/models/
├── lstm/
│   ├── v2026-01-09-a4fecab1bdcc/   # versioned artifact
│   │   ├── weights.pt
│   │   ├── feature_scaler.pkl
│   │   ├── config.json
│   │   └── metadata.json
│   ├── snapshot-2025-12-31/        # point-in-time snapshots
│   └── current                     # text file with active version
└── patchtst/
    └── (same structure)
```

- Active version tracked by `data/models/{model}/current` (text file with version string)
- RL experience buffer stored under `data/experience/<run_id>.json`
- All model artifacts must include `metadata.json` with: training timestamp, data window, config hash, eval metrics

## Agent workflow rules

Agents must produce **structured outputs** that can be stored and audited:

- Include citations/identifiers where possible (e.g., news URL, data source + timestamp)
- A `RiskCritic` (or equivalent) must be able to:
  - challenge contradictions
  - flag weak/insufficient evidence
  - downgrade confidence or veto a trade recommendation

Agents are used for **evidence synthesis**. Numeric optimization remains in deterministic code (feature engineering) + forecasters/RL.

### LLM summary (n8n orchestrated)

The Monday email includes an **AI summary** generated by OpenAI/GPT-4o-mini:

- n8n merges all signal data and sends to OpenAI
- LLM produces: market outlook, top opportunities, key risks, portfolio insights
- This is for **learning/interpretation**, not for trading decisions

## Testing policy

User preference / repo rule:

- In Python, **never write schema tests**. Schemas are exercised via API usage.
- In the router layer, add **explicit tests by calling the API** for constraint behaviors (e.g., `min_items`, `max_items`, min/max length/count).

If tests are added later, they should be:

- Integration-style API tests for routers/handlers
- Deterministic unit tests for pure functions (feature transforms, idempotency key generation, screening ranking)

## Non-negotiable invariants

### Run identity & rerun semantics

- `run_date` is the **Monday date in IST** (calendar date)
- `run_id = paper:YYYY-MM-DD`
- `attempt` starts at `1`
- **Rerun is read-only** if the latest attempt has any order not in a terminal canceled/expired/rejected state
- To allow a new submission: user cancels paper orders manually in Alpaca, then rerun creates `attempt += 1`

### Order idempotency

All submitted orders must include deterministic `client_order_id`:

- `paper:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>`

The system must:

- Check local DB for existing submissions before submitting
- Query Alpaca by `client_order_id` as a second guardrail

### Trading mode

- **Paper auto-submit** is allowed
- **Live trading is out of scope** unless explicitly added with additional safety controls

### Default execution choices

- Default order type: **limit**
- Default sizing: **fractional shares when supported**

### Model lifecycle

- **Monday runs are inference-only**. Never retrain inside the Monday inference run.
- **Training schedule**:

| When | What | Trigger |
|------|------|---------|
| Monthly (Saturday) | Full retrain all models | Manual |
| Weekly (Sunday) | Fine-tune PPO/SAC variants | Cron |
| Monday 6 PM IST | Inference only | Cron (n8n) |

- Training produces a **new versioned artifact**; inference loads from `current` pointer
- **Promotion requires evaluation**: new model must beat prior + baseline before becoming `current`
- **Rollback is always possible**: keep last known-good version; pointer swap is atomic

## Operational requirements

Any implementation must include:

- **Idempotency**: safe reruns
- **Timeouts + retries** with exponential backoff for external APIs
- **Rate limit awareness** and batching
- **Observability**:
  - run-level logs with `run_id` + `attempt`
  - stage duration metrics (even if just logged)
  - clear error propagation back to n8n

## Change safety checklists

Before merging changes that touch trading logic:

- [ ] Confirm rerun behavior is still read-only after any submission
- [ ] Confirm `client_order_id` format unchanged (or migration handled)
- [ ] Confirm safety caps exist and are enforced (max turnover, max orders, cash buffer)

Before merging changes that touch ML/model code:

- [ ] Confirm Monday inference does NOT trigger training
- [ ] Confirm training writes new versioned artifact (not overwrite)
- [ ] Confirm promotion requires evaluation gate
- [ ] Confirm endpoints remain stateless (no global model cache)
- [ ] Confirm storage abstraction is used (not hardcoded paths)
- [ ] Confirm LSTM remains pure-price (no signals in input)
- [ ] Confirm PatchTST/PPO/SAC receive correct signal state vector
