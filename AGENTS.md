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

- **Prefect** is the outer orchestrator:
  - schedule trigger (Monday 6 PM IST for inference, Sunday 11 AM UTC for training)
  - calling brain_api endpoints in the correct order
  - handling parallel task execution and skip logic
  - status tracking + flow observability
- **brain_api (Python brain)** owns:
  - universe build + screening
  - signal collection (news, fundamentals)
  - price forecasting (LSTM pure-price, PatchTST OHLCV 5-channel)
  - portfolio allocation (HRP math baseline, PPO variants, SAC variants)
  - order generation (convert weights to limit orders with idempotent IDs)
  - Alpaca integration (portfolio queries, order submission)
  - OpenAI/LLM integration (summaries via `/llm/weekly-summary`)
  - Gmail SMTP integration (emails via `/email/weekly-report`)
  - explanation generation
  - persistence of run artifacts

Avoid putting "business logic" inside Prefect tasks beyond orchestration.

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

**Inference** (called by Monday run via Prefect):

| Endpoint | Purpose |
|----------|---------|
| `POST /inference/lstm` | Price predictions (OHLCV only) |
| `POST /inference/patchtst` | Price predictions (OHLCV) |
| `POST /inference/ppo` | PPO allocation using dual forecasts (LSTM + PatchTST) |
| `POST /inference/sac` | SAC allocation using dual forecasts (LSTM + PatchTST) |
| `POST /allocation/hrp` | HRP risk-parity allocation |

**Orders** (called by Monday run via Prefect after allocations):

| Endpoint | Purpose |
|----------|---------|
| `POST /orders/generate` | Convert allocation weights to limit orders |

**Signals** (called by Monday run via Prefect):

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

**Alpaca** (called by Monday run via Prefect for order execution):

| Endpoint | Purpose |
|----------|---------|
| `GET /alpaca/portfolio` | Get account positions, cash, open orders count |
| `POST /alpaca/submit-orders` | Submit orders to Alpaca paper trading |
| `GET /alpaca/order-history` | Get order execution history |

**LLM & Email** (called by Monday run via Prefect for reporting):

| Endpoint | Purpose |
|----------|---------|
| `POST /llm/weekly-summary` | Generate AI summary of forecasts and allocations |
| `POST /email/weekly-report` | Send weekly portfolio analysis email via Gmail SMTP |

**Other**:

| Endpoint | Purpose |
|----------|---------|
| `GET /universe/halal` | Halal stock universe |
| `POST /etl/news-sentiment` | ETL pipeline for news sentiment |
| `POST /etl/sentiment-gaps` | Gap detection and backfill |
| `POST /etl/refresh-training-data` | Refresh training data (sentiment gaps + fundamentals) |
| `POST /experience/store` | Store RL experience |
| `POST /experience/update-execution` | Update experience with execution results |
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
4. Update `BRAIN_API_URL` in Prefect to use Cloud Function URL

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
- **LSTM** = pure price forecaster (close returns only)
- **PatchTST** = OHLCV forecaster (5-channel: open, high, low, close, volume log returns)
- **PPO/SAC** = RL allocators (receive signals + dual forecaster output)

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

### LLM summary (Prefect orchestrated)

The Monday email includes an **AI summary** generated by OpenAI/GPT-4o-mini:

- Prefect calls brain_api's `/llm/weekly-summary` endpoint with all signal data
- brain_api uses Jinja2 templates to construct prompts and calls OpenAI
- LLM produces: market outlook, top opportunities, key risks, portfolio insights
- This is for **learning/interpretation**, not for trading decisions

## Testing policy

User preference / repo rule:

- In Python, **never write schema tests**. Schemas are exercised via API usage.
- In the router layer, add **explicit tests by calling the API** for constraint behaviors (e.g., `min_items`, `max_items`, min/max length/count).

If tests are added later, they should be:

- Integration-style API tests for routers/handlers
- Deterministic unit tests for pure functions (feature transforms, idempotency key generation, screening ranking)

**Test ownership:**

- Agent must always write, fix, or modify tests for any code changes
- Strive for excellent test quality focused on business logic coverage
- 100% code coverage is not the goal; 100%+ business logic coverage is (all edge cases, error paths, boundary conditions)
- Every feature/fix should have corresponding test updates

## Code quality guidelines

### Code reuse

- Before writing new code, search for existing helpers, utilities, or similar implementations in the codebase
- Best programmers factor out and reuse similar code
- Avoid duplicating logic that already exists elsewhere

### Naming conventions

- Use real-world domain names that match DDD (Domain-Driven Design) principles
- Class, function, and variable names should reflect business concepts clearly
- Not mandatory to have infrastructure layers, but names must be intuitive and domain-accurate

### File size limits

- Keep files under 600 lines
- If a file exceeds this limit, refactor into smaller, focused modules
- Split by responsibility, not arbitrarily

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
| Weekly (Sunday) | Fine-tune PPO/SAC variants | Cron (Prefect) |
| Monday 6 PM IST | Inference only | Cron (Prefect) |

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
  - clear error propagation back to Prefect

### Prefect flow configuration

All Prefect flows must include:

- **`persist_result=True`** at the flow level: enables result persistence for all tasks, allowing flows to resume from failure point instead of restarting from scratch
- **Task-level retries**: critical tasks (especially external API calls) should have `retries` configured
- **Retry delays**: use `retry_delay_seconds` to avoid hammering failing services

Note: Prefect 3 does not have a "retry single task" button in the UI. To retry a failed task:
1. Re-run the entire flow (fast if tasks are idempotent/cached)
2. Call the underlying API endpoint directly
3. Use `persist_result=True` to skip already-completed tasks on re-run

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

## AI assistant planning rules

When operating in **plan mode**, the AI assistant must:

1. Always include these two final TODOs at the end of every plan:
   - [ ] Fix all ruff linting issues (related and unrelated to the change)
   - [ ] Run and fix all tests (related and unrelated to the change)

2. These cleanup tasks ensure the codebase stays healthy with every change.
