# CLAUDE.md

This file is the **working agreement** for humans + AI assistants contributing to this repo.

## Project intent (north star)

Build a **learning-focused** weekly paper-trading portfolio system for halal Nasdaq-500 stocks that **compares multiple approaches side-by-side**:

- **Safe-by-default** (paper auto-submit only; reruns cannot duplicate orders)
- **Audit-friendly** (every run reproducible and explainable)
- **Learning-focused** (compare LSTM vs PatchTST, SAC, all vs HRP baseline)
- **Cloud-ready** (local-first design that can migrate to Cloud Functions / HuggingFace Hub)

The goal is to learn which approaches work best, not to pick a single method upfront.

## Architecture boundaries

- **Temporal** is the outer orchestrator (replaced Prefect):
  - schedule trigger (Monday 6 PM IST for US inference, Monday 9 AM IST for India, Sunday 11 AM UTC for US training, Sunday 4:30 AM UTC for India training)
  - calling brain_api endpoints via HTTP activities
  - handling parallel task execution (asyncio.gather) and skip logic
  - durable sleep/wait for sell-wait-buy pattern (single workflow, no 3-flow hack)
  - automatic replay from event history (no cache policies needed)
  - status tracking + workflow observability via Temporal UI (port 8233)
  - Runs locally via `temporal server start-dev` (SQLite persistence, survives laptop shutdown)
  - India weekly allocation workflow (`IndiaWeeklyAllocationWorkflow`): full Nifty Shariah 500 universe -> PatchTST alpha screen (`/inference/patchtst/score-batch` with `market='india'`) -> rank-band sticky selection (`halal_india_alpha` partition, K_in=15 / K_hold=30) -> HRP allocation (lookback=252d) on the 15 chosen names -> record final weights -> AI summary -> email (paper-only, no broker)
  - India training workflow (`IndiaWeeklyTrainingWorkflow`): NiftyShariah500 universe -> PatchTST India train -> halal_india rank-band sticky top 15 (`halal_india_filtered_alpha` partition in `screening_history`, monthly cadence) -> LLM summary -> email
  - US weekly allocation workflow (`USWeeklyAllocationWorkflow`): signals + forecasts -> allocators -> sell-wait-buy with durable polling -> email
  - US weekly training workflow (`USWeeklyTrainingWorkflow`): full retrain pipeline
- **brain_api (Python brain)** owns:
  - universe build + screening
  - signal collection (news, fundamentals)
  - price forecasting (LSTM pure-price, PatchTST OHLCV 5-channel)
  - portfolio allocation (HRP math baseline, SAC variants)
  - order generation (convert weights to limit orders with idempotent IDs)
  - Alpaca integration (portfolio queries, order submission)
  - OpenAI/LLM integration (summaries via `/llm/sac-weekly-summary`)
  - Gmail SMTP integration (emails via `/email/sac-weekly-report`)
  - explanation generation
  - persistence of run artifacts

Avoid putting "business logic" inside Temporal activities beyond orchestration.

## Code structure

```
brain_api/
├── routes/
│   ├── inference/           # One file per model
│   │   ├── lstm.py
│   │   ├── patchtst.py
│   │   └── sac.py
│   ├── training/            # Same pattern as inference
│   │   ├── patchtst.py      # US PatchTST training
│   │   ├── patchtst_india.py # India PatchTST training
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

temporal/                         # Temporal workflow orchestration
├── pyproject.toml                # temporalio, httpx, pydantic
├── worker.py                     # Worker entry point (registers all workflows + activities)
├── schedules.py                  # One-time script to register cron schedules
├── workflows/
│   ├── us_weekly_allocation.py   # Sell-wait-buy with durable polling
│   ├── us_weekly_training.py     # Full US model training pipeline
│   ├── india_weekly_allocation.py # India Alpha-HRP (PatchTST screen + sticky + HRP + email)
│   └── india_weekly_training.py  # India PatchTST training pipeline
├── activities/
│   ├── client.py                 # Shared httpx client for brain_api
│   ├── inference.py              # Signals + forecasts + allocators
│   ├── portfolio.py              # Portfolios + order submission + history
│   ├── execution.py              # Order generation + experience
│   ├── reporting.py              # Summary + email
│   └── training.py               # Training activities (long timeouts + heartbeating)
├── models/                       # Pydantic models (shared with workflow orchestration)
└── tests/

# prefect/ and n8n/ were removed; Temporal is the sole orchestrator
```

## API design rules

### Endpoints

**Inference** (called by Monday run via Temporal):

| Endpoint | Purpose |
|----------|---------|
| `POST /inference/lstm` | Price predictions (symbols from model metadata) |
| `POST /inference/patchtst` | US PatchTST price predictions (symbols from model metadata) |
| `POST /inference/patchtst/india` | India PatchTST price predictions (loads `patchtst_india` storage) |
| `POST /inference/patchtst/score-batch` | Batch PatchTST score endpoint (US or India) -- returns `{symbol -> predicted_weekly_return_pct}` and enforces finite-score / `min_predictions` invariants used by Alpha-HRP |
| `POST /inference/sac` | SAC allocation using dual forecasts (LSTM + PatchTST) |
| `POST /allocation/hrp` | HRP risk-parity allocation (requires `universe` param) |

**Orders** (called by Monday run via Temporal after allocations):

| Endpoint | Purpose |
|----------|---------|
| `POST /orders/generate` | Convert allocation weights to limit orders |

**Signals** (called by Monday run via Temporal):

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
| `POST /train/patchtst` | Full PatchTST retrain (US) |
| `POST /train/patchtst/india` | Full PatchTST retrain (India NiftyShariah500) |
| `POST /train/sac/full` | Full SAC retrain (dual forecasts) |
| `POST /train/sac/finetune` | SAC fine-tune on experience buffer |

**Alpaca** (called by Monday run via Temporal for order execution):

| Endpoint | Purpose |
|----------|---------|
| `GET /alpaca/portfolio` | Get account positions, cash, open orders count |
| `POST /alpaca/submit-orders` | Submit orders to Alpaca paper trading |
| `GET /alpaca/order-history` | Get order execution history |

**LLM & Email** (called by Monday run via Temporal for reporting):

| Endpoint | Purpose |
|----------|---------|
| `POST /llm/sac-weekly-summary` | Generate AI summary of the SAC-only weekly run (US) |
| `POST /llm/us-alpha-hrp-summary` | Generate AI summary of US Alpha-HRP allocation (PatchTST alpha screen + rank-band sticky + HRP) |
| `POST /llm/india-alpha-hrp-summary` | Generate AI summary of India Alpha-HRP allocation (PatchTST alpha screen + rank-band sticky + HRP) |
| `POST /llm/india-training-summary` | Generate AI summary of India PatchTST training results |
| `POST /email/sac-weekly-report` | Send the SAC-only weekly portfolio analysis email via Gmail SMTP (US) |
| `POST /email/us-alpha-hrp-report` | Send US Alpha-HRP report email (alpha screen + sticky + HRP + Alpaca order execution) via Gmail SMTP |
| `POST /email/india-alpha-hrp-report` | Send India Alpha-HRP report email (alpha screen + sticky + HRP, paper-only / no broker) via Gmail SMTP |
| `POST /email/india-training-summary` | Send India training summary email via Gmail SMTP |

**Other**:

| Endpoint | Purpose |
|----------|---------|
| `GET /universe/halal` | Halal stock universe |
| `GET /universe/halal_india` | Top 15 PatchTST-scored from Nifty 500 Shariah (NSE India) |
| `GET /universe/nifty_shariah_500` | All ~210 Nifty 500 Shariah constituents (NSE India) |
| `GET /models/active-symbols` | Active symbols from SAC model metadata |
| `POST /etl/news-sentiment` | ETL pipeline for news sentiment |
| `POST /etl/sentiment-gaps` | Gap detection and backfill |
| `POST /etl/refresh-training-data` | Refresh training data (symbols from ETL_UNIVERSE config) |
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
4. Update `BRAIN_API_URL` in Temporal to use Cloud Function URL

## Universe pipeline (invariants)

Universes are produced by `brain_api.universe`. The pipeline is fixed; agents must not reintroduce factor scoring, momentum/quality/value blends, or ROE/Beta/SMA rules.

| Universe | Source code | How it is built |
|----------|-------------|------------------|
| `halal` | [`universe/halal.py`](brain_api/brain_api/universe/halal.py) | Legacy yfinance top-holdings of SPUS, HLAL, SPTE (~14 stocks). Kept for backwards compatibility only. |
| `halal_new` | [`universe/halal_new.py`](brain_api/brain_api/universe/halal_new.py) | Scrape **all** holdings from 5 ETFs (`SPUS`, `SPTE`, `SPWO` from sp-funds.com; `HLAL`, `UMMA` from Wahed Google Sheets), merge + dedupe, filter to Alpaca-tradable, append the 5 ETFs themselves. Size varies monthly (~400 stocks). US base universe. |
| `halal_filtered` | [`universe/halal_filtered.py`](brain_api/brain_api/universe/halal_filtered.py) | `halal_new` -> `filter_symbols_by_min_history` (~10 years of trading data, derived from `LSTM_TRAIN_LOOKBACK_YEARS=10` via `compute_min_walkforward_days`) -> US PatchTST batch inference -> rank-band sticky selection (`K_in=15`, `K_hold=30`, partition `halal_filtered_alpha` in the sibling `screening_history` table). Cold-start (no prior month) is byte-equivalent to the legacy blanket top-15. **Monthly cache cadence; no factor scoring.** |
| `nifty_shariah_500` | [`universe/nifty_shariah_500.py`](brain_api/brain_api/universe/nifty_shariah_500.py) | Full Nifty 500 Shariah constituents from NSE India (~210 stocks). Symbols carry `.NS` suffix end-to-end. India base universe. |
| `halal_india` | [`universe/halal_india.py`](brain_api/brain_api/universe/halal_india.py) | `nifty_shariah_500` -> same min-history filter -> India PatchTST batch inference (`PatchTSTIndiaModelStorage`) -> rank-band sticky selection (`K_in=15`, `K_hold=30`, partition `halal_india_filtered_alpha` in the `screening_history` sibling table; period_key anchored to first-Monday-of-month YYYYWW). Cold-start (no prior month) is byte-equivalent to the legacy blanket top-15. `.NS` suffix preserved end-to-end. **Monthly cache cadence; no factor scoring.** |

Invariants:

- For both `halal_filtered` (US, partition `halal_filtered_alpha`) and `halal_india` (India, partition `halal_india_filtered_alpha`), PatchTST predicted weekly return + rank-band sticky (`K_in=15`, `K_hold=30`) is the ONLY ranking step (cold-start = top-K_in by score). Both partitions live in the `screening_history` table; both are isolated from the weekly two-stage Alpha-HRP partitions in `stage1_weight_history`. Adding a momentum/quality/value layer requires explicit research approval; do not add silent fallbacks.
- `halal_india` symbols MUST keep `.NS` suffix throughout (storage, training, inference, allocation, email, screening_history.stock, evicted_from_previous keys). No append/strip transformations.
- US PatchTST and India PatchTST are independently versioned; promoting one MUST NOT touch the other's `current` pointer.
- Universe scrapes are cached monthly under `brain_api/data/cache/universe/<name>_YYYY-MM.json`. A new month auto-invalidates the cache.
- Sticky carry-set isolation: every strategy that reads/writes sticky history MUST own a unique `partition` string (see `brain_api/core/strategy_partitions.py`). Two-stage strategies (HRP-backed) live in `stage1_weight_history`; single-stage screening strategies live in the sibling `screening_history` table. Reusing a partition across strategies even when they sit in different tables corrupts the carry-set.

## Model hierarchy

### Price Forecasters

| Model | Market | Input | Output |
|-------|--------|-------|--------|
| LSTM | US | Close-only log returns (pure price) | Weekly return prediction |
| PatchTST | US | OHLCV (5-channel log returns) | Weekly return prediction |
| PatchTST India | India (NiftyShariah500) | OHLCV (5-channel log returns) | Weekly return prediction |

### Portfolio Allocators

| Model | Input | Output |
|-------|-------|--------|
| HRP | Covariance matrix | Allocation weights |
| SAC | State vector + dual forecasts (LSTM + PatchTST) | Allocation weights |

### Signal state vector (for SAC)

SAC consumes a flat state vector composed of **per-stock features** and **portfolio-level features**. Source of truth: `StateSchema` in [brain_api/brain_api/core/portfolio_rl/state.py](brain_api/brain_api/core/portfolio_rl/state.py).

**Per-stock features (9 per stock, x `n_stocks`):**

| Feature | Source |
|---------|--------|
| News sentiment score | `/signals/news` (FinBERT) |
| Gross margin | `/signals/fundamentals` |
| Operating margin | `/signals/fundamentals` |
| Net margin | `/signals/fundamentals` |
| Current ratio | `/signals/fundamentals` |
| Debt to equity | `/signals/fundamentals` |
| Fundamental data age | Days since last fundamentals update |
| LSTM predicted weekly return | `/inference/lstm` (US, re-run on the chosen 15) |
| PatchTST predicted weekly return | `/inference/patchtst` (US, re-run on the chosen 15) |

**Portfolio-level features (`n_stocks + 1`):**

| Feature | Source |
|---------|--------|
| Current weight per stock | Portfolio state |
| Current cash weight (CASH slot) | Portfolio state |

For `n_stocks = 15` -> `state_dim = 15*7 + 15*2 + 16 = 151`. Both LSTM and PatchTST run **on the 15-name slate** chosen by `halal_filtered` so that SAC's dual-forecast features cover the same symbols as its action space.

**Key distinction:**
- **LSTM** = pure price forecaster (close returns only, US only)
- **PatchTST** (US) = OHLCV forecaster (5-channel: open, high, low, close, volume log returns)
- **PatchTST India** = OHLCV forecaster (5-channel, India NiftyShariah500, independent storage + versioning under `data/models/patchtst_india/`)
- **SAC** = RL allocator that receives the 9-per-stock features (including dual LSTM + PatchTST forecasts) plus portfolio weights, US only

## Data storage rules

Store three classes of data:

- **Structured DB** (local Postgres via Docker)
  - runs, screening decisions, signals, decisions, orders
- **Local SQLite** (single file at `data/allocation/sticky_history.db`, two sibling tables)
  - `stage1_weight_history` -- two-stage strategies (HRP-backed, **weekly cadence**). Partitions: `halal_new` (US Double HRP), `halal_new_alpha` (US Alpha-HRP), `halal_india_alpha` (India Alpha-HRP). See `brain_api/storage/sticky_history.py` for rerun semantics (delete-then-insert per `(universe, year_week)`).
  - `screening_history` -- single-stage screening strategies (no Stage 2 HRP, **monthly cadence**). Partitions: `halal_filtered_alpha` (monthly halal_filtered builder, US) and `halal_india_filtered_alpha` (monthly halal_india builder, India NSE; `.NS`-suffixed stock values stored verbatim). Both anchor period_key to the first-Monday-of-month YYYYWW. See `brain_api/storage/screening_history.py` for rerun semantics (delete-then-insert per `(partition, period_key)`). Note: `screening_history` and `stage1_weight_history` are physically separate tables in the same `data/allocation/sticky_history.db` file; cross-table reads are forbidden by construction.
  - Partition strings MUST be unique across the union of both tables (see `brain_api/core/strategy_partitions.py`).
- **Raw evidence snapshots** (filesystem)
  - `data/raw/<run_id>/<attempt>/<source>/<symbol>.json`
- **Feature snapshots**
  - `data/features/<run_id>/<attempt>/...`

Every persisted record must include:

- `run_id`, `attempt`
- an `as_of` timestamp for time-sensitive signals

### Model storage

Models are stored under `data/models/{lstm,patchtst,patchtst_india}/<version>/`:

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
├── patchtst/
│   └── (same structure)
└── patchtst_india/
    └── (same structure, independent current pointer)
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

### LLM summary (Temporal orchestrated)

The Monday email includes an **AI summary** generated by OpenAI/GPT-4o-mini:

- Temporal workflow calls brain_api's `/llm/sac-weekly-summary` endpoint with the SAC-only signal data
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
- Reuse must never compromise per-algorithm math correctness; see "AI assistant behavioral rules" #2

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
| Monthly (Saturday) | Full retrain all US models | Manual |
| Weekly (Sunday 11 AM UTC) | Fine-tune SAC variants (US) | Cron (Temporal) |
| Weekly (Sunday 4:30 AM UTC) | Full PatchTST retrain (India) | Cron (Temporal) |
| Monday 6 PM IST | US inference only | Cron (Temporal) |

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
  - clear error propagation back to Temporal

### Temporal workflow configuration

Key configuration:

- **Activity timeouts**: `start_to_close_timeout` on every activity (5 min for API calls, 10h for training)
- **Activity retries**: `RetryPolicy(maximum_attempts=N)` on activities that call external APIs
- **Heartbeating**: Long-running training activities use `heartbeat_timeout` to detect stalled workers
- **Resume from failure**: Automatic. Temporal replays from event history -- completed activities are skipped automatically. No cache policies needed.
- **Durable sleep**: `workflow.sleep()` survives worker crashes, laptop shutdowns, and restarts
- **Parallel execution**: `asyncio.gather()` for concurrent activity execution within workflows
- **Pydantic data converter**: `pydantic_data_converter` used for correct Pydantic v2 serialization
- **Sell-wait-buy**: Single workflow with `while True: check -> sleep 15 min` durable polling loop

**Laptop-only setup** (3 terminal processes):
1. `devbox run temporal:server` -- Temporal dev server with SQLite persistence + UI at port 8233
2. `devbox run brain:run` -- brain_api FastAPI service
3. `devbox run temporal:worker` -- Python worker polling the `learnfinance` task queue

**Schedule registration** (run once): `devbox run temporal:schedule`

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
- [ ] Confirm PatchTST/SAC receive correct signal state vector
- [ ] Confirm India PatchTST uses `patchtst_india` storage (not US `patchtst`)
- [ ] Confirm India symbols retain `.NS` suffix throughout the pipeline (including `screening_history.stock` rows and `evicted_from_previous` keys for the `halal_india_filtered_alpha` partition)
- [ ] Confirm sticky carry-set isolation: no two strategies share a `partition` string in `brain_api/core/strategy_partitions.py` (uniqueness across `stage1_weight_history` AND `screening_history`)
- [ ] Confirm India universe builders (monthly `halal_india`) write to `screening_history` via `ScreeningHistoryRepository`, NOT `stage1_weight_history` -- the weekly India Alpha-HRP partition (`halal_india_alpha`) is the only India strategy that uses the two-stage table

## AI assistant behavioral rules

1. **Never add silent fallbacks without asking first.** Fallbacks mask real bugs and break correctness. For example, falling back to momentum when a snapshot fails to load means the system silently produces garbage instead of surfacing the error. Always raise exceptions for unexpected failures; ask the user before adding any degraded-mode fallback.

2. **Math correctness is the highest priority -- never break math to simplify code.** DRY, DDD, and clean code matter and you should factor out genuinely shared logic. The rule is about precedence, not duplication: when two algorithms have research-driven mathematical differences (even subtle ones), each must keep its own math even if the surface code looks similar. Concrete cautionary tale from this repo: PPO and SAC each have algorithm-specific mathematical steps; we once "reused" code between them for DRY and silently broke PPO's math. If the math is provably identical (e.g., a standard formula like Sharpe ratio, a generic covariance estimator, a shared data loader), share it; if there is any research-level difference, keep the implementations separate even if the code looks alike. When in doubt, ask before merging two model-specific code paths.

## AI assistant planning rules

When operating in **plan mode**, the AI assistant must:

1. Always include these two final TODOs at the end of every plan:
   - [ ] Fix all ruff linting issues (related and unrelated to the change)
   - [ ] Run and fix all tests (related and unrelated to the change)

2. These cleanup tasks ensure the codebase stays healthy with every change.
