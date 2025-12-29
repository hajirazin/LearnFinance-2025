# CLAUDE.md

This file is the **working agreement** for humans + AI assistants contributing to this repo.

## Project intent (north star)

Build a weekly, paper-trading portfolio decision system for halal Nasdaq-500 stocks that is:

- **Safe-by-default** (paper auto-submit only; reruns cannot duplicate orders)
- **Audit-friendly** (every run reproducible and explainable)
- **Learning-focused** (n8n orchestration + multi-agent workflows + LSTM + PPO)
- **Cloud-ready** (local-first design that can migrate to Cloud Functions)

## Non-negotiable invariants (do not break)

### Run identity & rerun semantics

- `run_date` is the **Monday date in IST** (calendar date).
- `run_id = paper:YYYY-MM-DD`.
- `attempt` starts at `1`.
- **Rerun is read-only** if the latest attempt has any order not in a terminal canceled/expired/rejected state.
- To allow a new submission: user cancels paper orders manually in Alpaca, then rerun creates `attempt += 1`.

### Order idempotency

All submitted orders must include deterministic `client_order_id`:

- `paper:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>`

The system must:

- Check local DB for existing submissions before submitting.
- Query Alpaca by `client_order_id` as a second guardrail.

### Trading mode

- **Paper auto-submit** is allowed.
- **Live trading is out of scope** unless explicitly added with additional safety controls.

### Default execution choices

- Default order type: **limit**
- Default sizing: **fractional shares when supported**

### Model lifecycle (training vs inference)

- **Monday runs are inference-only**. Never retrain inside the Monday inference run.
- **Training happens on Sundays** as separate jobs:
  - LSTM: full retrain on **first Sunday of month**
  - PPO: fine-tune **every Sunday** using 26-week rolling experience buffer
- Training produces a **new versioned artifact**; inference loads from `current` pointer.
- **Promotion requires evaluation**: new model must beat prior + baseline before becoming `current`.
- **Rollback is always possible**: keep last known-good version; pointer swap is atomic.

### Model storage (local artifacts)

- Models are stored under `data/models/{lstm,ppo}/<version>/`
- Active version tracked by `data/models/{model}/current` (text file with version string)
- PPO experience buffer stored under `data/experience/<run_id>.json`
- All model artifacts must include `metadata.json` with: training timestamp, data window, config hash, eval metrics

## Architecture boundaries (keep it modular)

- **n8n** is the outer orchestrator:
  - schedule trigger
  - calling APIs (Alpaca, email)
  - calling the Python brain API
  - status tracking + notifications
- **Python brain** owns:
  - universe build + screening
  - data ingestion and caching
  - multi-agent committee workflows (agent-to-agent)
  - LSTM inference + PPO decisions
  - explanation generation
  - persistence of run artifacts

Avoid putting "business logic" inside n8n nodes beyond simple orchestration.

## API design rules (cloud-ready)

Each ML operation must be a **separate REST endpoint** that can later become a standalone Cloud Function.

### Required endpoints

**Inference** (called by Monday run):

- `POST /inference/lstm` — price predictions
- `POST /inference/ppo` — portfolio allocation

**Training** (called by Sunday cron or manual):

- `POST /train/lstm` — full LSTM retrain
- `POST /train/ppo/finetune` — PPO fine-tune on 26-week buffer
- `POST /train/ppo/full` — PPO full retrain (drift recovery)

**Model management**:

- `GET /models/{model}/current` — active version + metadata
- `POST /models/{model}/promote` — promote version to current
- `POST /models/{model}/rollback` — revert to prior known-good

### Design rules (do not violate)

1. **Stateless**: load model from storage on each request; no in-memory state across requests
2. **Storage abstraction**: use `storage.load_model(path)` that works for local or GCS
3. **JSON in, JSON out**: core functions must not depend on FastAPI request objects
4. **Idempotent training**: version ID = `hash(data_window + config_hash)`, so re-runs produce same version
5. **Thin endpoints**: FastAPI route handlers only validate + call core functions + return response

### Code structure pattern

```
brain/
├── api/
│   └── routes/
│       ├── inference.py      # POST /inference/lstm, /inference/ppo
│       ├── training.py       # POST /train/lstm, /train/ppo/*
│       └── models.py         # GET/POST /models/*
├── core/
│   ├── lstm.py               # lstm_inference(), lstm_train() — pure functions
│   ├── ppo.py                # ppo_inference(), ppo_finetune(), ppo_train()
│   └── ...
├── storage/
│   ├── base.py               # abstract Storage class
│   ├── local.py              # LocalStorage(base_path="data/")
│   └── gcs.py                # GCSStorage(bucket="...") — swap via env var
└── ...
```

### Cloud Function migration

When migrating an endpoint to GCP:

1. Extract core function call into `main.py` with `def handler(request):`
2. Set `STORAGE_BACKEND=gcs` environment variable
3. Deploy: `gcloud functions deploy <name> --runtime python311 --trigger-http`
4. Update caller (n8n) to use Cloud Function URL

## Data storage rules (auditability)

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

## Agent workflow rules (multi-agent, but disciplined)

Agents must produce **structured outputs** that can be stored and audited:

- Include citations/identifiers where possible (e.g., news URL, tweet IDs, data source + timestamp).
- A `RiskCritic` (or equivalent) must be able to:
  - challenge contradictions
  - flag weak/insufficient evidence
  - downgrade confidence or veto a trade recommendation

Agents are used for **evidence synthesis**. Numeric optimization remains in deterministic code (feature engineering) + LSTM/RL.

## Testing policy (important)

User preference / repo rule:

- In Python, **never write schema tests**. Schemas are exercised via API usage.
- In the router layer, add **explicit tests by calling the API** for constraint behaviors (e.g., `min_items`, `max_items`, min/max length/count).

If tests are added later, they should be:

- Integration-style API tests for routers/handlers
- Deterministic unit tests for pure functions (feature transforms, idempotency key generation, screening ranking)

## Operational requirements

Any implementation must include:

- **Idempotency**: safe reruns
- **Timeouts + retries** with exponential backoff for external APIs
- **Rate limit awareness** and batching
- **Observability**:
  - run-level logs with `run_id` + `attempt`
  - stage duration metrics (even if just logged)
  - clear error propagation back to n8n

## Change safety checklist

Before merging changes that touch trading logic:

- Confirm rerun behavior is still read-only after any submission
- Confirm `client_order_id` format unchanged (or migration handled)
- Confirm safety caps exist and are enforced (max turnover, max orders, cash buffer)

Before merging changes that touch ML/model code:

- Confirm Monday inference does NOT trigger training
- Confirm training writes new versioned artifact (not overwrite)
- Confirm promotion requires evaluation gate
- Confirm endpoints remain stateless (no global model cache)
- Confirm storage abstraction is used (not hardcoded paths)


