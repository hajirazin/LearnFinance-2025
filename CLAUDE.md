# CLAUDE.md

This file is the **working agreement** for humans + AI assistants contributing to this repo.

## Project intent (north star)

Build a weekly, paper-trading portfolio decision system for halal Nasdaq-500 stocks that is:

- **Safe-by-default** (paper auto-submit only; reruns cannot duplicate orders)
- **Audit-friendly** (every run reproducible and explainable)
- **Learning-focused** (n8n orchestration + multi-agent workflows + LSTM + RL)

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
  - LSTM inference + RL decisions
  - explanation generation
  - persistence of run artifacts

Avoid putting “business logic” inside n8n nodes beyond simple orchestration.

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


