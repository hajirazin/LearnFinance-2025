# CLAUDE.md

This file is the **working agreement** for humans + AI assistants contributing to this repo.

## Project intent (north star)

Build a **learning-focused** weekly portfolio system (paper by default, per-account live opt-in via env) for halal Nasdaq-500 stocks that **compares multiple approaches side-by-side**:

- **Safe-by-default** (paper auto-submit; live requires explicit per-account env opt-in; reruns cannot duplicate orders)
- **Audit-friendly** (every run reproducible and explainable)
- **Learning-focused** (compare LSTM vs PatchTST, SAC, all vs HRP baseline)
- **Cloud-ready** (local-first design that can migrate to Cloud Functions / HuggingFace Hub)

The goal is to learn which approaches work best, not to pick a single method upfront.

## Architecture boundaries

- **Temporal** is the outer orchestrator (replaced Prefect):
  - schedule trigger (Monday 6 PM IST for US inference, Monday 9 AM IST for India, and four monthly training slots on the first Sunday of the month staggered 6h apart starting 00:01 UTC -- see "Training schedule" below)
  - calling brain_api endpoints via HTTP activities
  - handling parallel task execution (asyncio.gather) and skip logic
  - durable sleep/wait for sell-wait-buy pattern (single workflow, no 3-flow hack)
  - automatic replay from event history (no cache policies needed)
  - status tracking + workflow observability via Temporal UI (port 8233)
  - Runs locally via `temporal server start-dev` (SQLite persistence, survives laptop shutdown)
  - India weekly allocation workflow (`IndiaWeeklyAllocationWorkflow`): full Nifty Shariah 500 universe -> PatchTST alpha screen (`/inference/patchtst/score-batch` with `market='india'`) -> rank-band sticky selection (`halal_india_alpha` partition, K_in=15 / K_hold=20) -> HRP allocation (lookback=252d) on the 15 chosen names -> record final weights -> AI summary -> email (paper-only, no broker)
  - India training workflow (`IndiaWeeklyTrainingWorkflow`): NiftyShariah500 universe -> PatchTST India train -> halal_india rank-band sticky top 15 (`halal_india_filtered_alpha` partition in `screening_history`, monthly cadence) -> LLM summary -> email
  - US weekly allocation workflow (`USWeeklyAllocationWorkflow`): signals + forecasts -> SAC allocator (universe `halal_filtered`) -> sell-wait-buy with durable polling via the `sac` Alpaca account (orders tagged `algorithm='sac'`) -> AI summary -> email tagged `universe='halal_filtered'`. Run_id stays `paper:YYYY-MM-DD` (default form).
  - US SAC (halal) weekly allocation workflow (`USSACHalalAllocationWorkflow`): parallel A/B sibling of `USWeeklyAllocationWorkflow` running 30 minutes later on Mondays. Same SAC inference pipeline but with universe `halal` (the legacy yfinance halal universe, variable size 10-15). Trades through the dedicated `sac_halal` Alpaca account (orders tagged `algorithm='sac_halal'`); experience records share `model_type='sac'` but live in their own files because `run_id` is `paper:halal:YYYY-MM-DD`. AI summary + email tagged `universe='halal'`.
  - US Alpha-HRP workflow (`USAlphaHRPWorkflow`): halal_new universe -> PatchTST alpha screen (`/inference/patchtst/score-batch` with `market='us'`) -> rank-band sticky selection (`halal_new_alpha` partition, K_in=15 / K_hold=20) -> HRP allocation (lookback=252d) on the 15 chosen names -> record final weights -> sell-wait-buy via the `hrp` Alpaca account (orders tagged `algorithm='alpha_hrp'`) -> AI summary -> email. Replaced the retired naive-HRP path that used to run inside `USWeeklyAllocationWorkflow`.
  - US Double HRP workflow (`USDoubleHRPWorkflow`): halal_new universe -> Stage 1 HRP (lookback=756d) -> sticky top 15 (`halal_new` partition in `stage1_weight_history`, K_in=15, stickiness threshold 1.0pp) -> Stage 2 HRP (lookback=252d) on the chosen 15 -> record final weights -> sell-wait-buy via the dedicated `dhrp` Alpaca account (orders tagged `algorithm='dhrp'`) -> AI summary -> email.
  - India Double HRP workflow (`IndiaDoubleHRPWorkflow`): full Nifty Shariah 500 universe -> Stage 1 HRP (lookback=756d) -> weight-band sticky top 15 (`halal_india_double_hrp` partition in `stage1_weight_history`, K_in=15, stickiness threshold 1.0pp) -> Stage 2 HRP (lookback=252d) on the chosen 15 -> record final weights -> AI summary -> email (paper-only, no broker). Mirrors `USDoubleHRPWorkflow` minus the Alpaca order-execution legs; the strategy-named partition keeps its carry-set isolated from `halal_india_alpha` (rank-band, sister India strategy) and from `halal_new` (US Double HRP).
  - US forecasters training workflow (`USForecastersTrainingWorkflow`): halal_new universe -> train LSTM -> train PatchTST (strictly serial, single trainer at a time) -> forecasters-only LLM summary -> forecasters-only email
  - US SAC training workflow (`USSACTrainingWorkflow`): runs 6 hours after the forecasters workflow on the same first-Sunday-of-month slot (06:01 UTC); halal_filtered top-15 (uses whatever PatchTST `current` pointer is live at trigger time) -> fill sentiment gaps with mandatory Hugging Face publication -> train SAC -> SAC-only LLM summary -> SAC-only email. Serialization guarantee: the Mac training worker runs with `TEMPORAL_MAX_CONCURRENT_ACTIVITIES=1`, so even if the forecasters run overshoots 6h this activity waits for it to finish rather than starting in parallel.
  - US SAC (halal) training workflow (`USSACHalalTrainingWorkflow`): parallel A/B sibling of `USSACTrainingWorkflow` running 6 hours later on the same first-Sunday-of-month slot (12:01 UTC) on the legacy yfinance halal universe (variable size, ~12-15 stocks); fill sentiment gaps with mandatory Hugging Face publication -> train SAC (`sac_halal` bucket, independent `current` pointer) -> SAC-only LLM summary tagged `universe=halal` -> SAC-only email tagged `universe=halal`. Same single-activity serialization applies via the training worker's concurrency cap.
- **brain_api (Python brain)** owns:
  - universe build + screening
  - signal collection (news + price momentum)
  - price forecasting (LSTM pure-price, PatchTST close-only)
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
│   ├── sac/                 # SAC allocator (PatchTST forecast features)
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
│   ├── us_forecasters_training.py # US LSTM + PatchTST training (Saturday)
│   ├── us_sac_training.py        # US SAC training (Sunday, 12+ h after forecasters)
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
| `POST /inference/sac` | SAC allocation using PatchTST forecasts in the state vector; `universe` query param is mandatory (`halal_filtered` or `halal`) |
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
| `POST /signals/prices` | Adjusted daily closes, current execution prices, and provenance for SAC v3 eligibility/features |
| `POST /signals/market-history` | Gap-checked aligned SPY adjusted-close/VIX history after the active SAC artifact's HMM cutoff |

**Training** (called by Saturday/Sunday cron or manual):

| Endpoint | Purpose |
|----------|---------|
| `POST /train/lstm` | Full LSTM retrain |
| `POST /train/patchtst` | Full PatchTST retrain (US) |
| `POST /train/patchtst/india` | Full PatchTST retrain (India NiftyShariah500) |
| `POST /train/sac/full` | Full SAC retrain (PatchTST-only forecasts). Body `{"universe": "halal_filtered"\|"halal"}` selects the bucket; ``n_stocks`` and ``target_entropy`` are resized at training time from the bucket's symbol count via `make_sac_config_for_n_stocks`. |

**Alpaca** (called by Monday run via Temporal for order execution):

| Endpoint | Purpose |
|----------|---------|
| `GET /alpaca/portfolio` | Get account positions, cash, open orders count |
| `POST /alpaca/submit-orders` | Submit orders to Alpaca (paper by default, live when `ALPACA_{ACCOUNT}_URL` overrides the host) |
| `GET /alpaca/order-history` | Get order execution history |
| `GET /alpaca/clock` | Get the Alpaca market clock (`is_open`, `next_open`, `next_close`). Authenticates with the generic `ALPACA_API_KEY` / `ALPACA_API_SECRET` env pair (NOT per-account trading creds) and always hits the paper host -- the clock payload is account-agnostic and identical paper vs live. Consumed by Temporal's `sell_wait_buy` helper to sleep until the next NYSE open. |

**LLM & Email** (called by Monday run via Temporal for reporting):

| Endpoint | Purpose |
|----------|---------|
| `POST /llm/sac-weekly-summary` | Generate AI summary of the SAC-only weekly run (US); `universe` field is mandatory in the body (`halal_filtered` or `halal`) so the prompt can label the A/B run |
| `POST /llm/us-alpha-hrp-summary` | Generate AI summary of US Alpha-HRP allocation (PatchTST alpha screen + rank-band sticky + HRP) |
| `POST /llm/india-alpha-hrp-summary` | Generate AI summary of India Alpha-HRP allocation (PatchTST alpha screen + rank-band sticky + HRP) |
| `POST /llm/india-training-summary` | Generate AI summary of India PatchTST training results |
| `POST /llm/forecasters-training-summary` | Generate AI summary of US LSTM + PatchTST training (called by `USForecastersTrainingWorkflow`) |
| `POST /llm/sac-training-summary` | Generate AI summary of US SAC training (called by `USSACTrainingWorkflow` with `universe="halal_filtered"` (default) and by `USSACHalalTrainingWorkflow` with `universe="halal"`). The `universe` field branches the prompt so the summary identifies which bucket. |
| `POST /email/sac-weekly-report` | Send the SAC-only weekly portfolio analysis email via Gmail SMTP (US); `universe` is mandatory and renders into the subject (`US SAC ({universe}) Weekly Portfolio Analysis ...`) so the two A/B runs are distinguishable in the inbox |
| `POST /email/us-alpha-hrp-report` | Send US Alpha-HRP report email (alpha screen + sticky + HRP + Alpaca order execution) via Gmail SMTP |
| `POST /email/india-alpha-hrp-report` | Send India Alpha-HRP report email (alpha screen + sticky + HRP, paper-only / no broker) via Gmail SMTP |
| `POST /email/india-training-summary` | Send India training summary email via Gmail SMTP |
| `POST /email/forecasters-training-summary` | Send US Forecasters (LSTM + PatchTST) training summary email via Gmail SMTP |
| `POST /email/sac-training-summary` | Send US SAC training summary email via Gmail SMTP. Subject is `f"US SAC ({universe}) Training: ..."` so the two parallel A/B reports (`halal_filtered`, `halal`) are distinguishable in the inbox. |

**Other**:

| Endpoint | Purpose |
|----------|---------|
| `GET /universe/halal` | Halal stock universe |
| `GET /universe/halal_india` | Top 15 PatchTST-scored from Nifty 500 Shariah (NSE India) |
| `GET /universe/nifty_shariah_500` | All ~210 Nifty 500 Shariah constituents (NSE India) |
| `GET /models/active-symbols` | Active SAC symbols plus schema version and HMM training cutoff; `universe` query param is mandatory (`halal_filtered` or `halal`) |
| `POST /etl/news-sentiment` | ETL pipeline for news sentiment (`universe` required in body) |
| `POST /etl/sentiment-gaps` | Gap detection and backfill (`universe` required in body) |
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
2. Set `STORAGE_BACKEND=hf_first` environment variable. Cold start
   (HF `main` missing) on Cloud Functions surfaces as a 503 for
   inference and as an inaugural promotion for training -- writes
   always go to local AND HF whenever the bucket has an HF repo
   configured, so the first successful training run populates HF
   `main` for subsequent inference invocations. ETL (news / twitter
   sentiment datasets) is intentionally out of policy gating.
3. Deploy: `gcloud functions deploy <name> --runtime python311 --trigger-http`
4. Update `BRAIN_API_URL` in Temporal to use Cloud Function URL

## Universe pipeline (invariants)

Universes are produced by `brain_api.universe`. The pipeline is fixed; agents must not reintroduce factor scoring, momentum/quality/value blends, or ROE/Beta/SMA rules.

| Universe | Source code | How it is built |
|----------|-------------|------------------|
| `halal` | [`universe/halal.py`](brain_api/brain_api/universe/halal.py) | Legacy yfinance top-holdings of SPUS, HLAL, SPTE (~14 stocks). Kept for backwards compatibility only. |
| `halal_new` | [`universe/halal_new.py`](brain_api/brain_api/universe/halal_new.py) | Scrape **all** holdings from 5 ETFs (`SPUS`, `SPTE`, `SPWO` from sp-funds.com; `HLAL`, `UMMA` from Wahed Google Sheets), merge + dedupe, filter to Alpaca-tradable, append the 5 ETFs themselves. Size varies monthly (~400 stocks). US base universe. |
| `halal_filtered` | [`universe/halal_filtered.py`](brain_api/brain_api/universe/halal_filtered.py) | `halal_new` -> `filter_symbols_by_min_history` (~10 years of trading data, derived from `LSTM_TRAIN_LOOKBACK_YEARS=10` via `compute_min_walkforward_days`) -> US PatchTST batch inference -> rank-band sticky selection (`K_in=15`, `K_hold=20`, partition `halal_filtered_alpha` in the sibling `screening_history` table). Cold-start (no prior month) is byte-equivalent to the legacy blanket top-15. **Monthly cache cadence; no factor scoring.** |
| `nifty_shariah_500` | [`universe/nifty_shariah_500.py`](brain_api/brain_api/universe/nifty_shariah_500.py) | Full Nifty 500 Shariah constituents from NSE India (~210 stocks). Symbols carry `.NS` suffix end-to-end. India base universe. |
| `halal_india` | [`universe/halal_india.py`](brain_api/brain_api/universe/halal_india.py) | `nifty_shariah_500` -> same min-history filter -> India PatchTST batch inference (`PatchTSTIndiaModelStorage`) -> rank-band sticky selection (`K_in=15`, `K_hold=20`, partition `halal_india_filtered_alpha` in the `screening_history` sibling table; period_key anchored to first-Monday-of-month YYYYWW). Cold-start (no prior month) is byte-equivalent to the legacy blanket top-15. `.NS` suffix preserved end-to-end. **Monthly cache cadence; no factor scoring.** |

Invariants:

- For both `halal_filtered` (US, partition `halal_filtered_alpha`) and `halal_india` (India, partition `halal_india_filtered_alpha`), PatchTST predicted weekly return + rank-band sticky (`K_in=15`, `K_hold=20`) is the ONLY ranking step (cold-start = top-K_in by score). Both partitions live in the `screening_history` table; both are isolated from the weekly two-stage Alpha-HRP partitions in `stage1_weight_history`. Adding a momentum/quality/value layer requires explicit research approval; do not add silent fallbacks.
- `halal_india` symbols MUST keep `.NS` suffix throughout (storage, training, inference, allocation, email, screening_history.stock, evicted_from_previous keys). No append/strip transformations.
- US PatchTST and India PatchTST are independently versioned; promoting one MUST NOT touch the other's `current` pointer.
- Universe scrapes are cached monthly under `brain_api/data/cache/universe/<name>_YYYY-MM.json`. A new month auto-invalidates the cache.
- Sticky carry-set isolation: every strategy that reads/writes sticky history MUST own a unique `partition` string (see `brain_api/core/strategy_partitions.py`). Two-stage strategies (HRP-backed) live in `stage1_weight_history`; single-stage screening strategies live in the sibling `screening_history` table. Reusing a partition across strategies even when they sit in different tables corrupts the carry-set.
- Weight-band vs rank-band selectors: the two sticky primitives in `brain_api/core/sticky_selection.py` (`select_with_stickiness` for weight-band, `select_with_rank_band` for rank-band) are different mathematical operators per AGENTS.md rule #2. Any new strategy MUST pick one and own a partition that no other selector touches; never mix the two selectors against the same partition. The Double HRP family (US `halal_new`, India `halal_india_double_hrp`) uses weight-band; the Alpha-HRP family (US `halal_new_alpha`, India `halal_india_alpha`) uses rank-band. Likewise, the email + LLM template families (`double_hrp_email_base.html.j2` / `double_hrp_summary_prompt_base.j2` vs `alpha_hrp_email_base.html.j2` / `alpha_hrp_summary_prompt_base.j2`) are sibling families, NOT a unified family -- they describe different math and must not be merged.

## Model hierarchy

### Price Forecasters

| Model | Market | Input | Output |
|-------|--------|-------|--------|
| LSTM | US | Close-only log returns (pure price) | Weekly return prediction |
| PatchTST | US | Close log returns (1 channel) | Weekly return prediction |
| PatchTST India | India (NiftyShariah500) | Close log returns (1 channel) | Weekly return prediction |

### Portfolio Allocators

| Model | Input | Output |
|-------|-------|--------|
| HRP | Covariance matrix | Allocation weights |
| SAC | State vector + PatchTST forecast features | Allocation weights |

### SAC v3 token state and action contract

SAC v3 uses one unconditional fixed-shape contract for both SAC buckets; there
are no feature flags or legacy flat-model fallbacks. The serialized state is
245 values: `asset_features[30,7]` + `globals[5]` + the auxiliary binary
`asset_mask[30]`. The actor is a shared token encoder plus masked attention and
always emits 31 logits (30 stock slots + CASH). The critics encode shared
masked `(token, action)` pairs and pool them; padded slots are excluded from
attention, sampling, log probability, entropy, Q, allocation, reward, and
costs. Source of truth: `StateSchema` in
[brain_api/brain_api/core/portfolio_rl/state.py](brain_api/brain_api/core/portfolio_rl/state.py).

**Per-asset token (valid assets are exact average-tie CS-ranked):**

| Feature | Raw source |
|---------|------------|
| PatchTST weekly-return rank | `/inference/patchtst` |
| Momentum 1w rank (`P[t]/P[t-5]-1`) | adjusted closes |
| Momentum 4w rank (`P[t]/P[t-20]-1`) | adjusted closes |
| Momentum 12-1 rank (`P[t-21]/P[t-252]-1`) | adjusted closes |
| News-sentiment rank | provider-checked `/signals/news` |
| Realized-volatility rank | 20 adjusted-close log returns, `ddof=1`, annualized `sqrt(252)` |
| Current stock weight | portfolio state, unscaled |

The five globals are raw PatchTST median (training-fold standardized), raw
PatchTST fraction positive, filtered HMM calm/stress probabilities, and
unscaled cash weight. All ranks, probabilities, weights, cash, and masks remain
unscaled. The HMM is a deterministic three-state diagonal Gaussian model over
SPY 20-bar return/realized volatility plus positive VIX level/5-bar change. It
is fit on the train fold only and validation/test/live use causal forward
filtering. Artifacts persist the cutoff posterior and the final 21 aligned SPY
and VIX sessions so stateless live inference can derive the first post-cutoff
observation without overlapping history.

Live SPY/VIX evidence must cover the exact XNYS sessions after the artifact
cutoff through the latest completed session before the scheduled decision date.
The Monday 18:00 IST SAC decision is pre-open in New York, so the partial Monday
session is never requested, required, or used; the normal endpoint is the prior
Friday. Empty evidence is valid only when no completed post-cutoff XNYS session
is due.

Missing/non-finite PatchTST or insufficient/nonpositive feature history masks
that asset; provider/news failures abort. Production train/infer requires at
least 10 eligible assets. A held asset without a finite positive execution
price aborts the rebalance; padded zero prices are never costed. Training and
Temporal preserve identical adjusted-close momentum/volatility formulas. LSTM
remains standalone and is not a SAC input.

**Key distinction:**
- **LSTM** = pure price forecaster (close returns only, US only)
- **PatchTST** (US) = close-only forecaster (1-channel close log returns)
- **PatchTST India** = close-only forecaster (1-channel close log returns, India NiftyShariah500, independent storage + versioning under `data/models/patchtst_nifty_shariah_500/`)
- **SAC** = RL allocator that receives 5 signals + PatchTST forecast per stock plus portfolio weights, US only

## Data storage rules

Store three classes of data:

- **Structured DB** (local Postgres via Docker)
  - runs, screening decisions, signals, decisions, orders
- **Local SQLite** (single file at `data/allocation/sticky_history.db`, two sibling tables)
  - `stage1_weight_history` -- two-stage strategies (HRP-backed, **weekly cadence**). Partitions: `halal_new` (US Double HRP), `halal_new_alpha` (US Alpha-HRP), `halal_india_alpha` (India Alpha-HRP), `halal_india_double_hrp` (India Double HRP). See `brain_api/storage/sticky_history.py` for rerun semantics (delete-then-insert per `(universe, year_week)`). Note that `halal_india_alpha` and `halal_india_double_hrp` MUST stay in distinct partitions even though they screen the same NSE Shariah universe -- they use different selector primitives (rank-band vs weight-band) and merging their carry-sets would silently miscategorise "previously held" semantics across weeks.
  - `screening_history` -- single-stage screening strategies (no Stage 2 HRP, **monthly cadence**). Partitions: `halal_filtered_alpha` (monthly halal_filtered builder, US) and `halal_india_filtered_alpha` (monthly halal_india builder, India NSE; `.NS`-suffixed stock values stored verbatim). Both anchor period_key to the first-Monday-of-month YYYYWW. See `brain_api/storage/screening_history.py` for rerun semantics (delete-then-insert per `(partition, period_key)`). Note: `screening_history` and `stage1_weight_history` are physically separate tables in the same `data/allocation/sticky_history.db` file; cross-table reads are forbidden by construction.
  - Partition strings MUST be unique across the union of both tables (see `brain_api/core/strategy_partitions.py`).
- **Raw evidence snapshots** (filesystem)
  - `data/raw/<run_id>/<attempt>/<source>/<symbol>.json`
- **Feature snapshots**
  - `data/features/<run_id>/<attempt>/...`

Every persisted record must include:

- `run_id`, `attempt`
- an `as_of` timestamp for time-sensitive signals

### Model storage (universe-keyed buckets)

Each `(model_type, universe)` pair is its own *bucket*. The bucket name
is `{model}_{universe}` and drives both the on-disk path and the
HuggingFace repo, so two parallel A/B workflows (e.g. `sac_halal` vs
`sac_halal_filtered`) never collide on the `current` pointer.

```
data/models/
├── lstm_halal_new/
│   ├── v2026-01-09-a4fecab1bdcc/   # versioned artifact
│   │   ├── weights.pt
│   │   ├── feature_scaler.pkl
│   │   ├── config.json
│   │   └── metadata.json
│   ├── snapshot-2025-12-31/        # point-in-time snapshots (siblings)
│   └── current                     # text file with active version
├── patchtst_halal_new/
│   └── (same structure, US PatchTST trained on halal_new)
├── patchtst_nifty_shariah_500/
│   └── (same structure, India PatchTST; independent current pointer)
├── sac_halal_filtered/
│   └── (same structure, SAC trained on top-15 halal_filtered; n_stocks=15 fixed)
└── sac_halal/
    └── (same structure, SAC trained on the legacy yfinance halal universe;
         variable n_stocks bound to the bucket's symbol count at training
         time -- independent current pointer, parallel A/B sibling of
         sac_halal_filtered)
```

- Active version per bucket: `data/models/{bucket_name}/current`.
- RL experience buffer: `data/experience/<run_id>.json`.
- All model artifacts must include `metadata.json` with: training timestamp, data window, config hash, eval metrics.

The bucket registry lives in
[brain_api/brain_api/core/model_buckets.py](brain_api/brain_api/core/model_buckets.py).
Each `BucketConfig` records the local storage class, HF storage class,
HF repo getter, in-process symbol resolver, and an optional symbol
validator (e.g. `.NS` suffix enforcement for India). Training endpoints
take `{"universe": "<name>"}` in the request body, look up the bucket
via `get_bucket(model_type, universe)`, and dispatch to the existing
core training functions -- there is **no env-var-driven universe
selection** for forecasters, SAC, or ETL. ETL has its own sibling
registry at
[brain_api/brain_api/etl/universe_registry.py](brain_api/brain_api/etl/universe_registry.py)
keyed only on the universe string (no model dimension); the two
`/etl/*` job endpoints take `{"universe": "<name>"}` in the request body
and 422 on unknown values. This keeps two parallel SAC workflows
(e.g. `halal_filtered` and a future `halal`) from racing on a single
process-wide env var when each refreshes its own slate.

Adding a new bucket is one `_register(BucketConfig(...))` call plus a
sibling local-storage subclass and a new HF repo env var. Adding a new
ETL universe is one `_register(ETLUniverseConfig(...))` call. No other
endpoint, workflow, or test edits should be required.

## Agent workflow rules

Agents must produce **structured outputs** that can be stored and audited:

- Include citations/identifiers where possible (e.g., news URL, data source + timestamp)
- A `RiskCritic` (or equivalent) must be able to:
  - challenge contradictions
  - flag weak/insufficient evidence
  - downgrade confidence or veto a trade recommendation

Agents are used for **evidence synthesis**. Numeric optimization remains in deterministic code (feature engineering) + forecasters/RL.

### LLM summary (Temporal orchestrated)

The Monday email includes an **AI summary** generated by OpenAI/gpt-5-mini:

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
- `run_id = paper:YYYY-MM-DD` (default form for the original `sac` / `hrp` / `dhrp` strategies)
- `run_id = paper:<universe>:YYYY-MM-DD` is an **accepted variant** when a strategy uses a dedicated Alpaca account (currently only `sac_halal` -> `paper:halal:YYYY-MM-DD`). The variant exists so two strategies that share a Monday slot do not collide on `client_order_id` or experience-file paths. Only allowed when:
  1. The strategy submits orders through a **dedicated Alpaca account** (different `ALPACA_<ACCOUNT>_KEY` / `_SECRET`), so per-account `client_order_id` dedup is automatic, AND
  2. The variant prefix is exactly the strategy's `universe` string.
- `attempt` starts at `1`
- **Rerun is read-only** if the latest attempt has any order not in a terminal canceled/expired/rejected state
- To allow a new submission: user cancels paper orders manually in Alpaca, then rerun creates `attempt += 1`
- The `paper:` literal in `run_id` and `client_order_id` is a static audit-string prefix; it does NOT reflect the actual Alpaca host being used. To check whether a run hit live, inspect `ALPACA_{ACCOUNT}_URL` env at run time or the Alpaca dashboard.

### Order idempotency

All submitted orders must include deterministic `client_order_id`:

- `paper:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>` (default)
- `paper:<universe>:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>` (variant; only when the strategy uses a dedicated Alpaca account, e.g. `paper:halal:...` for `sac_halal`)

The system must:

- Check local DB for existing submissions before submitting
- Query Alpaca by `client_order_id` as a second guardrail

### Trading mode

- **Paper is the default.** With no Alpaca URL env vars set, every account uses `https://paper-api.alpaca.markets`.
- **Per-account live override.** Setting `ALPACA_{ACCOUNT}_URL=https://api.alpaca.markets` plus the matching live API key/secret flips that one account to live. Other accounts remain on paper.
- **Live mode runs without the full AGENTS.md safety stack.** The deferred safety gaps (limit orders, max turnover cap, max orders cap, DB pre-submit dedup, 48h sells-stuck auto-buy fallback) still apply on live runs. Treat live as a manual smoke-test capability until those gaps close. Do not register a live schedule in production.
- The `run_id` and `client_order_id` audit prefix stays `paper:` regardless of the actual broker host -- this is a known cosmetic mismatch in the smoke-test scope.

### Default execution choices

- Default order type: **limit**
- Default sizing: **fractional shares when supported**

### Model lifecycle

- **Monday runs are inference-only**. Never retrain inside the Monday inference run.
- **Training schedule**:

| When | What | Trigger |
|------|------|---------|
| Monthly (Saturday) | Full retrain all US models | Manual |
| Monthly (first Sunday 00:01 UTC) | US Forecasters training (LSTM then PatchTST, strictly serial because the host can only fit one trainer at a time) | Cron (Temporal, Mac training queue) |
| Monthly (first Sunday 06:01 UTC) | US SAC training on `halal_filtered` (consumes whatever PatchTST `current` pointer is live at trigger time; 6 h gap from forecasters slot) | Cron (Temporal, Mac training queue) |
| Monthly (first Sunday 12:01 UTC) | US SAC training on the legacy `halal` universe (parallel A/B sibling of `sac_halal_filtered`; 6 h gap from halal_filtered SAC) | Cron (Temporal, Mac training queue) |
| Monthly (first Sunday 18:01 UTC) | Full PatchTST retrain (India) | Cron (Temporal, Mac training queue) |
| Monday 6 PM IST | US inference only | Cron (Temporal, Pi inference queue) |

Training cadence cannot be expressed via cron "first Sunday of month" (Vixie cron OR's day-of-month with day-of-week). Schedules use `ScheduleCalendarSpec(day_of_month=[1..7], day_of_week=[0], hour=H, minute=M)` instead; see `temporal/schedules.py::first_sunday_of_month_at`. The four training slots are staggered 6 h apart so the single Mac trainer runs them serially (enforced by `TEMPORAL_MAX_CONCURRENT_ACTIVITIES=1` on the training worker).

- Training produces a **new versioned artifact**; inference loads from `current` pointer
- **Promotion requires guardrails**: new model must pass model-specific health checks (artifact integrity, finite metrics, SAC CAGR floor of 0.12, SAC symbol-count match against the bucket). Universe drift no longer suppresses healthy promotions; rollback is the recovery mechanism (separate story).
- **Rollback is always possible**: keep last known-good version; pointer swap is atomic
- **No HF cold-start fallback**: HuggingFace `make_current` is set to `promoted` only. A failed inaugural run leaves HF `main` empty (and inference 503s) by design -- AGENTS.md rule #1 forbids the silent "ship the broken inaugural" fallback.

## Operational requirements

Any implementation must include:

- **Idempotency**: safe reruns
- **Timeouts + retries** with exponential backoff for external APIs
- **Rate limit awareness** and batching
- **Observability**:
  - run-level logs with `run_id` + `attempt`
  - stage duration metrics (even if just logged)
  - clear error propagation back to Temporal

### Known limitations

- `/inference/sac` and `/models/active-symbols` accept a **mandatory**
  `universe` query parameter; both routes resolve the SAC bucket via
  `get_bucket(ModelType.SAC, universe)`. There is no implicit default
  -- callers must pass `halal_filtered` or `halal` explicitly so the
  two A/B paths cannot accidentally share state.
- `/experience/label/sac` routes each record to its Alpaca account via
  `brain_api.core.alpaca_client.resolve_alpaca_account(model_type,
  universe)` (mapping `halal_filtered` -> `sac`, `halal` -> `sac_halal`).
  New SAC writes MUST set `universe` on the experience record (the
  Temporal workflows do this); legacy records that pre-date the field
  fall back to inferring the universe from the run_id prefix
  (`paper:halal:...` -> `halal`, else `halal_filtered`). Per AGENTS.md
  rule #1, the labeller raises (rather than defaulting to a fallback
  account) for any unknown `(model_type, universe)` pair.

### Temporal workflow configuration

Key configuration:

- **Activity timeouts**: `start_to_close_timeout` on every activity (5 min for API calls, 10h for training)
- **Activity retries**: `RetryPolicy(maximum_attempts=N)` on activities that call external APIs
- **Heartbeating**: Long-running training activities use `heartbeat_timeout` to detect stalled workers
- **Resume from failure**: Automatic. Temporal replays from event history -- completed activities are skipped automatically. No cache policies needed.
- **Durable sleep**: `workflow.sleep()` survives worker crashes, laptop shutdowns, and restarts
- **Parallel execution**: `asyncio.gather()` for concurrent activity execution within workflows
- **Pydantic data converter**: `pydantic_data_converter` used for correct Pydantic v2 serialization
- **Sell-wait-buy**: Single workflow with a market-aware durable polling loop. After submitting sells, the helper fetches the Alpaca market clock once (`GET /alpaca/clock`); if the market is closed it sleeps until exactly the advertised `next_open` (no lead-time fudge), then polls `check_order_statuses` every `POLL_INTERVAL = 1 min` until all sells reach a terminal status or the 48h `SELL_DEADLINE` is hit. Replaces the legacy flat 15-min poll cadence.
- **Task queue routing** (role-based, not host-based): two queues -- `learnfinance-inference` for weekly allocation / HRP workflows, `learnfinance-training` for monthly training workflows. Each worker subscribes to exactly one queue via `TEMPORAL_TASK_QUEUE` env. Activities inherit the workflow's task queue by default, so ETL activities inside training workflows (e.g. `run_sentiment_gap_fill`) automatically land on the training worker.
- **Activity concurrency cap** (per worker): `TEMPORAL_MAX_CONCURRENT_ACTIVITIES` env (default `10`) drives BOTH `Worker(max_concurrent_activities=N)` and the `ThreadPoolExecutor(max_workers=N)`. The Mac training worker sets this to `1` so heavy training activities are serialized; Pi inference keeps the default `10` so fast allocation activities run in parallel.

**Host topology** (current production deployment):
- **Pi** (`docker compose up`): runs Temporal server, brain_api, and one worker subscribed to `learnfinance-inference`. `temporal-schedules-init` is the one-shot container that registers all schedules on the server. **brain-api `data/`** is bind-mounted to `${BRAIN_DATA_DIR:-/home/razin/learnfinance/brain-data}` -> `/app/data` (create the host dir once; `chmod 777` if the container user cannot write). Set **`STORAGE_BACKEND=hf_first`** in the Pi's `brain_api/.env` with `HF_TOKEN` and all required `HF_*_MODEL_REPO` vars so models lazy-download from Hugging Face after training on the Mac—rebuild the image for code changes only, not for model promotion.
- **Mac** (manual, via `devbox`): user starts brain_api plus one or two workers as needed:
  - `devbox run temporal:worker:training` -- subscribes to `learnfinance-training`, concurrency cap 1. Required for the monthly training slots to actually execute.
  - `devbox run temporal:worker:inference` -- subscribes to `learnfinance-inference` as a redundant/faster backup to the Pi worker. Optional.
- Workers connect outbound to the Pi's Temporal server at `TEMPORAL_ADDRESS=<pi-host>:7233` (LAN or Tailscale). No inbound port needed on the Mac.

**Single-host dev setup** (no Pi, everything on one laptop):
1. `devbox run temporal:server` -- Temporal dev server with SQLite persistence + UI at port 8233
2. `devbox run brain:run` -- brain_api FastAPI service
3. `devbox run temporal:worker:inference` and/or `devbox run temporal:worker:training` in separate terminals

**Schedule registration** (run once): `devbox run temporal:schedule` -- registers all 9 schedules on whatever server `TEMPORAL_ADDRESS` points at. Idempotent; never updates existing schedules (delete on the server first, then re-register, if you need to change one).

## Change safety checklists

Before merging changes that touch trading logic:

- [ ] Confirm rerun behavior is still read-only after any submission
- [ ] Confirm `client_order_id` format is the default (`paper:YYYY-MM-DD:...`) or a documented variant (`paper:<universe>:YYYY-MM-DD:...`) tied to a dedicated Alpaca account
- [ ] If introducing a new strategy that shares a Monday slot with an existing one, the new strategy MUST use a dedicated Alpaca account AND the `paper:<universe>:YYYY-MM-DD` run_id variant so `client_order_id`s and experience-file paths stay disjoint
- [ ] Confirm safety caps exist and are enforced (max turnover, max orders, cash buffer)
- [ ] If touching live trading: confirm that paper accounts (any with no `ALPACA_{ACCOUNT}_URL` set) still hit the paper host

Before merging changes that touch ML/model code:

- [ ] Confirm Monday inference does NOT trigger training
- [ ] Confirm training writes new versioned artifact (not overwrite)
- [ ] Confirm promotion uses the per-model guardrail check (forecaster: `evaluate_forecaster_artifact_health`; SAC full: `evaluate_sac_artifact_health`) and that HF `make_current = promoted` with no cold-start fallback
- [ ] Confirm `failure_reasons` is threaded end-to-end (metadata.json -> training response -> Temporal workflow return -> Jinja prompt + email template)
- [ ] Confirm endpoints remain stateless (no global model cache)
- [ ] Confirm storage abstraction is used (not hardcoded paths)
- [ ] Confirm LSTM remains pure-price (no signals in input)
- [ ] Confirm PatchTST/SAC receive correct signal state vector
- [ ] Confirm India PatchTST uses `patchtst_india` storage (not US `patchtst`)
- [ ] Confirm India symbols retain `.NS` suffix throughout the pipeline (including `screening_history.stock` rows and `evicted_from_previous` keys for the `halal_india_filtered_alpha` partition)
- [ ] Confirm sticky carry-set isolation: no two strategies share a `partition` string in `brain_api/core/strategy_partitions.py` (uniqueness across `stage1_weight_history` AND `screening_history`)
- [ ] Confirm India universe builders (monthly `halal_india`) write to `screening_history` via `ScreeningHistoryRepository`, NOT `stage1_weight_history` -- the weekly India Alpha-HRP partition (`halal_india_alpha`) is the only India strategy that uses the two-stage table
- [ ] Confirm SAC `n_stocks` is resolved from the bucket symbol count via `make_sac_config_for_n_stocks(DEFAULT_SAC_CONFIG, len(symbols))` and NOT from a process-wide config (the parallel `sac_halal_filtered` / `sac_halal` workflows share a process and must each pick their own action dim)
- [ ] Confirm `sac_halal_filtered` and `sac_halal` never share a `current` pointer, on-disk path, or HF repo (`HF_SAC_HALAL_FILTERED_MODEL_REPO` vs `HF_SAC_HALAL_MODEL_REPO`)

## AI assistant behavioral rules

1. **Never add silent fallbacks without asking first.** Fallbacks mask real bugs and break correctness. For example, falling back to momentum when a snapshot fails to load means the system silently produces garbage instead of surfacing the error. Always raise exceptions for unexpected failures; ask the user before adding any degraded-mode fallback.

2. **Math correctness is the highest priority -- never break math to simplify code.** DRY, DDD, and clean code matter and you should factor out genuinely shared logic. The rule is about precedence, not duplication: when two algorithms have research-driven mathematical differences (even subtle ones), each must keep its own math even if the surface code looks similar. Concrete cautionary tale from this repo: PPO and SAC each have algorithm-specific mathematical steps; we once "reused" code between them for DRY and silently broke PPO's math. If the math is provably identical (e.g., a standard formula like Sharpe ratio, a generic covariance estimator, a shared data loader), share it; if there is any research-level difference, keep the implementations separate even if the code looks alike. When in doubt, ask before merging two model-specific code paths.

3. **Never do `-n` / `--no-verify` — pre-commit hooks MUST run; fix ruff and test failures instead of bypassing.** Commits that bypass verification hide real failures and break the invariant that `main` is always green.

## AI assistant planning rules

When operating in **plan mode**, the AI assistant must:

1. Always include these two final TODOs at the end of every plan:
   - [ ] Fix all ruff linting issues (related and unrelated to the change)
   - [ ] Run and fix all tests (related and unrelated to the change)

2. These cleanup tasks ensure the codebase stays healthy with every change.
