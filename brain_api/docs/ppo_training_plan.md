# PPO + LSTM/PatchTST Training Plan (Portfolio RL Allocators)

> **Overview**: Add a shared, realistic weekly portfolio-rebalancing RL environment and **two PPO allocator variants**: `ppo_lstm` and `ppo_patchtst`. Both share env/reward/constraints/eval/promotion logic; they differ only in which forecaster output feature is injected into the PPO state (LSTM vs PatchTST). Includes evaluation gates vs baselines and safeguards learned from the last commits (no horizon mismatch, no memory blow-ups).

---

## Implementation Todos

| ID | Task | Status | Dependencies |
|----|------|--------|--------------|
| ppo-shared-core | Implement shared weekly portfolio RL core (env, reward, constraints, eval helpers) | ✅ done | — |
| ppo-state-schema | Define PPO state vector schema with injectable forecast feature | ✅ done | ppo-shared-core |
| ppo-lstm-variant | Implement ppo_lstm full training + inference (uses LSTM predicted weekly return) | ✅ done | ppo-shared-core, ppo-state-schema |
| ppo-lstm-finetune | Implement ppo_lstm fine-tuning (26-week rolling buffer) | ✅ done | ppo-lstm-variant |
| ppo-patchtst-variant | Implement ppo_patchtst full training + inference (uses PatchTST predicted weekly return) | ✅ done | ppo-shared-core, ppo-state-schema |
| ppo-patchtst-finetune | Implement ppo_patchtst fine-tuning (26-week rolling buffer) | ✅ done | ppo-patchtst-variant |
| ppo-eval-gate | Add evaluation vs HRP/equal-weight + prior PPO (first-ever auto-promotes) | ✅ done | ppo-lstm-variant, ppo-patchtst-variant |
| ppo-experience-labeling | Add Sunday job or endpoint to label experience tuples with realized reward | pending | ppo-shared-core |
| ppo-api-tests-lstm | Router-level API tests for ppo_lstm endpoints (full + finetune) | ✅ done | ppo-lstm-variant, ppo-lstm-finetune |
| ppo-api-tests-patchtst | Router-level API tests for ppo_patchtst endpoints (full + finetune) | ✅ done | ppo-patchtst-variant, ppo-patchtst-finetune |
| docs-update-top15 | Update README.md and CLAUDE.md to change Top-30 to Top-15 | pending | — |

---

## Key Expert Choices (and Why)

- **Decision interval**: **weekly portfolio rebalancing** (one action per week) as the default.
- **Why (expert reasoning)**: markets are noisy at high frequency; without market-impact modeling + tight execution, shorter intervals often devolve into overtrading where costs dominate. Weekly also reduces non-stationarity pressure and makes PPO markedly more stable (fewer steps, lower variance advantages). Daily can be explored later as an ablation once weekly is solid.
- **What PPO learns**: **allocation weights**, not next-price/return.
- **Action**: **long-only** weights on the simplex (plus optional `CASH` weight).
- **Explicit constraints (non-negotiable)**:
  - No shorting
  - No leverage / margin
  - No options / derivatives
  - No live trading (paper-only; halal universe only)
  - One rebalance decision per week (single delivery call)
  - **Cash buffer**: `CASH >= 2%` at all times
  - **Max position size**: `max_position_weight = 0.20` (20%) per stock
- **Reward**: realized **portfolio log-return** over the next week **minus explicit transaction costs** (and optional drawdown/vol penalty).
- **State inputs**: use **actual market data + signals**, with **LSTM output as one feature**, not as the reward driver.
- **Reward is always computed from realized prices/returns**, otherwise you train on your own model's errors.

---

## Design Decisions from Expert Review

### 1) Episode Structure for Training

- **One episode = one contiguous year** of weekly steps.
- During training, randomly sample start years across the training window to improve generalization.
- Episode terminates when: (a) end of year reached, or (b) end of data reached.
- No "bankruptcy" termination — we enforce constraints (cash buffer, max position) that prevent catastrophic loss.
- **Initial portfolio at episode start**: `CASH = 1.0` (all cash, no positions).

### 2) Fixed Universe Size (Top-15 + Holdings)

PPO requires fixed observation/action dimensions. The halal universe changes over time.

**Solution for v1**: use a **fixed subset** of **Top-15** stocks by liquidity (plus current holdings) for both training and inference. This:

- Keeps action/observation dimensions fixed
- Matches the screening strategy in the repo
- Is simple to implement

**Note**: README.md and CLAUDE.md will be updated to change "Top-30" to "Top-15".

### 3) Action Space Mechanics

- **Continuous action space**: policy outputs raw logits for each asset (15 stocks + CASH = 16 outputs).
- **Softmax** applied to enforce simplex (weights sum to 1, all ≥ 0).
- **Post-processing** enforces:
  - `max_position_weight = 0.20` (clip and renormalize)
  - `CASH >= 0.02` (enforce cash buffer, renormalize)

### 4) State/Feature Normalization

- Fit a `StandardScaler` on training data features (like LSTM/PatchTST).
- Store the fitted scaler with the policy artifact.
- Apply the same scaler at inference time.

### 5) Reward Scaling

- Raw log returns are tiny (~±0.01/week). Transaction costs also tiny.
- PPO learns better with larger reward magnitudes.
- **Scale reward by 100**: so 1% weekly return → reward of 1.0.
- Document this clearly in code.

### 6) PPO Hyperparameters (PPOConfig)

Add a `PPOConfig` dataclass with sensible defaults (based on SB3 recommendations):

```python
@dataclass
class PPOConfig:
    # Policy network
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    
    # PPO algorithm
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99  # discount factor
    
    # Training
    n_epochs: int = 10  # epochs per rollout batch
    batch_size: int = 64
    rollout_steps: int = 52  # ~1 year of weekly steps per rollout
    total_timesteps: int = 10_000  # total training steps
    
    # Environment
    cost_bps: int = 10  # transaction cost in basis points
    cash_buffer: float = 0.02  # minimum cash weight
    max_position_weight: float = 0.20  # max weight per stock
    reward_scale: float = 100.0  # multiply returns by this
    
    # Reproducibility
    seed: int = 42
```

### 7) Walk-Forward Evaluation Details

- **Split strategy**: expanding window.
  - Train on years 1–N, test on year N+1.
  - Repeat for multiple folds (e.g., train on 2012–2020, test 2021; train 2012–2021, test 2022; etc.).
- **Metrics computed**:
  - **Sharpe ratio** (annualized, after costs) — primary metric for promotion
  - CAGR (compound annual growth rate)
  - Max drawdown
  - Turnover (average weekly)
  - Hit rate (% of weeks with positive return)
- **Promotion rule**: new model must beat **HRP baseline** and **equal-weight baseline** and **prior PPO** on **Sharpe after costs**.

### 8) Experience Buffer Schema

Each experience record stored in `data/experience/<run_id>.json`:

```json
{
  "run_id": "paper:2026-01-12",
  "week_start": "2026-01-12",
  "week_end": "2026-01-16",
  "state": {
    "current_weights": {"AAPL": 0.10, "MSFT": 0.08, "CASH": 0.82},
    "signals": { ... },
    "forecast_feature": 0.015
  },
  "action": {"AAPL": 0.12, "MSFT": 0.10, "CASH": 0.78},
  "turnover": 0.04,
  "reward": null,
  "realized_return": null,
  "next_state": null,
  "labeled_at": null
}
```

Fields `reward`, `realized_return`, `next_state`, `labeled_at` are **null at decision time** and filled in by the reward-labeling job after the week ends.

### 9) Reward Labeling Pipeline

A **Sunday job** (or on-demand endpoint `POST /experience/label`) that:

1. Reads experience records where `reward` is null and `week_end < today`.
2. Fetches realized weekly returns for each symbol.
3. Computes portfolio return = weighted sum of symbol returns.
4. Computes transaction cost = `cost_bps/10000 * turnover`.
5. Computes `reward = (portfolio_return - transaction_cost) * reward_scale`.
6. Updates the experience record with `reward`, `realized_return`, `next_state`, `labeled_at`.

### 10) Max Position Size

- **Default**: `max_position_weight = 0.20` (20% of portfolio per stock).
- Configurable via `PPOConfig`.
- Enforced via post-processing after softmax (clip + renormalize).

### 11) Execution Timing

- **Monday open** execution model.
- PPO receives state computed from data available up to **Friday close**.
- Target weights are executed at **Monday open prices** (simulated).
- This matches the repo's Monday 6 PM IST run (before US market opens).

### 12) Survivorship Bias (Known Limitation)

- For v1, we use **today's halal universe** to train on historical data.
- This introduces survivorship bias (we only train on stocks that "survived" to today).
- **Accepted for learning purposes**; documented as a known limitation.
- Future improvement: reconstruct point-in-time universe membership.

---

## Answers to Your Questions

### Should we worry about fundamentals age like PatchTST?

**Yes**. Your PatchTST pipeline already forward-fills quarterly fundamentals and adds a `fundamental_age` staleness feature (days since last update normalized by 90). For RL, it's even more important: PPO will otherwise treat forward-filled fundamentals as "fresh" and can overfit to stale values.

### Do the last 5 commits matter for PPO?

**Yes, directly.** The last 5 commits highlight three RL-relevant pitfalls:

- **Horizon mismatch** (PatchTST weekly target vs daily dynamics): for PPO we must keep the environment step, reward horizon, and signals aligned (weekly step → weekly reward).
- **Staleness features** (`fundamental_age`): reuse the same idea in PPO state.
- **Memory/perf** (OOM, vectorization, batching): PPO training will also need batched rollouts, careful logging, and vectorized feature building.

### Should we train RL on actual prices or on LSTM output?

- **Train RL reward on actual prices/returns** (realized portfolio PnL).
- **Optionally include LSTM predicted weekly return as a feature** (an "alpha"). If the LSTM is weak, PPO can learn to downweight it.

### LSTM not beating baseline: normal or bug?

- **It can be normal** for a pure-price model to not beat a strong baseline on **MSE of returns** (returns are close to 0 mean and very noisy).
- **But it can also be a mismatch of objective**: a model can have mediocre MSE yet still be useful for ranking or sign prediction (economic utility ≠ MSE).
- **What we'll do**: add an evaluation panel for LSTM that reports both prediction metrics (MSE, sign hit-rate) and trading utility metrics (top-K long-only simulated portfolio vs equal-weight/HRP) to determine if you're "doing something wrong" vs "market is hard."

---

## Implementation Outline

### 1) Shared Portfolio RL Core (Weekly Step)

Create shared RL core module used by both variants in `brain_api/brain_api/core/portfolio_rl/`:

- `env.py`: weekly portfolio environment (long-only simplex + optional CASH).
- `state.py` / `features.py`: builds state vectors per week (prices → returns + existing signals + `fundamental_age` + injected forecast feature).
- `rewards.py`: log-return minus turnover/slippage penalty, scaled by 100.
- `constraints.py`: max position size (20%), cash buffer (2%), halal-only universe guardrails.
- `config.py`: `PPOConfig` dataclass with all hyperparameters.
- `scaler.py`: feature normalization (fit on training, apply at inference).

#### Portfolio State is an Explicit Input (Required)

- The PPO state must include **current portfolio state**, otherwise the agent cannot reason about turnover/costs and will learn unrealistic "flip portfolios weekly" behavior.
- Include at minimum:
  - `current_weights` (including `CASH`)
  - `cash_value` (or cash weight)
  - optional: `positions_count`, `portfolio_value`, last-week turnover

#### Transaction Costs are Explicit (Required)

- Costs will be included in the environment step and in the reward.
- Default cost model (simple, robust):
  - **proportional cost per turnover**: `cost_bps * turnover` (bps = basis points)
  - turnover defined as 0.5 * sum(|w_t - w_{t-1}|)
- Default parameter choice: start with **`cost_bps = 10`** (0.10% per 1.0 turnover).
- Rationale: for liquid US equities, explicit commissions are often $0, but **spread + slippage/market impact** dominates; 5–20 bps is a common backtest range depending on liquidity and execution. We start at 10 bps to discourage pathological churn without killing signal.
- Keep `cost_bps` configurable and tune it later using observed Alpaca paper fills.
- Optional extensions later (kept out of v1 unless needed): spread vs slippage decomposition, liquidity caps, per-order fees.

### 2) PPO Model + Training Loop (Shared; Forecast Injected)

- Use a battle-tested PPO implementation (e.g., SB3-style) with **recurrent policy optional**.
- Start **non-recurrent PPO** because state already includes a 60-day context via features (and recurrent PPO is more fragile).
- Add LSTM policy later if needed.
- Policy network: 2-layer MLP (64, 64) with tanh activations (configurable via `PPOConfig`).

### 3) Training Endpoints: Full vs Fine-tune

Both PPO variants have **two training modes** (4 endpoints total):

| Endpoint | Mode | Schedule | Data Window | Timesteps | Learning Rate |
|----------|------|----------|-------------|-----------|---------------|
| `POST /train/ppo_lstm/full` | Full | Monthly (manual) | 10+ years | 10,000 | 3e-4 |
| `POST /train/ppo_lstm/finetune` | Fine-tune | Weekly (Sunday cron) | 26 weeks | 2,000 | 1e-4 |
| `POST /train/ppo_patchtst/full` | Full | Monthly (manual) | 10+ years | 10,000 | 3e-4 |
| `POST /train/ppo_patchtst/finetune` | Fine-tune | Weekly (Sunday cron) | 26 weeks | 2,000 | 1e-4 |

#### Full Training
- Starts from **random weights** (no prior model required)
- Uses **full historical data** (10+ years)
- Fits a **new scaler** on training states
- First model auto-promotes; subsequent models must beat baselines + prior

#### Fine-tuning
- **Requires a prior promoted model** (returns 400 if none exists)
- Loads **prior model weights** and continues training
- Uses **26-week rolling buffer** (recent data only)
- **Reuses prior scaler** (no re-fitting)
- Lower learning rate (1e-4 vs 3e-4) to avoid catastrophic forgetting
- Fewer timesteps (2,000 vs 10,000) for faster weekly runs
- Version string includes `-ft` suffix (e.g., `v2025-01-08-abc123-ft`)

#### Fine-tune Configuration (`PPOFinetuneConfig`)

```python
@dataclass
class PPOFinetuneConfig:
    lookback_weeks: int = 26       # 6-month rolling buffer
    total_timesteps: int = 2_000   # Much smaller than full training
    learning_rate: float = 1e-4    # Lower LR for fine-tuning
```

#### HuggingFace Hub Upload

All 4 PPO training endpoints support optional HuggingFace Hub upload, controlled by environment variables:

| Environment Variable | Model Type | Description |
|---------------------|------------|-------------|
| `HF_PPO_LSTM_MODEL_REPO` | PPO + LSTM | HuggingFace repo for PPO LSTM allocator |
| `HF_PPO_PATCHTST_MODEL_REPO` | PPO + PatchTST | HuggingFace repo for PPO PatchTST allocator |
| `STORAGE_BACKEND` | All | Set to `"hf"` to enable upload |
| `HF_TOKEN` | All | HuggingFace API token |

Example configuration:
```bash
export STORAGE_BACKEND="hf"
export HF_TOKEN="hf_xxxxxxxxxxxx"
export HF_PPO_LSTM_MODEL_REPO="username/learnfinance-ppo-lstm"
export HF_PPO_PATCHTST_MODEL_REPO="username/learnfinance-ppo-patchtst"
```

Response includes `hf_repo` and `hf_url` when upload is enabled and successful.

#### Experience Buffer (for future reward labeling)

- Persist weekly transitions to `data/experience/<run_id>.json` as described in `CLAUDE.md`.
- Both full and fine-tune training can optionally consume labeled experience for offline RL.

### 4) Inference Endpoint

- Implement `POST /inference/ppo_lstm` that:
  - loads current PPO policy artifact,
  - consumes the same state vector schema,
  - outputs target weights + turnover.

### 4b) Parallel Implementation for PPO + PatchTST

- We will implement parallel endpoints for `ppo_patchtst` in the same rollout.
- The **only intended difference** is the state feature that represents the forecaster output:
  - `ppo_lstm` uses LSTM predicted weekly return as the forecast feature
  - `ppo_patchtst` uses PatchTST predicted weekly return as the forecast feature
- The environment (action constraints, costs, reward definition) stays the same so results are comparable.

### 5) Evaluation + Promotion Gate

- Promotion semantics match LSTM/PatchTST:
  - **First-ever PPO model**: promote automatically (so inference has a usable `current`).
  - **Second onward**: promote only if it beats:
    - **HRP allocation** baseline,
    - **equal-weight** baseline,
    - and the **prior PPO** version.
  - Evaluate on walk-forward splits (train on past, validate on later), include costs.
  - Primary metric: **Sharpe ratio after costs**.

---

## Files Changed/Added

### Core Modules (✅ Implemented)

- `brain_api/core/portfolio_rl/` (shared core)
  - `__init__.py`, `config.py`, `constraints.py`, `env.py`, `eval.py`, `rewards.py`, `scaler.py`, `state.py`
- `brain_api/core/ppo_lstm/` (LSTM variant)
  - `__init__.py`, `config.py`, `inference.py`, `model.py`, `training.py`, `version.py`
  - Training exports: `train_ppo_lstm`, `finetune_ppo_lstm`, `PPOFinetuneConfig`
- `brain_api/core/ppo_patchtst/` (PatchTST variant)
  - `__init__.py`, `config.py`, `inference.py`, `training.py`, `version.py`
  - Training exports: `train_ppo_patchtst`, `finetune_ppo_patchtst`

### Storage Modules (✅ Implemented)

- `brain_api/storage/ppo_lstm/local.py` — `PPOLSTMLocalStorage`, `PPOLSTMArtifacts`
- `brain_api/storage/ppo_patchtst/local.py` — `PPOPatchTSTLocalStorage`, `PPOPatchTSTArtifacts`

### Routes (✅ Implemented)

- `brain_api/routes/training.py` — Added 4 endpoints:
  - `POST /train/ppo_lstm/full`
  - `POST /train/ppo_lstm/finetune`
  - `POST /train/ppo_patchtst/full`
  - `POST /train/ppo_patchtst/finetune`
- `brain_api/routes/inference.py` — Added 2 endpoints:
  - `POST /inference/ppo_lstm`
  - `POST /inference/ppo_patchtst`

### Tests (✅ Implemented)

- `brain_api/tests/test_ppo_lstm.py` — 16 tests (training, inference, fine-tuning)
- `brain_api/tests/test_ppo_patchtst.py` — 16 tests (training, inference, fine-tuning)

### Still Pending

- `brain_api/routes/experience.py` — Add `/experience/label` endpoint for reward labeling
- Update `README.md` (change Top-30 to Top-15)
- Update `CLAUDE.md` (change Top-30 to Top-15)

---

## Naming + Dedup Strategy (Explicit)

- **Endpoint naming**: `ppo_<forecaster>` where `<forecaster>` is `lstm` or `patchtst`.
  - This prevents ambiguity and makes A/B comparisons clean.
- **Shared code**: everything except the forecast provider lives in `core/portfolio_rl/`.
- **Forecast provider injection**:
  - `ppo_lstm` and `ppo_patchtst` call a shared state builder but supply a different `forecast_feature` source.
- **Artifact naming**: store under `data/models/ppo_lstm/<version>/...` and `data/models/ppo_patchtst/<version>/...` so version promotion/rollback is independent.
- **Test dedup**: share fixtures/helpers for creating requests and stubbing signal loaders, but keep separate test files per endpoint family so failures are localized.

---

## API Contract Decision (Portfolio Input)

- **Inference requests use Option B (raw Alpaca-like portfolio snapshot)**:
  - Caller provides `cash_value` and `positions[]` with `symbol` + `market_value`.
  - API normalizes into `current_weights` (including `CASH`) before passing to the PPO policy.
- **Training endpoints** do not accept live portfolio state; they simulate portfolio evolution inside the environment.
- **Fine-tune endpoints** read portfolio state from the persisted experience buffer (which stores the normalized portfolio state captured at decision time).

---

## Notes on Research Results

I attempted targeted web searches for canonical references (FinRL, classic portfolio-RL papers, SB3 RecurrentPPO docs). The tool results I'm getting back are largely high-level summaries and a few arXiv links, not clean citations. I'll still implement the plan using widely accepted best practices (realistic costs, walk-forward eval, long-only simplex actions, reward=log portfolio return), and we'll validate via empirical backtests in your setup.

