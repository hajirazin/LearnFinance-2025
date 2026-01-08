# SAC Training Plan

## Overview

Soft Actor-Critic (SAC) is an off-policy, model-free RL algorithm for continuous action spaces. Key advantages over PPO:

- **Sample efficiency**: Off-policy learning with experience replay
- **Stable learning**: Twin Q-critics mitigate overestimation bias
- **Automatic exploration**: Entropy maximization balances exploration/exploitation

## Architecture

### SAC Variants

| Variant | Forecast Source | HF Repo Env Var |
|---------|----------------|-----------------|
| SAC + LSTM | LSTM weekly return predictions | `HF_SAC_LSTM_MODEL_REPO` |
| SAC + PatchTST | PatchTST weekly return predictions | `HF_SAC_PATCHTST_MODEL_REPO` |

### Network Architecture

```
GaussianActor:
  ├── Feature MLP: state_dim → 64 → 64
  ├── Mean Head: 64 → action_dim
  └── LogStd Head: 64 → action_dim (clamped [-20, 2])

TwinCritic:
  ├── Q1: (state_dim + action_dim) → 64 → 64 → 1
  └── Q2: (state_dim + action_dim) → 64 → 64 → 1
```

### State Vector (same as PPO for comparability)

```
state = [
    signals (n_stocks × 7): news_sentiment, fundamentals, fundamental_age
    forecasts (n_stocks × 1): LSTM or PatchTST predicted weekly return
    weights (n_stocks + 1): current portfolio weights including CASH
]
```

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_sizes` | (64, 64) | Smaller networks for limited data (~500 weekly transitions) |
| `actor_lr` | 3e-4 | Standard SAC default |
| `critic_lr` | 3e-4 | Standard SAC default |
| `alpha_lr` | 3e-4 | Auto-entropy tuning rate |
| `tau` | 0.005 | Polyak averaging for target networks |
| `gamma` | 0.99 | Discount factor (weekly horizon) |
| `init_alpha` | 0.2 | Initial entropy coefficient |
| `batch_size` | 64 | Small batches for limited data |
| `buffer_size` | 10,000 | Replay buffer capacity |
| `warmup_steps` | 100 | Random actions before training starts |
| `weight_decay` | 1e-4 | L2 regularization for overfitting prevention |

## Constraints (same as PPO)

| Constraint | Value | Enforcement |
|------------|-------|-------------|
| Cash buffer | 2% | Post-softmax clipping |
| Max position | 20% | Post-softmax capping |
| Transaction cost | 10 bps | Subtracted from return in reward |

## Training Data

### Historical Data Sources

1. **Prices**: Yahoo Finance via `load_prices_yfinance()`
2. **News Sentiment**: `data/output/daily_sentiment.parquet` (from Alpaca News API + FinBERT)
3. **Fundamentals**: `data/raw/fundamentals/` (cached Alpha Vantage JSONs)
4. **Forecasts**: Walk-forward LSTM/PatchTST predictions (no look-ahead bias)

### Walk-Forward Forecast Generation

To avoid look-ahead bias, forecast features are generated using a walk-forward approach:

1. For each evaluation year, train LSTM/PatchTST on prior 5 years of data
2. Generate predictions for the evaluation year
3. Use these predictions as the "forecast" feature for SAC training

This ensures SAC never sees "future" forecast information during training.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train/sac_lstm/full` | POST | Full training from scratch |
| `/train/sac_lstm/finetune` | POST | Fine-tune on recent 26-week data |
| `/train/sac_patchtst/full` | POST | Full training from scratch |
| `/train/sac_patchtst/finetune` | POST | Fine-tune on recent 26-week data |
| `/inference/sac_lstm` | POST | Get target portfolio weights |
| `/inference/sac_patchtst` | POST | Get target portfolio weights |

## Promotion Logic

CAGR-first promotion gating:

1. **First model**: Auto-promoted
2. **Subsequent models**: Promoted only if `eval_cagr > prior_cagr`

Evaluation uses expanding-window walk-forward validation (last 2 years by default).

## Storage

Artifacts stored under `data/models/sac_{lstm,patchtst}/{version}/`:

```
{version}/
├── actor.pt           # Actor network weights
├── critic.pt          # Twin critic weights
├── critic_target.pt   # Target critic weights
├── log_alpha.pt       # Entropy coefficient
├── scaler.pkl         # PortfolioScaler for state normalization
├── config.json        # SAC hyperparameters
├── symbol_order.json  # Ordered symbol list
└── metadata.json      # Training info, metrics, data window
```

Current version pointer: `data/models/sac_{lstm,patchtst}/current`

## Comparison with PPO

| Aspect | PPO | SAC |
|--------|-----|-----|
| Policy type | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Exploration | Action space clipping | Entropy maximization |
| Networks | Actor-Critic (shared) | Actor + Twin Critics |
| Target networks | No | Yes (Polyak averaging) |
| Replay buffer | Rolling (on-policy) | Persistent (off-policy) |

## Files Changed/Added

### Core Modules

- `brain_api/core/portfolio_rl/sac_config.py` - SAC configuration
- `brain_api/core/portfolio_rl/sac_buffer.py` - Replay buffer
- `brain_api/core/portfolio_rl/sac_networks.py` - Actor, TwinCritic
- `brain_api/core/portfolio_rl/sac_trainer.py` - Training loop
- `brain_api/core/sac_lstm/` - SAC + LSTM variant
- `brain_api/core/sac_patchtst/` - SAC + PatchTST variant

### Storage

- `brain_api/storage/sac_lstm/` - Local/HF storage for SAC LSTM
- `brain_api/storage/sac_patchtst/` - Local/HF storage for SAC PatchTST

### Routes

- `brain_api/routes/training.py` - Added 4 training endpoints
- `brain_api/routes/inference.py` - Added 2 inference endpoints

### Config

- `brain_api/core/config.py` - Added `HF_SAC_*_MODEL_REPO` env vars

### Tests

- `tests/test_sac_lstm.py` - SAC LSTM endpoint tests
- `tests/test_sac_patchtst.py` - SAC PatchTST endpoint tests

