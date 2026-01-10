# Training Endpoints

## Overview

The training endpoints train and fine-tune ML models. Training is idempotent (same inputs = cached result) and supports automatic promotion based on evaluation metrics.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/train/lstm` | Train LSTM forecaster |
| POST | `/train/patchtst` | Train PatchTST forecaster |
| POST | `/train/ppo_lstm/full` | Full PPO+LSTM training |
| POST | `/train/ppo_lstm/finetune` | Fine-tune PPO+LSTM |
| POST | `/train/ppo_patchtst/full` | Full PPO+PatchTST training |
| POST | `/train/ppo_patchtst/finetune` | Fine-tune PPO+PatchTST |
| POST | `/train/sac_lstm/full` | Full SAC+LSTM training |
| POST | `/train/sac_lstm/finetune` | Fine-tune SAC+LSTM |
| POST | `/train/sac_patchtst/full` | Full SAC+PatchTST training |
| POST | `/train/sac_patchtst/finetune` | Fine-tune SAC+PatchTST |

---

## Forecaster Training

### POST /train/lstm

**Train LSTM Weekly Return Forecaster**

Trains a shared LSTM model on historical price data for the halal universe.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /train/lstm] --> B[Resolve Training Window<br/>from config]
    B --> C[Get Halal Symbols]
    C --> D[Compute Version Hash]
    
    D --> E{Version Exists?}
    E -->|Yes| F[Return Cached Result<br/>Idempotent]
    E -->|No| G[Load Price Data<br/>yfinance]
    
    G --> H[Build Dataset<br/>Weekly Returns]
    H --> I[Train PyTorch Model]
    I --> J[Evaluate on Val Set]
    
    J --> K{Better than Prior?}
    K -->|Yes| L[Promote to Current]
    K -->|No| M[Keep Prior as Current]
    K -->|First Model| L
    
    L & M --> N[Save Artifacts]
    N --> O{HF Backend?}
    O -->|Yes| P[Upload to HuggingFace]
    O -->|No| Q[Local Only]
    
    P & Q --> R[Save Snapshots<br/>for Walk-Forward]
    R --> S[Return Result]
```

### Training Pipeline Details

```mermaid
flowchart LR
    subgraph "Data Preparation"
        A[Raw Prices] --> B[Weekly Returns<br/>Mon-Fri]
        B --> C[Feature Sequences<br/>60 days each]
        C --> D[Train/Val Split<br/>80/20]
    end
    
    subgraph "Training"
        D --> E[Fit StandardScaler]
        E --> F[Train LSTM<br/>MSE Loss]
        F --> G[Early Stopping]
    end
    
    subgraph "Evaluation"
        G --> H[Val Loss]
        H --> I[Compare to Baseline<br/>Predict Zero]
    end
```

### Request Schema

```json
{
  "skip_snapshot": false
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_snapshot` | bool | false | Skip saving historical snapshots |

### Response Schema

```json
{
  "version": "v2026.01.05-abc123",
  "data_window_start": "2011-01-01",
  "data_window_end": "2026-01-05",
  "metrics": {
    "train_loss": 0.000234,
    "val_loss": 0.000312,
    "baseline_loss": 0.000456
  },
  "promoted": true,
  "prior_version": "v2025.12.29-xyz789",
  "hf_repo": "user/lstm-model",
  "hf_url": "https://huggingface.co/user/lstm-model/tree/v2026.01.05-abc123"
}
```

---

### POST /train/patchtst

**Train PatchTST Multi-Signal Forecaster**

Trains PatchTST with OHLCV + news sentiment + fundamentals.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /train/patchtst] --> B[Resolve Training Window]
    B --> C[Get Halal Symbols]
    
    subgraph "Data Loading"
        C --> D[Load Prices<br/>yfinance]
        C --> E[Load News Sentiment<br/>Parquet]
        C --> F[Load Fundamentals<br/>Cache/Alpha Vantage]
    end
    
    D & E & F --> G[Align Multivariate Data<br/>11 Channels]
    G --> H[Build Dataset]
    H --> I[Train PatchTST]
    I --> J[Evaluate & Promote]
    J --> K[Save Artifacts + Snapshots]
```

### PatchTST Model

```mermaid
flowchart LR
    subgraph "Input"
        A[11 × 60<br/>channels × time]
    end
    
    subgraph "PatchTST"
        B[Patch Embedding<br/>patch_len=8]
        C[Transformer Encoder<br/>4 layers]
        D[Flatten + Linear]
    end
    
    subgraph "Output"
        E[Weekly Return]
    end
    
    A --> B --> C --> D --> E
```

---

## RL Training

### POST /train/ppo_lstm/full

**Full PPO + LSTM Training**

Trains PPO policy from scratch using LSTM forecasts.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /train/ppo_lstm/full] --> B[Resolve Training Window]
    B --> C[Get Top 15 Symbols]
    
    C --> D[Load Price Data]
    D --> E[Resample to Weekly]
    E --> F[Load Historical Signals]
    
    F --> G{Snapshots Available?}
    G -->|Yes| H[Walk-Forward Forecasts<br/>Using Snapshots]
    G -->|No| I[Walk-Forward Forecasts<br/>Using Final Model]
    
    H & I --> J[Build Training Data]
    J --> K[Train PPO<br/>Gym Environment]
    K --> L[Evaluate on Held-Out Data]
    L --> M[Compute Sharpe, CAGR, Max DD]
    M --> N[Promote if Better]
```

### PPO Training Loop

```mermaid
flowchart LR
    subgraph "Episode"
        A[Reset Environment<br/>Random Start Week] --> B[Get State]
        B --> C[PPO Policy → Action]
        C --> D[Execute Trades]
        D --> E[Get Reward<br/>Portfolio Return - Costs]
        E --> F{Episode Done?}
        F -->|No| B
        F -->|Yes| G[Store Trajectory]
    end
    
    subgraph "Update"
        G --> H[Compute Advantages<br/>GAE]
        H --> I[PPO Clipped Update]
        I --> J[Update Value Function]
    end
```

### Response Schema

```json
{
  "version": "v2026.01.05-ppo123",
  "data_window_start": "2011-01-01",
  "data_window_end": "2026-01-05",
  "metrics": {
    "policy_loss": 0.0234,
    "value_loss": 0.0456,
    "avg_episode_return": 15.23,
    "avg_episode_sharpe": 1.45,
    "eval_sharpe": 1.62,
    "eval_cagr": 0.18,
    "eval_max_drawdown": 0.12
  },
  "promoted": true,
  "prior_version": null,
  "symbols_used": ["AAPL", "MSFT", "GOOGL", "..."]
}
```

---

### POST /train/ppo_lstm/finetune

**Fine-tune PPO + LSTM**

Adapts existing model to recent market conditions using 26 weeks of data.

### Flow Diagram

```mermaid
flowchart TD
    A[POST /train/ppo_lstm/finetune] --> B{Prior Model Exists?}
    B -->|No| C[Return 400 Error]
    B -->|Yes| D[Load Prior Model]
    
    D --> E[Get Last 26 Weeks Data]
    E --> F[Load Recent Signals]
    F --> G[Build Training Data]
    
    G --> H[Fine-tune PPO<br/>Lower LR, Fewer Steps]
    H --> I[Evaluate]
    I --> J{Better than Prior?}
    J -->|Yes| K[Promote]
    J -->|No| L[Keep Prior]
```

### Fine-tuning Config

```mermaid
flowchart LR
    subgraph "Full Training"
        A[100K timesteps]
        B[LR: 3e-4]
        C[15 years data]
    end
    
    subgraph "Fine-tuning"
        D[10K timesteps]
        E[LR: 1e-4]
        F[26 weeks data]
    end
```

---

### SAC Training Endpoints

SAC (Soft Actor-Critic) training follows the same pattern as PPO but uses:
- Actor-Critic architecture with entropy bonus
- Off-policy learning with replay buffer
- Automatic temperature tuning

```mermaid
flowchart LR
    subgraph "SAC Components"
        A[Actor Network] --> D[Action Distribution]
        B[Critic 1] --> E[Q-Value 1]
        C[Critic 2] --> F[Q-Value 2]
        G[Alpha] --> H[Entropy Coefficient]
    end
```

---

## Model Promotion Logic

```mermaid
flowchart TD
    A[New Model Trained] --> B{First Model?}
    B -->|Yes| C[Auto-Promote]
    B -->|No| D{Beats Prior?}
    
    D -->|Forecaster| E[val_loss < prior_val_loss?]
    D -->|PPO| F[eval_sharpe > prior_sharpe?]
    D -->|SAC| G[eval_cagr > prior_cagr?]
    
    E & F & G -->|Yes| H[Promote to Current]
    E & F & G -->|No| I[Keep Prior]
```

---

## Snapshot System

For walk-forward forecast generation during RL training:

```mermaid
flowchart TD
    A[LSTM/PatchTST Training] --> B[Save Current Model]
    B --> C[Backfill Historical Snapshots]
    
    C --> D[For year in 2015..2025]
    D --> E[Filter Data to Year End]
    E --> F[Train Snapshot Model]
    F --> G[Save to snapshots/lstm/YYYY-12-31/]
    G --> D
    
    subgraph "Walk-Forward Usage"
        H[RL Training Week 2020-06-15]
        I[Load Snapshot 2019-12-31]
        J[Generate Forecast]
        H --> I --> J
    end
```

---

## Storage Locations

```
data/models/
├── lstm/
│   ├── current → v2026.01.05-abc123
│   └── v2026.01.05-abc123/
│       ├── model.pt
│       ├── scaler.pkl
│       ├── config.json
│       └── metadata.json
├── patchtst/
├── ppo_lstm/
├── ppo_patchtst/
├── sac_lstm/
└── sac_patchtst/

data/models/snapshots/
├── lstm/
│   ├── 2015-12-31/
│   ├── 2016-12-31/
│   └── ...
└── patchtst/
    ├── 2015-12-31/
    └── ...
```
