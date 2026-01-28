# Architecture Research Comparison

Research-backed comparison of the portfolio allocation architecture. Based on published research, academic papers, and industry reviews.

**Date**: January 2026

---

## Summary of Actionable Items

| # | Action | Research Basis | Priority |
|---|--------|----------------|----------|
| 1 | Train LSTM/PatchTST on full S&P 500 | Transfer learning: ~94% of stock prediction is "global" | HIGH |
| 2 | Add ROE, momentum, earnings growth to signals | Fama-French: ROE stocks earned 15.2% abnormal returns | HIGH |
| 3 | Add HRP weights to RL state vector | Imperial College: RL + HRP outperformed pure HRP | MEDIUM |
| 4 | Send both LSTM + PatchTST forecasts to RL | RL-based ensemble research shows dynamic forecaster weighting | MEDIUM |

---

## 1. Do We Really Need Forecasting (LSTM/PatchTST) in RL?

**Research Verdict: YES, forecasts as features improve RL performance**

### Evidence

- **AAAI 2020 Paper** "Reinforcement-Learning Based Portfolio Management with Augmented Asset Movement Prediction States" demonstrates that **augmenting RL state spaces with price forecasts significantly improves allocation decisions** by addressing data heterogeneity and market uncertainty.
- **2024 Multimodal DRL Study** shows that integrating LSTM forecasts with PPO achieved **annualized Sharpe Ratio of 0.86** on Dow Jones, outperforming RL without forecasts.
- **January 2025 Research (arXiv:2501.17992)** combining deep learning embeddings with RL meta-learning for top 500 US stocks shows **outperformance during market stress** when using forecast features.

### Current Architecture Assessment

The design is **correctly aligned with research best practices**:

- LSTM predicted return as a feature in RL state vector
- RL reward computed from **actual realized returns** (not forecaster predictions)
- Forecaster treated as an "alpha signal" that RL can learn to weight

### Recommendation

**Keep forecasters**. They provide valuable signals. The key insight from research is that forecasters should be **features, not the reward function** - which the architecture correctly implements.

---

## 2. Do We Need Multi-Signal PatchTST or Just Price?

**Research Verdict: Multivariate (price + signals) outperforms price-only**

### Evidence

- **2024 Springer Study** on S&P 500: Adding sentiment to fundamentals + technicals **increased accuracy by 1.5%** for 66% of companies analyzed.
- **PLOS ONE 2023 Study**: LSTM/GRU with news features produced **better prediction accuracy** than fundamental features alone.
- **Multi-method Survey (Entropy Journal)**: Testing 67 feature combinations across 30 algorithms found multivariate approaches consistently outperformed univariate, though improvements vary by sector and timeframe.

### PatchTST-Specific Research

The original **PatchTST paper (arXiv:2211.14730)** uses **channel-independence** where each univariate series shares weights. However, **HDMixer (AAAI 2024)** extends this to model:

- Short-term dependencies within patches
- Long-term dependencies across patches  
- **Cross-variable interactions** (multivariate relationships)

### Current Architecture Assessment

Current design uses **PatchTST with OHLCV + news + fundamentals (12 channels)** which is supported by research. However, there's a valid counter-argument:

**Alternative View**: Since RL already receives all signals, having **LSTM (price-only)** and **PatchTST (price-only)** gives RL two "pure forecaster" views that it can combine with raw signals itself. This avoids **signal double-counting**.

### Recommendation

**Current approach is valid**, but consider running an **ablation experiment**:

- PatchTST with all 12 channels (current)
- PatchTST with price-only (5 channels)

Compare which variant the RL agent performs better with.

---

## 3. Train on Full S&P 500, Inference on Halal Subset?

**Research Verdict: YES, training on larger universe improves predictions**

### Evidence

- **SSRN Paper "How Global is Predictability?"** (2024): A global model trained across multiple countries **predicts stock returns more effectively** than locally-estimated models, even when excluding local data. Key finding: **stock return prediction is ~94% global** - nearly identical functions predict returns across markets.
- **Large Investment Model (arXiv:2408.10255)**: Foundation model trained on diverse financial data spanning multiple exchanges/instruments/frequencies learns **"global patterns"** that transfer downstream to specific tasks. Shows advantages in **cross-instrument prediction**.
- **Transfer Learning Survey (arXiv:2409.17183)**: Emphasizes transfer learning is particularly valuable for stock prediction given non-linear nature of financial time series.

### Current Architecture Assessment

Current approach trains on **halal universe only**, which limits training data significantly (maybe ~100-150 stocks vs 500).

### Recommendation

**Train forecasters (LSTM, PatchTST) on full S&P 500, inference on halal universe**. Benefits:

- 3-5x more training data
- Better learned representations of market dynamics
- Halal stocks aren't fundamentally different in price behavior - just filtered by business activities

**For RL (PPO/SAC)**: Train on halal universe only since the allocator needs to learn the specific portfolio constraints.

---

## 4. Is SAC Being Used Properly? SAC vs PPO for Portfolio?

**Research Verdict: SAC implementation is correct; SAC generally outperforms PPO for portfolios**

### Implementation Review

SAC implementation includes all required components:

| Component                | Implementation                      | Standard SAC  | Status  |
| ------------------------ | ----------------------------------- | ------------- | ------- |
| Twin Q-critics           | TwinCritic with Q1, Q2              | Required      | Correct |
| Target networks          | Polyak averaging (tau=0.005)        | Required      | Correct |
| Automatic entropy tuning | log_alpha with target entropy       | Best practice | Correct |
| Gaussian policy          | GaussianActor with tanh squashing   | Standard      | Correct |
| Replay buffer            | Off-policy with experience replay   | Required      | Correct |
| Gradient clipping        | max_grad_norm applied               | Best practice | Correct |
| Q-value clipping         | Clamp targets to prevent divergence | Stability fix | Correct |

### Research on SAC vs PPO

- **Stevens Institute Study (2024)**: Comparing A2C, DDPG, PPO, SAC on DJIA trading found **SAC delivered superior cumulative and annual returns**, with PPO ranking third.
- **SAC Advantages per Research**:
  - Higher sample efficiency (important with limited financial data)
  - Better for continuous action spaces (portfolio weights)
  - Entropy maximization prevents premature convergence
- **SAC Limitations (2022 Market Downturn Study)**: SAC struggled during regime changes, required retraining on down-market scenarios.

### Recommendation

**Keep both PPO and SAC** - the comparison approach is scientifically sound. Research suggests:

- SAC for better absolute performance
- PPO for stability in deployment
- The comparison will reveal which works better for the specific use case

---

## 5. Are There Better Algorithms for Portfolio Allocation?

**Research Verdict: SAC is already the best choice; current setup is optimal**

### Head-to-Head: SAC vs TD3

Benchmark comparisons (MuJoCo/Tianshou) show **SAC outperforms TD3** on all standard tasks:

| Task        | TD3    | SAC    | Winner |
| ----------- | ------ | ------ | ------ |
| Ant         | 5,116  | 5,850  | SAC    |
| HalfCheetah | 10,201 | 12,139 | SAC    |
| Walker2d    | 3,982  | 5,007  | SAC    |

**SAC advantages over TD3** (per original SAC paper arXiv:1812.05905):

- Better sample efficiency
- More stable across random seeds
- Less sensitive to hyperparameters
- Better exploration via entropy regularization

### 2024-2025 Financial Research

| Algorithm    | Performance                     | Notes                          |
| ------------ | ------------------------------- | ------------------------------ |
| **SAC**      | Best cumulative returns on DJIA | Preferred for volatile markets |
| **PPO**      | Third place, stable baseline    | Good for deployment            |
| **TD3**      | Good but not better than SAC    | Simpler but less robust        |
| **Ensemble** | Can improve Sharpe ratio        | Higher complexity              |

### Recommendation

**Current PPO + SAC comparison is already optimal**. No need to add TD3 - research shows SAC is the superior algorithm.

---

## 6. Are Fundamentals Being Used Right? What's Missing?

**Research Verdict: Missing key factors; current factors are reasonable but incomplete**

### Current Fundamentals

| Factor           | Category      | Status           |
| ---------------- | ------------- | ---------------- |
| Gross margin     | Profitability | Included         |
| Operating margin | Profitability | Included         |
| Net margin       | Profitability | Included         |
| Current ratio    | Liquidity     | Included         |
| Debt to equity   | Leverage      | Included         |
| Fundamental age  | Staleness     | Included (good!) |

### Missing Factors per Research

**Fama-French Five-Factor Model** (foundational academic research) suggests:

| Factor                           | Description                          | Research Finding                                                   | Priority |
| -------------------------------- | ------------------------------------ | ------------------------------------------------------------------ | -------- |
| **ROE (Return on Equity)**       | Profitability per shareholder equity | Highest ROE stocks earned **15.2% annual abnormal excess returns** | HIGH     |
| **Momentum (12-1 month)**        | Prior price performance              | **~1.16% monthly returns**, strongest predictor                    | HIGH     |
| **Book-to-Market (P/B inverse)** | Value factor                         | Core Fama-French factor                                            | MEDIUM   |
| **Investment (Asset Growth)**    | Conservative vs aggressive           | CMA factor in 5-factor model                                       | MEDIUM   |
| **Earnings Growth / EPS**        | Growth trajectory                    | Core valuation metric                                              | MEDIUM   |

**Quality Factor Research (MSCI, T. Rowe Price)**:

- ROE combined with debt-to-equity and earnings variability **outperformed individual metrics**
- **Earnings variability** (low is better) is missing from current set

### Recommendation

**High priority additions**:

1. **ROE** - strong research backing, easy to add from same data source
2. **12-1 Month Momentum** - calculated from prices already available
3. **Earnings Growth Rate** - if available from Alpha Vantage

**Medium priority**:

4. Price-to-Book ratio
5. Earnings variability (std dev of quarterly EPS)

**Keep current factors** - margins and ratios are valid. Just incomplete.

---

## 7. Should HRP Output Be Input to RL?

**Research Verdict: YES, this is a validated approach**

### Evidence

- **Imperial College / MDPI 2023 Study**: "Using Deep Reinforcement Learning with Hierarchical Risk Parity for Portfolio Optimization" - A two-level system where **DRL agent receives HRP performance as input state** and selects among multiple HRP variants. Results show **RL portfolios outperformed pure HRP on returns, risk, and Sharpe ratio** across crypto, stock, and forex markets.
- **RL-BHRP (2025, arXiv:2508.11856)**: Two-level learning-based method that uses HRP risk allocation as a component. Achieved **120% wealth compounding** vs 101% for static portfolios.

### Implementation Options

**Option A: HRP weights as features**

```
state = [...existing signals..., hrp_weight_AAPL, hrp_weight_MSFT, ...]
```

Lets RL learn when to follow/deviate from HRP.

**Option B: HRP performance as features**

```
state = [..., hrp_trailing_return, hrp_trailing_sharpe, ...]
```

Lets RL adapt based on how well HRP is doing.

**Option C: Hierarchical selection**

RL chooses between HRP output, PPO output, equal-weight, etc. (meta-controller approach from Imperial College paper)

### Current Architecture

Currently HRP is computed separately and compared at email-sending time. HRP output is **not** fed to PPO/SAC.

### Recommendation

**Add HRP weights as input features to RL state vector**. This is low-effort and research-backed:

- Add `n_stocks` additional features: HRP target weights
- RL can learn when HRP allocation is good vs when to deviate
- Provides a "risk parity anchor" signal

---

## 8. Do We Need LSTM If We Have PatchTST?

**Research Verdict: KEEP BOTH - ensemble benefits outweigh simplification**

### PatchTST vs LSTM Head-to-Head

| Aspect | LSTM | PatchTST | Winner |
|--------|------|----------|--------|
| General accuracy | Good | Better | PatchTST |
| Consistency across timeseries | Varies | Consistent | PatchTST |
| Limited data scenarios | Better | Worse | LSTM |
| Volatile stocks | More robust | "Struggled" | LSTM |
| Price difference prediction | Better | Good | LSTM |

### Key Research

- **SSRN 2024**: "PatchTST outperforms LSTM in terms of accuracy and reliability"
- **arXiv 2024**: "Vanilla LSTMs consistently achieve superior predictive accuracy" with **limited data and default hyperparameters**
- **HAELT Framework**: LSTM + Transformer ensemble achieved **highest F1-Score** for stock forecasting

### Recommendation

Keep both LSTM and PatchTST because:

1. Ensemble approaches show 40-60% improvement over single models
2. LSTM catches things PatchTST misses (volatile stocks, small data)
3. Different "views" of the market add robustness

---

## 9. Should RL Receive Both Forecasts?

**Research Verdict: YES - RL can learn to dynamically weight forecasters**

### Current Architecture (4 agents)

```
PPO + LSTM     → receives only LSTM forecast
PPO + PatchTST → receives only PatchTST forecast  
SAC + LSTM     → receives only LSTM forecast
SAC + PatchTST → receives only PatchTST forecast
```

### Research on Multi-Forecaster RL

| Study | Finding |
|-------|---------|
| **arXiv 2025** | RL framework for dynamic model selection to combine diverse forecasting methods |
| **Amazon Science** | Deep RL for dynamic ensembles enables "adaptive combination of multiple forecasters" |
| **IEEE** | Online ensemble aggregation using deep RL for real-time adaptation |

### Proposed Architecture (2 agents)

```
PPO → receives LSTM forecast + PatchTST forecast + signals
SAC → receives LSTM forecast + PatchTST forecast + signals
```

### Benefits

1. RL learns **when to trust which forecaster** (market-condition dependent)
2. Reduces from 4 agents to 2 (simpler deployment)
3. Research-backed ensemble approach
4. Maintains PPO vs SAC comparison

---

## Final Summary Table

| Question                           | Research Answer            | Current Architecture       | Action Needed                               |
| ---------------------------------- | -------------------------- | -------------------------- | ------------------------------------------- |
| 1. Need forecasters in RL?         | Yes, as features           | Correct                    | None                                        |
| 2. PatchTST multivariate vs price? | Multivariate better        | Correct (12 channels)      | Optional ablation                           |
| 3. Train full S&P500?              | Yes for forecasters        | Training on halal only     | **Change**: Train LSTM/PatchTST on S&P500   |
| 4. SAC implementation correct?     | Yes                        | Correct                    | None                                        |
| 5. Better algorithms?              | SAC is best                | PPO + SAC                  | None (already optimal)                      |
| 6. Fundamentals complete?          | Missing ROE, momentum      | 5 ratios                   | **Add**: ROE, momentum, earnings growth     |
| 7. HRP as input to RL?             | Yes, validated             | Not implemented            | **Add**: HRP weights to RL state            |
| 8. Need LSTM with PatchTST?        | Yes, ensemble benefits     | Both exist separately      | None (keep both)                            |
| 9. Both forecasts to RL?           | Yes, dynamic weighting     | Separate agents            | **Change**: Send both forecasts to each RL  |
