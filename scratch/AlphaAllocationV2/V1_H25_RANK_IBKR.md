# H25 rank-only filter with IBKR costs

The seven established features use 60 daily observations. PatchTST pools their encoded patches, appends a binary flag for the stock's membership in the model's own last top 15, and predicts one scalar score. Sorting scores gives ranks. The model trains on the existing 356-stock cohort over 2021–2024 and validates on the broader 2025 sample. It starts each fold in cash. Its carry-set comes from its prior predictions. Realized ranks are labels for training and evaluation.

For each pair from the same decision whose actual five-session returns satisfy `R_i > R_j`, minimize `softplus(-(s_i - s_j))`. Return magnitudes determine order only. Add `50 × IBKR fees / $10,000` for the proposed hard top-15 basket. The fixed coefficient adds 0.05 loss units for 10 bps in fees, against a logistic loss of about 0.693 for tied predictions. Fees use the previous equal-weight basket drifted to the decision close and SAC's IBKR Singapore schedule. The training loss sizes estimated shares from adjusted closes and uses a differentiable top-15 approximation in the backward pass with temperature 0.5. The replay sizes shares using the archived unadjusted execution opens. Use chronological batches, seed 42, the 60-epoch cap, and eight-epoch patience. Select checkpoints by validation Spearman correlation, with validation objective breaking ties.

Use strict top 15 for H25 so its previous-membership feature and fee term can learn retention. Preserve the saved close-only V1 refit's production-style 15/20 rule. Report strict hits for both models as an additional ranking check. Actual stock ranks use Friday-close to next-Friday-close realized returns. Portfolio graphs use the same 40 execution-open to execution-open intervals as the earlier 2026 replay. The hindsight line selects the 15 largest realized adjusted open-to-open returns from the common eligible roster with observed prices. The HRP line applies HRP weights to those same hindsight winners.

The two portfolio replays hold the selections fixed and use either equal weights or the production 252-session HRP math. All active portfolios start at $10,000, carry NAV forward, and pay IBKR fees plus 10 bps per traded dollar. SPY uses adjusted open-to-open buy-and-hold returns. Report compounded net returns, drawdown, top-15 hits, and the one-sided block bootstrap for the change in hits. The reused 2026 sample and one seed make this an exploratory experiment. H25 changes the forecast head, the objective, the previous-membership input, and the retention policy together.

```sh
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/train_v1_h25.py --device mps
PYTHONPATH=brain_api python scratch/AlphaAllocationV2/run_v1_h25.py --device mps
```

## Results

The replay scored 406–421 candidates per decision from January 12 through November 9, 2026. H25's epoch 7 checkpoint had 2025 validation Spearman correlation of +0.0195 and a mean estimated IBKR fee of 3.07 bps per rebalance.

| Ranking over 37 matched Friday pairs | V1 | H25 |
|---|---:|---:|
| Realized top-15 hits per Friday, selected basket | 1.65 | 0.46 |
| Realized top-15 hits per Friday, strict predicted top 15 | 1.76 | 0.46 |
| Mean Spearman correlation | +0.0057 | +0.0026 |
| Names replaced per rebalance after entry | 4.31 | 0.31 |

The random-selection reference was 0.54 hits per Friday. H25 averaged 1.19 fewer hits than V1, with a 95% block-bootstrap interval of −1.54 to −0.84 and a one-sided improvement p-value of 1.00. The learned membership coefficient added 0.148 score units to a prior holding. H25 selected 27 distinct names across the replay and V1 selected 114.

| Net portfolio results over 40 periods | V1 | H25 | SPY | Actual top 15, hindsight |
|---|---:|---:|---:|---:|
| Equal-weight return | +42.94% | +11.39% | +8.26% | 270.14× starting NAV |
| HRP return | +27.95% | +6.13% | +8.26% | 216.05× starting NAV |
| Equal-weight maximum drawdown | −30.41% | −7.66% | | |
| HRP maximum drawdown | −23.52% | −12.76% | | |

H25 beat SPY in 15 of its 18 down periods with equal weights and 16 of 18 with HRP. V1 earned more overall with both weightings. The hindsight portfolios assume each rebalance has perfect knowledge of the next execution return and scale with the simulated NAV at the same assumed slippage.

[Stock ranks](data/v1_hypotheses/loss_comparison/h25_rank_ibkr/stock_ranks.csv) · [Weekly ranks and actual Friday winners](data/v1_hypotheses/loss_comparison/h25_rank_ibkr/rank_weekly.csv) · [Picks and weights](data/v1_hypotheses/loss_comparison/h25_rank_ibkr/picks_and_weights.csv) · [Weekly portfolios](data/v1_hypotheses/loss_comparison/h25_rank_ibkr/portfolio_weekly.csv) · [Summary](data/v1_hypotheses/loss_comparison/h25_rank_ibkr/summary.json)
