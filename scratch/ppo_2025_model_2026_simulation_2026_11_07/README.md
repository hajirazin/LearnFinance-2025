# Frozen December-2025 PPO: rolling 2026 account simulation

Starting with $10,000 on Monday 5 January 2026, the account is worth
**$12,072.77 at Friday 6 November's close**, the latest completed trading
session as of Saturday 7 November. The total return is **20.7277% after
$136.06 in modeled buy/sell charges**. The model was not retrained.

| Portfolio | Ending NAV | Total return | Weekly-mark max drawdown | Fees |
| --- | ---: | ---: | ---: | ---: |
| Frozen PPO | $12,072.77 | +20.7277% | 11.5643% | $136.0638 |
| SPY buy and hold | $10,793.35 | +7.9335% | 8.3923% | $0.3537 |
| QQQ buy and hold | $10,629.58 | +6.2958% | 12.6679% | $0.3540 |

PPO's return exceeds SPY by 12.7941 percentage points and QQQ by 14.4318
percentage points in this retrospective simulation. Drawdowns are based on
weekly marks, not every daily or intraday high/low.

## Ending account

| Symbol | Fractional units | 6 November close | Market value | Account weight |
| --- | ---: | ---: | ---: | ---: |
| BSX | 37.74539985 | $42.75 | $1,613.62 | 13.3658% |
| INTC | 26.06050478 | $56.28 | $1,466.69 | 12.1487% |
| NVDA | 7.69108628 | $186.64 | $1,435.46 | 11.8901% |
| SNDK | 1.57244836 | $835.06 | $1,313.09 | 10.8765% |
| CASH | — | — | $6,243.91 | 51.7190% |

Units and valuations use the adjusted-price convention below. Stock values
plus cash reconcile to the full NAV. There is no final liquidation charge,
since the account still holds these stocks.

## Exact checkpoint and scope

The checkpoint is
`scratch/ppo_allocator_learning_2026_11_06/market_seed42_final.pt`, SHA-256
`5b12f183891961b2d4ff29f0184652fe0d1c0341830278b22444b0eda73c9300`.
It completed 10,296 sampled decisions in the earlier scratch experiment.
The outer training cutoff is 31 December 2025; the last actual training
reward outcome ends 29 December 2025. Its channel/count scalers use training
weeks only, and its HMM is fitted through 19 December 2025. The saved scalers,
HMM, configuration and policy are reused unchanged. No 2026 outcome chooses
a checkpoint or changes model parameters in this simulation.

The exact original 432-stock frozen roster is retained. It was assembled on
12 September 2026 and applied historically, so **survivorship/universe-selection
bias remains**. This is not a reconstruction of a live January-2026 universe.

## Decision and accounting rules

1. There are 44 scheduled Monday decisions, 5 January through 2 November.
   News uses the exact Monday 09:00 New York window and requires complete
   coverage, including verified zero-news records. No neutral substitution.
2. All OHLCV features and SPY globals exclude the entire decision session.
   The HMM continues causally from its saved pre-2026 cutoff posterior.
   Current portfolio-weight features use the preceding completed session's
   close, never the yet-unknown Monday open.
3. After the allocation is fixed, execution occurs at the first XNYS open of
   that week. Monday holidays therefore execute Tuesday. Stocks are not sold
   automatically on Friday: Friday is a valuation point. Good Friday and
   other Friday holidays use the week's last actual session.
4. Actual fractional holdings and cash carry across weeks. Weekend price gaps
   affect Monday's pre-trade NAV. Rebalances trade only differences between
   existing holdings and the new target; there is no $10,000 weekly reset.
5. Both buys and sells use the existing IBKR Singapore Tiered cost function,
   sized against the account's **actual rolling NAV** at each rebalance.
   This scratch account simulation follows the user's compounding request;
   production training's fixed-$10,000 cost convention is unchanged.
   Fees are paid from target cash. Every rebalance checks self-financing and
   rejects negative cash or missing held/traded prices.
6. SPY and QQQ each buy and hold from the same first Monday open. Their
   purchase quantities reserve and pay the modeled entry fee without
   borrowing. No fee is charged for a week without a trade.

The PPO account makes 318 modeled trade legs: buy fees total $69.2576 and
sell fees $66.8061. There is no HHI/reward-shaping deduction from money: only
economic price changes and modeled broker charges change NAV.

## Data and limitations

All 435 adjusted OHLCV histories (432 stocks plus SPY, QQQ and VIX) loaded
from Yahoo through the native price loader. Adjusted prices approximate
split/dividend reinvestment with fractional positions; dividend cash is not
added again. Cash earns no interest. Slippage, bid/ask spread, taxes,
fractional-share availability and actual execution fills are not modeled.

News was copied to an independent APFS clone under this folder. Only 240
missing symbol/window cells across eight weeks were filled there, using the
existing strict Alpaca/Benzinga and pinned FinBERT service. All 19,008 required
2026 news cells are complete. No production news database was modified.

Multiplying all future asset/market price bars by seven left the first
decision state digest and weights unchanged. Every decision also checks its
HMM evidence dates and input cutoff. Eleven accounting, calendar, causality
and output-reconciliation tests passed.

## Artifacts

- `weekly_performance_2026.png` / `.svg`: weekly NAV and drawdown versus SPY/QQQ.
- `weekly_nav.csv`: the initial $10,000 point plus all 44 weekly closing marks.
- `ending_holdings.csv`: fractional quantities, closing prices and values.
- `trades.csv`: every buy/sell leg and its cost components.
- `weekly_ledger.json`: each allocation, pre/post-trade NAV, Friday book and fees.
- `summary.json`: complete final results and conventions.
- `model_identity.json`, `price_manifest.json`, `news_preparation_result.json`:
  checkpoint and data provenance. Raw evidence is ignored under `data/`.

Source model hashes, the production current pointer and original news-file
size/modification time were verified unchanged. No production-code edits,
model promotion, HF publication, broker orders or Pi access occurred.

Run offline with the existing scratch plotting dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api:scratch/ppo_2025_model_2026_simulation_2026_11_07 brain_api/.venv/bin/python scratch/ppo_2025_model_2026_simulation_2026_11_07/simulate.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=scratch/ppo_prod_refresh_2026_11_07/data/mpl_cache PYTHONPATH=scratch/ppo_prod_refresh_2026_11_07/data/chart_dependencies brain_api/.venv/bin/python scratch/ppo_2025_model_2026_simulation_2026_11_07/plot.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain_api:scratch/ppo_2025_model_2026_simulation_2026_11_07 brain_api/.venv/bin/python -m pytest scratch/ppo_2025_model_2026_simulation_2026_11_07 -q
```
