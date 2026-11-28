# The same snapshot replay with repository IBKR fees

Drawdown is the percentage fall from the highest portfolio value reached so
far. A fall from $12,000 to $10,800 is a 10% drawdown, even though the account
remains above its initial $10,000. Maximum drawdown is the largest such fall.
These results use sampled weekly marks and the final partial-week mark; daily
or intraday drawdowns may be larger.

All results use the original January 5–November 25, 2026 dates, stock selections,
execution opens, valuation closes and saved HRP targets. Initial capital is
$10,000. Fees reduce invested capital at each rebalance and affect subsequent
compounding; they are not simply subtracted from the final gross value.

| Allocation and fee convention | Ending value | Return | Fees paid | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Equal weight, zero fees | $14,543.95 | +45.44% | $0 | 10.22% |
| HRP, zero fees | $14,115.23 | +41.15% | $0 | 11.23% |
| Equal weight, repo default | $14,283.46 | +42.83% | $206.91 | 10.39% |
| HRP, repo default | $13,728.47 | +37.28% | $299.85 | 11.75% |
| Equal weight, charge every trade | $14,190.62 | +41.91% | $282.85 | 10.45% |
| HRP, charge every trade | $13,693.81 | +36.94% | $327.23 | 11.79% |

HRP still underperforms equal weight in this period. With the repo default,
its return trails by 5.55 percentage points and its maximum observed drawdown
is 1.35 percentage points larger. One retrospective, reused period without
uncertainty estimates does not establish a general ranking of allocators.

## Fee assumptions

The runner imports production `IBKRSingaporeCostConfig` and
`compute_ibkr_rebalance_cost` from `brain_api.core.portfolio_rl.broker_costs`.
No production rates or fee code were edited. The configured US-stock IBKR
Singapore Tiered model includes:

- $0.0035 commission per share, $0.35 minimum per order, capped at 1% of
  trade value. The cap takes precedence over the minimum for tiny trades.
- Clearing of $0.00020 and FINRA CAT of $0.000033 per share on both sides.
- Sell-side SEC charges of 0.0000206 times sale value, plus FINRA TAF of
  $0.000195 per share sold, capped at $9.27 per trade.
- Pass-through charges equal to commission times the configured NYSE and
  FINRA rates, 0.000175 and 0.000565.

These are **repository assumptions held constant over the replay**, not verified
current or historically changing broker quotations.

The helper defaults to `weight_epsilon=0.005`: actual stock-weight changes
below 0.5 percentage points execute without a modeled fee. The `repo_default`
rows preserve this convention, exempting 377 of 945 equal-weight trades and
122 of 945 HRP trades. The `every_trade` sensitivity uses `1e-12` to suppress
numerical dust only; all 945 trades per strategy are charged. Neither version
changes selections or introduces a no-trade band.

Each rebalance uses its account's **actual current opening NAV**, starting at
$10,000, through `with_nav(nav)`. This follows the repository's self-financing
account replay convention; it differs from the fixed $10,000 training cost
anchor. Invested capital solves invested dollars plus fees on the resulting
actual trades equals pre-fee NAV. Targets remain fully invested in the same
stocks, with fractional adjusted-price units and positions held over weekends.

SPY and QQQ pay their modeled entry fee, about $0.35 each. Their ending values
become $10,520.47 (+5.20%) and $10,270.40 (+2.70%). The S&P 500 remains a price
index comparison without a broker trade. No terminal liquidation is modeled.

Per-share charges are approximate because the original frozen experiment uses
adjusted-price units instead of reconstructing historical broker share counts
through corporate actions. FX, spread, slippage, tax and dividend withholding
remain omitted. Original retrospective fitting, current-roster survivorship
bias and the final incomplete-week limitation still apply.

## Evidence and checks

- `weekly_nav.csv`: five original zero-fee curves, four fee-adjusted strategy
  curves and fee-funded SPY/QQQ benchmarks on the same dates.
- `weekly_fees.csv`: opening NAV, invested capital, closing NAV and fee
  component totals for 188 weekly books.
- `trades.csv`: 3,780 executed legs with quantities, prices, notional,
  itemized charges and explicit fee-exemption flags.
- `weekly_ledger.json`: complete holdings, targets and account reconciliation.
- `summary.json`: metrics, assumptions, limitations, source/code/output
  hashes and verification counts.

Every exported leg was checked against the production per-leg fee function.
All 188 books conserve cash and NAV; the maximum accounting residual is below
$0.0000001. Original evidence and zero-fee result hashes remain unchanged.
No new forecasts, HRP fits, downloads, training, promotion or orders were needed.

Nine additional synthetic offline tests cover fee-funded entry, sell charges,
no trades, fee exemptions, changing NAV, invalid held prices and a two-week
replay carrying fee-reduced capital through a weekend gap.

From the repository root:

```sh
PYTHONPATH=.:brain_api brain_api/.venv/bin/python scratch/patchtst_2025_snapshot_2026_hrp_replay/fee_run.py
PYTHONPATH=.:brain_api:scratch/patchtst_2025_snapshot_2026_hrp_replay brain_api/.venv/bin/python -m pytest -q scratch/patchtst_2025_snapshot_2026_hrp_replay/test_fee_accounting.py
```

The runner skips a matching completed comparison and refuses to overwrite
different inputs, settings, code identities or output hashes. The original
replay folders are read-only evidence dependencies.
