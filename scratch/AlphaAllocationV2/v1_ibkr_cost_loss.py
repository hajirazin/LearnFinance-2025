"""SAC's IBKR fee schedule for the cost-aware PatchTST experiment."""

import numpy as np
import torch
from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_leg_cost,
)
from brain_api.core.sticky_selection import select_with_rank_band
from patchtst_replay_models import simple_returns
from v1_loss_objectives import TOP_K, soft_top15, weekly_mse

COST_CONFIG = IBKRSingaporeCostConfig.default()
COST_WEIGHT = 0.02


def ibkr_leg_fees(delta_usd, prices):
    """Vectorized SAC per-leg fee math, including the minimum and cap."""
    cfg = COST_CONFIG
    notional = delta_usd.abs()
    shares = notional / prices
    commission = torch.minimum(
        (shares * cfg.commission_per_share).clamp_min(cfg.commission_min),
        notional * cfg.commission_max_pct,
    )
    sell_fees = notional * cfg.sec_fee_rate + (
        shares * cfg.finra_taf_per_share
    ).clamp_max(cfg.finra_taf_cap)
    return (
        commission * (1 + cfg.nyse_pass_through_rate + cfg.finra_pass_through_rate)
        + shares * (cfg.clearing_per_share + cfg.cat_per_share)
        + (delta_usd < 0) * sell_fees
    )


def ibkr_transaction_fees(delta, raw_open, bps):
    """Use the production per-leg implementation at simulated execution prices."""
    fees = 0.0
    for symbol, change in delta.items():
        if abs(change) < 1e-9:
            continue
        notional = abs(change)
        fees += compute_ibkr_leg_cost(
            symbol,
            notional,
            notional / raw_open[symbol],
            "buy" if change > 0 else "sell",
            COST_CONFIG,
        ).total
    return float(delta.abs().sum()) * bps / 10_000, fees


class CostAwareWeeklyMSE:
    """Charge a hard sticky basket with soft top-15 gradients."""

    def __init__(self, index, train, closes, device, *, cost_weight=COST_WEIGHT):
        self.cost_weight = cost_weight
        self.symbols = list(closes.columns)
        self.device = device
        self.fold = {}
        marked = closes.where(closes.gt(0)).ffill()
        for training, keys in ((True, index[train]), (False, index[~train])):
            self.fold[training] = []
            for date in keys.unique("friday").sort_values():
                symbols = keys.get_level_values("symbol")[
                    keys.get_level_values("friday") == date
                ].tolist()
                positions = closes.columns.get_indexer(symbols)
                values = marked.loc[date].to_numpy(dtype=np.float32, copy=True)
                if (positions < 0).any() or not np.isfinite(values[positions]).all():
                    raise ValueError(f"Missing candidate fee-sizing prices on {date}")
                self.fold[training].append((symbols, positions, values))

    def reset(self, *, training):
        self.weeks = self.fold[training]
        self.previous = None
        self.last_prices = None

    def __call__(self, output, target, week):
        if self.cost_weight == 0:
            return weekly_mse(output, target)
        scores = simple_returns(output, daily=True)
        return weekly_mse(output, target) + self.cost_weight * self.fee_fraction(
            scores, week
        )

    def membership(self, week):
        symbols, _, _ = self.weeks[week]
        return [
            self.previous is not None and symbol in self.previous for symbol in symbols
        ]

    def fee_fraction(self, scores, week, *, temperature=None, sticky=True):
        symbols, positions, prices = self.weeks[week]
        selection = select_with_rank_band(
            dict(zip(symbols, scores.detach().cpu().tolist(), strict=True)),
            self.previous if sticky else None,
            TOP_K,
            20,
        )
        selected = set(selection.selected)
        held = np.zeros(len(prices), dtype=np.float32)
        if self.previous is not None:
            retained_positions = [
                self.symbols.index(symbol) for symbol in self.previous
            ]
            values = prices[retained_positions] / self.last_prices[retained_positions]
            held[retained_positions] = values / values.sum()
        self.previous, self.last_prices = selected, prices

        cfg = COST_CONFIG
        current = torch.as_tensor(held[positions], device=self.device) * cfg.nav_usd
        candidate_prices = torch.as_tensor(prices[positions], device=self.device)
        enter_or_hold = ibkr_leg_fees(cfg.nav_usd / TOP_K - current, candidate_prices)
        exclude = ibkr_leg_fees(-current, candidate_prices)
        hard = scores.new_tensor([symbol in selected for symbol in symbols])
        soft = (
            soft_top15(scores)
            if temperature is None
            else soft_top15(scores, temperature=temperature)
        ) * TOP_K
        inclusion = hard + (soft - soft.detach())
        fee = (exclude + inclusion * (enter_or_hold - exclude)).sum()
        outside = np.flatnonzero((held > 0) & ~np.isin(np.arange(len(held)), positions))
        if len(outside):
            fee = (
                fee
                + ibkr_leg_fees(
                    -torch.as_tensor(held[outside], device=self.device) * cfg.nav_usd,
                    torch.as_tensor(prices[outside], device=self.device),
                ).sum()
            )
        return fee / cfg.nav_usd
