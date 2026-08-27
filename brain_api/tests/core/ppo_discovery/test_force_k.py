"""Forced-K uses selection-logit ranking, not allocation-weight or alpha zeros."""

from __future__ import annotations

import torch

from brain_api.core.ppo_discovery.config import MAX_SELECTED
from brain_api.core.ppo_discovery.distributions import deterministic_weights


def _count_logits(native_k: int) -> torch.Tensor:
    logits = torch.full((MAX_SELECTED + 1,), -10.0)
    logits[native_k] = 5.0
    return logits


def test_force_k_larger_than_native_uses_next_selection_logits() -> None:
    symbols = ("ZZZ", "AAA", "MMM")
    mask = torch.tensor([True, True, True])
    weights = deterministic_weights(
        count_logits=_count_logits(1),
        selection_logits=torch.tensor([3.0, 1.0, 2.0]),
        cash_raw=torch.tensor([1.0, 1.0]),
        allocation_raw=torch.tensor([0.1, 10.0, 0.1]),
        asset_mask=mask,
        symbols=symbols,
        force_k=2,
    )
    selected = {name for name, value in weights.items() if name != "CASH" and value > 0}
    assert selected == {"ZZZ", "MMM"}
    assert "AAA" not in selected


def test_force_k_smaller_than_native_keeps_top_selection_logits() -> None:
    symbols = ("ZZZ", "AAA", "MMM")
    mask = torch.tensor([True, True, True])
    weights = deterministic_weights(
        count_logits=_count_logits(2),
        selection_logits=torch.tensor([3.0, 1.0, 2.0]),
        cash_raw=torch.tensor([1.0, 1.0]),
        allocation_raw=torch.tensor([0.1, 10.0, 0.1]),
        asset_mask=mask,
        symbols=symbols,
        force_k=1,
    )
    selected = {name for name, value in weights.items() if name != "CASH" and value > 0}
    assert selected == {"ZZZ"}


def test_force_k_zero_is_all_cash() -> None:
    symbols = ("ZZZ", "AAA", "MMM")
    weights = deterministic_weights(
        count_logits=_count_logits(2),
        selection_logits=torch.tensor([3.0, 1.0, 2.0]),
        cash_raw=torch.tensor([1.0, 1.0]),
        allocation_raw=torch.tensor([1.0, 1.0, 1.0]),
        asset_mask=torch.tensor([True, True, True]),
        symbols=symbols,
        force_k=0,
    )
    assert weights == {"CASH": 1.0}


def test_force_k_clamps_to_eligible() -> None:
    symbols = ("ZZZ", "AAA")
    weights = deterministic_weights(
        count_logits=_count_logits(1),
        selection_logits=torch.tensor([3.0, 1.0]),
        cash_raw=torch.tensor([1.0, 1.0]),
        allocation_raw=torch.tensor([1.0, 1.0]),
        asset_mask=torch.tensor([True, True]),
        symbols=symbols,
        force_k=99,
    )
    selected = {name for name, value in weights.items() if name != "CASH" and value > 0}
    assert selected == {"ZZZ", "AAA"}
