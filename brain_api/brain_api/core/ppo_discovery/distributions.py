"""Factored PPO action: count, Plackett-Luce selection, Beta cash, Dirichlet weights."""

from __future__ import annotations

import torch
from torch.distributions import Beta, Categorical, Dirichlet

from brain_api.core.ppo_discovery.config import CASH_FLOOR
from brain_api.core.ppo_discovery.schemas import ActionLogProb, SampledAction


def masked_count_logits(logits: torch.Tensor, n_eligible: torch.Tensor) -> torch.Tensor:
    """Mask K greater than eligible count. logits: [batch, 16], n_eligible: [batch]."""
    k_values = torch.arange(logits.size(-1), device=logits.device)
    valid = k_values.unsqueeze(0) <= n_eligible.unsqueeze(-1)
    return logits.masked_fill(~valid, float("-inf"))


def clamp_concentration(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + 1.0, max=50.0)


def cash_from_z(z_cash: torch.Tensor, cash_floor: float = CASH_FLOOR) -> torch.Tensor:
    return cash_floor + (1.0 - cash_floor) * z_cash


def sample_count_and_selection(
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    asset_mask: torch.Tensor,
    symbols: tuple[str, ...],
) -> tuple[int, tuple[int, ...], tuple[str, ...], float, float]:
    """Sample K and the Plackett-Luce order. Cash/Dirichlet are sampled later."""
    if count_logits.ndim != 1 or selection_logits.ndim != 1:
        raise ValueError("sample_count_and_selection expects a single unbatched state")
    n_eligible = asset_mask.to(dtype=torch.long).sum()
    masked_counts = masked_count_logits(
        count_logits.unsqueeze(0), n_eligible.unsqueeze(0)
    )[0]
    count_dist = Categorical(logits=masked_counts)
    k_tensor = count_dist.sample()
    k = int(k_tensor.item())
    log_p_k = float(count_dist.log_prob(k_tensor).item())
    if k == 0:
        return 0, (), (), log_p_k, 0.0
    available = asset_mask.clone()
    selected_idx: list[int] = []
    log_p_sel = 0.0
    for _ in range(k):
        logits = selection_logits.masked_fill(~available, float("-inf"))
        dist = Categorical(logits=logits)
        idx = dist.sample()
        log_p_sel += float(dist.log_prob(idx).item())
        selected_idx.append(int(idx.item()))
        available[idx] = False
    order = tuple(symbols[index] for index in selected_idx)
    return k, tuple(selected_idx), order, log_p_k, log_p_sel


def sample_cash_and_weights(
    *,
    k: int,
    selected_idx: tuple[int, ...],
    order: tuple[str, ...],
    log_p_k: float,
    log_p_sel: float,
    cash_raw: torch.Tensor,
    allocation_raw: torch.Tensor,
    cash_floor: float = CASH_FLOOR,
) -> SampledAction:
    """Sample Beta cash and Dirichlet weights for a already-chosen set."""
    if k == 0:
        return SampledAction(
            k=0,
            selection_order=(),
            selection_indices=(),
            z_cash=None,
            dirichlet_weights=None,
            percentage_weights={"CASH": 1.0},
            log_p_k=log_p_k,
            log_p_selection=0.0,
            log_p_cash=0.0,
            log_p_dirichlet=0.0,
            log_p_total=log_p_k,
        )
    alpha_cash, beta_cash = clamp_concentration(cash_raw)
    cash_dist = Beta(alpha_cash, beta_cash)
    z_cash = cash_dist.sample()
    log_p_cash = float(cash_dist.log_prob(z_cash).item())
    cash_weight = float(cash_from_z(z_cash, cash_floor).item())
    selected_raw = allocation_raw[
        torch.tensor(selected_idx, device=allocation_raw.device)
    ]
    concentrations = clamp_concentration(selected_raw)
    dirichlet = Dirichlet(concentrations)
    simplex = dirichlet.sample()
    log_p_dir = float(dirichlet.log_prob(simplex).item())
    stock_mass = 1.0 - cash_weight
    weights = {
        symbol: float(stock_mass * simplex[i].item()) for i, symbol in enumerate(order)
    }
    weights["CASH"] = cash_weight
    total = log_p_k + log_p_sel + log_p_cash + log_p_dir
    return SampledAction(
        k=k,
        selection_order=order,
        selection_indices=selected_idx,
        z_cash=float(z_cash.item()),
        dirichlet_weights=tuple(float(value) for value in simplex.tolist()),
        percentage_weights=weights,
        log_p_k=log_p_k,
        log_p_selection=log_p_sel,
        log_p_cash=log_p_cash,
        log_p_dirichlet=log_p_dir,
        log_p_total=total,
    )


def sample_factored_action(
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    cash_raw: torch.Tensor,
    allocation_raw: torch.Tensor,
    asset_mask: torch.Tensor,
    symbols: tuple[str, ...],
    cash_floor: float = CASH_FLOOR,
) -> SampledAction:
    """Sample one complete action. Prefer the two-stage helpers when heads depend on K."""
    k, selected_idx, order, log_p_k, log_p_sel = sample_count_and_selection(
        count_logits=count_logits,
        selection_logits=selection_logits,
        asset_mask=asset_mask,
        symbols=symbols,
    )
    return sample_cash_and_weights(
        k=k,
        selected_idx=selected_idx,
        order=order,
        log_p_k=log_p_k,
        log_p_sel=log_p_sel,
        cash_raw=cash_raw,
        allocation_raw=allocation_raw,
        cash_floor=cash_floor,
    )


def recompute_action_log_prob_tensors(
    action: SampledAction,
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    cash_raw: torch.Tensor,
    allocation_raw: torch.Tensor,
    asset_mask: torch.Tensor,
    cash_floor: float = CASH_FLOOR,
) -> torch.Tensor:
    """Replay the stored selection order as a differentiable scalar logp."""
    del cash_floor  # cash transform is deterministic; density is on z_cash.
    n_eligible = asset_mask.to(dtype=torch.long).sum()
    masked_counts = masked_count_logits(
        count_logits.unsqueeze(0), n_eligible.unsqueeze(0)
    )[0]
    count_dist = Categorical(logits=masked_counts)
    k_tensor = torch.tensor(action.k, device=count_logits.device)
    log_p_k = count_dist.log_prob(k_tensor)
    if action.k == 0:
        return log_p_k

    available = asset_mask.clone()
    log_p_sel = count_logits.new_zeros(())
    for index in action.selection_indices:
        logits = selection_logits.masked_fill(~available, float("-inf"))
        dist = Categorical(logits=logits)
        idx = torch.tensor(index, device=selection_logits.device)
        log_p_sel = log_p_sel + dist.log_prob(idx)
        available[idx] = False

    alpha_cash, beta_cash = clamp_concentration(cash_raw)
    z = torch.tensor(action.z_cash, device=cash_raw.device, dtype=cash_raw.dtype)
    log_p_cash = Beta(alpha_cash, beta_cash).log_prob(z)
    concentrations = clamp_concentration(
        allocation_raw[
            torch.tensor(action.selection_indices, device=allocation_raw.device)
        ]
    )
    simplex = torch.tensor(
        action.dirichlet_weights,
        device=allocation_raw.device,
        dtype=allocation_raw.dtype,
    )
    log_p_dir = Dirichlet(concentrations).log_prob(simplex)
    return log_p_k + log_p_sel + log_p_cash + log_p_dir


def recompute_action_log_prob(
    action: SampledAction,
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    cash_raw: torch.Tensor,
    allocation_raw: torch.Tensor,
    asset_mask: torch.Tensor,
    cash_floor: float = CASH_FLOOR,
) -> ActionLogProb:
    """Replay the stored selection order. Sorting before this call is forbidden."""
    n_eligible = asset_mask.to(dtype=torch.long).sum()
    masked_counts = masked_count_logits(
        count_logits.unsqueeze(0), n_eligible.unsqueeze(0)
    )[0]
    count_dist = Categorical(logits=masked_counts)
    k_tensor = torch.tensor(action.k, device=count_logits.device)
    log_p_k = count_dist.log_prob(k_tensor)
    if action.k == 0:
        total = float(log_p_k.item())
        return ActionLogProb(total, 0.0, 0.0, 0.0, total)

    available = asset_mask.clone()
    log_p_sel = count_logits.new_zeros(())
    for index in action.selection_indices:
        logits = selection_logits.masked_fill(~available, float("-inf"))
        dist = Categorical(logits=logits)
        idx = torch.tensor(index, device=selection_logits.device)
        log_p_sel = log_p_sel + dist.log_prob(idx)
        available[idx] = False

    alpha_cash, beta_cash = clamp_concentration(cash_raw)
    z = torch.tensor(action.z_cash, device=cash_raw.device, dtype=cash_raw.dtype)
    log_p_cash = Beta(alpha_cash, beta_cash).log_prob(z)
    concentrations = clamp_concentration(
        allocation_raw[
            torch.tensor(action.selection_indices, device=allocation_raw.device)
        ]
    )
    simplex = torch.tensor(
        action.dirichlet_weights,
        device=allocation_raw.device,
        dtype=allocation_raw.dtype,
    )
    log_p_dir = Dirichlet(concentrations).log_prob(simplex)
    total = log_p_k + log_p_sel + log_p_cash + log_p_dir
    return ActionLogProb(
        float(log_p_k.item()),
        float(log_p_sel.item()),
        float(log_p_cash.item()),
        float(log_p_dir.item()),
        float(total.item()),
    )


def deterministic_weights(
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    cash_raw: torch.Tensor,
    allocation_raw: torch.Tensor,
    asset_mask: torch.Tensor,
    symbols: tuple[str, ...],
    cash_floor: float = CASH_FLOOR,
) -> dict[str, float]:
    """Inference action: argmax K, lex-stable top-K, Beta/Dirichlet means."""
    n_eligible = int(asset_mask.sum().item())
    masked_counts = masked_count_logits(
        count_logits.unsqueeze(0),
        torch.tensor([n_eligible], device=count_logits.device),
    )[0]
    k = int(torch.argmax(masked_counts).item())
    if k == 0:
        return {"CASH": 1.0}
    valid_indices = [index for index, flag in enumerate(asset_mask.tolist()) if flag]
    ranked = sorted(
        valid_indices,
        key=lambda index: (-float(selection_logits[index].item()), symbols[index]),
    )
    selected = ranked[:k]
    alpha_cash, beta_cash = clamp_concentration(cash_raw)
    z_mean = alpha_cash / (alpha_cash + beta_cash)
    cash_weight = float(cash_from_z(z_mean, cash_floor).item())
    concentrations = clamp_concentration(
        allocation_raw[torch.tensor(selected, device=allocation_raw.device)]
    )
    simplex = concentrations / concentrations.sum()
    stock_mass = 1.0 - cash_weight
    weights = {
        symbols[index]: float(stock_mass * simplex[i].item())
        for i, index in enumerate(selected)
    }
    weights["CASH"] = cash_weight
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"inference weights sum to {total}, not 1")
    if abs(total - 1.0) > 0:
        weights["CASH"] += 1.0 - total
    return weights


__all__ = [
    "cash_from_z",
    "clamp_concentration",
    "deterministic_weights",
    "masked_count_logits",
    "recompute_action_log_prob",
    "recompute_action_log_prob_tensors",
    "sample_cash_and_weights",
    "sample_count_and_selection",
    "sample_factored_action",
]
