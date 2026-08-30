"""Factored PPO action: count, Plackett-Luce selection, Beta cash, Dirichlet weights."""

from __future__ import annotations

import torch
from torch.distributions import Beta, Categorical, Dirichlet

from brain_api.core.ppo_discovery.config import CASH_FLOOR, MAX_SELECTED
from brain_api.core.ppo_discovery.schemas import ActionLogProb, SampledAction


def masked_count_logits(logits: torch.Tensor, n_eligible: torch.Tensor) -> torch.Tensor:
    """Mask K greater than eligible count. logits: [batch, 16], n_eligible: [batch]."""
    k_values = torch.arange(logits.size(-1), device=logits.device)
    valid = k_values.unsqueeze(0) <= n_eligible.unsqueeze(-1)
    return logits.masked_fill(~valid, float("-inf"))


def clamp_concentration(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nn.functional.softplus(raw) + 1.0, max=50.0)


def _dirichlet_log_prob(
    concentrations: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Dirichlet log density using device-native primitives."""
    return (
        torch.xlogy(concentrations - 1.0, value).sum(dim=-1)
        + torch.lgamma(concentrations.sum(dim=-1))
        - torch.lgamma(concentrations).sum(dim=-1)
    )


def _beta_log_prob(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Beta log density as a two-part Dirichlet."""
    concentrations = torch.stack((alpha, beta), dim=-1)
    parts = torch.stack((value, 1.0 - value), dim=-1)
    return _dirichlet_log_prob(concentrations, parts)


def cash_from_z(z_cash: torch.Tensor, cash_floor: float = CASH_FLOOR) -> torch.Tensor:
    return cash_floor + (1.0 - cash_floor) * z_cash


def _sample_cpu_if_mps(dist: Beta | Dirichlet) -> torch.Tensor:
    """Sample Beta/Dirichlet on CPU when params live on MPS (PyTorch 2.9)."""
    if isinstance(dist, Beta):
        if dist.concentration1.device.type != "mps":
            return dist.sample()
        return Beta(
            dist.concentration1.detach().cpu(),
            dist.concentration0.detach().cpu(),
        ).sample()
    if dist.concentration.device.type != "mps":
        return dist.sample()
    return Dirichlet(dist.concentration.detach().cpu()).sample()


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
    z_cash = _sample_cpu_if_mps(cash_dist).to(
        device=alpha_cash.device, dtype=alpha_cash.dtype
    )
    log_p_cash = float(_beta_log_prob(alpha_cash, beta_cash, z_cash).item())
    cash_weight = float(cash_from_z(z_cash, cash_floor).item())
    selected_raw = allocation_raw[
        torch.tensor(selected_idx, device=allocation_raw.device)
    ]
    concentrations = clamp_concentration(selected_raw)
    dirichlet = Dirichlet(concentrations)
    simplex = _sample_cpu_if_mps(dirichlet).to(
        device=concentrations.device, dtype=concentrations.dtype
    )
    log_p_dir = float(_dirichlet_log_prob(concentrations, simplex).item())
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
    log_p_cash = _beta_log_prob(alpha_cash, beta_cash, z)
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
    log_p_dir = _dirichlet_log_prob(concentrations, simplex)
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
    log_p_cash = _beta_log_prob(alpha_cash, beta_cash, z)
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
    log_p_dir = _dirichlet_log_prob(concentrations, simplex)
    total = log_p_k + log_p_sel + log_p_cash + log_p_dir
    return ActionLogProb(
        float(log_p_k.item()),
        float(log_p_sel.item()),
        float(log_p_cash.item()),
        float(log_p_dir.item()),
        float(total.item()),
    )


def count_and_selection_entropy(
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    asset_mask: torch.Tensor,
    selection_indices: tuple[int, ...],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical entropy of the count head and the sampled Plackett-Luce path.

    Cash and Dirichlet terms are excluded: their coefficients are not
    applied to those heads. Selection entropy is the sum of sequential
    Categorical entropies along ``selection_indices``, matching the
    sampled action's log-probability.
    """
    if count_logits.ndim != 1 or selection_logits.ndim != 1:
        raise ValueError("count_and_selection_entropy expects an unbatched state")
    if k != len(selection_indices):
        raise ValueError("selection entropy k must match selection_indices length")
    n_eligible = asset_mask.to(dtype=torch.long).sum()
    masked_counts = masked_count_logits(
        count_logits.unsqueeze(0), n_eligible.unsqueeze(0)
    )[0]
    h_count = Categorical(logits=masked_counts).entropy()
    if k == 0:
        return h_count, count_logits.new_zeros(())
    remaining = asset_mask.clone()
    h_selection = count_logits.new_zeros(())
    for index in selection_indices:
        masked_selection = selection_logits.masked_fill(~remaining, float("-inf"))
        h_selection = h_selection + Categorical(logits=masked_selection).entropy()
        remaining[index] = False
    return h_count, h_selection


def deterministic_weights(
    *,
    count_logits: torch.Tensor,
    selection_logits: torch.Tensor,
    cash_raw: torch.Tensor,
    allocation_raw: torch.Tensor,
    asset_mask: torch.Tensor,
    symbols: tuple[str, ...],
    cash_floor: float = CASH_FLOOR,
    force_k: int | None = None,
) -> dict[str, float]:
    """Inference action: argmax K (or forced K), lex-stable top-K, Beta/Dirichlet means."""
    n_eligible = int(asset_mask.sum().item())
    if force_k is not None:
        k = min(max(int(force_k), 0), n_eligible, MAX_SELECTED)
    else:
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
    "count_and_selection_entropy",
    "deterministic_weights",
    "masked_count_logits",
    "recompute_action_log_prob",
    "recompute_action_log_prob_tensors",
    "sample_cash_and_weights",
    "sample_count_and_selection",
    "sample_factored_action",
]
