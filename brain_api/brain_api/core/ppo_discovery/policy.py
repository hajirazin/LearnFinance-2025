"""Actor-critic for ppo_discovery: temporal encoder + set encoder + factored heads."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from brain_api.core.ppo_discovery.config import (
    CASH_FLOOR,
    EXPLICIT_ASSET_FEATURES,
    GLOBAL_FEATURES,
    MAX_SELECTED,
    SET_D_MODEL,
    TEMPORAL_D_MODEL,
    TOKEN_WIDTH,
    PPODiscoveryConfig,
)
from brain_api.core.ppo_discovery.distributions import (
    count_and_selection_entropy as factored_count_and_selection_entropy,
)
from brain_api.core.ppo_discovery.distributions import (
    deterministic_weights,
    recompute_action_log_prob_tensors,
    sample_cash_and_weights,
    sample_count_and_selection,
)
from brain_api.core.ppo_discovery.schemas import CanonicalPPOState, SampledAction
from brain_api.core.ppo_discovery.set_encoder import PPODiscoverySetEncoder
from brain_api.core.ppo_discovery.temporal_encoder import PPODiscoveryTemporalEncoder


def tensors_from_state(
    state: CanonicalPPOState, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a packed state into unbatched tensors on ``device``."""
    history = torch.as_tensor(state.price_history, dtype=torch.float32, device=device)
    features = torch.as_tensor(state.asset_features, dtype=torch.float32, device=device)
    globals_ = torch.as_tensor(state.globals, dtype=torch.float32, device=device)
    mask = torch.as_tensor(state.asset_mask, dtype=torch.bool, device=device)
    return history, features, globals_, mask


def tensors_from_states(
    states: list[CanonicalPPOState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack packed states into a microbatch on ``device``."""
    histories = torch.stack(
        [
            torch.as_tensor(state.price_history, dtype=torch.float32, device=device)
            for state in states
        ]
    )
    features = torch.stack(
        [
            torch.as_tensor(state.asset_features, dtype=torch.float32, device=device)
            for state in states
        ]
    )
    globals_ = torch.stack(
        [
            torch.as_tensor(state.globals, dtype=torch.float32, device=device)
            for state in states
        ]
    )
    masks = torch.stack(
        [
            torch.as_tensor(state.asset_mask, dtype=torch.bool, device=device)
            for state in states
        ]
    )
    return histories, features, globals_, masks


class PPODiscoveryActorCritic(nn.Module):
    """Count, Plackett-Luce selection, Beta cash, Dirichlet weights, value."""

    def __init__(self, config: PPODiscoveryConfig | None = None) -> None:
        super().__init__()
        config = config or PPODiscoveryConfig()
        self.config = config
        self.temporal = PPODiscoveryTemporalEncoder(
            d_model=config.temporal_d_model,
            n_heads=config.temporal_heads,
            n_layers=config.temporal_layers,
            ffn_dim=config.temporal_ff,
            dropout=config.dropout,
        )
        self.set_encoder = PPODiscoverySetEncoder(
            token_width=TOKEN_WIDTH,
            d_model=config.set_d_model,
            n_heads=config.set_heads,
            n_layers=config.set_layers,
            ffn_dim=config.set_ff,
            dropout=config.dropout,
            n_globals=config.max_assets and GLOBAL_FEATURES,
        )
        pooled_width = SET_D_MODEL + GLOBAL_FEATURES
        self.count_head = nn.Linear(pooled_width, MAX_SELECTED + 1)
        self.selection_head = nn.Linear(SET_D_MODEL, 1)
        self.cash_head = nn.Linear(pooled_width + SET_D_MODEL, 2)
        self.allocation_head = nn.Linear(SET_D_MODEL + SET_D_MODEL, 1)
        self.value_head = nn.Linear(pooled_width, 1)
        self.pretrain_head = nn.Linear(TEMPORAL_D_MODEL, 1)

    def freeze_temporal(self) -> None:
        for parameter in self.temporal.parameters():
            parameter.requires_grad = False

    def unfreeze_temporal(self) -> None:
        for parameter in self.temporal.parameters():
            parameter.requires_grad = True

    def encode(
        self,
        history: torch.Tensor,
        asset_features: torch.Tensor,
        globals_: torch.Tensor,
        asset_mask: torch.Tensor,
        temporal_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-asset encodings and pooled state.

        history: [batch, assets, 250, 4]
        asset_features: [batch, assets, 9]
        """
        if asset_features.size(-1) != EXPLICIT_ASSET_FEATURES:
            raise ValueError("asset feature width must be 9")
        if globals_.size(-1) != GLOBAL_FEATURES:
            raise ValueError("global feature width must be 7")
        embeddings = (
            temporal_embeddings
            if temporal_embeddings is not None
            else self.temporal(history)
        )
        tokens = torch.cat([embeddings, asset_features], dim=-1)
        encoded, pooled, _ = self.set_encoder(tokens, asset_mask, globals_)
        return encoded, pooled

    def heads(
        self, encoded: torch.Tensor, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return count logits, selection logits, and value."""
        count_logits = self.count_head(pooled)
        selection_logits = self.selection_head(encoded).squeeze(-1)
        value = self.value_head(pooled).squeeze(-1)
        return count_logits, selection_logits, value

    def cash_and_allocation_raw(
        self, encoded: torch.Tensor, pooled: torch.Tensor, selected_indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cash Beta raw params and per-asset Dirichlet raw concentrations."""
        if encoded.ndim != 2:
            raise ValueError("cash_and_allocation_raw expects unbatched encodings")
        selected = encoded[torch.tensor(selected_indices, device=encoded.device)]
        selected_context = selected.mean(dim=0)
        cash_raw = self.cash_head(torch.cat([pooled, selected_context], dim=-1))
        alloc_context = selected_context.unsqueeze(0).expand(encoded.size(0), -1)
        allocation_raw = self.allocation_head(
            torch.cat([encoded, alloc_context], dim=-1)
        ).squeeze(-1)
        return cash_raw, allocation_raw

    def sample_action(
        self, state: CanonicalPPOState, *, device: torch.device | None = None
    ) -> SampledAction:
        """Sample a training action from one canonical state."""
        device = device or next(self.parameters()).device
        history, features, globals_, mask = tensors_from_state(state, device)
        encoded, pooled = self.encode(
            history.unsqueeze(0),
            features.unsqueeze(0),
            globals_.unsqueeze(0),
            mask.unsqueeze(0),
        )
        count_logits, selection_logits, _ = self.heads(encoded, pooled)
        encoded_u = encoded[0]
        pooled_u = pooled[0]
        k, selected_idx, order, log_p_k, log_p_sel = sample_count_and_selection(
            count_logits=count_logits[0],
            selection_logits=selection_logits[0],
            asset_mask=mask,
            symbols=state.symbols,
        )
        if k == 0:
            return sample_cash_and_weights(
                k=0,
                selected_idx=(),
                order=(),
                log_p_k=log_p_k,
                log_p_sel=0.0,
                cash_raw=pooled_u.new_zeros(2),
                allocation_raw=encoded_u.new_zeros(encoded_u.size(0)),
                cash_floor=self.config.cash_floor,
            )
        cash_raw, allocation_raw = self.cash_and_allocation_raw(
            encoded_u, pooled_u, list(selected_idx)
        )
        return sample_cash_and_weights(
            k=k,
            selected_idx=selected_idx,
            order=order,
            log_p_k=log_p_k,
            log_p_sel=log_p_sel,
            cash_raw=cash_raw,
            allocation_raw=allocation_raw,
            cash_floor=self.config.cash_floor,
        )

    def sample_action_value_log_prob(
        self,
        state: CanonicalPPOState,
        *,
        temporal_embeddings: torch.Tensor | None = None,
    ) -> tuple[SampledAction, float, float]:
        """One encoder pass: action, value, log_p_total."""
        device = next(self.parameters()).device
        history, features, globals_, mask = tensors_from_state(state, device)
        if temporal_embeddings is not None and temporal_embeddings.ndim == 2:
            temporal_embeddings = temporal_embeddings.unsqueeze(0)
        encoded, pooled = self.encode(
            history.unsqueeze(0),
            features.unsqueeze(0),
            globals_.unsqueeze(0),
            mask.unsqueeze(0),
            temporal_embeddings=temporal_embeddings,
        )
        count_logits, selection_logits, value = self.heads(encoded, pooled)
        encoded_u = encoded[0]
        pooled_u = pooled[0]
        k, selected_idx, order, log_p_k, log_p_sel = sample_count_and_selection(
            count_logits=count_logits[0],
            selection_logits=selection_logits[0],
            asset_mask=mask,
            symbols=state.symbols,
        )
        if k == 0:
            action = sample_cash_and_weights(
                k=0,
                selected_idx=(),
                order=(),
                log_p_k=log_p_k,
                log_p_sel=0.0,
                cash_raw=pooled_u.new_zeros(2),
                allocation_raw=encoded_u.new_zeros(encoded_u.size(0)),
                cash_floor=self.config.cash_floor,
            )
        else:
            cash_raw, allocation_raw = self.cash_and_allocation_raw(
                encoded_u, pooled_u, list(selected_idx)
            )
            action = sample_cash_and_weights(
                k=k,
                selected_idx=selected_idx,
                order=order,
                log_p_k=log_p_k,
                log_p_sel=log_p_sel,
                cash_raw=cash_raw,
                allocation_raw=allocation_raw,
                cash_floor=self.config.cash_floor,
            )
        return action, float(value[0].item()), float(action.log_p_total)

    def evaluate_actions(
        self,
        states: list[CanonicalPPOState],
        actions: list[SampledAction],
        *,
        temporal_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One encode per microbatch; loop only over factored action metadata."""
        if len(states) != len(actions):
            raise ValueError("evaluate_actions requires aligned states and actions")
        device = next(self.parameters()).device
        history, features, globals_, mask = tensors_from_states(states, device)
        encoded, pooled = self.encode(
            history,
            features,
            globals_,
            mask,
            temporal_embeddings=temporal_embeddings,
        )
        count_logits, selection_logits, values = self.heads(encoded, pooled)
        log_probs: list[torch.Tensor] = []
        count_entropies: list[torch.Tensor] = []
        selection_entropies: list[torch.Tensor] = []
        for index, action in enumerate(actions):
            encoded_u = encoded[index]
            pooled_u = pooled[index]
            mask_u = mask[index]
            if action.k == 0:
                cash_raw = pooled_u.new_zeros(2)
                allocation_raw = encoded_u.new_zeros(encoded_u.size(0))
            else:
                cash_raw, allocation_raw = self.cash_and_allocation_raw(
                    encoded_u, pooled_u, list(action.selection_indices)
                )
            log_probs.append(
                recompute_action_log_prob_tensors(
                    action,
                    count_logits=count_logits[index],
                    selection_logits=selection_logits[index],
                    cash_raw=cash_raw,
                    allocation_raw=allocation_raw,
                    asset_mask=mask_u,
                    cash_floor=self.config.cash_floor,
                )
            )
            h_count, h_selection = factored_count_and_selection_entropy(
                count_logits=count_logits[index],
                selection_logits=selection_logits[index],
                asset_mask=mask_u,
                selection_indices=action.selection_indices,
                k=action.k,
            )
            count_entropies.append(h_count)
            selection_entropies.append(h_selection)
        return (
            torch.stack(log_probs),
            values,
            torch.stack(count_entropies),
            torch.stack(selection_entropies),
        )

    def log_prob(self, state: CanonicalPPOState, action: SampledAction) -> torch.Tensor:
        """Recompute the stored action's log probability under current params."""
        device = next(self.parameters()).device
        history, features, globals_, mask = tensors_from_state(state, device)
        encoded, pooled = self.encode(
            history.unsqueeze(0),
            features.unsqueeze(0),
            globals_.unsqueeze(0),
            mask.unsqueeze(0),
        )
        count_logits, selection_logits, _ = self.heads(encoded, pooled)
        encoded_u = encoded[0]
        pooled_u = pooled[0]
        if action.k == 0:
            cash_raw = pooled_u.new_zeros(2)
            allocation_raw = encoded_u.new_zeros(encoded_u.size(0))
        else:
            cash_raw, allocation_raw = self.cash_and_allocation_raw(
                encoded_u, pooled_u, list(action.selection_indices)
            )
        return recompute_action_log_prob_tensors(
            action,
            count_logits=count_logits[0],
            selection_logits=selection_logits[0],
            cash_raw=cash_raw,
            allocation_raw=allocation_raw,
            asset_mask=mask,
            cash_floor=self.config.cash_floor,
        )

    def value(self, state: CanonicalPPOState) -> torch.Tensor:
        device = next(self.parameters()).device
        history, features, globals_, mask = tensors_from_state(state, device)
        _, pooled = self.encode(
            history.unsqueeze(0),
            features.unsqueeze(0),
            globals_.unsqueeze(0),
            mask.unsqueeze(0),
        )
        return self.value_head(pooled).squeeze(-1)

    def infer_decision_value(
        self, state: CanonicalPPOState, force_k: int | None = None
    ) -> tuple[dict[str, float], tuple[str, ...], float]:
        """One encode: deterministic weights, selection order, and value."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            history, features, globals_, mask = tensors_from_state(state, device)
            encoded, pooled = self.encode(
                history.unsqueeze(0),
                features.unsqueeze(0),
                globals_.unsqueeze(0),
                mask.unsqueeze(0),
            )
            count_logits, selection_logits, value = self.heads(encoded, pooled)
            value_f = float(value[0].item())
            n_eligible = int(mask.sum().item())
            if force_k is not None:
                k = min(max(int(force_k), 0), n_eligible, MAX_SELECTED)
            else:
                k_values = torch.arange(count_logits.size(-1), device=device)
                valid = (k_values <= n_eligible) & (k_values <= MAX_SELECTED)
                masked_counts = count_logits[0].masked_fill(~valid, float("-inf"))
                k = int(torch.argmax(masked_counts).item())
            if k == 0:
                return {"CASH": 1.0}, (), value_f
            valid_indices = [index for index, flag in enumerate(mask.tolist()) if flag]
            ranked = sorted(
                valid_indices,
                key=lambda index: (
                    -float(selection_logits[0, index].item()),
                    state.symbols[index],
                ),
            )
            selected = ranked[:k]
            order = tuple(state.symbols[index] for index in selected)
            cash_raw, allocation_raw = self.cash_and_allocation_raw(
                encoded[0], pooled[0], selected
            )
            weights = deterministic_weights(
                count_logits=count_logits[0],
                selection_logits=selection_logits[0],
                cash_raw=cash_raw,
                allocation_raw=allocation_raw,
                asset_mask=mask,
                symbols=state.symbols,
                cash_floor=self.config.cash_floor,
                force_k=force_k,
            )
            return weights, order, value_f

    def infer_decision(
        self, state: CanonicalPPOState, force_k: int | None = None
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        """Deterministic weights plus selection-logit order (symbol tie-break)."""
        weights, order, _value = self.infer_decision_value(state, force_k=force_k)
        return weights, order

    def infer_weights(
        self, state: CanonicalPPOState, force_k: int | None = None
    ) -> dict[str, float]:
        """Deterministic inference action."""
        weights, _order = self.infer_decision(state, force_k=force_k)
        return weights

    def count_and_selection_entropy(
        self, state: CanonicalPPOState, action: SampledAction
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        history, features, globals_, mask = tensors_from_state(state, device)
        encoded, pooled = self.encode(
            history.unsqueeze(0),
            features.unsqueeze(0),
            globals_.unsqueeze(0),
            mask.unsqueeze(0),
        )
        count_logits, selection_logits, _ = self.heads(encoded, pooled)
        return factored_count_and_selection_entropy(
            count_logits=count_logits[0],
            selection_logits=selection_logits[0],
            asset_mask=mask,
            selection_indices=action.selection_indices,
            k=action.k,
        )

    def pretrain_forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict next-week open-to-open log return from the temporal encoder."""
        embeddings = self.temporal(history)
        return self.pretrain_head(embeddings).squeeze(-1)


def validate_inference_weights(
    weights: dict[str, float], eligible: set[str], cash_floor: float = CASH_FLOOR
) -> dict[str, float]:
    """Reject non-simplex or ineligible inference output."""
    if "CASH" not in weights:
        raise ValueError("inference weights must include CASH")
    stocks = [symbol for symbol in weights if symbol != "CASH"]
    if len(stocks) > MAX_SELECTED:
        raise ValueError("inference selected more than 15 stocks")
    if len(stocks) != len(set(stocks)):
        raise ValueError("inference selected duplicate symbols")
    for symbol in stocks:
        if symbol not in eligible:
            raise ValueError(f"ineligible symbol {symbol} in inference weights")
    values = [float(weights[symbol]) for symbol in weights]
    if any(
        not np.isfinite(value) or value < -1e-12 or value > 1 + 1e-12
        for value in values
    ):
        raise ValueError("inference weights must be finite and in [0, 1]")
    total = float(sum(max(value, 0.0) for value in values))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"inference weights sum to {total}, not 1")
    cleaned = {symbol: max(float(weight), 0.0) for symbol, weight in weights.items()}
    residue = 1.0 - sum(cleaned.values())
    cleaned["CASH"] += residue
    if cleaned["CASH"] < 0:
        raise ValueError("CASH weight became negative after residue correction")
    if len(stocks) > 0 and cleaned["CASH"] + 1e-9 < cash_floor:
        raise ValueError("CASH weight below the 2% floor")
    return cleaned


__all__ = [
    "PPODiscoveryActorCritic",
    "tensors_from_state",
    "tensors_from_states",
    "validate_inference_weights",
]
