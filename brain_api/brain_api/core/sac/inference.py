"""Stateless SAC v3 inference from a canonical raw decision context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brain_api.core.portfolio_rl.constraints import (
    apply_softmax_to_weights,
    compute_turnover,
    enforce_constraints,
)
from brain_api.core.portfolio_rl.state import MAX_ASSETS, build_state_vector
from brain_api.core.sac.decision_context import SACDecisionContext
from brain_api.core.sac.regime_hmm import (
    RegimeHMMArtifact,
    continue_live_filter,
    live_market_observations,
    regime_probabilities,
)


@dataclass
class SACInferenceResult:
    allocation: dict[str, float]
    turnover: float
    model_version: str
    raw_action: np.ndarray
    state_vector: np.ndarray
    asset_mask: np.ndarray
    regime_posterior: np.ndarray


def run_sac_inference(
    actor,
    scaler,
    config,
    decision_context: SACDecisionContext,
    regime_hmm: RegimeHMMArtifact,
    model_version: str,
) -> SACInferenceResult:
    """Build v3 state, causally advance its HMM, and allocate over valid slots."""
    bundle = decision_context.feature_bundle
    current_weights = decision_context.weight_array()
    asset_mask, signals, forecasts = bundle.eligible_inputs(
        decision_context.current_weights, production=True
    )
    market_observation_rows, market_dates = live_market_observations(
        regime_hmm, list(bundle.market_history), decision_context.as_of_date
    )
    posterior = (
        regime_hmm.terminal_posterior.copy()
        if len(market_observation_rows) == 0
        else continue_live_filter(
            regime_hmm,
            market_observation_rows,
            market_dates,
            decision_context.as_of_date,
        )
    )
    state = build_state_vector(
        signals=signals,
        patchtst_forecasts=forecasts,
        portfolio_weights=current_weights,
        symbol_order=list(bundle.symbols),
        asset_mask=asset_mask,
        regime_probabilities=regime_probabilities(posterior, regime_hmm),
    )
    normalized = scaler.transform(state)
    with torch.no_grad():
        action = actor.get_action(normalized, deterministic=True)
    raw_weights = apply_softmax_to_weights(action, asset_mask)
    weights = enforce_constraints(raw_weights, cash_buffer=config.cash_buffer)
    # Constraint enforcement cannot resurrect masked zeros because it only
    # rescales stock weights proportionally.
    if np.any(weights[:MAX_ASSETS][~asset_mask] != 0.0):
        raise RuntimeError("SAC v3 constraint processing allocated to a padded asset")
    turnover = compute_turnover(current_weights, weights)
    allocation = {
        symbol: float(weights[index]) for index, symbol in enumerate(bundle.symbols)
    }
    allocation["CASH"] = float(weights[-1])
    return SACInferenceResult(
        allocation,
        float(turnover),
        model_version,
        action,
        state,
        asset_mask,
        posterior,
    )
