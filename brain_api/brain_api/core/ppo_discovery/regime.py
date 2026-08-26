"""Train-fold HMM fit and causal live continuation for ppo_discovery.

Reuses the SAC three-state diagonal Gaussian HMM. PPO never copies
frozen ``p_calm`` / ``p_stress`` into live state.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.sac.market_history import extract_aligned_market_history
from brain_api.core.sac.regime_hmm import (
    RegimeHMMArtifact,
    causal_filter,
    continue_live_filter,
    fit_regime_hmm,
    live_market_observations,
    market_observations,
    regime_probabilities,
)


def fit_ppo_regime_hmm(
    prices: dict[str, pd.DataFrame],
    *,
    start_date: date,
    cutoff: date,
) -> RegimeHMMArtifact:
    """Fit on train-fold SPY/VIX through ``cutoff`` only."""
    market_dates, spy, vix = extract_aligned_market_history(
        prices, start_date=start_date, completed_through=cutoff
    )
    observations = market_observations(list(spy), list(vix))
    observation_dates = market_dates[20:]
    if len(observation_dates) < 3:
        raise PPODiscoveryError("insufficient train-fold HMM observations")
    tail_indices = [
        index for index, value in enumerate(market_dates) if value <= cutoff
    ][-21:]
    if len(tail_indices) != 21:
        raise PPODiscoveryError("HMM requires 21 market-tail sessions at cutoff")
    return fit_regime_hmm(
        observations,
        observation_dates,
        spy_tail=spy[tail_indices],
        vix_tail=vix[tail_indices],
        tail_dates=[market_dates[index] for index in tail_indices],
    )


def weekly_regime_probabilities(
    artifact: RegimeHMMArtifact,
    prices: dict[str, pd.DataFrame],
    *,
    start_date: date,
    completed_through: date,
    weekly_cutoffs: list[date],
) -> dict[date, tuple[float, float]]:
    """Causal ``(p_calm, p_stress)`` at each weekly cutoff."""
    market_dates, spy, vix = extract_aligned_market_history(
        prices, start_date=start_date, completed_through=completed_through
    )
    observations = market_observations(list(spy), list(vix))
    observation_dates = market_dates[20:]
    train_count = sum(
        value <= artifact.training_cutoff_date for value in observation_dates
    )
    train_posts = causal_filter(observations[:train_count], artifact)
    if train_count < len(observations):
        live_posts = causal_filter(
            observations[train_count:],
            artifact,
            artifact.terminal_posterior,
        )
        all_posts = np.vstack((train_posts, live_posts))
    else:
        all_posts = train_posts
    result: dict[date, tuple[float, float]] = {}
    for cutoff in weekly_cutoffs:
        posterior_index = (
            int(np.searchsorted(observation_dates, cutoff, side="right")) - 1
        )
        if posterior_index < 0:
            raise PPODiscoveryError(f"no causal HMM posterior for {cutoff.isoformat()}")
        result[cutoff] = regime_probabilities(all_posts[posterior_index], artifact)
    return result


def live_regime_probabilities(
    artifact_payload: dict[str, Any],
    *,
    spy_vix_rows: list[dict[str, Any]],
    decision_date: date,
) -> tuple[float, float]:
    """Continue the persisted posterior through post-cutoff SPY/VIX."""
    try:
        artifact = RegimeHMMArtifact.from_dict(artifact_payload)
    except ValueError as exc:
        raise PPODiscoveryError(
            f"invalid ppo_discovery regime_hmm artifact: {exc}"
        ) from exc
    observations, observation_dates = live_market_observations(
        artifact, spy_vix_rows, decision_date
    )
    posterior = continue_live_filter(
        artifact, observations, observation_dates, decision_date
    )
    return regime_probabilities(posterior, artifact)


def spy_vix_rows_after_cutoff(
    prices: dict[str, pd.DataFrame],
    *,
    cutoff: date,
    decision_date: date,
) -> list[dict[str, Any]]:
    """Raw post-cutoff rows for ``live_market_observations``. Empty is valid."""
    from brain_api.core.sac.market_sessions import completed_xnys_session_dates

    expected = completed_xnys_session_dates(cutoff + timedelta(days=1), decision_date)
    if not expected:
        return []
    spy = prices.get("SPY")
    vix = prices.get("^VIX")
    if spy is None or vix is None:
        raise PPODiscoveryError("live HMM requires SPY and ^VIX frames")

    def _close_on(frame: pd.DataFrame, session: date) -> float:
        index = frame.index
        if index.tz is not None:
            index = index.tz_localize(None)
        normalized = pd.DatetimeIndex(index).normalize()
        matches = frame.loc[normalized.date == session]
        if matches.empty or "close" not in matches.columns:
            raise PPODiscoveryError(f"missing close on {session.isoformat()}")
        value = float(matches.iloc[-1]["close"])
        if not np.isfinite(value) or value <= 0:
            raise PPODiscoveryError(f"non-positive close on {session.isoformat()}")
        return value

    return [
        {
            "date": session.isoformat(),
            "spy_adjusted_close": _close_on(spy, session),
            "vix_close": _close_on(vix, session),
        }
        for session in expected
    ]


__all__ = [
    "fit_ppo_regime_hmm",
    "live_regime_probabilities",
    "spy_vix_rows_after_cutoff",
    "weekly_regime_probabilities",
]
