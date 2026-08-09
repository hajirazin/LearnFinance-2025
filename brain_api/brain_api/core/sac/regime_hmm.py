"""Deterministic three-state diagonal Gaussian HMM for SAC market regime inputs.

``HMM_SEED`` is part of the frozen artifact config contract. The current EM
initialization is closed-form (volatility tertiles + fixed sticky transitions)
and does not draw from a PRNG; the seed documents the reproducible recipe and
is validated on load so future stochastic inits cannot silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from scipy.special import logsumexp

from brain_api.core.sac.market_sessions import (
    completed_xnys_session_dates,
    require_exact_session_dates,
)
from brain_api.core.sac.momentum_signals import compute_realized_vol_20d

N_STATES = 3
N_OBSERVATIONS = 4
HMM_SEED = 42
HMM_MAX_ITERATIONS = 200
HMM_TOLERANCE = 1e-4
MIN_VARIANCE = 1e-4


@dataclass(frozen=True)
class RegimeHMMArtifact:
    """Complete fitted state required for causal stateless inference."""

    start_probability: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    label_map: dict[str, int]
    terminal_posterior: np.ndarray
    training_cutoff_date: date
    fit_start_date: date
    iterations: int
    log_likelihood: float
    spy_tail: np.ndarray
    vix_tail: np.ndarray
    tail_dates: tuple[date, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "config": {
                "n_states": N_STATES,
                "covariance": "diag",
                "seed": HMM_SEED,
                "max_iterations": HMM_MAX_ITERATIONS,
                "tolerance": HMM_TOLERANCE,
            },
            "start_probability": self.start_probability.tolist(),
            "transition": self.transition.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "label_map": self.label_map,
            "terminal_posterior": self.terminal_posterior.tolist(),
            "training_cutoff_date": self.training_cutoff_date.isoformat(),
            "fit_start_date": self.fit_start_date.isoformat(),
            "iterations": self.iterations,
            "log_likelihood": self.log_likelihood,
            "spy_tail": self.spy_tail.tolist(),
            "vix_tail": self.vix_tail.tolist(),
            "tail_dates": [value.isoformat() for value in self.tail_dates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeHMMArtifact:
        if data.get("schema_version") != 3:
            raise ValueError("legacy or missing SAC v3 HMM metadata")
        config = data.get("config") or {}
        if int(config.get("seed", HMM_SEED)) != HMM_SEED:
            raise ValueError(
                f"SAC v3 HMM artifact seed must be {HMM_SEED}, got {config.get('seed')}"
            )
        artifact = cls(
            start_probability=np.asarray(data["start_probability"], dtype=float),
            transition=np.asarray(data["transition"], dtype=float),
            means=np.asarray(data["means"], dtype=float),
            variances=np.asarray(data["variances"], dtype=float),
            scaler_mean=np.asarray(data["scaler_mean"], dtype=float),
            scaler_scale=np.asarray(data["scaler_scale"], dtype=float),
            label_map={key: int(value) for key, value in data["label_map"].items()},
            terminal_posterior=np.asarray(data["terminal_posterior"], dtype=float),
            training_cutoff_date=date.fromisoformat(data["training_cutoff_date"]),
            fit_start_date=date.fromisoformat(data["fit_start_date"]),
            iterations=int(data["iterations"]),
            log_likelihood=float(data["log_likelihood"]),
            spy_tail=np.asarray(data["spy_tail"], dtype=float),
            vix_tail=np.asarray(data["vix_tail"], dtype=float),
            tail_dates=tuple(date.fromisoformat(value) for value in data["tail_dates"]),
        )
        _validate_artifact(artifact)
        return artifact


def _validate_observations(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_OBSERVATIONS or len(values) < 1:
        raise ValueError("HMM observations must have shape (n>=1, 4)")
    if not np.all(np.isfinite(values)) or np.any(values[:, 2] <= 0):
        raise ValueError("HMM observations must be finite with positive VIX")
    return values


def _log_emissions(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    return -0.5 * np.sum(
        np.log(2 * np.pi * variances)[None, :, :]
        + (observations[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :],
        axis=2,
    )


def _forward_backward(
    observations: np.ndarray,
    start: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    emissions = _log_emissions(observations, means, variances)
    alpha = np.empty((len(observations), N_STATES))
    alpha[0] = np.log(start) + emissions[0]
    log_transition = np.log(transition)
    for index in range(1, len(observations)):
        alpha[index] = emissions[index] + logsumexp(
            alpha[index - 1][:, None] + log_transition, axis=0
        )
    likelihood = float(logsumexp(alpha[-1]))
    beta = np.zeros_like(alpha)
    for index in range(len(observations) - 2, -1, -1):
        beta[index] = logsumexp(
            log_transition + emissions[index + 1][None, :] + beta[index + 1][None, :],
            axis=1,
        )
    gamma = np.exp(alpha + beta - likelihood)
    gamma /= gamma.sum(axis=1, keepdims=True)
    counts = np.zeros((N_STATES, N_STATES))
    for index in range(len(observations) - 1):
        counts += np.exp(
            alpha[index][:, None]
            + log_transition
            + emissions[index + 1][None, :]
            + beta[index + 1][None, :]
            - likelihood
        )
    return likelihood, gamma, counts


def causal_filter(
    observations: np.ndarray,
    artifact: RegimeHMMArtifact,
    initial_posterior: np.ndarray | None = None,
) -> np.ndarray:
    """Return forward-only posteriors; future rows cannot change a prefix."""
    raw = _validate_observations(observations)
    scaled = (raw - artifact.scaler_mean) / artifact.scaler_scale
    emissions = _log_emissions(scaled, artifact.means, artifact.variances)
    continuing = initial_posterior is not None
    posterior = np.asarray(
        artifact.start_probability if not continuing else initial_posterior,
        dtype=np.float64,
    ).copy()
    if posterior.shape != (N_STATES,) or np.any(posterior < 0):
        raise ValueError("initial posterior must be a nonnegative three-vector")
    posterior /= posterior.sum()
    rows = []
    for index, log_emission in enumerate(emissions):
        # A fresh sequence applies its first emission to the fitted initial
        # distribution. A continuation starts after an already-filtered
        # terminal posterior and therefore predicts through one transition.
        predicted = (
            posterior @ artifact.transition if continuing or index > 0 else posterior
        )
        emission = np.exp(log_emission - np.max(log_emission))
        posterior = predicted * emission
        posterior /= posterior.sum()
        rows.append(posterior.copy())
    return np.asarray(rows)


def fit_regime_hmm(
    observations: np.ndarray,
    observation_dates: list[date],
    *,
    spy_tail: np.ndarray,
    vix_tail: np.ndarray,
    tail_dates: list[date],
) -> RegimeHMMArtifact:
    """Fit scaler and HMM using the supplied training fold only."""
    raw = _validate_observations(observations)
    if len(raw) < 3:
        raise ValueError("HMM fitting requires at least three observations")
    if len(observation_dates) != len(raw) or observation_dates != sorted(
        observation_dates
    ):
        raise ValueError("HMM observation dates must be complete and ordered")
    scaler_mean = raw.mean(axis=0)
    scaler_scale = raw.std(axis=0, ddof=1)
    if np.any(~np.isfinite(scaler_scale)) or np.any(scaler_scale <= 0):
        raise ValueError("HMM training scaler is degenerate")
    values = (raw - scaler_mean) / scaler_scale
    volatility_order = np.argsort(values[:, 1], kind="stable")
    labels = np.empty(len(values), dtype=int)
    for state, indices in enumerate(np.array_split(volatility_order, N_STATES)):
        labels[indices] = state
    means = np.asarray([values[labels == state].mean(0) for state in range(N_STATES)])
    variances = np.asarray(
        [values[labels == state].var(0) + MIN_VARIANCE for state in range(N_STATES)]
    )
    start = np.full(N_STATES, 1 / N_STATES)
    transition = np.full((N_STATES, N_STATES), 0.05)
    np.fill_diagonal(transition, 0.90)
    previous = -np.inf
    for iteration in range(1, HMM_MAX_ITERATIONS + 1):
        likelihood, gamma, counts = _forward_backward(
            values, start, transition, means, variances
        )
        start = np.maximum(gamma[0], 1e-8)
        start /= start.sum()
        transition = np.maximum(counts, 1e-8)
        transition /= transition.sum(axis=1, keepdims=True)
        mass = gamma.sum(axis=0)[:, None]
        means = np.einsum("ts,tf->sf", gamma, values) / mass
        centered = values[:, None, :] - means[None, :, :]
        variances = np.sum(gamma[:, :, None] * centered**2, axis=0) / mass
        variances = np.maximum(variances, MIN_VARIANCE)
        if likelihood < previous - 1e-6:
            raise ValueError("HMM likelihood decreased during deterministic fit")
        if iteration > 1 and abs(likelihood - previous) < HMM_TOLERANCE:
            original_vol_means = means[:, 1] * scaler_scale[1] + scaler_mean[1]
            ordering = sorted(
                range(N_STATES), key=lambda item: (original_vol_means[item], item)
            )
            label_map = {
                "calm": ordering[0],
                "transition": ordering[1],
                "stress": ordering[2],
            }
            tail_spy = np.asarray(spy_tail, dtype=float)
            tail_vix = np.asarray(vix_tail, dtype=float)
            tail_date_values = tuple(tail_dates)
            if not (len(tail_spy) == len(tail_vix) == len(tail_date_values) == 21):
                raise ValueError(
                    "HMM artifact requires exactly 21 aligned market-tail rows"
                )
            if (
                tail_date_values != tuple(sorted(set(tail_date_values)))
                or tail_date_values[-1] != observation_dates[-1]
            ):
                raise ValueError(
                    "HMM market tail must be ordered through the training cutoff"
                )
            if (
                not np.all(np.isfinite(tail_spy))
                or not np.all(np.isfinite(tail_vix))
                or np.any(tail_spy <= 0)
                or np.any(tail_vix <= 0)
            ):
                raise ValueError("HMM market tail values must be finite and positive")
            provisional = RegimeHMMArtifact(
                start,
                transition,
                means,
                variances,
                scaler_mean,
                scaler_scale,
                label_map,
                np.full(N_STATES, 1 / N_STATES),
                observation_dates[-1],
                observation_dates[0],
                iteration,
                likelihood,
                tail_spy,
                tail_vix,
                tail_date_values,
            )
            terminal = causal_filter(raw, provisional)[-1]
            return RegimeHMMArtifact(
                **{**provisional.__dict__, "terminal_posterior": terminal}
            )
        previous = likelihood
    raise ValueError("HMM failed to converge within 200 iterations")


def market_observations(
    spy_adjusted_closes: list[float], vix_closes: list[float]
) -> np.ndarray:
    """Build four market-only observations from aligned daily closes."""
    spy = np.asarray(spy_adjusted_closes, dtype=np.float64)
    vix = np.asarray(vix_closes, dtype=np.float64)
    if spy.shape != vix.shape or spy.ndim != 1 or len(spy) < 21:
        raise ValueError("aligned SPY/VIX history requires at least 21 rows")
    if (
        not np.all(np.isfinite(spy))
        or not np.all(np.isfinite(vix))
        or np.any(spy <= 0)
        or np.any(vix <= 0)
    ):
        raise ValueError("SPY/VIX closes must be finite and positive")
    rows = []
    for index in range(20, len(spy)):
        rows.append(
            (
                spy[index] / spy[index - 20] - 1.0,
                compute_realized_vol_20d(spy, as_of_index=index),
                vix[index],
                vix[index] / vix[index - 5] - 1.0,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def continue_live_filter(
    artifact: RegimeHMMArtifact,
    observations: np.ndarray,
    observation_dates: list[date],
    decision_date: date,
) -> np.ndarray:
    """Validate post-cutoff chronology and continue the persisted posterior."""
    if len(observation_dates) != len(observations):
        raise ValueError("market history dates and observations must align")
    expected = completed_xnys_session_dates(
        artifact.training_cutoff_date + timedelta(days=1), decision_date
    )
    require_exact_session_dates(
        observation_dates, expected, context="post-cutoff market history"
    )
    if not observation_dates:
        return artifact.terminal_posterior.copy()
    return causal_filter(observations, artifact, artifact.terminal_posterior)[-1]


def live_market_observations(
    artifact: RegimeHMMArtifact,
    rows: list[dict[str, Any]],
    decision_date: date,
) -> tuple[np.ndarray, list[date]]:
    """Build post-cutoff observations using the persisted 20-session tail."""
    dates = [date.fromisoformat(str(row["date"])) for row in rows]
    expected = completed_xnys_session_dates(
        artifact.training_cutoff_date + timedelta(days=1), decision_date
    )
    require_exact_session_dates(dates, expected, context="post-cutoff market history")
    if not rows:
        return np.empty((0, N_OBSERVATIONS)), []
    if len(artifact.spy_tail) != 21 or len(artifact.vix_tail) != 21:
        raise ValueError("SAC v3 HMM artifact lacks the required market tail")
    spy_new = np.asarray([row["spy_adjusted_close"] for row in rows], dtype=float)
    vix_new = np.asarray([row["vix_close"] for row in rows], dtype=float)
    combined_spy = np.concatenate((artifact.spy_tail, spy_new))
    combined_vix = np.concatenate((artifact.vix_tail, vix_new))
    observations = market_observations(combined_spy, combined_vix)
    return observations[-len(rows) :], dates


def regime_probabilities(
    posterior: np.ndarray, artifact: RegimeHMMArtifact
) -> tuple[float, float]:
    values = np.asarray(posterior, dtype=float)
    return float(values[artifact.label_map["calm"]]), float(
        values[artifact.label_map["stress"]]
    )


def _validate_artifact(artifact: RegimeHMMArtifact) -> None:
    if artifact.transition.shape != (N_STATES, N_STATES) or artifact.means.shape != (
        N_STATES,
        N_OBSERVATIONS,
    ):
        raise ValueError("invalid SAC v3 HMM artifact shapes")
    if artifact.variances.shape != artifact.means.shape or np.any(
        artifact.variances < MIN_VARIANCE
    ):
        raise ValueError("invalid SAC v3 HMM variances")
    if set(artifact.label_map) != {"calm", "transition", "stress"}:
        raise ValueError("invalid SAC v3 HMM label map")
    if not (
        len(artifact.spy_tail)
        == len(artifact.vix_tail)
        == len(artifact.tail_dates)
        == 21
    ):
        raise ValueError("invalid SAC v3 HMM market tail")
