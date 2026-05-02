"""SAC promotion guardrails (always-promote-with-guardrails policy).

SAC keeps an absolute CAGR floor on the new model's own backtest
(``SAC_PROMOTION_CAGR_FLOOR = 0.12``); unlike the forecaster gate,
this is **not** subject to universe-drift confounding because eval
is computed on the same fresh universe SAC will trade.

Per AGENTS.md rule #2 (math correctness over DRY), SAC checks live
in this module rather than being merged with the forecaster checks
even though both return :class:`ArtifactHealthCheck`.
"""

import math
from pathlib import Path

from brain_api.core.training_health import ArtifactHealthCheck

SAC_PROMOTION_CAGR_FLOOR: float = 0.12
"""Minimum eval CAGR (decimal, not percent) for SAC promotion.

Renamed from ``MIN_PROMOTION_CAGR`` (kept the same numeric value) so
the new name reflects the policy: this is a guardrail floor, not a
prior-comparison gate. SAC is allowed to ship a model worse than the
prior as long as its own backtest clears 12% CAGR.
"""

# All artifact files SACLocalStorage.write_artifacts persists. Drifting
# from this list will cause file-existence guardrails to silently pass
# when they shouldn't, so the constant lives next to the function.
_SAC_REQUIRED_FILES: tuple[str, ...] = (
    "actor.pt",
    "critic.pt",
    "critic_target.pt",
    "log_alpha.pt",
    "scaler.pkl",
    "config.json",
    "symbol_order.json",
    "metadata.json",
)


def _check_finite_metric(value: float, name: str, reasons: list[str]) -> None:
    """Append a reason when ``value`` is NaN/Inf."""
    if not math.isfinite(value):
        reasons.append(f"{name} is not finite")


def _check_artifact_files(artifact_dir: Path, reasons: list[str]) -> None:
    """Append one reason per missing or zero-byte SAC artifact file."""
    for filename in _SAC_REQUIRED_FILES:
        path = artifact_dir / filename
        if not path.exists() or path.stat().st_size <= 0:
            reasons.append(f"{filename} missing or zero bytes")


def evaluate_sac_artifact_health(
    *,
    actor_loss: float,
    critic_loss: float,
    eval_cagr: float,
    eval_sharpe: float,
    eval_max_drawdown: float,
    expected_symbol_count: int,
    actual_symbol_count: int,
    artifact_dir: Path,
) -> ArtifactHealthCheck:
    """Run SAC full-training guardrails.

    Replaces the prior ``promoted = prior_version is None or
    result.eval_cagr > MIN_PROMOTION_CAGR`` gate. The new policy
    drops the inaugural special-case (an inaugural that fails
    guardrails should NOT be promoted just to populate HF main --
    that was a silent fallback per AGENTS.md rule #1) and adds
    finite-metric + artifact-existence + symbol-count guardrails.

    Guardrails (each failure appends a stable, human-readable string):

    1. ``eval_cagr`` finite AND ``> SAC_PROMOTION_CAGR_FLOOR``
    2. ``eval_sharpe`` finite
    3. ``eval_max_drawdown`` finite
    4. ``actor_loss`` finite
    5. ``critic_loss`` finite
    6. ``actual_symbol_count == expected_symbol_count`` (action-space
       dimension must match the bucket's symbol resolver)
    7-14. Each of the eight SAC artifact files
       (actor.pt, critic.pt, critic_target.pt, log_alpha.pt, scaler.pkl,
       config.json, symbol_order.json, metadata.json) exists with
       non-zero size

    Returns:
        :class:`ArtifactHealthCheck` whose ``is_healthy`` is the new
        promotion decision.
    """
    failure_reasons: list[str] = []

    if not math.isfinite(eval_cagr):
        failure_reasons.append("eval_cagr is not finite")
    elif eval_cagr <= SAC_PROMOTION_CAGR_FLOOR:
        failure_reasons.append(
            f"eval_cagr {eval_cagr:.4f} below floor {SAC_PROMOTION_CAGR_FLOOR}"
        )

    _check_finite_metric(eval_sharpe, "eval_sharpe", failure_reasons)
    _check_finite_metric(eval_max_drawdown, "eval_max_drawdown", failure_reasons)
    _check_finite_metric(actor_loss, "actor_loss", failure_reasons)
    _check_finite_metric(critic_loss, "critic_loss", failure_reasons)

    if actual_symbol_count != expected_symbol_count:
        failure_reasons.append(
            f"actual_symbol_count {actual_symbol_count} does not match "
            f"expected_symbol_count {expected_symbol_count}"
        )

    _check_artifact_files(artifact_dir, failure_reasons)

    return ArtifactHealthCheck(
        is_healthy=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def evaluate_sac_finetune_artifact_health(
    *,
    actor_loss: float,
    critic_loss: float,
    eval_cagr: float,
    eval_sharpe: float,
    eval_max_drawdown: float,
    prior_symbol_order: list[str],
    actual_symbol_order: list[str],
    artifact_dir: Path,
) -> ArtifactHealthCheck:
    """Run SAC finetune guardrails.

    Same finite-metric + CAGR-floor + artifact-existence guardrails as
    :func:`evaluate_sac_artifact_health`, with the symbol-count check
    replaced by a stricter symbol-ORDER equality check: SAC's actor /
    critic action-space dimension is positional, so a finetune that
    drops a delisted symbol or reorders the slate would silently
    misalign the action distribution from the prior model's weights.

    Guardrails (each failure appends a stable, human-readable string):

    1. ``eval_cagr`` finite AND ``> SAC_PROMOTION_CAGR_FLOOR``
    2. ``eval_sharpe`` finite
    3. ``eval_max_drawdown`` finite
    4. ``actor_loss`` finite
    5. ``critic_loss`` finite
    6. ``actual_symbol_order == prior_symbol_order`` (list equality
       INCLUDING ordering, not just set equality)
    7-14. Same eight SAC artifact files as full training

    Returns:
        :class:`ArtifactHealthCheck` whose ``is_healthy`` is the new
        promotion decision.
    """
    failure_reasons: list[str] = []

    if not math.isfinite(eval_cagr):
        failure_reasons.append("eval_cagr is not finite")
    elif eval_cagr <= SAC_PROMOTION_CAGR_FLOOR:
        failure_reasons.append(
            f"eval_cagr {eval_cagr:.4f} below floor {SAC_PROMOTION_CAGR_FLOOR}"
        )

    _check_finite_metric(eval_sharpe, "eval_sharpe", failure_reasons)
    _check_finite_metric(eval_max_drawdown, "eval_max_drawdown", failure_reasons)
    _check_finite_metric(actor_loss, "actor_loss", failure_reasons)
    _check_finite_metric(critic_loss, "critic_loss", failure_reasons)

    if list(actual_symbol_order) != list(prior_symbol_order):
        failure_reasons.append(
            f"actual_symbol_order {actual_symbol_order} does not match "
            f"prior_symbol_order {prior_symbol_order} (finetune action "
            "space is positional; symbols must be identical and in the "
            "same order)"
        )

    _check_artifact_files(artifact_dir, failure_reasons)

    return ArtifactHealthCheck(
        is_healthy=not failure_reasons,
        failure_reasons=failure_reasons,
    )


__all__ = [
    "SAC_PROMOTION_CAGR_FLOOR",
    "evaluate_sac_artifact_health",
    "evaluate_sac_finetune_artifact_health",
]
