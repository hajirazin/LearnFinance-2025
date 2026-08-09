"""SAC promotion policy.

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
prior as long as its own backtest reaches 12% CAGR.
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
    """Apply the sole full-training promotion rule: net eval CAGR >= 12%.

    Losses, Sharpe, drawdown, prior-model performance, baseline performance,
    symbol count, and artifact presence do not participate in the product
    promotion decision. They remain available for reporting and operational
    diagnostics. Non-finite CAGR is rejected because it cannot establish that
    the absolute floor was met.
    """
    del (
        actor_loss,
        critic_loss,
        eval_sharpe,
        eval_max_drawdown,
        expected_symbol_count,
        actual_symbol_count,
        artifact_dir,
    )
    failure_reasons = []
    if not math.isfinite(eval_cagr):
        failure_reasons.append("eval_cagr is not finite")
    elif eval_cagr < SAC_PROMOTION_CAGR_FLOOR:
        failure_reasons.append(
            f"eval_cagr {eval_cagr:.4f} below floor {SAC_PROMOTION_CAGR_FLOOR}"
        )

    return ArtifactHealthCheck(
        is_healthy=not failure_reasons,
        failure_reasons=failure_reasons,
    )


__all__ = [
    "SAC_PROMOTION_CAGR_FLOOR",
    "evaluate_sac_artifact_health",
]
