"""Domain types for the always-promote-with-guardrails promotion model.

Both forecaster and SAC training routes evaluate model artifact health
post-training and emit one of these checks. The dataclass is shared;
the per-model evaluation functions live in their own modules so SAC
math stays isolated from forecaster math (AGENTS.md rule #2).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactHealthCheck:
    """Result of a per-model post-training health evaluation.

    The promotion decision is exactly ``is_healthy``: when the model
    passes all guardrails it gets promoted to ``current`` (local +
    HF main); when any guardrail fails it stays as a version branch
    only and the operator is informed via ``failure_reasons``.

    Attributes:
        is_healthy: Promotion outcome. ``True`` iff every guardrail
            evaluated against the artifact passed.
        failure_reasons: Human-readable strings, one per failing
            guardrail, in the order the checks were applied. Always
            empty when ``is_healthy`` is ``True``; always non-empty
            when ``is_healthy`` is ``False``.
    """

    is_healthy: bool
    failure_reasons: list[str]

    def __post_init__(self) -> None:
        # Invariant: is_healthy <=> empty failure_reasons. Enforced at
        # construction so downstream code can treat the two as
        # interchangeable proofs of the same condition.
        if self.is_healthy and self.failure_reasons:
            raise ValueError(
                "ArtifactHealthCheck inconsistent: is_healthy=True but "
                f"failure_reasons={self.failure_reasons!r}"
            )
        if not self.is_healthy and not self.failure_reasons:
            raise ValueError(
                "ArtifactHealthCheck inconsistent: is_healthy=False but "
                "failure_reasons is empty"
            )


__all__ = ["ArtifactHealthCheck"]
