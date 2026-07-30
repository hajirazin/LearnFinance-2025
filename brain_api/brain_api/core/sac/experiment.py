"""Fixed-seed SAC training experiment policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from brain_api.core.sac.config import SACConfig

SAC_EXPERIMENT_SEEDS = (42, 123, 2026)
T = TypeVar("T")


@dataclass(frozen=True)
class SACCandidate(Generic[T]):
    """One independently trained fixed-seed SAC candidate."""

    seed: int
    eval_cagr: float
    result: T


@dataclass(frozen=True)
class SACTrainingExperiment(Generic[T]):
    """Train exactly three seeds and select the median-CAGR candidate."""

    candidates: tuple[SACCandidate[T], ...]
    selected: SACCandidate[T]

    @classmethod
    def run(
        cls,
        *,
        config: SACConfig,
        train_candidate: Callable[[SACConfig], T],
        cagr_of: Callable[[T], float],
    ) -> SACTrainingExperiment[T]:
        """Run seeds 42, 123, and 2026 from fresh copied configurations."""
        candidates_list: list[SACCandidate[T]] = []
        for seed in SAC_EXPERIMENT_SEEDS:
            result = train_candidate(replace(config, seed=seed))
            candidates_list.append(
                SACCandidate(
                    seed=seed,
                    eval_cagr=float(cagr_of(result)),
                    result=result,
                )
            )
        candidates = tuple(candidates_list)
        selected = sorted(candidates, key=lambda item: (item.eval_cagr, item.seed))[1]
        return cls(candidates=candidates, selected=selected)
