"""Walk-forward splits with a 21-day embargo for ppo_discovery."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from brain_api.core.ppo_discovery.config import PPODiscoveryConfig
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError

PURGE_DAYS = 21
FULL_VARIANT = "full"
DIAGNOSTIC_VARIANT = "diagnostic"


def _purge_before(
    weeks: Sequence[WeeklyTransition],
    next_start,
    *,
    purge_days: int = PURGE_DAYS,
) -> list[WeeklyTransition]:
    boundary = next_start - timedelta(days=purge_days)
    return [week for week in weeks if week.cutoff <= boundary]


def split_walk_forward(
    transitions: Sequence[WeeklyTransition],
    *,
    experiment_variant: str,
) -> tuple[list[WeeklyTransition], list[WeeklyTransition], list[WeeklyTransition]]:
    """Return ``(train, val, test)``.

    ``full``: last 20% test, 21-day purge, last 20% of remainder val,
    21-day purge, rest train. Raises if either purge is empty.
    ``diagnostic``: disjoint 60/20/20 with no purge.
    """
    if len(transitions) < 5:
        raise PPODiscoveryError("need at least five weekly transitions to split")
    ordered = list(transitions)
    test_n = max(1, len(ordered) // 5)
    test = ordered[-test_n:]
    remainder = ordered[:-test_n]
    if experiment_variant == DIAGNOSTIC_VARIANT:
        val_n = max(1, len(remainder) // 5)
        val = remainder[-val_n:]
        train = remainder[:-val_n]
        if not train or not val or not test:
            raise PPODiscoveryError("diagnostic split produced an empty fold")
        return train, val, test
    if experiment_variant != FULL_VARIANT:
        raise PPODiscoveryError(
            f"unknown experiment_variant {experiment_variant!r} for splits"
        )
    purged_remainder = _purge_before(remainder, test[0].cutoff)
    if not purged_remainder:
        raise PPODiscoveryError("21-day test purge emptied train/val remainder")
    val_n = max(1, len(purged_remainder) // 5)
    val = purged_remainder[-val_n:]
    pre_val = purged_remainder[:-val_n]
    train = _purge_before(pre_val, val[0].cutoff)
    if not train:
        raise PPODiscoveryError("21-day val purge emptied the train fold")
    if not val:
        raise PPODiscoveryError("21-day test purge emptied the val fold")
    return train, val, test


def is_locked_full_training(
    config: PPODiscoveryConfig,
    *,
    skip_supervised_pretraining: bool = False,
    freeze_encoder: bool = False,
) -> bool:
    """True only for the locked seeds, 10_000 timesteps, and both stages."""
    locked = PPODiscoveryConfig()
    return (
        tuple(config.seeds) == tuple(locked.seeds)
        and int(config.total_timesteps) == int(locked.total_timesteps)
        and not skip_supervised_pretraining
        and not freeze_encoder
    )


def resolve_experiment_variant(
    config: PPODiscoveryConfig,
    *,
    skip_supervised_pretraining: bool = False,
    freeze_encoder: bool = False,
) -> str:
    """``full`` is derived from locked defaults, never from a request label."""
    if is_locked_full_training(
        config,
        skip_supervised_pretraining=skip_supervised_pretraining,
        freeze_encoder=freeze_encoder,
    ):
        return FULL_VARIANT
    return DIAGNOSTIC_VARIANT


__all__ = [
    "DIAGNOSTIC_VARIANT",
    "FULL_VARIANT",
    "PURGE_DAYS",
    "is_locked_full_training",
    "resolve_experiment_variant",
    "split_walk_forward",
]
