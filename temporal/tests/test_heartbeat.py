"""Sidecar heartbeat helper for long-blocking sync activities."""

from __future__ import annotations

import inspect

from activities.heartbeat import heartbeat_until_done
from workflows.us_ppo_discovery_training import HEARTBEAT_TIMEOUT, TRAIN_TIMEOUT
from workflows.us_sac_training import HEARTBEAT_TIMEOUT as SAC_HEARTBEAT_TIMEOUT


def test_heartbeat_until_done_copies_activity_context() -> None:
    source = inspect.getsource(heartbeat_until_done)
    assert "copy_context" in source
    assert "ctx.run" in source
    assert "activity.heartbeat" in source


def test_heartbeat_until_done_emits_initial_beat(monkeypatch) -> None:
    import activities.heartbeat as heartbeat_module

    beats: list[tuple] = []
    monkeypatch.setattr(
        heartbeat_module.activity, "heartbeat", lambda *details: beats.append(details)
    )
    with heartbeat_until_done("job-1"):
        pass
    assert beats == [("job-1",)]


def test_ppo_training_heartbeat_timeout_matches_other_trainers() -> None:
    assert HEARTBEAT_TIMEOUT == SAC_HEARTBEAT_TIMEOUT
    assert TRAIN_TIMEOUT.total_seconds() == 24 * 3600


def test_preflight_activity_uses_sidecar_heartbeat() -> None:
    from datetime import timedelta

    from activities import training as training_module
    from workflows._sac_training_readiness import (
        PREFLIGHT_HEARTBEAT_TIMEOUT,
        PREFLIGHT_TIMEOUT,
    )

    source = inspect.getsource(training_module.preflight_sac_training)
    assert "heartbeat_until_done" in source
    assert timedelta(minutes=30) == PREFLIGHT_TIMEOUT
    assert timedelta(minutes=2) == PREFLIGHT_HEARTBEAT_TIMEOUT
