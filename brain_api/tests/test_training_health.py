"""Tests for the shared ``ArtifactHealthCheck`` dataclass."""

from dataclasses import FrozenInstanceError

import pytest

from brain_api.core.training_health import ArtifactHealthCheck


def test_healthy_construction_with_empty_reasons():
    """Healthy check must have empty failure_reasons."""
    health = ArtifactHealthCheck(is_healthy=True, failure_reasons=[])
    assert health.is_healthy is True
    assert health.failure_reasons == []


def test_unhealthy_construction_preserves_reason_order():
    """Unhealthy check must keep failure_reasons in the order supplied
    so the operator-facing email lists checks in the order they were
    applied (downstream Jinja templates render the list as-is)."""
    reasons = [
        "val_loss is not finite",
        "weights.pt missing or zero bytes",
        "feature_scaler.pkl missing or zero bytes",
    ]
    health = ArtifactHealthCheck(is_healthy=False, failure_reasons=reasons)
    assert health.is_healthy is False
    assert health.failure_reasons == reasons


def test_healthy_with_nonempty_reasons_raises():
    """is_healthy=True + non-empty reasons is a structural bug."""
    with pytest.raises(ValueError, match="is_healthy=True but failure_reasons"):
        ArtifactHealthCheck(is_healthy=True, failure_reasons=["something"])


def test_unhealthy_with_empty_reasons_raises():
    """is_healthy=False + empty reasons leaves the operator no way to
    diagnose what failed -- forbidden so callers always supply a reason."""
    with pytest.raises(ValueError, match="is_healthy=False but failure_reasons"):
        ArtifactHealthCheck(is_healthy=False, failure_reasons=[])


def test_dataclass_is_frozen():
    """Health checks travel through the metadata + email pipeline; we
    don't want any of those consumers to mutate the failure list."""
    health = ArtifactHealthCheck(is_healthy=True, failure_reasons=[])
    with pytest.raises(FrozenInstanceError):
        health.is_healthy = False  # type: ignore[misc]
