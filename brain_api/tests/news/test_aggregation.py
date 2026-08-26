from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from brain_api.news.aggregation import (
    confidence_recency_weighted_mean,
    population_std,
)
from brain_api.news.errors import NewsError

NY = ZoneInfo("America/New_York")


def test_empty_mean_is_zero() -> None:
    cutoff = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
    assert confidence_recency_weighted_mean([], [], [], cutoff) == 0.0


def test_zero_weight_nonempty_raises() -> None:
    cutoff = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
    created = datetime(2026, 8, 20, 9, 0, tzinfo=NY)
    with pytest.raises(NewsError, match="weight sum"):
        confidence_recency_weighted_mean([0.5], [0.0], [created], cutoff)


def test_tau_recency_matches_closed_form() -> None:
    cutoff = datetime(2026, 8, 24, 9, 0, tzinfo=NY)
    created = datetime(2026, 8, 17, 9, 0, tzinfo=NY)
    tau = 168.0
    age = 7 * 24
    weight = 1.0 * math.exp(-age / tau)
    got = confidence_recency_weighted_mean([0.4], [1.0], [created], cutoff, tau=tau)
    assert got == pytest.approx(0.4)
    assert weight == pytest.approx(math.exp(-1.0))


def test_population_std_two_values() -> None:
    assert population_std([0.0, 2.0]) == pytest.approx(1.0)
    assert population_std([1.0]) == 0.0
