"""HRP weights must use full prior history and ignore future observations."""

import numpy as np
import pandas as pd
import pytest
from add_hrp import target_from_history


def test_future_price_changes_cannot_affect_the_decision():
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2022-08-01", periods=500)
    closes = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (500, 4)), axis=0)),
        index=days,
        columns=list("ABCD"),
    )
    target, evidence = target_from_history(closes, "2024-02-05")
    changed = closes.copy()
    changed.loc[changed.index >= "2024-02-05", "A"] *= 10
    other, _ = target_from_history(changed, "2024-02-05")
    np.testing.assert_array_equal(target, other)
    assert evidence["daily_returns_used"] == 252
    assert evidence["history_end"] == "2024-02-02"
    assert target[-1] == pytest.approx(0.02)
    assert target.sum() == pytest.approx(1)


def test_short_history_is_rejected_instead_of_using_a_shorter_window():
    days = pd.bdate_range("2024-01-01", periods=25)
    closes = pd.DataFrame(100.0, index=days, columns=list("ABCD"))
    with pytest.raises(ValueError, match="253 complete prior closes"):
        target_from_history(closes, "2024-02-05")
