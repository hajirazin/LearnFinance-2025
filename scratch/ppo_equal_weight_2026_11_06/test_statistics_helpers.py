"""Check interpretations needed for a practical-equivalence experiment."""

import numpy as np
import pytest

from statistics_helpers import bootstrap, dispersion, rank_correlation


def test_equal_weights_and_cash_invariance():
    assert dispersion([0.04] * 10)["cv"] == pytest.approx(0)
    assert dispersion([0.1, 0.2])["cv"] == pytest.approx(dispersion([0.2, 0.4])["cv"])


def test_relative_deviation_and_effective_names():
    d = dispersion([0.09, 0.11])
    assert d["cv"] == pytest.approx(0.10)
    assert d["max_relative_deviation"] == pytest.approx(0.10)
    assert d["equity_tv"] == pytest.approx(0.05)
    assert d["effective_names_fraction"] == pytest.approx(1 / 1.01)


def test_degenerate_portfolios_do_not_count_as_evidence():
    assert dispersion([]) is None
    assert dispersion([1]) is None
    assert rank_correlation([1, 1, 1], [1, 2, 3]) is None


def test_bootstrap_preserves_constant_and_is_reproducible():
    assert bootstrap([0.04] * 20)["ci95"] == pytest.approx([0.04, 0.04])
    assert bootstrap(np.arange(20))["ci95"] == bootstrap(np.arange(20))["ci95"]


def test_invalid_and_insufficient_evidence_raises():
    with pytest.raises(ValueError):
        dispersion([1, -1])
    with pytest.raises(ValueError):
        bootstrap([0.01, 0.02])
