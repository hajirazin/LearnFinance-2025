"""Unit tests for HRP domain service (pure functions)."""

import numpy as np
import pandas as pd
import pytest

from brain_api.domain.services.hrp import (
    _compute_correlation_distance,
    _get_cluster_variance,
    _recursive_bisection,
    compute_hrp_weights,
)


class TestComputeCorrelationDistance:
    """Tests for _compute_correlation_distance."""

    def test_perfect_positive_correlation(self):
        """Perfect positive correlation (1.0) should give distance 0."""
        corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], columns=["A", "B"], index=["A", "B"])
        dist = _compute_correlation_distance(corr)

        assert dist.loc["A", "B"] == pytest.approx(0.0, abs=1e-6)
        assert dist.loc["A", "A"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_correlation(self):
        """Zero correlation should give distance ~0.707."""
        corr = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=["A", "B"], index=["A", "B"])
        dist = _compute_correlation_distance(corr)

        expected = np.sqrt(0.5)  # ~0.707
        assert dist.loc["A", "B"] == pytest.approx(expected, rel=1e-6)

    def test_perfect_negative_correlation(self):
        """Perfect negative correlation (-1.0) should give distance 1.0."""
        corr = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], columns=["A", "B"], index=["A", "B"])
        dist = _compute_correlation_distance(corr)

        assert dist.loc["A", "B"] == pytest.approx(1.0, rel=1e-6)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        corr = pd.DataFrame(
            [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        dist = _compute_correlation_distance(corr)

        assert dist.loc["A", "B"] == dist.loc["B", "A"]
        assert dist.loc["A", "C"] == dist.loc["C", "A"]
        assert dist.loc["B", "C"] == dist.loc["C", "B"]


class TestGetClusterVariance:
    """Tests for _get_cluster_variance."""

    def test_single_asset(self):
        """Single asset cluster should return its variance."""
        cov = pd.DataFrame([[0.04]], columns=["A"], index=["A"])
        var = _get_cluster_variance(cov, ["A"])
        assert var == pytest.approx(0.04)

    def test_two_assets_uncorrelated(self):
        """Two uncorrelated assets with equal variance."""
        # Variance 0.04 each, zero covariance
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.04]], columns=["A", "B"], index=["A", "B"]
        )
        var = _get_cluster_variance(cov, ["A", "B"])

        # Inverse variance weights: 50% each
        # Cluster variance = 0.5^2 * 0.04 + 0.5^2 * 0.04 = 0.02
        assert var == pytest.approx(0.02, rel=1e-6)

    def test_two_assets_different_variance(self):
        """Two uncorrelated assets with different variances."""
        # A: var=0.01, B: var=0.04
        cov = pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.04]], columns=["A", "B"], index=["A", "B"]
        )
        var = _get_cluster_variance(cov, ["A", "B"])

        # Inverse variance weights: A gets 4x weight of B
        # w_A = (1/0.01) / (1/0.01 + 1/0.04) = 100/125 = 0.8
        # w_B = (1/0.04) / (1/0.01 + 1/0.04) = 25/125 = 0.2
        # Cluster variance = 0.8^2 * 0.01 + 0.2^2 * 0.04 = 0.0064 + 0.0016 = 0.008
        assert var == pytest.approx(0.008, rel=1e-6)


class TestRecursiveBisection:
    """Tests for _recursive_bisection."""

    def test_single_asset(self):
        """Single asset should get 100% weight."""
        cov = pd.DataFrame([[0.04]], columns=["A"], index=["A"])
        weights = _recursive_bisection(cov, ["A"])

        assert weights == {"A": 1.0}

    def test_two_assets_equal_variance(self):
        """Two assets with equal variance should get equal weights."""
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.04]], columns=["A", "B"], index=["A", "B"]
        )
        weights = _recursive_bisection(cov, ["A", "B"])

        assert weights["A"] == pytest.approx(0.5, rel=1e-6)
        assert weights["B"] == pytest.approx(0.5, rel=1e-6)

    def test_weights_sum_to_one(self):
        """All weights should sum to 1.0."""
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.005], [0.01, 0.09, 0.02], [0.005, 0.02, 0.16]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        weights = _recursive_bisection(cov, ["A", "B", "C"])

        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_all_weights_positive(self):
        """All weights should be positive."""
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.005], [0.01, 0.09, 0.02], [0.005, 0.02, 0.16]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        weights = _recursive_bisection(cov, ["A", "B", "C"])

        for symbol, weight in weights.items():
            assert weight > 0, f"Weight for {symbol} should be positive"


class TestComputeHRPWeights:
    """Tests for compute_hrp_weights (full algorithm)."""

    def test_empty_returns(self):
        """Empty returns should give empty weights."""
        returns = pd.DataFrame()
        weights = compute_hrp_weights(returns)
        assert weights == {}

    def test_single_asset(self):
        """Single asset should get 100% weight."""
        returns = pd.DataFrame({"A": [0.01, -0.02, 0.015, 0.005, -0.01]})
        weights = compute_hrp_weights(returns)

        assert weights == {"A": 1.0}

    def test_two_assets(self):
        """Two assets should produce valid weights."""
        returns = pd.DataFrame({
            "A": [0.01, -0.02, 0.015, 0.005, -0.01],
            "B": [0.02, -0.01, 0.01, -0.005, 0.015],
        })
        weights = compute_hrp_weights(returns)

        assert len(weights) == 2
        assert "A" in weights
        assert "B" in weights
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_three_assets(self):
        """Three assets should use full HRP algorithm."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0.01, 0.02, 50),
            "B": np.random.normal(0.005, 0.03, 50),
            "C": np.random.normal(0.008, 0.015, 50),
        })
        weights = compute_hrp_weights(returns)

        assert len(weights) == 3
        assert all(w > 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_lower_volatility_gets_higher_weight(self):
        """Lower volatility assets should generally get higher weights."""
        # A has much lower volatility than B and C
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0.01, 0.01, 100),  # Low vol
            "B": np.random.normal(0.01, 0.05, 100),  # High vol
            "C": np.random.normal(0.01, 0.04, 100),  # High vol
        })
        weights = compute_hrp_weights(returns)

        # A should have highest weight due to low volatility
        assert weights["A"] > weights["B"]
        assert weights["A"] > weights["C"]

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "A": np.random.normal(0.01, 0.02, 50),
            "B": np.random.normal(0.005, 0.03, 50),
            "C": np.random.normal(0.008, 0.015, 50),
        })

        weights1 = compute_hrp_weights(returns)
        weights2 = compute_hrp_weights(returns)

        for symbol in weights1:
            assert weights1[symbol] == weights2[symbol]

