"""Check the streaming ridge fit against the standard ridge solver."""

import unittest

import numpy as np
from compare_ridge_lookback import RidgeMoments, design
from sklearn.linear_model import Ridge


class RidgeLookbackTest(unittest.TestCase):
    def test_streaming_coefficients_match_ridge_with_intercept(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 5)) + 3
        y = rng.normal(size=80) - 2
        penalties = np.array([0.001, 0.1, 1.0])
        moments = RidgeMoments(x.shape[1])
        for start in (0, 40):
            moments.update(x[start : start + 40], y[start : start + 40])
        weights, intercept = moments.coefficients(penalties)
        for j, penalty in enumerate(penalties):
            reference = Ridge(alpha=len(y) * penalty).fit(x, y)
            np.testing.assert_allclose(
                x @ weights[:, j] + intercept[j], reference.predict(x)
            )

    def test_masked_prices_cannot_change_either_design(self):
        raw = np.arange(64, dtype=np.float32).reshape(2, 4, 8)
        mask = np.ones_like(raw, dtype=bool)
        mask[:, 0] = False
        before = design(raw, mask)
        raw[:, 0] = -1e6
        after = design(raw, mask)
        for former, latter in zip(before, after, strict=True):
            np.testing.assert_array_equal(former, latter)


if __name__ == "__main__":
    unittest.main()
