"""Synthetic attribution checks, isolated from installed artifacts and providers."""

import numpy as np
from attribution import exact_channel_shapley

pytest_plugins = ["tests.offline"]


def test_linear_channels_recover_their_exact_effects():
    effects = np.array([0.2, -0.1, 0.4, 0.05, -0.03, 0.06, 0.01])
    coalitions = np.array(
        [
            0.7 + sum(effects[i] for i in range(7) if mask & (1 << i))
            for mask in range(128)
        ]
    )
    np.testing.assert_allclose(exact_channel_shapley(coalitions), effects, atol=1e-14)


def test_joint_effect_is_shared_and_constant_offset_has_no_attribution():
    np.testing.assert_allclose(exact_channel_shapley([5, 5, 5, 7]), [1, 1])
