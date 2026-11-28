"""Both objectives must learn a strong, known synthetic relationship."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_training import synthetic_panel, tiny_config
from training import fit_arm


@pytest.mark.parametrize("arm", ["mse", "huber"])
def test_known_mapping_reduces_fitting_error(arm, tmp_path):
    torch.set_num_threads(1)
    panel = synthetic_panel()
    rng = np.random.default_rng(71)
    signal = rng.normal(size=len(panel.y)).astype(np.float32)
    panel.X = np.broadcast_to(signal[:, None, None], panel.X.shape).copy()
    panel.y = (0.03 * signal).astype(np.float32)
    config = replace(
        tiny_config(),
        learning_rate=0.01,
        dropout=0.0,
        epochs=15,
        early_stopping_patience=15,
    )
    result = fit_arm(panel, config, arm, torch.device("cpu"), tmp_path)
    assert result["history"][-1]["training_mse"] < (
        0.5 * result["history"][0]["training_mse"]
    )
