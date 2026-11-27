"""Weekly training records cross-sectional rank IC as a diagnostic."""

from dataclasses import replace

import numpy as np
import pytest

from brain_api.core.patchtst.config import INDIA_WEEKLY_CONFIG
from brain_api.core.patchtst.weekly_data import InsufficientWeeklyHistory
from brain_api.core.patchtst.weekly_training import (
    fit_weekly_patchtst,
    weekly_fold_indices,
)
from tests.test_weekly_patchtst_progress import (  # noqa: F401
    synthetic_weekly_data,
    tiny_regression_model,
)


@pytest.mark.usefixtures("tiny_regression_model")
def test_weekly_trainer_reports_finite_rank_ic():
    result = fit_weekly_patchtst(
        synthetic_weekly_data(), replace(INDIA_WEEKLY_CONFIG, epochs=2)
    )
    assert np.isfinite(result.val_rank_ic)
    assert result.stopped_epoch == 2
    assert 1 <= result.best_epoch <= 2


def test_empty_data_is_rejected_without_fitting():
    data = synthetic_weekly_data()
    data.anchor_dates = data.anchor_dates[:0]
    with pytest.raises(InsufficientWeeklyHistory):
        weekly_fold_indices(data, 0.2)
