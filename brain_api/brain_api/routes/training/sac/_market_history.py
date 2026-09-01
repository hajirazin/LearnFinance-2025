"""Strict market-history repair and extraction for SAC v3 training."""

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from brain_api.core.sac.market_history import extract_aligned_market_history
from brain_api.core.sac.market_sessions import xnys_session_dates
from brain_api.core.vix_fallback import VixFallbackAudit, apply_cboe_vix_fallback


def repair_and_extract_sac_market_history(
    prices: Mapping[str, pd.DataFrame],
    *,
    start_date: date,
    completed_through: date,
) -> tuple[
    dict[str, pd.DataFrame], list[date], np.ndarray, np.ndarray, VixFallbackAudit
]:
    """Repair consumed VIX sessions, then retain exact SPY/VIX validation."""
    result = apply_cboe_vix_fallback(
        prices,
        required_dates=xnys_session_dates(start_date, completed_through),
    )
    market_dates, spy_closes, vix_closes = extract_aligned_market_history(
        result.prices,
        start_date=start_date,
        completed_through=completed_through,
    )
    return result.prices, market_dates, spy_closes, vix_closes, result.audit


def record_sac_vix_audit(experiment: Any, audit: VixFallbackAudit) -> None:
    """Attach identical VIX provenance to every persisted SAC candidate."""
    for candidate in experiment.candidates:
        candidate.result.audit_metadata["vix_fallback"] = audit.to_dict()


__all__ = [
    "extract_aligned_market_history",
    "record_sac_vix_audit",
    "repair_and_extract_sac_market_history",
]
