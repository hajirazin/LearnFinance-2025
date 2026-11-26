"""Check that a weekly artifact actually reaches its advertised cutoff."""

import logging
from datetime import date, timedelta

from brain_api.core.patchtst.calendar import FORECAST_SESSIONS, load_patchtst_calendar

logger = logging.getLogger(__name__)


def validate_label_evidence(
    evidence: dict, market: str, cutoff: date, symbols: int
) -> None:
    calendar = load_patchtst_calendar(market, cutoff - timedelta(days=40), cutoff)
    sessions = calendar.sessions
    anchors = sessions.to_series().groupby(sessions.to_period("W-SUN")).last()
    matured = [
        sessions[index + FORECAST_SESSIONS].date()
        for stamp in anchors
        if (index := sessions.get_loc(stamp)) + FORECAST_SESSIONS < len(sessions)
    ]
    expected = max(matured)
    latest = evidence.get("latest_label_end")
    if latest != expected.isoformat() or evidence.get("latest_label_symbols", 0) < min(
        10, symbols
    ):
        raise ValueError(
            f"PatchTST artifact is missing current label evidence for {expected} "
            f"at cutoff {cutoff}; latest_label_end={latest}. Repair evidence and retrain."
        )


def can_reuse_weekly_fit(metadata: dict, config, symbols: int) -> bool:
    if not config.weekly_regression:
        return True
    try:
        validate_label_evidence(
            metadata.get("training_evidence", {}),
            config.market,
            date.fromisoformat(metadata["data_window"]["end"]),
            symbols,
        )
    except ValueError as exc:
        logger.warning("Saved PatchTST fit requires retraining: %s", exc)
        return False
    return True
