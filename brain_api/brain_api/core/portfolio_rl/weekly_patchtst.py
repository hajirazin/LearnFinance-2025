"""Generate snapshot-based forecasts with an explicit historical news policy."""

import logging
from datetime import date

import numpy as np
import pandas as pd

from brain_api.core.forecaster_snapshot_identity import annual_snapshot_cutoffs
from brain_api.core.patchtst.calendar import training_calendar
from brain_api.core.patchtst.config import US_WEEKLY_CONFIG, patchtst_recipe
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTInputs
from brain_api.core.patchtst.weekly_training import predict_weekly_returns
from brain_api.core.prices import load_prices_yfinance
from brain_api.news.models import HISTORICAL_PATCHTST_NEWS_REVISION_POLICY
from brain_api.storage.forecaster_snapshot_policy import ensure_snapshot_for_bucket
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage

logger = logging.getLogger(__name__)


def ready_weekly_snapshots(storage, start: date, end: date) -> dict[date, dict]:
    """Resolve the same expected inventory in preflight and forecast generation."""
    snapshots = {}
    for cutoff in annual_snapshot_cutoffs(start, end):
        if not ensure_snapshot_for_bucket(snapshot_storage=storage, cutoff_date=cutoff):
            raise ValueError(f"Missing US PatchTST annual snapshot {cutoff}")
        metadata = storage.read_metadata(cutoff)
        evidence = (metadata or {}).get("training_evidence", {})
        label_end = evidence.get("latest_label_end")
        if not label_end or date.fromisoformat(label_end) > cutoff:
            raise ValueError(
                f"US PatchTST snapshot {cutoff} lacks valid label-cutoff evidence"
            )
        snapshots[cutoff] = metadata
    if not snapshots:
        raise ValueError("No compatible US PatchTST annual snapshots")
    return snapshots


def score_sac_forecasts(
    symbols: list[str],
    weekly_dates: pd.DatetimeIndex,
    decision_sessions: pd.DatetimeIndex,
    *,
    start: date,
    end: date,
    evidence: dict,
    shutdown_event=None,
) -> dict[str, np.ndarray]:
    from brain_api.core.training_utils import TrainingCancelledError

    if len(decision_sessions) != len(weekly_dates):
        raise ValueError("Weekly prices and SAC forecast decisions are not aligned")
    storage = SnapshotLocalStorage("patchtst_halal_new")
    snapshots = ready_weekly_snapshots(storage, start, end)
    calendar = training_calendar("us", weekly_dates[0].date(), end)
    anchors = [calendar.anchor(cutoff.date()) for cutoff in weekly_dates]
    groups = {}
    for index, (anchor, decision) in enumerate(
        zip(anchors, decision_sessions, strict=True)
    ):
        available = [
            cutoff
            for cutoff, metadata in snapshots.items()
            if cutoff < decision.date()
            and date.fromisoformat(metadata["training_evidence"]["latest_label_end"])
            <= anchor
        ]
        if not available:
            raise ValueError(
                f"No causal US PatchTST snapshot for decision {decision.date()}"
            )
        groups.setdefault(max(available), []).append((index, anchor))
    prices = load_prices_yfinance(
        symbols,
        calendar.sessions[0].date(),
        max(anchors),
        required_columns=("close", "volume"),
    )
    if not prices:
        raise ValueError("Missing historical prices for US PatchTST forecasts")
    inputs = WeeklyPatchTSTInputs(
        US_WEEKLY_CONFIG,
        calendar,
        prices,
        news_revision_policy=HISTORICAL_PATCHTST_NEWS_REVISION_POLICY,
    )
    later_revision_cells = inputs.news_evidence.get("later_revision_cells", 0)
    evidence.update(
        patchtst_news_revision_policy=HISTORICAL_PATCHTST_NEWS_REVISION_POLICY,
        patchtst_news_later_revision_cells=later_revision_cells,
        patchtst_news_lookahead_bias=bool(later_revision_cells),
    )
    logger.warning(
        "Historical PatchTST news policy=%s later_revision_cells=%s; "
        "training/evaluation may include look-ahead bias from edited articles",
        HISTORICAL_PATCHTST_NEWS_REVISION_POLICY,
        later_revision_cells,
    )
    forecasts = {symbol: np.full(len(anchors), np.nan) for symbol in symbols}
    for cutoff, decisions in groups.items():
        artifacts = storage.load_snapshot(cutoff)
        if patchtst_recipe(artifacts.config) != patchtst_recipe(US_WEEKLY_CONFIG):
            raise ValueError(f"Incompatible US PatchTST snapshot recipe for {cutoff}")
        for index, anchor in decisions:
            if shutdown_event and shutdown_event.is_set():
                raise TrainingCancelledError("SAC forecast generation cancelled")
            windows = {}
            for symbol in symbols:
                window, _ = inputs.window(symbol, anchor)
                if window is not None:
                    windows[symbol] = window
            if len(windows) < 10:
                raise ValueError(
                    f"Only {len(windows)} PatchTST inputs available for SAC at {anchor}"
                )
            predictions = predict_weekly_returns(
                artifacts.model,
                np.stack(list(windows.values())),
                artifacts.feature_scaler,
            )
            for symbol, forecast in zip(windows, predictions, strict=True):
                forecasts[symbol][index] = forecast
    evidence["patchtst_snapshots"] = {
        cutoff.isoformat(): snapshots[cutoff]["config_symbols_hash"]
        for cutoff in groups
    }
    return forecasts
