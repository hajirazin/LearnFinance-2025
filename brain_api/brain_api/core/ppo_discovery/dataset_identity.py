"""Bind ppo_discovery artifacts to the actual train/val/test evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from brain_api.core.ppo_discovery.config import HISTORY_BARS
from brain_api.core.ppo_discovery.environment import WeeklyTransition
from brain_api.core.ppo_discovery.price_features import validate_ohlcv_frame
from brain_api.core.ppo_discovery.schemas import (
    PPODiscoveryError,
    UniverseSnapshot,
    canonical_json_bytes,
)
from brain_api.core.ppo_discovery.weeks import open_to_open_return, prices_as_of


@dataclass(frozen=True)
class DatasetIdentity:
    """Separate hashes so a corrected slice cannot reuse a version id."""

    training_dataset_hash: str
    validation_dataset_hash: str
    evaluation_dataset_hash: str
    news_weeks: list[dict[str, Any]]
    price_sessions: dict[str, str]


def build_dataset_identity(
    train_weeks: Sequence[WeeklyTransition],
    val_weeks: Sequence[WeeklyTransition],
    test_weeks: Sequence[WeeklyTransition],
    *,
    snapshot: UniverseSnapshot,
    ohlcv: Mapping[str, pd.DataFrame],
    spy: pd.DataFrame,
) -> DatasetIdentity:
    return DatasetIdentity(
        training_dataset_hash=_weeks_hash(train_weeks, snapshot, ohlcv, spy),
        validation_dataset_hash=_weeks_hash(val_weeks, snapshot, ohlcv, spy),
        evaluation_dataset_hash=_weeks_hash(test_weeks, snapshot, ohlcv, spy),
        news_weeks=[_news_week_payload(week, snapshot) for week in train_weeks],
        price_sessions=_session_hashes(ohlcv, spy, snapshot),
    )


def frame_session_hash(frame: pd.DataFrame) -> str:
    """Hash session dates plus OHLCV so corrected prices change identity."""
    if frame is None or frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    rows: list[str] = []
    for timestamp, row in frame.iterrows():
        session = pd.Timestamp(timestamp).strftime("%Y-%m-%d")
        rows.append(
            ",".join(
                [
                    session,
                    _finite_cell(row, "open"),
                    _finite_cell(row, "high"),
                    _finite_cell(row, "low"),
                    _finite_cell(row, "close"),
                    _finite_cell(row, "volume"),
                ]
            )
        )
    payload = "\n".join(rows) + f"\n{len(rows)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _weeks_hash(
    weeks: Sequence[WeeklyTransition],
    snapshot: UniverseSnapshot,
    ohlcv: Mapping[str, pd.DataFrame],
    spy: pd.DataFrame,
) -> str:
    payload = [_week_payload(week, snapshot, ohlcv, spy) for week in weeks]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _week_payload(
    week: WeeklyTransition,
    snapshot: UniverseSnapshot,
    ohlcv: Mapping[str, pd.DataFrame],
    spy: pd.DataFrame,
) -> dict[str, Any]:
    cutoff = week.cutoff.date()
    eligible, exclusions = _week_eligibility(week, snapshot, ohlcv)
    return {
        "cutoff": week.cutoff.isoformat(),
        "rebalance_session": pd.Timestamp(week.rebalance_session).isoformat(),
        "next_rebalance_session": pd.Timestamp(week.next_rebalance_session).isoformat(),
        "p_calm": float(week.p_calm),
        "p_stress": float(week.p_stress),
        "snapshot_sha256": snapshot.snapshot_sha256,
        "eligible": eligible,
        "exclusions": exclusions,
        "news": _news_week_payload(week, snapshot)["symbols"],
        "ohlcv": {
            symbol: _pit_ohlcv_hash(ohlcv.get(symbol), cutoff)
            for symbol in snapshot.sorted_symbols
        },
        "spy": _pit_ohlcv_hash(spy, cutoff),
        "next_open": _next_open_payload(week, snapshot, ohlcv),
    }


def _week_eligibility(
    week: WeeklyTransition,
    snapshot: UniverseSnapshot,
    ohlcv: Mapping[str, pd.DataFrame],
) -> tuple[list[str], dict[str, str]]:
    cutoff = week.cutoff.date()
    eligible: list[str] = []
    exclusions: dict[str, str] = {}
    for symbol in snapshot.sorted_symbols:
        frame = ohlcv.get(symbol)
        if frame is None:
            exclusions[symbol] = "missing OHLCV frame"
            continue
        try:
            validate_ohlcv_frame(symbol, prices_as_of(frame, cutoff))
        except PPODiscoveryError as exc:
            exclusions[symbol] = str(exc)
            continue
        eligible.append(symbol)
    return eligible, exclusions


def _news_week_payload(
    week: WeeklyTransition, snapshot: UniverseSnapshot
) -> dict[str, Any]:
    symbols = {}
    for symbol in snapshot.sorted_symbols:
        row = week.news_by_symbol.get(symbol)
        if row is None:
            symbols[symbol] = None
            continue
        symbols[symbol] = {
            "article_ids_sha256": row.article_ids_sha256,
            "request_manifest_sha256": row.request_manifest_sha256,
            "raw_sentiment": float(row.raw_sentiment),
            "article_count": int(row.article_count),
            "average_confidence": float(row.average_confidence),
            "sentiment_dispersion": float(row.sentiment_dispersion),
            "hours_since_latest": float(row.hours_since_latest),
            "unique_source_count": int(row.unique_source_count),
            "has_news": int(row.has_news),
            "query_complete": bool(row.query_complete),
            "news_recency": float(row.news_recency),
            "log1p_article_count": float(row.log1p_article_count),
        }
    return {"cutoff": week.cutoff.isoformat(), "symbols": symbols}


def _pit_ohlcv_hash(frame: pd.DataFrame | None, cutoff) -> str:
    if frame is None:
        return ""
    try:
        sliced = prices_as_of(frame, cutoff).tail(HISTORY_BARS)
    except Exception:
        return ""
    return frame_session_hash(sliced)


def _next_open_payload(
    week: WeeklyTransition,
    snapshot: UniverseSnapshot,
    ohlcv: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for symbol in snapshot.sorted_symbols:
        frame = ohlcv.get(symbol)
        if frame is None:
            continue
        try:
            start_open, simple = open_to_open_return(
                frame,
                week.rebalance_session,
                week.next_rebalance_session,
                symbol=symbol,
            )
        except Exception:
            continue
        payload[symbol] = {
            "start_open": float(start_open),
            "simple_return": float(simple),
        }
    return payload


def _session_hashes(
    ohlcv: Mapping[str, pd.DataFrame],
    spy: pd.DataFrame,
    snapshot: UniverseSnapshot,
) -> dict[str, str]:
    hashes = {symbol: frame_session_hash(ohlcv[symbol]) for symbol in ohlcv}
    hashes["SPY"] = frame_session_hash(spy)
    hashes["universe"] = ",".join(snapshot.sorted_symbols)
    return hashes


def _finite_cell(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    value = row[column]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number or abs(number) == float("inf"):
        return "nan"
    return repr(number)


__all__ = ["DatasetIdentity", "build_dataset_identity", "frame_session_hash"]
