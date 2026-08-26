"""Signals, forecasts, and allocator activities."""

import logging
from datetime import date, datetime, timedelta

from temporalio import activity

from activities.client import get_client
from activities.sac_context import (
    MOM_12_1_LOOKBACK_BARS,
    build_sac_feature_bundle,
)
from models import (
    AdjustedClosesResponse,
    AlpacaPortfolioResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    MarketHistoryResponse,
    PatchTSTBatchScores,
    PatchTSTInferenceResponse,
    PreviousFinalAllocationResponse,
    RankBandTopNResponse,
    RecordFinalWeightsResponse,
    SACInferenceResponse,
    StickyTopNResponse,
)
from models.news import MondayDecisionWindowResponse, NewsWindowResult

logger = logging.getLogger(__name__)

# Minimum trailing daily closes for SAC's momentum_12_1 (skip 21 bars,
# then a 252-bar/~12-month lookback -- see activities.sac_context).
SAC_MOMENTUM_LOOKBACK_BARS = MOM_12_1_LOOKBACK_BARS + 1


@activity.defn
def get_adjusted_closes(
    symbols: list[str],
    as_of_date: str,
    lookback_bars: int = SAC_MOMENTUM_LOOKBACK_BARS,
) -> AdjustedClosesResponse:
    """Fetch point-in-time adjusted-close history for SAC features."""
    logger.info(
        f"Fetching {lookback_bars}-bar adjusted closes for {len(symbols)} symbols..."
    )
    with get_client() as client:
        response = client.post(
            "/signals/prices",
            json={
                "symbols": symbols,
                "as_of_date": as_of_date,
                "lookback_bars": lookback_bars,
            },
        )
        response.raise_for_status()
    result = AdjustedClosesResponse(**response.json())
    logger.info(f"Got adjusted closes for {len(result.adjusted_closes)} symbols")
    return result


@activity.defn
def get_market_history(
    training_cutoff_date: str, as_of_date: str
) -> MarketHistoryResponse:
    """Fetch SPY/VIX history through the last completed pre-decision day."""
    cutoff = date.fromisoformat(training_cutoff_date)
    decision_date = date.fromisoformat(as_of_date)
    if cutoff > decision_date:
        raise ValueError("SAC training cutoff cannot be after the decision date")
    market_as_of = decision_date - timedelta(days=1)
    start_date = (cutoff + timedelta(days=1)).isoformat()
    if cutoff >= market_as_of:
        return MarketHistoryResponse(
            start_date=start_date,
            as_of_date=market_as_of.isoformat(),
            rows=[],
            provenance={
                "source": "training_cutoff",
                "no_completed_post_cutoff_sessions": True,
            },
        )
    market_as_of_date = market_as_of.isoformat()
    logger.info(
        f"Fetching SAC market history ({start_date} through {market_as_of_date})..."
    )
    with get_client() as client:
        response = client.post(
            "/signals/market-history",
            # The Brain endpoint takes the pre-open decision date and owns
            # exchange-calendar exclusion of that still-incomplete session.
            json={"start_date": start_date, "as_of_date": as_of_date},
        )
        response.raise_for_status()
    result = MarketHistoryResponse(**response.json())
    if result.start_date != start_date or result.as_of_date != as_of_date:
        raise ValueError("Market-history response range does not match the request")
    # Downstream evidence metadata names the last eligible calendar date, not
    # the endpoint's pre-open decision date. Rows remain the source of truth
    # for exact XNYS-session validation in Brain.
    result = result.model_copy(update={"as_of_date": market_as_of_date})
    logger.info(f"Got {len(result.rows)} market-history rows")
    return result


@activity.defn
def get_monday_decision_window(run_date: str) -> MondayDecisionWindowResponse:
    """Resolve Monday 09:00 NY bounds from the Brain calendar (not locally)."""
    logger.info("Resolving Monday decision window for run_date=%s", run_date)
    with get_client() as client:
        response = client.post(
            "/calendar/monday-decision-window",
            json={"run_date": run_date},
        )
        response.raise_for_status()
    result = MondayDecisionWindowResponse(**response.json())
    logger.info(
        "Monday decision window cutoff=%s start=%s end=%s",
        result.cutoff.isoformat(),
        result.start_exclusive.isoformat(),
        result.end_inclusive.isoformat(),
    )
    return result


def _news_window_body(symbols: list[str], window: MondayDecisionWindowResponse) -> dict:
    return {
        "symbols": symbols,
        "start_exclusive": window.start_exclusive.isoformat(),
        "end_inclusive": window.end_inclusive.isoformat(),
    }


@activity.defn
def materialize_news_window(
    symbols: list[str], window: MondayDecisionWindowResponse
) -> NewsWindowResult:
    """Fetch+score the exact Monday window. Parse-only; no aggregation."""
    logger.info(
        "Materializing news window for %s symbols end=%s",
        len(symbols),
        window.end_inclusive.isoformat(),
    )
    with get_client() as client:
        response = client.post(
            "/news/windows/materialize",
            json=_news_window_body(symbols, window),
        )
        response.raise_for_status()
    result = NewsWindowResult(**response.json())
    logger.info(
        "Materialized news window events=%s coverage=%s",
        len(result.events),
        len(result.coverage),
    )
    return result


@activity.defn
def query_news_window(
    symbols: list[str], window: MondayDecisionWindowResponse
) -> NewsWindowResult:
    """Read-only query of the exact Monday window. Parse-only DTO."""
    logger.info(
        "Querying news window for %s symbols end=%s",
        len(symbols),
        window.end_inclusive.isoformat(),
    )
    with get_client() as client:
        response = client.post(
            "/news/windows/query",
            json=_news_window_body(symbols, window),
        )
        response.raise_for_status()
    result = NewsWindowResult(**response.json())
    logger.info(
        "Queried news window events=%s coverage=%s",
        len(result.events),
        len(result.coverage),
    )
    return result


def _as_of_iso(as_of: datetime | str) -> str:
    if isinstance(as_of, datetime):
        return as_of.isoformat()
    return as_of


@activity.defn
def get_lstm_forecast(
    as_of_date: str, symbols: list[str] | None = None
) -> LSTMInferenceResponse:
    """Get LSTM price predictions.

    When ``symbols`` is set, scopes inference to that list; otherwise brain_api
    uses model metadata symbols.
    """
    if symbols:
        logger.info(f"Getting LSTM forecast for {len(symbols)} requested symbols...")
    else:
        logger.info("Getting LSTM forecast (symbols from model metadata)...")
    payload: dict = {"as_of_date": as_of_date}
    if symbols:
        payload["symbols"] = symbols
    with get_client() as client:
        response = client.post("/inference/lstm", json=payload)
        response.raise_for_status()
    result = LSTMInferenceResponse(**response.json())
    logger.info(
        f"Got LSTM predictions: {len(result.predictions)} symbols, "
        f"version={result.model_version}"
    )
    return result


@activity.defn
def get_patchtst_forecast(
    as_of_date: str, symbols: list[str] | None = None
) -> PatchTSTInferenceResponse:
    """Get PatchTST predictions.

    When ``symbols`` is set, scopes inference to that list; otherwise brain_api
    uses model metadata symbols.
    """
    if symbols:
        n = len(symbols)
        logger.info(f"Getting PatchTST forecast for {n} requested symbols...")
    else:
        logger.info("Getting PatchTST forecast (symbols from model metadata)...")
    payload: dict = {"as_of_date": as_of_date}
    if symbols:
        payload["symbols"] = symbols
    with get_client() as client:
        response = client.post("/inference/patchtst", json=payload)
        response.raise_for_status()
    result = PatchTSTInferenceResponse(**response.json())
    logger.info(
        f"Got PatchTST predictions: {len(result.predictions)} symbols, "
        f"version={result.model_version}"
    )
    return result


@activity.defn
def select_sticky_top_n(
    stage1: HRPAllocationResponse,
    universe: str,
    year_week: str,
    as_of_date: str,
    run_id: str,
    top_n: int = 15,
    stickiness_threshold_pp: float = 1.0,
) -> StickyTopNResponse:
    """Apply sticky-selection to a Stage 1 HRP result.

    POSTs to brain_api's /allocation/sticky-top-n which persists Stage 1
    weights and returns the chosen symbols + provenance (sticky vs
    top_rank, evicted previous holdings).
    """
    logger.info(
        f"[Sticky] {universe}/{year_week}: top_n={top_n} "
        f"threshold={stickiness_threshold_pp}pp"
    )
    with get_client() as client:
        response = client.post(
            "/allocation/sticky-top-n",
            json={
                "stage1": stage1.model_dump(),
                "universe": universe,
                "year_week": year_week,
                "as_of_date": as_of_date,
                "run_id": run_id,
                "top_n": top_n,
                "stickiness_threshold_pp": stickiness_threshold_pp,
            },
        )
        response.raise_for_status()
    result = StickyTopNResponse(**response.json())
    logger.info(
        f"[Sticky] kept={result.kept_count} fillers={result.fillers_count} "
        f"prev_yw={result.previous_year_week_used}"
    )
    return result


@activity.defn
def get_previous_final_allocation(
    universe: str,
    current_year_week: str,
) -> PreviousFinalAllocationResponse:
    """Return the prior week's Stage 2 final weights for a partition.

    Used by paper-only India workflows to populate the "Going Into
    This Week" email block when there is no live broker to query.
    Cold-start (no prior row) returns ``year_week=None`` -- the
    workflow surfaces it as a "(cold start)" label.
    """
    logger.info(
        f"[Sticky] Reading prior final allocation for {universe}/{current_year_week}"
    )
    with get_client() as client:
        response = client.get(
            "/allocation/previous-final-allocation",
            params={
                "universe": universe,
                "current_year_week": current_year_week,
            },
        )
        response.raise_for_status()
    result = PreviousFinalAllocationResponse(**response.json())
    logger.info(
        f"[Sticky] Prior final allocation: "
        f"year_week={result.year_week} "
        f"stocks={len(result.final_weights_pct)}"
    )
    return result


@activity.defn
def record_final_weights(
    universe: str,
    year_week: str,
    final_weights_pct: dict[str, float],
) -> RecordFinalWeightsResponse:
    """Record Stage 2 final HRP weights for the just-completed week."""
    logger.info(
        f"[Sticky] Recording final weights for {universe}/{year_week} "
        f"({len(final_weights_pct)} stocks)"
    )
    with get_client() as client:
        response = client.post(
            "/allocation/record-final-weights",
            json={
                "universe": universe,
                "year_week": year_week,
                "final_weights_pct": final_weights_pct,
            },
        )
        response.raise_for_status()
    result = RecordFinalWeightsResponse(**response.json())
    logger.info(
        f"[Sticky] Recorded {result.rows_updated} final weights "
        f"for {universe}/{year_week}"
    )
    return result


@activity.defn
def infer_sac(
    portfolio: AlpacaPortfolioResponse,
    as_of_date: str,
    universe: str,
    symbols: list[str],
    as_of: datetime | str,
    news_window: NewsWindowResult,
    patchtst: PatchTSTInferenceResponse,
    prices: AdjustedClosesResponse,
    market: MarketHistoryResponse,
) -> SACInferenceResponse:
    """Get SAC allocation for the requested SAC bucket.

    The ``universe`` arg is mandatory (no default) so each parallel A/B
    SAC workflow declares its bucket explicitly. brain_api resolves
    the bucket via ``get_bucket(ModelType.SAC, universe)`` and loads
    that bucket's frozen ``symbol_order``. Per AGENTS.md rule #1.
    Temporal sends the parse-only ``NewsWindowResult`` plus raw price
    evidence. Brain owns adapter math, eligibility, ranks, HMM, and
    state packing.
    """
    feature_bundle = build_sac_feature_bundle(
        symbols=symbols,
        as_of_date=as_of_date,
        patchtst=patchtst,
        prices=prices,
        market=market,
    )
    logger.info(f"Getting SAC allocation (universe={universe})...")
    with get_client() as client:
        response = client.post(
            "/inference/sac",
            params={"universe": universe},
            json={
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "as_of": _as_of_iso(as_of),
                "as_of_date": as_of_date,
                "news_window": news_window.model_dump(mode="json"),
                "feature_bundle": feature_bundle,
            },
        )
        if response.status_code >= 400:
            logger.error(
                "SAC inference failed (universe=%s, status=%s): %s",
                universe,
                response.status_code,
                response.text,
            )
        response.raise_for_status()
    result = SACInferenceResponse(**response.json())
    logger.info(
        f"SAC allocation: {len(result.target_weights)} positions, "
        f"turnover={result.turnover:.2%}"
    )
    return result


@activity.defn
def allocate_hrp(
    symbols: list[str], as_of_date: str, lookback_days: int = 252
) -> HRPAllocationResponse:
    """Get HRP allocation for the given symbols."""
    logger.info(
        f"Getting HRP allocation ({len(symbols)} symbols, lookback={lookback_days})..."
    )
    with get_client() as client:
        response = client.post(
            "/allocation/hrp",
            json={
                "symbols": symbols,
                "as_of_date": as_of_date,
                "lookback_days": lookback_days,
            },
        )
        response.raise_for_status()
    result = HRPAllocationResponse(**response.json())
    logger.info(
        f"HRP allocation: {result.symbols_used} symbols, "
        f"lookback={result.lookback_days}, "
        f"excluded={len(result.symbols_excluded)}"
    )
    return result


def _score_with_patchtst(
    market: str,
    symbols: list[str],
    as_of_date: str,
    min_predictions: int,
    log_prefix: str,
) -> PatchTSTBatchScores:
    """Thin HTTP wrapper around ``POST /inference/patchtst/score-batch``.

    Both US and India alpha-screen activities funnel through this
    helper. The math invariants (non-finite rejection, ``min_predictions``
    floor) live in :mod:`brain_api.core.patchtst.score_validation` and
    are enforced inside the brain_api endpoint -- this layer does not
    re-implement them. A 422 from the endpoint is re-raised as
    ``RuntimeError`` to preserve the existing AlphaHRP failure
    semantics (workflow-level retry policy treats it as terminal).
    """
    logger.info(
        f"{log_prefix} PatchTST batch scoring on {len(symbols)} symbols "
        f"(market={market}, as_of_date={as_of_date})"
    )
    with get_client() as client:
        response = client.post(
            "/inference/patchtst/score-batch",
            json={
                "market": market,
                "symbols": symbols,
                "as_of_date": as_of_date,
                "min_predictions": min_predictions,
            },
        )
        if response.status_code == 422:
            # Math invariants live behind the endpoint; surface the
            # validation message verbatim so the workflow log captures
            # it without re-implementing the policy here.
            detail = response.json().get("detail", response.text)
            raise RuntimeError(detail)
        response.raise_for_status()
    result = PatchTSTBatchScores(**response.json())
    logger.info(
        f"{log_prefix} PatchTST scores: {result.predicted_count} valid / "
        f"{result.requested_count} requested, model_version={result.model_version}"
    )
    return result


@activity.defn
def score_halal_new_with_patchtst(
    symbols: list[str],
    as_of_date: str,
    min_predictions: int = 15,
) -> PatchTSTBatchScores:
    """Score the US halal_new universe with PatchTST (Alpha-HRP Stage 1).

    Thin HTTP wrapper around ``POST /inference/patchtst/score-batch``
    with ``market='us'``. The math invariants (rank-band selector
    contract: non-finite rejection + ``min_predictions`` floor) live
    in :mod:`brain_api.core.patchtst.score_validation` so US and India
    cannot drift. The activity name and signature are preserved for
    Temporal replay safety.

    Raises:
        RuntimeError: Re-raised from a 422 brain_api response when
            either invariant is violated. Per AGENTS.md "no silent
            fallbacks", this is terminal -- the operator must fix the
            underlying batch (typically an exploded model) before
            rerunning.
    """
    return _score_with_patchtst(
        market="us",
        symbols=symbols,
        as_of_date=as_of_date,
        min_predictions=min_predictions,
        log_prefix="[AlphaHRP US]",
    )


@activity.defn
def score_halal_india_with_patchtst(
    symbols: list[str],
    as_of_date: str,
    min_predictions: int = 15,
) -> PatchTSTBatchScores:
    """Score the India Nifty Shariah 500 universe with PatchTST.

    Thin HTTP wrapper around ``POST /inference/patchtst/score-batch``
    with ``market='india'``. Identical structure to the US activity --
    same math invariants, different trained weights. The brain_api
    endpoint resolves to ``PatchTSTIndiaModelStorage`` based on the
    ``market`` field; the rank-band score validation policy is shared.
    """
    return _score_with_patchtst(
        market="india",
        symbols=symbols,
        as_of_date=as_of_date,
        min_predictions=min_predictions,
        log_prefix="[AlphaHRP India]",
    )


@activity.defn
def select_rank_band_top_n(
    scores: dict[str, float],
    universe: str,
    year_week: str,
    as_of_date: str,
    run_id: str,
    top_n: int = 15,
    hold_threshold: int = 20,
) -> RankBandTopNResponse:
    """Apply rank-band sticky selection to a PatchTST batch result.

    POSTs to brain_api's /allocation/rank-band-top-n which persists the
    score rows (universe-scoped) and returns the chosen symbols with
    sticky vs top_rank provenance. Activity name mirrors the route
    path so synonyms (``alpha`` vs ``rank-band``) do not drift across
    layers.
    """
    logger.info(
        f"[RankBand] {universe}/{year_week}: top_n={top_n} K_hold={hold_threshold}"
    )
    with get_client() as client:
        response = client.post(
            "/allocation/rank-band-top-n",
            json={
                "current_scores": scores,
                "universe": universe,
                "year_week": year_week,
                "as_of_date": as_of_date,
                "run_id": run_id,
                "top_n": top_n,
                "hold_threshold": hold_threshold,
            },
        )
        response.raise_for_status()
    result = RankBandTopNResponse(**response.json())
    logger.info(
        f"[RankBand] kept={result.kept_count} fillers={result.fillers_count} "
        f"prev_yw={result.previous_year_week_used} "
        f"evicted={len(result.evicted_from_previous)}"
    )
    return result
