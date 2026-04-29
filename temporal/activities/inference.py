"""Signals, forecasts, and allocator activities."""

import logging

from temporalio import activity

from activities.client import get_client
from models import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
    RecordFinalWeightsResponse,
    SACInferenceResponse,
    StickyTopNResponse,
)

logger = logging.getLogger(__name__)


@activity.defn
def get_fundamentals(symbols: list[str]) -> FundamentalsResponse:
    """Fetch fundamental data for symbols."""
    logger.info(f"Fetching fundamentals for {len(symbols)} symbols...")
    with get_client() as client:
        response = client.post("/signals/fundamentals", json={"symbols": symbols})
        response.raise_for_status()
    result = FundamentalsResponse(**response.json())
    logger.info(f"Got fundamentals for {len(result.per_symbol)} symbols")
    return result


@activity.defn
def get_news_sentiment(
    symbols: list[str], as_of_date: str, run_id: str
) -> NewsSignalResponse:
    """Fetch news sentiment for symbols."""
    logger.info(f"Fetching news sentiment for {len(symbols)} symbols...")
    with get_client() as client:
        response = client.post(
            "/signals/news",
            json={
                "symbols": symbols,
                "as_of_date": as_of_date,
                "run_id": run_id,
                "max_articles_per_symbol": 10,
                "return_top_k": 3,
            },
        )
        response.raise_for_status()
    result = NewsSignalResponse(**response.json())
    logger.info(f"Got news sentiment for {len(result.per_symbol)} symbols")
    return result


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
def get_halal_india_universe() -> dict:
    """Validate and fetch the halal_india universe from NSE Nifty 500 Shariah."""
    logger.info("Fetching halal_india universe...")
    with get_client() as client:
        response = client.get("/universe/halal_india")
        response.raise_for_status()
    data = response.json()
    stock_count = len(data.get("stocks", []))
    logger.info(
        f"Halal India universe: {stock_count} stocks, "
        f"source={data.get('source', 'unknown')}"
    )
    return data


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
    portfolio: AlpacaPortfolioResponse, as_of_date: str
) -> SACInferenceResponse:
    """Get SAC allocation."""
    logger.info("Getting SAC allocation...")
    with get_client() as client:
        response = client.post(
            "/inference/sac",
            json={
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "as_of_date": as_of_date,
            },
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
