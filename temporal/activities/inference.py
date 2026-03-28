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
    SACInferenceResponse,
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
    as_of_date: str, universe: str = "halal_filtered"
) -> HRPAllocationResponse:
    """Get HRP allocation for the given universe."""
    logger.info(f"Getting HRP allocation (universe={universe})...")
    with get_client() as client:
        response = client.post(
            "/allocation/hrp",
            json={"as_of_date": as_of_date, "universe": universe},
        )
        response.raise_for_status()
    result = HRPAllocationResponse(**response.json())
    logger.info(
        f"HRP allocation: {result.symbols_used} symbols, "
        f"universe={result.universe}, excluded={len(result.symbols_excluded)}"
    )
    return result
