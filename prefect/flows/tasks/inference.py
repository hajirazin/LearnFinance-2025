"""Signals, forecasts, and allocator tasks."""

from prefect import task
from prefect.logging import get_run_logger

from flows.models import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    PatchTSTInferenceResponse,
    PPOInferenceResponse,
    SACInferenceResponse,
)
from flows.tasks.client import get_client

# =============================================================================
# Signals + Forecasts Tasks
# =============================================================================


@task(name="Get Fundamentals", retries=2, retry_delay_seconds=30)
def get_fundamentals(symbols: list[str]) -> FundamentalsResponse:
    """Fetch fundamental data for symbols."""
    logger = get_run_logger()
    logger.info(f"Fetching fundamentals for {len(symbols)} symbols...")

    with get_client() as client:
        response = client.post("/signals/fundamentals", json={"symbols": symbols})
        response.raise_for_status()
        data = response.json()

    result = FundamentalsResponse(**data)
    logger.info(f"Got fundamentals for {len(result.per_symbol)} symbols")
    return result


@task(name="Get News Sentiment", retries=2, retry_delay_seconds=60)
def get_news_sentiment(
    symbols: list[str], as_of_date: str, run_id: str
) -> NewsSignalResponse:
    """Fetch news sentiment for symbols."""
    logger = get_run_logger()
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
        data = response.json()

    result = NewsSignalResponse(**data)
    logger.info(f"Got news sentiment for {len(result.per_symbol)} symbols")
    return result


@task(name="Get LSTM Forecast", retries=2, retry_delay_seconds=60)
def get_lstm_forecast(as_of_date: str) -> LSTMInferenceResponse:
    """Get LSTM price predictions. Symbols resolved by brain_api from model metadata."""
    logger = get_run_logger()
    logger.info("Getting LSTM forecast (symbols from model metadata)...")

    with get_client() as client:
        response = client.post(
            "/inference/lstm",
            json={"as_of_date": as_of_date},
        )
        response.raise_for_status()
        data = response.json()

    result = LSTMInferenceResponse(**data)
    logger.info(
        f"Got LSTM predictions: {len(result.predictions)} symbols, "
        f"version={result.model_version}"
    )
    return result


@task(name="Get PatchTST Forecast", retries=2, retry_delay_seconds=60)
def get_patchtst_forecast(as_of_date: str) -> PatchTSTInferenceResponse:
    """Get PatchTST price predictions.

    Symbols resolved by brain_api from model metadata.
    """
    logger = get_run_logger()
    logger.info("Getting PatchTST forecast (symbols from model metadata)...")

    with get_client() as client:
        response = client.post(
            "/inference/patchtst",
            json={"as_of_date": as_of_date},
        )
        response.raise_for_status()
        data = response.json()

    result = PatchTSTInferenceResponse(**data)
    logger.info(
        f"Got PatchTST predictions: {len(result.predictions)} symbols, "
        f"version={result.model_version}"
    )
    return result


# =============================================================================
# Allocator Tasks
# =============================================================================


@task(name="Infer PPO", retries=1, retry_delay_seconds=60)
def infer_ppo(
    portfolio: AlpacaPortfolioResponse, as_of_date: str
) -> PPOInferenceResponse:
    """Get PPO allocation."""
    logger = get_run_logger()
    logger.info("Getting PPO allocation...")

    with get_client() as client:
        response = client.post(
            "/inference/ppo",
            json={
                "portfolio": {
                    "cash": portfolio.cash,
                    "positions": [p.model_dump() for p in portfolio.positions],
                },
                "as_of_date": as_of_date,
            },
        )
        response.raise_for_status()
        data = response.json()

    result = PPOInferenceResponse(**data)
    logger.info(
        f"PPO allocation: {len(result.target_weights)} positions, "
        f"turnover={result.turnover:.2%}"
    )
    return result


@task(name="Infer SAC", retries=1, retry_delay_seconds=60)
def infer_sac(
    portfolio: AlpacaPortfolioResponse, as_of_date: str
) -> SACInferenceResponse:
    """Get SAC allocation."""
    logger = get_run_logger()
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
        data = response.json()

    result = SACInferenceResponse(**data)
    logger.info(
        f"SAC allocation: {len(result.target_weights)} positions, "
        f"turnover={result.turnover:.2%}"
    )
    return result


@task(name="Allocate HRP", retries=1, retry_delay_seconds=60)
def allocate_hrp(as_of_date: str) -> HRPAllocationResponse:
    """Get HRP allocation."""
    logger = get_run_logger()
    logger.info("Getting HRP allocation...")

    with get_client() as client:
        response = client.post(
            "/allocation/hrp",
            json={"as_of_date": as_of_date},
        )
        response.raise_for_status()
        data = response.json()

    result = HRPAllocationResponse(**data)
    logger.info(
        f"HRP allocation: {result.symbols_used} symbols, "
        f"excluded={len(result.symbols_excluded)}"
    )
    return result
