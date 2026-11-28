"""Score the current weekly regressor with complete exchange-session evidence."""

import logging
from datetime import date

import numpy as np

from brain_api.core.model_types import classify_direction
from brain_api.core.patchtst.calendar import inference_calendar
from brain_api.core.patchtst.config import patchtst_recipe
from brain_api.core.patchtst.india_prices import restore_nse_cash_bar_gaps
from brain_api.core.patchtst.inference import BatchInferenceResult, SymbolPrediction
from brain_api.core.patchtst.weekly_data import WeeklyPatchTSTInputs
from brain_api.core.patchtst.weekly_training import predict_weekly_returns
from brain_api.core.prices import load_prices_yfinance
from brain_api.news.store import NewsStore

logger = logging.getLogger(__name__)


def run_weekly_batch(
    symbols, cutoff: date, artifacts, *, base_path=None
) -> BatchInferenceResult:
    config = artifacts.config
    calendar = inference_calendar(config.market, cutoff)
    anchor = calendar.anchor(cutoff)
    end = calendar.forecast_end(anchor)
    boundaries = calendar.week(cutoff)
    prices = (
        load_prices_yfinance(
            symbols,
            calendar.sessions[0].date(),
            anchor,
            required_columns=("close", "volume"),
        )
        if symbols
        else {}
    )
    if symbols and not prices:
        raise ValueError("No price evidence available for weekly PatchTST")
    closes = calendar.closes.loc[: str(anchor)]
    if config.market == "india":
        prices = restore_nse_cash_bar_gaps(prices, closes.index)
    inputs = WeeklyPatchTSTInputs(
        config,
        type(calendar)(closes),
        prices,
        NewsStore(base_path) if "news_sentiment" in config.feature_names else None,
    )
    windows, omitted = {}, []
    for symbol in dict.fromkeys(symbols):
        window, count = inputs.window(symbol, anchor)
        if window is None:
            omitted.append((symbol, count))
        else:
            windows[symbol] = (window, count)
    logger.info(
        "PatchTST weekly inputs model=%s anchor=%s usable=%s requested=%s",
        artifacts.version,
        anchor,
        len(windows),
        len(set(symbols)),
    )
    if symbols and not windows:
        available = [inputs.features[s] for s in set(symbols) if s in inputs.features]
        if available and "news_sentiment" in config.feature_names:
            channel = config.feature_names.index("news_sentiment")
            missing_news = sum(
                not np.isfinite(values[-1, channel]) for values in available
            )
            if missing_news == len(available):
                raise ValueError(
                    f"Missing compatible news sentiment at US session close {anchor} "
                    f"for {missing_news}/{len(set(symbols))} requested symbols "
                    f"(PatchTST {artifacts.version}); no usable forecasts. "
                    "Refresh closed-session news with POST /etl/news/patchtst "
                    f'body {{"session_date": "{anchor}"}}, poll its job until completed, '
                    "then retry. Monday news coverage alone does not establish "
                    "session-close coverage. No ranking fallback was used."
                )
        raise ValueError(
            f"PatchTST {artifacts.version} has no usable input windows at {anchor} "
            f"({len(set(symbols))} requested). Check price history and compatible "
            "feature/news coverage over the model's context window."
        )
    scores = (
        predict_weekly_returns(
            artifacts.model,
            np.stack([value[0] for value in windows.values()]),
            artifacts.feature_scaler,
        )
        if windows
        else []
    )
    results = []
    usable = [
        (symbol, count, float(score))
        for (symbol, (_, count)), score in zip(windows.items(), scores, strict=True)
    ]
    for symbol, count, score in [
        *usable,
        *((symbol, count, None) for symbol, count in omitted),
    ]:
        results.append(
            SymbolPrediction(
                symbol=symbol,
                predicted_weekly_return_pct=None if score is None else score * 100,
                direction="FLAT" if score is None else classify_direction(score),
                has_enough_history=score is not None,
                history_days_used=count,
                data_end_date=anchor.isoformat() if score is not None else None,
                target_week_start=boundaries.target_week_start.isoformat(),
                target_week_end=boundaries.target_week_end.isoformat(),
            )
        )
    results.sort(
        key=lambda item: (
            item.predicted_weekly_return_pct is None,
            -(item.predicted_weekly_return_pct or 0),
        )
    )
    return BatchInferenceResult(
        results, artifacts.version, boundaries, anchor, end, patchtst_recipe(config)
    )
