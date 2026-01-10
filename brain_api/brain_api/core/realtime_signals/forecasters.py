"""Forecaster classes for RL inference.

Each forecaster is responsible for:
1. Loading its model artifacts
2. Fetching price data
3. Building features
4. Running inference
5. Returning predictions as decimal returns
"""

import logging
from abc import ABC, abstractmethod
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for forecasters."""

    @abstractmethod
    def build_forecasts(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, float]:
        """Build forecast features for RL inference.

        Args:
            symbols: List of stock ticker symbols
            as_of_date: Reference date for inference

        Returns:
            Dict mapping symbol -> predicted weekly return (decimal, not percentage)
        """
        ...


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecaster for RL inference."""

    def build_forecasts(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, float]:
        """Build LSTM forecast features.

        Args:
            symbols: List of stock ticker symbols
            as_of_date: Reference date for inference

        Returns:
            Dict mapping symbol -> predicted weekly return (decimal)
        """
        from brain_api.core.lstm import (
            InferenceFeatures,
            build_inference_features,
            compute_week_boundaries,
            load_prices_yfinance,
            run_inference,
        )
        from brain_api.storage.local import LocalModelStorage

        logger.info(f"[LSTMForecaster] Building forecasts for {len(symbols)} symbols")

        # Initialize with zeros (fallback)
        forecasts: dict[str, float] = dict.fromkeys(symbols, 0.0)

        try:
            # Load model
            storage = LocalModelStorage()
            artifacts = storage.load_current_artifacts()
            config = artifacts.config

            # Compute week boundaries
            week_boundaries = compute_week_boundaries(as_of_date)

            # Fetch prices
            buffer_days = config.sequence_length * 2 + 30
            data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
            data_end = week_boundaries.target_week_start - timedelta(days=1)
            prices = load_prices_yfinance(symbols, data_start, data_end)

            # Build features
            features_list = []
            for symbol in symbols:
                prices_df = prices.get(symbol)
                if prices_df is None or prices_df.empty:
                    features_list.append(
                        InferenceFeatures(
                            symbol=symbol,
                            features=None,
                            has_enough_history=False,
                            history_days_used=0,
                            data_end_date=None,
                        )
                    )
                else:
                    features = build_inference_features(
                        symbol=symbol,
                        prices_df=prices_df,
                        config=config,
                        cutoff_date=week_boundaries.target_week_start,
                    )
                    features_list.append(features)

            # Run inference
            predictions = run_inference(
                model=artifacts.model,
                feature_scaler=artifacts.feature_scaler,
                features_list=features_list,
                week_boundaries=week_boundaries,
            )

            # Extract forecasts (convert from pct to decimal)
            valid_count = 0
            for pred in predictions:
                if pred.predicted_weekly_return_pct is not None:
                    forecasts[pred.symbol] = pred.predicted_weekly_return_pct / 100.0
                    valid_count += 1

            logger.info(
                f"[LSTMForecaster] Generated forecasts for {valid_count} symbols"
            )

        except Exception as e:
            logger.warning(f"[LSTMForecaster] Failed to generate forecasts: {e}")

        return forecasts


class PatchTSTForecaster(BaseForecaster):
    """PatchTST-based forecaster for RL inference."""

    def build_forecasts(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, float]:
        """Build PatchTST forecast features.

        Args:
            symbols: List of stock ticker symbols
            as_of_date: Reference date for inference

        Returns:
            Dict mapping symbol -> predicted weekly return (decimal)
        """
        from brain_api.core.lstm import compute_week_boundaries
        from brain_api.core.patchtst import (
            InferenceFeatures,
            build_inference_features,
            load_prices_yfinance,
            run_inference,
        )
        from brain_api.storage.local import PatchTSTModelStorage

        logger.info(
            f"[PatchTSTForecaster] Building forecasts for {len(symbols)} symbols"
        )

        # Initialize with zeros (fallback)
        forecasts: dict[str, float] = dict.fromkeys(symbols, 0.0)

        try:
            # Load model
            storage = PatchTSTModelStorage()
            artifacts = storage.load_current_artifacts()
            config = artifacts.config

            # Compute week boundaries
            week_boundaries = compute_week_boundaries(as_of_date)

            # Fetch prices
            buffer_days = config.context_length * 2 + 30
            data_start = week_boundaries.target_week_start - timedelta(days=buffer_days)
            data_end = week_boundaries.target_week_start - timedelta(days=1)
            prices = load_prices_yfinance(symbols, data_start, data_end)

            # Build features
            features_list = []
            for symbol in symbols:
                prices_df = prices.get(symbol)
                if prices_df is None or prices_df.empty:
                    features_list.append(
                        InferenceFeatures(
                            symbol=symbol,
                            features=None,
                            has_enough_history=False,
                            history_days_used=0,
                            data_end_date=None,
                        )
                    )
                else:
                    features = build_inference_features(
                        symbol=symbol,
                        prices_df=prices_df,
                        config=config,
                        cutoff_date=week_boundaries.target_week_start,
                    )
                    features_list.append(features)

            # Run inference
            predictions = run_inference(
                model=artifacts.model,
                feature_scaler=artifacts.feature_scaler,
                features_list=features_list,
                week_boundaries=week_boundaries,
                config=config,
            )

            # Extract forecasts (convert from pct to decimal)
            valid_count = 0
            for pred in predictions:
                if pred.predicted_weekly_return_pct is not None:
                    forecasts[pred.symbol] = pred.predicted_weekly_return_pct / 100.0
                    valid_count += 1

            logger.info(
                f"[PatchTSTForecaster] Generated forecasts for {valid_count} symbols"
            )

        except Exception as e:
            logger.warning(f"[PatchTSTForecaster] Failed to generate forecasts: {e}")

        return forecasts
