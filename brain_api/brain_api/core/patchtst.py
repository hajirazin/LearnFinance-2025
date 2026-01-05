"""PatchTST model training and evaluation logic.

PatchTST is a multi-signal transformer that uses OHLCV + external signals
(news sentiment, fundamentals) to predict weekly returns. This contrasts
with the pure-price LSTM baseline.

Key differences from LSTM:
- Multi-channel input: OHLCV (5) + News (1) + Fundamentals (5) = 11 features
- Transformer architecture with patching for efficiency
- Same prediction target: weekly return (Mon open → Fri close)
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTConfig as HFPatchTSTConfig
from transformers import PatchTSTForPrediction

from brain_api.core.prices import load_prices_yfinance as _load_prices_shared

# ============================================================================
# Device Detection (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
# ============================================================================


def get_device() -> torch.device:
    """Get the best available device for training.

    Priority:
    1. MPS (Apple Silicon GPU) - for M1/M2/M3 Macs
    2. CUDA (NVIDIA GPU)
    3. CPU (fallback)

    Returns:
        torch.device for the best available accelerator
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Configuration / Hyperparameters
# ============================================================================


@dataclass
class PatchTSTConfig:
    """PatchTST model hyperparameters and training config.

    PatchTST predicts weekly returns using multi-channel input:
    - OHLCV log returns (5 channels)
    - News sentiment (1 channel)
    - Fundamentals (5 channels): gross_margin, operating_margin, net_margin,
      current_ratio, debt_to_equity

    Total: 11 input channels (without Twitter, which is skipped for now)
    """

    # Model architecture
    num_input_channels: int = 11  # OHLCV (5) + News (1) + Fundamentals (5)
    context_length: int = 60  # 60 trading days lookback (same as LSTM)
    prediction_length: int = 1  # Single output: weekly return
    patch_length: int = 16  # Standard patch size for PatchTST
    stride: int = 8  # 50% overlap between patches

    # Transformer architecture
    d_model: int = 64  # Hidden dimension
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    ffn_dim: int = 128  # Feed-forward network dimension
    dropout: float = 0.2

    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2

    # Feature engineering
    use_returns: bool = True  # Use log returns for OHLCV (more stationary)

    # Week filtering
    min_week_days: int = 3  # Skip weeks with fewer than 3 trading days

    # Feature channel names for documentation
    feature_names: list[str] = field(default_factory=lambda: [
        "open_ret", "high_ret", "low_ret", "close_ret", "volume_ret",  # OHLCV
        "news_sentiment",  # News
        "gross_margin", "operating_margin", "net_margin",  # Fundamentals
        "current_ratio", "debt_to_equity",
    ])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_input_channels": self.num_input_channels,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "d_model": self.d_model,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "use_returns": self.use_returns,
            "min_week_days": self.min_week_days,
            "feature_names": self.feature_names,
        }

    def to_hf_config(self) -> HFPatchTSTConfig:
        """Convert to HuggingFace PatchTSTConfig."""
        return HFPatchTSTConfig(
            num_input_channels=self.num_input_channels,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            patch_length=self.patch_length,
            patch_stride=self.stride,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            # Use distribution output for regression
            loss="mse",
            # Scaling for better training
            scaling="std",
        )


DEFAULT_CONFIG = PatchTSTConfig()


# ============================================================================
# Version computation (idempotent)
# ============================================================================


def compute_version(
    start_date: date,
    end_date: date,
    symbols: list[str],
    config: PatchTSTConfig,
) -> str:
    """Compute a deterministic version string for the training run.

    The version is a hash of (window, symbols, config) so that reruns with
    the same inputs produce the same version (idempotent training).

    Returns:
        Version string in format 'v{date_prefix}-{hash_suffix}'
    """
    canonical = {
        "model": "patchtst",  # Distinguish from LSTM versions
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": sorted(symbols),
        "config": config.to_dict(),
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    hash_digest = hashlib.sha256(canonical_json.encode()).hexdigest()[:12]
    date_prefix = end_date.strftime("%Y-%m-%d")
    return f"v{date_prefix}-{hash_digest}"


# ============================================================================
# Data loading
# ============================================================================


def load_prices_yfinance(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV price data for symbols using yfinance.

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window

    Returns:
        Dict mapping symbol -> DataFrame with OHLCV columns and DatetimeIndex
    """
    return _load_prices_shared(symbols, start_date, end_date, log_prefix="[PatchTST]")


def load_historical_news_sentiment(
    symbols: list[str],
    start_date: date,
    end_date: date,
    parquet_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load historical news sentiment from parquet file.

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        parquet_path: Path to daily_sentiment.parquet (defaults to project data/)

    Returns:
        Dict mapping symbol -> DataFrame with 'sentiment_score' column and DatetimeIndex
    """
    if parquet_path is None:
        # Default path: project_root/data/output/daily_sentiment.parquet
        parquet_path = Path(__file__).parent.parent.parent.parent / "data" / "output" / "daily_sentiment.parquet"

    sentiment: dict[str, pd.DataFrame] = {}

    if not parquet_path.exists():
        print(f"[PatchTST] Warning: News sentiment parquet not found at {parquet_path}")
        return sentiment

    try:
        df = pd.read_parquet(parquet_path)
        # Convert date column to string for filtering
        df["date"] = pd.to_datetime(df["date"]).dt.date

        for symbol in symbols:
            symbol_df = df[
                (df["symbol"] == symbol) &
                (df["date"] >= start_date) &
                (df["date"] <= end_date)
            ][["date", "sentiment_score"]].copy()

            if len(symbol_df) > 0:
                symbol_df["date"] = pd.to_datetime(symbol_df["date"])
                symbol_df = symbol_df.set_index("date").sort_index()
                sentiment[symbol] = symbol_df

    except Exception as e:
        print(f"[PatchTST] Error loading news sentiment: {e}")

    return sentiment


def load_historical_fundamentals(
    symbols: list[str],
    start_date: date,
    end_date: date,
    cache_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load historical fundamentals from cache.

    Fundamentals are quarterly data that should be forward-filled to daily.
    Expects cached JSON files from Alpha Vantage at:
    brain_api/data/fundamentals_cache/{symbol}/

    Args:
        symbols: List of ticker symbols
        start_date: Start of data window
        end_date: End of data window
        cache_path: Base path for fundamentals cache

    Returns:
        Dict mapping symbol -> DataFrame with fundamental ratio columns
        and DatetimeIndex (quarterly dates, to be forward-filled later)
    """
    if cache_path is None:
        cache_path = Path(__file__).parent.parent / "data" / "raw" / "fundamentals"

    fundamentals: dict[str, pd.DataFrame] = {}

    # Import fundamentals parsing utilities
    try:
        from brain_api.core.fundamentals import (
            compute_ratios,
            load_raw_response,
            parse_quarterly_statements,
        )
    except ImportError:
        print("[PatchTST] Warning: Could not import fundamentals utilities")
        return fundamentals

    for symbol in symbols:
        try:
            # Load cached responses
            income_data = load_raw_response(cache_path, symbol, "income_statement")
            balance_data = load_raw_response(cache_path, symbol, "balance_sheet")

            if income_data is None and balance_data is None:
                continue

            # Parse statements
            income_stmts = []
            balance_stmts = []

            if income_data:
                income_stmts = parse_quarterly_statements(
                    symbol, "income_statement", income_data
                )
            if balance_data:
                balance_stmts = parse_quarterly_statements(
                    symbol, "balance_sheet", balance_data
                )

            # Collect ratios for each fiscal date
            rows = []
            fiscal_dates = set()

            for stmt in income_stmts:
                if start_date <= date.fromisoformat(stmt.fiscal_date_ending) <= end_date:
                    fiscal_dates.add(stmt.fiscal_date_ending)
            for stmt in balance_stmts:
                if start_date <= date.fromisoformat(stmt.fiscal_date_ending) <= end_date:
                    fiscal_dates.add(stmt.fiscal_date_ending)

            for fiscal_date in sorted(fiscal_dates):
                income_stmt = next(
                    (s for s in income_stmts if s.fiscal_date_ending == fiscal_date),
                    None,
                )
                balance_stmt = next(
                    (s for s in balance_stmts if s.fiscal_date_ending == fiscal_date),
                    None,
                )

                ratios = compute_ratios(income_stmt, balance_stmt)
                if ratios:
                    rows.append({
                        "date": pd.to_datetime(fiscal_date),
                        "gross_margin": ratios.gross_margin,
                        "operating_margin": ratios.operating_margin,
                        "net_margin": ratios.net_margin,
                        "current_ratio": ratios.current_ratio,
                        "debt_to_equity": ratios.debt_to_equity,
                    })

            if rows:
                df = pd.DataFrame(rows).set_index("date").sort_index()
                fundamentals[symbol] = df

        except Exception as e:
            print(f"[PatchTST] Error loading fundamentals for {symbol}: {e}")
            continue

    return fundamentals


# ============================================================================
# Data alignment and feature engineering
# ============================================================================


def align_multivariate_data(
    prices: dict[str, pd.DataFrame],
    news_sentiment: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> dict[str, pd.DataFrame]:
    """Align OHLCV, news sentiment, and fundamentals into multi-channel features.

    Creates a unified DataFrame per symbol with all 11 feature channels:
    - OHLCV log returns (5 channels)
    - News sentiment (1 channel) - forward-filled for missing days
    - Fundamentals (5 channels) - forward-filled quarterly data

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        news_sentiment: Dict of symbol -> sentiment DataFrame
        fundamentals: Dict of symbol -> fundamentals DataFrame
        config: PatchTST configuration

    Returns:
        Dict of symbol -> aligned multi-channel DataFrame
    """
    aligned: dict[str, pd.DataFrame] = {}

    for symbol, price_df in prices.items():
        if len(price_df) < config.context_length + 5:
            continue

        # Start with OHLCV features
        if config.use_returns:
            features_df = pd.DataFrame(
                {
                    "open_ret": np.log(price_df["open"] / price_df["open"].shift(1)),
                    "high_ret": np.log(price_df["high"] / price_df["high"].shift(1)),
                    "low_ret": np.log(price_df["low"] / price_df["low"].shift(1)),
                    "close_ret": np.log(price_df["close"] / price_df["close"].shift(1)),
                    "volume_ret": np.log(
                        price_df["volume"] / price_df["volume"].shift(1).replace(0, 1)
                    ),
                },
                index=price_df.index,
            )
            # Drop first row (NaN from shift)
            features_df = features_df.iloc[1:]
        else:
            features_df = price_df[["open", "high", "low", "close", "volume"]].copy()
            features_df.columns = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]

        # Replace infinities with 0
        features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

        # Add news sentiment (forward-fill missing days)
        if symbol in news_sentiment:
            sentiment_df = news_sentiment[symbol]
            # Reindex to match price dates and forward-fill
            sentiment_aligned = sentiment_df.reindex(features_df.index, method="ffill")
            features_df["news_sentiment"] = sentiment_aligned["sentiment_score"].fillna(0.0)
        else:
            features_df["news_sentiment"] = 0.0  # Neutral if no news data

        # Add fundamentals (forward-fill quarterly data)
        fundamental_cols = ["gross_margin", "operating_margin", "net_margin",
                          "current_ratio", "debt_to_equity"]
        if symbol in fundamentals:
            fund_df = fundamentals[symbol]
            # Reindex to match price dates and forward-fill
            fund_aligned = fund_df.reindex(features_df.index, method="ffill")
            for col in fundamental_cols:
                if col in fund_aligned.columns:
                    features_df[col] = fund_aligned[col].fillna(0.0)
                else:
                    features_df[col] = 0.0
        else:
            # No fundamentals - use zeros
            for col in fundamental_cols:
                features_df[col] = 0.0

        # Ensure column order matches config.feature_names
        features_df = features_df[config.feature_names]

        if len(features_df) >= config.context_length:
            aligned[symbol] = features_df

    return aligned


# ============================================================================
# Dataset building (weekly return prediction)
# ============================================================================


@dataclass
class DatasetResult:
    """Result of dataset building for weekly return prediction."""

    X: np.ndarray  # Input sequences: (n_samples, context_length, n_channels)
    y: np.ndarray  # Targets: (n_samples, prediction_length) - weekly returns
    feature_scaler: StandardScaler  # Scaler for input features


def _extract_trading_weeks(df: pd.DataFrame, min_days: int = 3) -> list[pd.DataFrame]:
    """Extract trading weeks from a price DataFrame.

    Groups data by ISO week and filters out weeks with too few trading days.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return []

    df = df.copy()
    df["_year_week"] = df.index.to_period("W")

    weeks = []
    for _, week_df in df.groupby("_year_week"):
        if len(week_df) >= min_days:
            weeks.append(week_df.drop(columns=["_year_week"]))

    return weeks


def _compute_weekly_return(
    prices_df: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> float | None:
    """Compute weekly return for a trading week.

    Args:
        prices_df: DataFrame with 'open' and 'close' columns
        week_start: First trading day of the week
        week_end: Last trading day of the week

    Returns:
        Weekly return = (week_end_close - week_start_open) / week_start_open
        or None if data is missing
    """
    try:
        week_data = prices_df.loc[week_start:week_end]
        if len(week_data) < 1:
            return None

        first_day_open = week_data["open"].iloc[0]
        last_day_close = week_data["close"].iloc[-1]

        if first_day_open == 0:
            return 0.0

        return (last_day_close - first_day_open) / first_day_open
    except Exception:
        return None


def build_dataset(
    aligned_features: dict[str, pd.DataFrame],
    prices: dict[str, pd.DataFrame],
    config: PatchTSTConfig,
) -> DatasetResult:
    """Build training dataset for weekly return prediction.

    Creates samples aligned to trading weeks:
    - Input: context_length days of multi-channel features ending at week start
    - Target: weekly return (fri_close - mon_open) / mon_open

    Args:
        aligned_features: Dict of symbol -> aligned multi-channel DataFrame
        prices: Dict of symbol -> raw OHLCV DataFrame (for computing weekly returns)
        config: PatchTST configuration

    Returns:
        DatasetResult with X, y (weekly returns), and feature_scaler
    """
    all_sequences = []
    all_targets = []

    print(f"[PatchTST] Building dataset from {len(aligned_features)} symbols...")
    symbols_used = 0
    total_weeks = 0

    for symbol, features_df in aligned_features.items():
        if symbol not in prices:
            continue

        price_df = prices[symbol]

        # Ensure price_df has same index type
        if not isinstance(price_df.index, pd.DatetimeIndex):
            continue

        # Extract trading weeks from features
        weeks = _extract_trading_weeks(features_df, min_days=config.min_week_days)

        if len(weeks) < 2:
            continue

        symbol_samples = 0

        # For each week (except the last), create a training sample
        for i in range(len(weeks) - 1):
            week = weeks[i + 1]  # Target week
            week_start = week.index[0]
            week_end = week.index[-1]

            # Find position in features_df
            try:
                week_start_idx = features_df.index.get_loc(week_start)
            except KeyError:
                continue

            # Check if we have enough history
            if week_start_idx < config.context_length:
                continue

            # Extract input sequence
            seq_start_idx = week_start_idx - config.context_length
            seq_end_idx = week_start_idx

            sequence = features_df.iloc[seq_start_idx:seq_end_idx].values

            if len(sequence) != config.context_length:
                continue

            # Compute target: weekly return
            weekly_return = _compute_weekly_return(price_df, week_start, week_end)
            if weekly_return is None:
                continue

            all_sequences.append(sequence)
            all_targets.append([weekly_return])
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_weeks += symbol_samples

    print(f"[PatchTST] Dataset built: {total_weeks} weekly samples from {symbols_used} symbols")

    if not all_sequences:
        empty_X = np.array([]).reshape(0, config.context_length, config.num_input_channels)
        empty_y = np.array([]).reshape(0, config.prediction_length)
        return DatasetResult(
            X=empty_X,
            y=empty_y,
            feature_scaler=StandardScaler(),
        )

    X = np.array(all_sequences)
    y = np.array(all_targets)

    # Fit feature scaler on input sequences
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    feature_scaler = StandardScaler()
    X_flat_scaled = feature_scaler.fit_transform(X_flat)
    X = X_flat_scaled.reshape(original_shape)

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )


# ============================================================================
# PyTorch Training with HuggingFace PatchTST
# ============================================================================


@dataclass
class TrainingResult:
    """Result of PatchTST training."""

    model: PatchTSTForPrediction
    feature_scaler: StandardScaler
    config: PatchTSTConfig
    train_loss: float
    val_loss: float
    baseline_loss: float  # Baseline: predict 0 return


def train_model_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    config: PatchTSTConfig,
) -> TrainingResult:
    """Train PatchTST model using HuggingFace Transformers.

    Args:
        X: Input sequences, shape (n_samples, context_length, n_channels)
        y: Targets, shape (n_samples, prediction_length)
        feature_scaler: Fitted scaler for input features
        config: Model configuration

    Returns:
        TrainingResult with trained model and metrics
    """
    device = get_device()
    print(f"[PatchTST] Training on device: {device}")

    if len(X) == 0:
        print("[PatchTST] No training data - returning untrained model")
        hf_config = config.to_hf_config()
        model = PatchTSTForPrediction(hf_config)
        return TrainingResult(
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            train_loss=float("inf"),
            val_loss=float("inf"),
            baseline_loss=float("inf"),
        )

    # Train/val split (time-based)
    split_idx = int(len(X) * (1 - config.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"[PatchTST] Dataset: {len(X)} samples ({len(X_train)} train, {len(X_val)} val)")
    print(f"[PatchTST] Config: {config.epochs} epochs, batch_size={config.batch_size}, lr={config.learning_rate}")

    # HuggingFace PatchTST expects shape (batch, sequence_length, num_channels)
    # Our X is already in that shape from build_dataset, no transpose needed

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create model
    hf_config = config.to_hf_config()
    model = PatchTSTForPrediction(hf_config).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    log_interval = max(1, config.epochs // 5)

    print("[PatchTST] Starting training...")

    for epoch in range(config.epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train_t), device=device)
        total_train_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), config.batch_size):
            batch_idx = indices[i : i + config.batch_size]
            batch_X = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]

            optimizer.zero_grad()

            # PatchTST forward pass
            outputs = model(past_values=batch_X)
            # Get predictions - shape depends on model config
            predictions = outputs.prediction_outputs
            if predictions.dim() == 3:
                predictions = predictions.mean(dim=-1)  # Average over samples if distribution
            if predictions.dim() == 2 and predictions.shape[1] > 1:
                predictions = predictions[:, :config.prediction_length]

            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(past_values=X_val_t)
            val_predictions = val_outputs.prediction_outputs
            if val_predictions.dim() == 3:
                val_predictions = val_predictions.mean(dim=-1)
            if val_predictions.dim() == 2 and val_predictions.shape[1] > 1:
                val_predictions = val_predictions[:, :config.prediction_length]

            val_loss = criterion(val_predictions, y_val_t).item()

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"[PatchTST] Epoch {epoch + 1}/{config.epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"[PatchTST] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")

    # Restore best model on CPU
    model_cpu = PatchTSTForPrediction(hf_config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    # Final metrics
    model.eval()
    with torch.no_grad():
        train_outputs = model(past_values=X_train_t)
        train_predictions = train_outputs.prediction_outputs
        if train_predictions.dim() == 3:
            train_predictions = train_predictions.mean(dim=-1)
        if train_predictions.dim() == 2 and train_predictions.shape[1] > 1:
            train_predictions = train_predictions[:, :config.prediction_length]
        final_train_loss = criterion(train_predictions, y_train_t).item()

    # Baseline: predict 0 return
    # Transpose y_val back to original shape for baseline calculation
    y_val_original = y[split_idx:]
    baseline_loss = float(np.mean(y_val_original ** 2))

    print(f"[PatchTST] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}")
    beats_baseline = best_val_loss < baseline_loss
    print(f"[PatchTST] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    return TrainingResult(
        model=model_cpu,
        feature_scaler=feature_scaler,
        config=config,
        train_loss=final_train_loss,
        val_loss=best_val_loss,
        baseline_loss=baseline_loss,
    )


# ============================================================================
# Evaluation / Promotion Gate
# ============================================================================


def evaluate_for_promotion(
    val_loss: float,
    baseline_loss: float,
    prior_val_loss: float | None,
) -> bool:
    """Decide whether to promote the new model to current.

    Promotion requires:
    1. Beat the baseline (persistence model)
    2. Beat the prior model (if one exists)

    Args:
        val_loss: Validation loss of new model
        baseline_loss: Loss of persistence baseline
        prior_val_loss: Validation loss of prior model (None if first model)

    Returns:
        True if model should be promoted
    """
    # Must beat baseline
    if val_loss >= baseline_loss:
        return False

    # Must beat prior (if exists)
    return prior_val_loss is None or val_loss < prior_val_loss


# ============================================================================
# Inference helpers
# ============================================================================


@dataclass
class WeekBoundaries:
    """Trading week boundaries for inference.

    Represents the target week for prediction, computed with holiday awareness.
    """

    target_week_start: date  # First trading day of the week (Mon or later if holiday)
    target_week_end: date  # Last trading day of the week (Fri or earlier if holiday)
    calendar_monday: date  # Calendar Monday of the ISO week
    calendar_friday: date  # Calendar Friday of the ISO week


def compute_week_boundaries(as_of_date: date) -> WeekBoundaries:
    """Compute holiday-aware week boundaries for the week containing as_of_date.

    Uses the NYSE calendar (XNYS) to determine actual trading days.
    The target week is the ISO week that contains as_of_date.

    Args:
        as_of_date: Reference date (typically the Monday when inference runs)

    Returns:
        WeekBoundaries with actual trading day start/end for the week
    """
    from datetime import timedelta

    import exchange_calendars as xcals

    # Get NYSE calendar
    nyse = xcals.get_calendar("XNYS")

    # Find the Monday of the ISO week containing as_of_date
    days_since_monday = as_of_date.weekday()
    calendar_monday = as_of_date - timedelta(days=days_since_monday)
    calendar_friday = calendar_monday + timedelta(days=4)

    # Convert to pandas Timestamp for exchange_calendars
    monday_ts = pd.Timestamp(calendar_monday)
    friday_ts = pd.Timestamp(calendar_friday)

    # Find trading days in the week
    schedule = nyse.sessions_in_range(monday_ts, friday_ts)

    if len(schedule) == 0:
        # Entire week is holiday
        return WeekBoundaries(
            target_week_start=calendar_monday,
            target_week_end=calendar_friday,
            calendar_monday=calendar_monday,
            calendar_friday=calendar_friday,
        )

    target_week_start = schedule[0].date()
    target_week_end = schedule[-1].date()

    return WeekBoundaries(
        target_week_start=target_week_start,
        target_week_end=target_week_end,
        calendar_monday=calendar_monday,
        calendar_friday=calendar_friday,
    )


@dataclass
class InferenceFeatures:
    """Multi-channel features prepared for inference for a single symbol."""

    symbol: str
    features: np.ndarray | None  # Shape: (context_length, num_channels) or None
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None
    has_news_data: bool  # Whether news sentiment data was available
    has_fundamentals_data: bool  # Whether fundamentals data was available


def build_inference_features(
    symbol: str,
    prices_df: pd.DataFrame,
    news_df: pd.DataFrame | None,
    fundamentals_df: pd.DataFrame | None,
    config: PatchTSTConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build multi-channel feature sequence for inference.

    Constructs the same features as training:
    - OHLCV log returns (5 channels)
    - News sentiment (1 channel) - forward-filled
    - Fundamentals (5 channels) - forward-filled

    Args:
        symbol: Ticker symbol
        prices_df: DataFrame with OHLCV columns and DatetimeIndex
        news_df: DataFrame with 'sentiment_score' column and DatetimeIndex (or None)
        fundamentals_df: DataFrame with fundamental ratio columns and DatetimeIndex (or None)
        config: PatchTST config with context_length and feature settings
        cutoff_date: Features end before this date (typically target_week_start)

    Returns:
        InferenceFeatures with prepared multi-channel feature sequence
    """
    if prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            has_news_data=False,
            has_fundamentals_data=False,
        )

    # Ensure DatetimeIndex
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
            has_news_data=False,
            has_fundamentals_data=False,
        )

    # Filter to data before cutoff_date
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = prices_df[prices_df.index < cutoff_ts].copy()

    if len(df) < config.context_length + 1:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(df),
            data_end_date=df.index[-1].date() if len(df) > 0 else None,
            has_news_data=news_df is not None and len(news_df) > 0,
            has_fundamentals_data=fundamentals_df is not None and len(fundamentals_df) > 0,
        )

    # Compute OHLCV features (log returns)
    if config.use_returns:
        features_df = pd.DataFrame(
            {
                "open_ret": np.log(df["open"] / df["open"].shift(1)),
                "high_ret": np.log(df["high"] / df["high"].shift(1)),
                "low_ret": np.log(df["low"] / df["low"].shift(1)),
                "close_ret": np.log(df["close"] / df["close"].shift(1)),
                "volume_ret": np.log(
                    df["volume"] / df["volume"].shift(1).replace(0, 1)
                ),
            },
            index=df.index,
        )
        features_df = features_df.iloc[1:]  # Drop first row (NaN from shift)
    else:
        features_df = df[["open", "high", "low", "close", "volume"]].copy()
        features_df.columns = ["open_ret", "high_ret", "low_ret", "close_ret", "volume_ret"]

    # Replace infinities with 0
    features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Add news sentiment (forward-fill missing days)
    has_news = False
    if news_df is not None and len(news_df) > 0:
        news_aligned = news_df.reindex(features_df.index, method="ffill")
        features_df["news_sentiment"] = news_aligned["sentiment_score"].fillna(0.0)
        has_news = True
    else:
        features_df["news_sentiment"] = 0.0

    # Add fundamentals (forward-fill quarterly data)
    fundamental_cols = ["gross_margin", "operating_margin", "net_margin",
                       "current_ratio", "debt_to_equity"]
    has_fundamentals = False
    if fundamentals_df is not None and len(fundamentals_df) > 0:
        fund_aligned = fundamentals_df.reindex(features_df.index, method="ffill")
        for col in fundamental_cols:
            if col in fund_aligned.columns:
                features_df[col] = fund_aligned[col].fillna(0.0)
            else:
                features_df[col] = 0.0
        has_fundamentals = True
    else:
        for col in fundamental_cols:
            features_df[col] = 0.0

    # Ensure column order matches config.feature_names
    features_df = features_df[config.feature_names]

    # Check if we have enough data after computing returns
    if len(features_df) < config.context_length:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(features_df),
            data_end_date=features_df.index[-1].date() if len(features_df) > 0 else None,
            has_news_data=has_news,
            has_fundamentals_data=has_fundamentals,
        )

    # Take the last context_length rows
    sequence = features_df.iloc[-config.context_length:].values
    data_end_date = features_df.index[-1].date()

    return InferenceFeatures(
        symbol=symbol,
        features=sequence,
        has_enough_history=True,
        history_days_used=len(features_df),
        data_end_date=data_end_date,
        has_news_data=has_news,
        has_fundamentals_data=has_fundamentals,
    )


@dataclass
class SymbolPrediction:
    """Prediction result for a single symbol."""

    symbol: str
    predicted_weekly_return_pct: float | None  # Percentage (e.g., 2.5 for +2.5%)
    direction: str  # "UP", "DOWN", or "FLAT"
    has_enough_history: bool
    history_days_used: int
    data_end_date: str | None  # ISO format
    target_week_start: str  # ISO format
    target_week_end: str  # ISO format
    has_news_data: bool
    has_fundamentals_data: bool


def run_inference(
    model: PatchTSTForPrediction,
    feature_scaler: StandardScaler,
    features_list: list[InferenceFeatures],
    week_boundaries: WeekBoundaries,
) -> list[SymbolPrediction]:
    """Run PatchTST inference on prepared multi-channel feature sequences.

    Args:
        model: Loaded PatchTSTForPrediction model in eval mode
        feature_scaler: Fitted StandardScaler from training
        features_list: List of InferenceFeatures (one per symbol)
        week_boundaries: Target week info for the response

    Returns:
        List of SymbolPrediction results
    """
    predictions = []

    # Separate symbols with/without sufficient data
    valid_features = [(f.symbol, f) for f in features_list if f.features is not None]
    invalid_features = [f for f in features_list if f.features is None]

    # Handle symbols without enough data
    for feat in invalid_features:
        predictions.append(
            SymbolPrediction(
                symbol=feat.symbol,
                predicted_weekly_return_pct=None,
                direction="FLAT",
                has_enough_history=False,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat() if feat.data_end_date else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                has_news_data=feat.has_news_data,
                has_fundamentals_data=feat.has_fundamentals_data,
            )
        )

    if not valid_features:
        return predictions

    # Batch inference for valid symbols
    # Shape: (n_samples, context_length, num_channels)
    X = np.array([f.features for _, f in valid_features])

    # Scale features using the training scaler
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = feature_scaler.transform(X_flat)
    X = X_scaled.reshape(original_shape)

    # HuggingFace PatchTST expects shape (batch, sequence_length, num_channels)
    # Our X is already in that shape, no transpose needed

    # Convert to tensor and run model
    X_tensor = torch.FloatTensor(X)

    model.eval()
    with torch.no_grad():
        outputs = model(past_values=X_tensor)
        raw_predictions = outputs.prediction_outputs
        if raw_predictions.dim() == 3:
            raw_predictions = raw_predictions.mean(dim=-1)
        raw_predictions = raw_predictions.cpu().numpy().flatten()

    # Build prediction results
    for i, (symbol, feat) in enumerate(valid_features):
        weekly_return = float(raw_predictions[i])
        weekly_return_pct = weekly_return * 100  # Convert to percentage

        # Determine direction
        if weekly_return > 0.001:  # > 0.1% threshold for "UP"
            direction = "UP"
        elif weekly_return < -0.001:  # < -0.1% threshold for "DOWN"
            direction = "DOWN"
        else:
            direction = "FLAT"

        predictions.append(
            SymbolPrediction(
                symbol=symbol,
                predicted_weekly_return_pct=round(weekly_return_pct, 4),
                direction=direction,
                has_enough_history=True,
                history_days_used=feat.history_days_used,
                data_end_date=feat.data_end_date.isoformat() if feat.data_end_date else None,
                target_week_start=week_boundaries.target_week_start.isoformat(),
                target_week_end=week_boundaries.target_week_end.isoformat(),
                has_news_data=feat.has_news_data,
                has_fundamentals_data=feat.has_fundamentals_data,
            )
        )

    return predictions

