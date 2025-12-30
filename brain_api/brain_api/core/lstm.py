"""LSTM model training and evaluation logic."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler

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
class LSTMConfig:
    """LSTM model hyperparameters and training config.

    The LSTM predicts weekly returns (Mon open → Fri close), not daily prices.
    This aligns with the RL agent's weekly decision horizon.
    """

    # Model architecture
    input_size: int = 5  # OHLCV features (log returns)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Forecast settings
    # Single output: weekly return = (fri_close - mon_open) / mon_open
    forecast_horizon: int = 1

    # Training
    sequence_length: int = 60  # 60 trading days lookback
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    validation_split: float = 0.2

    # Feature engineering
    use_returns: bool = True  # Use log returns for input features (more stationary)

    # Week filtering
    min_week_days: int = 3  # Skip weeks with fewer than 3 trading days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "forecast_horizon": self.forecast_horizon,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "use_returns": self.use_returns,
            "min_week_days": self.min_week_days,
        }


DEFAULT_CONFIG = LSTMConfig()


# ============================================================================
# Version computation (idempotent)
# ============================================================================


def compute_version(
    start_date: date,
    end_date: date,
    symbols: list[str],
    config: LSTMConfig,
) -> str:
    """Compute a deterministic version string for the training run.

    The version is a hash of (window, symbols, config) so that reruns with
    the same inputs produce the same version (idempotent training).

    Returns:
        Version string in format 'v{timestamp_prefix}-{hash_suffix}'
    """
    # Create a canonical representation
    canonical = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "symbols": sorted(symbols),  # Sort for determinism
        "config": config.to_dict(),
    }
    canonical_json = json.dumps(canonical, sort_keys=True)

    # Hash it
    hash_digest = hashlib.sha256(canonical_json.encode()).hexdigest()[:12]

    # Include end_date in version for human readability
    date_prefix = end_date.strftime("%Y-%m-%d")

    return f"v{date_prefix}-{hash_digest}"


# ============================================================================
# Data loading (yfinance)
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
        Dict mapping symbol -> DataFrame with OHLCV columns
    """
    prices: dict[str, pd.DataFrame] = {}

    # Batch download for efficiency
    # yfinance accepts space-separated tickers
    tickers_str = " ".join(symbols)

    try:
        data = yf.download(
            tickers_str,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            group_by="ticker",
        )

        if len(symbols) == 1:
            # Single ticker: data is a simple DataFrame
            symbol = symbols[0]
            if not data.empty:
                df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df = df.dropna()
                if len(df) > 0:
                    prices[symbol] = df
        else:
            # Multiple tickers: data is multi-level columns
            for symbol in symbols:
                try:
                    if symbol in data.columns.get_level_values(0):
                        df = data[symbol][
                            ["Open", "High", "Low", "Close", "Volume"]
                        ].copy()
                        df.columns = ["open", "high", "low", "close", "volume"]
                        df = df.dropna()
                        if len(df) > 0:
                            prices[symbol] = df
                except (KeyError, TypeError):
                    continue

    except Exception:
        # Fallback: fetch individually
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                )
                if not df.empty:
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.columns = ["open", "high", "low", "close", "volume"]
                    df = df.dropna()
                    if len(df) > 0:
                        prices[symbol] = df
            except Exception:
                continue

    return prices


# ============================================================================
# Dataset building (weekly return prediction)
# ============================================================================


@dataclass
class DatasetResult:
    """Result of dataset building for weekly return prediction."""

    X: np.ndarray  # Input sequences: (n_samples, seq_len, n_features)
    y: np.ndarray  # Targets: (n_samples, 1) - weekly returns
    feature_scaler: StandardScaler  # Scaler for input features


def _extract_trading_weeks(df: pd.DataFrame, min_days: int = 3) -> list[pd.DataFrame]:
    """Extract trading weeks from a price DataFrame.

    Groups data by ISO week and filters out weeks with too few trading days.

    Args:
        df: DataFrame with DatetimeIndex containing OHLCV data
        min_days: Minimum trading days required for a valid week

    Returns:
        List of DataFrames, one per valid trading week
    """
    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return []

    # Group by ISO week (year + week number)
    df = df.copy()
    df["_year_week"] = df.index.to_period("W")

    weeks = []
    for _, week_df in df.groupby("_year_week"):
        if len(week_df) >= min_days:
            weeks.append(week_df.drop(columns=["_year_week"]))

    return weeks


def _compute_weekly_return(week_df: pd.DataFrame) -> float:
    """Compute weekly return from a trading week DataFrame.

    Args:
        week_df: DataFrame for a single trading week with OHLCV data

    Returns:
        Weekly return = (last_day_close - first_day_open) / first_day_open
    """
    first_day_open = week_df["open"].iloc[0]
    last_day_close = week_df["close"].iloc[-1]

    if first_day_open == 0:
        return 0.0

    return (last_day_close - first_day_open) / first_day_open


def build_dataset(
    prices: dict[str, pd.DataFrame],
    config: LSTMConfig,
) -> DatasetResult:
    """Build training dataset for weekly return prediction.

    Creates samples aligned to trading weeks:
    - Input: 60 trading days of features ending at week start
    - Target: weekly return (fri_close - mon_open) / mon_open

    This naturally handles holidays - a "week" is simply the first to last
    trading day of each ISO week.

    Args:
        prices: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        config: LSTM configuration

    Returns:
        DatasetResult with X, y (weekly returns), and feature_scaler
    """
    all_sequences = []
    all_targets = []

    print(f"[LSTM] Building dataset from {len(prices)} symbols...")
    symbols_used = 0
    total_weeks = 0

    for _symbol, df in prices.items():
        # Skip if not enough data
        if len(df) < config.sequence_length + 5:  # Need at least one week after lookback
            continue

        # Compute features (log returns for stationarity)
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
            # Drop first row (NaN from shift)
            features_df = features_df.iloc[1:]
            df_aligned = df.iloc[1:]
        else:
            features_df = df[["open", "high", "low", "close", "volume"]].copy()
            df_aligned = df

        # Replace infinities with 0
        features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

        # Extract trading weeks from the aligned data
        weeks = _extract_trading_weeks(df_aligned, min_days=config.min_week_days)

        if len(weeks) < 2:  # Need at least 2 weeks (1 for target)
            continue

        symbol_samples = 0

        # For each week (except the last), create a training sample
        # Input: features from sequence_length days ending at week start
        # Target: that week's return
        for i in range(len(weeks) - 1):
            week = weeks[i + 1]  # Target week (we predict the NEXT week)
            week_start = week.index[0]

            # Find the position of week_start in features_df
            try:
                week_start_idx = features_df.index.get_loc(week_start)
            except KeyError:
                continue

            # Check if we have enough history
            if week_start_idx < config.sequence_length:
                continue

            # Extract input sequence (sequence_length days ending just before week start)
            seq_start_idx = week_start_idx - config.sequence_length
            seq_end_idx = week_start_idx  # Exclusive

            sequence = features_df.iloc[seq_start_idx:seq_end_idx].values

            if len(sequence) != config.sequence_length:
                continue

            # Compute target: weekly return for the target week
            weekly_return = _compute_weekly_return(week)

            all_sequences.append(sequence)
            all_targets.append([weekly_return])
            symbol_samples += 1

        if symbol_samples > 0:
            symbols_used += 1
            total_weeks += symbol_samples

    print(f"[LSTM] Dataset built: {total_weeks} weekly samples from {symbols_used} symbols")

    if not all_sequences:
        # Return empty arrays if no data
        empty_X = np.array([]).reshape(0, config.sequence_length, config.input_size)
        empty_y = np.array([]).reshape(0, 1)
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

    # Note: We do NOT scale the targets (weekly returns).
    # Returns are already naturally bounded (typically -0.1 to +0.1) and
    # keeping them in original scale makes interpretation straightforward.

    return DatasetResult(
        X=X,
        y=y,
        feature_scaler=feature_scaler,
    )


# ============================================================================
# PyTorch LSTM Model (weekly return prediction)
# ============================================================================


class LSTMModel(nn.Module):
    """PyTorch LSTM model for weekly return prediction.

    Predicts a single scalar: the expected weekly return
    (friday_close - monday_open) / monday_open.
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer: single value (weekly return)
        self.fc = nn.Linear(config.hidden_size, config.forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, 1) - predicted weekly return
        """
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


# ============================================================================
# Training
# ============================================================================


@dataclass
class TrainingResult:
    """Result of LSTM training for weekly return prediction."""

    model: LSTMModel
    feature_scaler: StandardScaler
    config: LSTMConfig
    train_loss: float
    val_loss: float
    baseline_loss: float  # Baseline: predict 0 return (no change)


def train_model_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    config: LSTMConfig,
) -> TrainingResult:
    """Train LSTM model using PyTorch.

    Automatically uses the best available device:
    - MPS (Apple Silicon GPU) on M1/M2/M3 Macs
    - CUDA (NVIDIA GPU) if available
    - CPU as fallback

    Args:
        X: Input sequences, shape (n_samples, seq_len, n_features)
        y: Targets, shape (n_samples, 1) - weekly returns (unscaled)
        feature_scaler: Fitted scaler for input features
        config: Model configuration

    Returns:
        TrainingResult with trained model and metrics
    """
    # Detect best available device
    device = get_device()
    print(f"[LSTM] Training on device: {device}")

    if len(X) == 0:
        print("[LSTM] No training data - returning dummy model")
        model = LSTMModel(config)
        return TrainingResult(
            model=model,
            feature_scaler=feature_scaler,
            config=config,
            train_loss=float("inf"),
            val_loss=float("inf"),
            baseline_loss=float("inf"),
        )

    # Train/val split (time-based, not random, to prevent data leakage)
    split_idx = int(len(X) * (1 - config.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"[LSTM] Dataset: {len(X)} samples ({len(X_train)} train, {len(X_val)} val)")
    print(f"[LSTM] Config: {config.epochs} epochs, batch_size={config.batch_size}, lr={config.learning_rate}")

    # Convert to tensors and move to device
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Create model and move to device
    model = LSTMModel(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    log_interval = max(1, config.epochs // 5)  # Log ~5 times during training

    print("[LSTM] Starting training...")

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
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_train_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        # Log progress at intervals
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"[LSTM] Epoch {epoch + 1}/{config.epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    print(f"[LSTM] Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")

    # Restore best model (on CPU for portability when saving)
    model_cpu = LSTMModel(config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    # Final metrics (compute on device for speed)
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_t)
        final_train_loss = criterion(train_outputs, y_train_t).item()

    # Baseline: predict 0 return (no change from week to week)
    # This is the naive "persistence" baseline for returns
    baseline_loss = float(np.mean(y_val**2))

    print(f"[LSTM] Training complete: train_loss={final_train_loss:.6f}, val_loss={best_val_loss:.6f}, baseline={baseline_loss:.6f}")
    beats_baseline = best_val_loss < baseline_loss
    print(f"[LSTM] Model {'BEATS' if beats_baseline else 'does NOT beat'} baseline")

    return TrainingResult(
        model=model_cpu,  # Return CPU model for portable saving
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
    # weekday(): Monday=0, Tuesday=1, ..., Sunday=6
    days_since_monday = as_of_date.weekday()
    calendar_monday = as_of_date - timedelta(days=days_since_monday)
    calendar_friday = calendar_monday + timedelta(days=4)

    # Convert to pandas Timestamp for exchange_calendars
    monday_ts = pd.Timestamp(calendar_monday)
    friday_ts = pd.Timestamp(calendar_friday)

    # Find first trading day on or after Monday (up to Friday)
    # and last trading day on or before Friday (down to Monday)
    schedule = nyse.sessions_in_range(monday_ts, friday_ts)

    if len(schedule) == 0:
        # Entire week is holiday - rare but possible (e.g., week between Christmas and New Year)
        # Fall back to calendar dates; inference will note this in quality
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
    """Features prepared for inference for a single symbol."""

    symbol: str
    features: np.ndarray | None  # Shape: (seq_len, n_features) or None if insufficient data
    has_enough_history: bool
    history_days_used: int
    data_end_date: date | None  # Last date of data used (should be before target_week_start)


def build_inference_features(
    symbol: str,
    prices_df: pd.DataFrame,
    config: LSTMConfig,
    cutoff_date: date,
) -> InferenceFeatures:
    """Build feature sequence for inference from OHLCV data.

    Constructs the same log-return features as training, ending just before
    cutoff_date (which should be target_week_start).

    Args:
        symbol: Ticker symbol
        prices_df: DataFrame with OHLCV columns (open, high, low, close, volume)
                   and DatetimeIndex
        config: LSTM config with sequence_length and use_returns settings
        cutoff_date: Features end before this date (typically target_week_start)

    Returns:
        InferenceFeatures with prepared feature sequence or None if insufficient data
    """
    if prices_df.empty:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
        )

    # Ensure DatetimeIndex
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=0,
            data_end_date=None,
        )

    # Filter to data before cutoff_date
    cutoff_ts = pd.Timestamp(cutoff_date)
    df = prices_df[prices_df.index < cutoff_ts].copy()

    if len(df) < config.sequence_length + 1:  # +1 for the shift in log returns
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(df),
            data_end_date=df.index[-1].date() if len(df) > 0 else None,
        )

    # Compute features (log returns, same as training)
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
        # Drop first row (NaN from shift)
        features_df = features_df.iloc[1:]
    else:
        features_df = df[["open", "high", "low", "close", "volume"]].copy()

    # Replace infinities with 0
    features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Check if we have enough data after computing returns
    if len(features_df) < config.sequence_length:
        return InferenceFeatures(
            symbol=symbol,
            features=None,
            has_enough_history=False,
            history_days_used=len(features_df),
            data_end_date=features_df.index[-1].date() if len(features_df) > 0 else None,
        )

    # Take the last sequence_length rows
    sequence = features_df.iloc[-config.sequence_length :].values
    data_end_date = features_df.index[-1].date()

    return InferenceFeatures(
        symbol=symbol,
        features=sequence,
        has_enough_history=True,
        history_days_used=len(features_df),
        data_end_date=data_end_date,
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


def run_inference(
    model: "LSTMModel",
    feature_scaler: StandardScaler,
    features_list: list[InferenceFeatures],
    week_boundaries: WeekBoundaries,
) -> list[SymbolPrediction]:
    """Run LSTM inference on prepared feature sequences.

    Args:
        model: Loaded LSTMModel in eval mode
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
            )
        )

    if not valid_features:
        return predictions

    # Batch inference for valid symbols
    X = np.array([f.features for _, f in valid_features])

    # Scale features using the training scaler
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled = feature_scaler.transform(X_flat)
    X = X_scaled.reshape(original_shape)

    # Convert to tensor and run model
    X_tensor = torch.FloatTensor(X)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        raw_predictions = outputs.cpu().numpy().flatten()

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
            )
        )

    return predictions
