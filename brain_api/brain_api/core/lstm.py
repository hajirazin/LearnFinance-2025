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
    """LSTM model hyperparameters and training config."""

    # Model architecture
    input_size: int = 5  # OHLCV features
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Forecast settings
    forecast_horizon: int = 7  # Predict next 7 days of prices

    # Training
    sequence_length: int = 60  # 60 days lookback
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    validation_split: float = 0.2

    # Feature engineering
    use_returns: bool = True  # Use returns for input features (more stationary)

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
# Dataset building (7-day price prediction)
# ============================================================================


@dataclass
class DatasetResult:
    """Result of dataset building."""

    X: np.ndarray  # Input sequences: (n_samples, seq_len, n_features)
    y: np.ndarray  # Targets: (n_samples, forecast_horizon) - next 7 days close prices
    feature_scaler: StandardScaler  # Scaler for input features
    price_scaler: StandardScaler  # Scaler for target prices (for denormalization)


def build_dataset(
    prices: dict[str, pd.DataFrame],
    config: LSTMConfig,
) -> DatasetResult:
    """Build training dataset from price data.

    Creates sequences of features for LSTM input and corresponding targets
    (next 7 days of close prices).

    Args:
        prices: Dict of symbol -> OHLCV DataFrame
        config: LSTM configuration

    Returns:
        DatasetResult with X, y, feature_scaler, and price_scaler
    """
    all_sequences = []
    all_targets = []
    all_close_prices = []  # For fitting price scaler

    horizon = config.forecast_horizon

    for _symbol, df in prices.items():
        # Need enough data for sequence + forecast horizon
        min_length = config.sequence_length + horizon
        if len(df) < min_length:
            continue

        # Collect all close prices for scaler fitting
        all_close_prices.extend(df["close"].values.tolist())

        # Feature engineering: use returns for input (more stationary)
        if config.use_returns:
            features = pd.DataFrame(
                {
                    "open_ret": np.log(df["open"] / df["open"].shift(1)),
                    "high_ret": np.log(df["high"] / df["high"].shift(1)),
                    "low_ret": np.log(df["low"] / df["low"].shift(1)),
                    "close_ret": np.log(df["close"] / df["close"].shift(1)),
                    "volume_ret": np.log(
                        df["volume"] / df["volume"].shift(1).replace(0, 1)
                    ),
                }
            )
            # Drop first row (NaN from shift) but keep index aligned with df
            features = features.iloc[1:].reset_index(drop=True)
            close_prices = df["close"].iloc[1:].reset_index(drop=True)
        else:
            features = (
                df[["open", "high", "low", "close", "volume"]]
                .copy()
                .reset_index(drop=True)
            )
            close_prices = df["close"].reset_index(drop=True)

        # Replace infinities with 0
        features = features.replace([np.inf, -np.inf], 0).fillna(0)

        values = features.values

        # Create sequences with 7-day price targets
        # We need: [day_0 ... day_59] as input -> [day_60 ... day_66] as target
        for i in range(len(values) - config.sequence_length - horizon + 1):
            seq = values[i : i + config.sequence_length]

            # Target: next 7 days of close prices
            target_start = i + config.sequence_length
            target_end = target_start + horizon
            target_prices = close_prices.iloc[target_start:target_end].values

            if len(target_prices) == horizon:
                all_sequences.append(seq)
                all_targets.append(target_prices)

    if not all_sequences:
        # Return empty arrays if no data
        empty_X = np.array([]).reshape(0, config.sequence_length, config.input_size)
        empty_y = np.array([]).reshape(0, horizon)
        return DatasetResult(
            X=empty_X,
            y=empty_y,
            feature_scaler=StandardScaler(),
            price_scaler=StandardScaler(),
        )

    X = np.array(all_sequences)
    y = np.array(all_targets)

    # Fit feature scaler on input sequences
    original_shape = X.shape
    X_flat = X.reshape(-1, X.shape[-1])
    feature_scaler = StandardScaler()
    X_flat_scaled = feature_scaler.fit_transform(X_flat)
    X = X_flat_scaled.reshape(original_shape)

    # Fit price scaler on all close prices (for target normalization)
    price_scaler = StandardScaler()
    price_scaler.fit(np.array(all_close_prices).reshape(-1, 1))

    # Scale the targets
    y_shape = y.shape
    y_flat = y.reshape(-1, 1)
    y_scaled = price_scaler.transform(y_flat).reshape(y_shape)

    return DatasetResult(
        X=X,
        y=y_scaled,
        feature_scaler=feature_scaler,
        price_scaler=price_scaler,
    )


# ============================================================================
# PyTorch LSTM Model (7-day output)
# ============================================================================


class LSTMModel(nn.Module):
    """PyTorch LSTM model for 7-day price prediction."""

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

        # Output layer: predict forecast_horizon days (default 7)
        self.fc = nn.Linear(config.hidden_size, config.forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, forecast_horizon) - predicted prices for next 7 days
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
    """Result of LSTM training."""

    model: LSTMModel
    feature_scaler: StandardScaler
    price_scaler: StandardScaler  # For denormalizing predictions back to prices
    config: LSTMConfig
    train_loss: float
    val_loss: float
    baseline_loss: float  # Persistence model loss


def train_model_pytorch(
    X: np.ndarray,
    y: np.ndarray,
    feature_scaler: StandardScaler,
    price_scaler: StandardScaler,
    config: LSTMConfig,
) -> TrainingResult:
    """Train LSTM model using PyTorch.

    Automatically uses the best available device:
    - MPS (Apple Silicon GPU) on M1/M2/M3 Macs
    - CUDA (NVIDIA GPU) if available
    - CPU as fallback

    Args:
        X: Input sequences, shape (n_samples, seq_len, n_features)
        y: Targets, shape (n_samples, forecast_horizon) - scaled prices
        feature_scaler: Fitted scaler for input features
        price_scaler: Fitted scaler for target prices
        config: Model configuration

    Returns:
        TrainingResult with trained model and metrics
    """
    # Detect best available device
    device = get_device()
    print(f"Training on device: {device}")

    if len(X) == 0:
        # No data - return dummy result
        model = LSTMModel(config)
        return TrainingResult(
            model=model,
            feature_scaler=feature_scaler,
            price_scaler=price_scaler,
            config=config,
            train_loss=float("inf"),
            val_loss=float("inf"),
            baseline_loss=float("inf"),
        )

    # Train/val split (time-based, not random, to prevent data leakage)
    split_idx = int(len(X) * (1 - config.validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

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

    for _epoch in range(config.epochs):
        model.train()

        # Mini-batch training
        indices = torch.randperm(len(X_train_t), device=device)
        total_train_loss = 0.0

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    # Restore best model (on CPU for portability when saving)
    model_cpu = LSTMModel(config)
    if best_model_state is not None:
        model_cpu.load_state_dict(best_model_state)

    # Final metrics (compute on device for speed)
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_t)
        final_train_loss = criterion(train_outputs, y_train_t).item()

    # Baseline: persistence model (predict last known price for all 7 days)
    # Since targets are scaled, baseline predicts scaled "0" which represents the mean
    # A better baseline: MSE if we predicted y_val[:, 0] (day 1) for all days
    baseline_loss = float(np.mean((y_val - y_val[:, :1]) ** 2))

    return TrainingResult(
        model=model_cpu,  # Return CPU model for portable saving
        feature_scaler=feature_scaler,
        price_scaler=price_scaler,
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
