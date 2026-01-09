"""Feature scaling for portfolio RL.

Fits a StandardScaler on training data and applies it at inference.
The scaler is stored with the policy artifact.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class PortfolioScaler:
    """Scaler for portfolio RL state features.

    Wraps sklearn StandardScaler with portfolio-specific logic:
    - Only scales signal and forecast features
    - Portfolio weights are NOT scaled (they're already in [0, 1])
    """

    scaler: StandardScaler
    n_features_to_scale: int  # number of features to scale (signals + forecasts)
    n_portfolio_weights: int  # number of portfolio weight features (not scaled)
    is_fitted: bool = False

    @classmethod
    def create(cls, n_stocks: int = 15) -> PortfolioScaler:
        """Create a new unfitted scaler.

        Args:
            n_stocks: Number of stocks in universe.

        Returns:
            New PortfolioScaler instance.
        """
        n_signals_per_stock = 7  # news + 5 fundamentals + fundamental_age
        n_forecast_features = n_stocks
        n_features_to_scale = n_stocks * n_signals_per_stock + n_forecast_features
        n_portfolio_weights = n_stocks + 1  # +1 for CASH

        return cls(
            scaler=StandardScaler(),
            n_features_to_scale=n_features_to_scale,
            n_portfolio_weights=n_portfolio_weights,
            is_fitted=False,
        )

    def fit(self, states: np.ndarray) -> PortfolioScaler:
        """Fit scaler on training states.

        Only fits on the signal/forecast portion of the state.
        Portfolio weights are left unscaled.

        Args:
            states: Training states, shape (n_samples, state_dim).

        Returns:
            Self for chaining.
        """
        # Extract only the features to scale
        features_to_scale = states[:, :self.n_features_to_scale]
        self.scaler.fit(features_to_scale)
        self.is_fitted = True
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        """Transform states using fitted scaler.

        Args:
            states: States to transform, shape (n_samples, state_dim) or (state_dim,).

        Returns:
            Scaled states with same shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        # Handle single state (1D) vs batch (2D)
        single_state = states.ndim == 1
        if single_state:
            states = states.reshape(1, -1)

        result = states.copy()

        # Scale only the signal/forecast features
        features_to_scale = states[:, :self.n_features_to_scale]
        scaled_features = self.scaler.transform(features_to_scale)
        result[:, :self.n_features_to_scale] = scaled_features

        # Portfolio weights remain unchanged

        if single_state:
            result = result.flatten()

        return result

    def fit_transform(self, states: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            states: Training states.

        Returns:
            Scaled states.
        """
        self.fit(states)
        return self.transform(states)

    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        """Inverse transform scaled states.

        Args:
            states: Scaled states.

        Returns:
            Original-scale states.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")

        single_state = states.ndim == 1
        if single_state:
            states = states.reshape(1, -1)

        result = states.copy()

        scaled_features = states[:, :self.n_features_to_scale]
        original_features = self.scaler.inverse_transform(scaled_features)
        result[:, :self.n_features_to_scale] = original_features

        if single_state:
            result = result.flatten()

        return result

    def save(self, path: Path | str) -> None:
        """Save scaler to file.

        Args:
            path: Path to save scaler pickle.
        """
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "n_features_to_scale": self.n_features_to_scale,
                "n_portfolio_weights": self.n_portfolio_weights,
                "is_fitted": self.is_fitted,
            }, f)

    @classmethod
    def load(cls, path: Path | str) -> PortfolioScaler:
        """Load scaler from file.

        Args:
            path: Path to scaler pickle.

        Returns:
            Loaded PortfolioScaler instance.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            scaler=data["scaler"],
            n_features_to_scale=data["n_features_to_scale"],
            n_portfolio_weights=data["n_portfolio_weights"],
            is_fitted=data["is_fitted"],
        )

