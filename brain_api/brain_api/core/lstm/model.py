"""LSTM PyTorch model architecture."""

import torch
import torch.nn as nn

from brain_api.core.lstm.config import LSTMConfig


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
