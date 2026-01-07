"""Version computation for LSTM models."""

from datetime import date

from brain_api.core.lstm.config import LSTMConfig
from brain_api.core.version import compute_model_version


def compute_version(
    start_date: date,
    end_date: date,
    symbols: list[str],
    config: LSTMConfig,
) -> str:
    """Compute a deterministic version string for the LSTM training run.

    The version is a hash of (window, symbols, config) so that reruns with
    the same inputs produce the same version (idempotent training).

    Returns:
        Version string in format 'v{timestamp_prefix}-{hash_suffix}'
    """
    return compute_model_version(
        model_type="lstm",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        config_dict=config.to_dict(),
    )
