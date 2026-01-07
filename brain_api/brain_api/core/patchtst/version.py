"""Version computation for PatchTST models."""

from datetime import date

from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.version import compute_model_version


def compute_version(
    start_date: date,
    end_date: date,
    symbols: list[str],
    config: PatchTSTConfig,
) -> str:
    """Compute a deterministic version string for the PatchTST training run.

    The version is a hash of (window, symbols, config) so that reruns with
    the same inputs produce the same version (idempotent training).

    Returns:
        Version string in format 'v{timestamp_prefix}-{hash_suffix}'
    """
    return compute_model_version(
        model_type="patchtst",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        config_dict=config.to_dict(),
    )
