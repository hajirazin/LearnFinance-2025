"""SAC + PatchTST inference implementation.

Re-exports run_sac_inference from sac_lstm since the inference
logic is identical - only the forecast feature source differs.
"""

# Import and re-export from sac_lstm
from brain_api.core.sac_lstm.inference import SACInferenceResult, run_sac_inference

__all__ = ["SACInferenceResult", "run_sac_inference"]

