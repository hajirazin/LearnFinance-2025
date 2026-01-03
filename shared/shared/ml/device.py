"""Device detection for ML workloads.

Provides unified device detection across all ML code, supporting:
- Apple Silicon MPS
- NVIDIA CUDA
- CPU fallback
"""

import torch


def get_device() -> str:
    """Detect the best available compute device.

    Priority order:
    1. MPS (Apple Silicon) - if available and functional
    2. CUDA (NVIDIA GPU) - if available
    3. CPU - fallback

    Returns:
        Device string: "mps", "cuda", or "cpu"
    """
    # Try MPS (Apple Silicon) first
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Verify MPS is actually functional
            _ = torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass

    # Try CUDA next
    if torch.cuda.is_available():
        return "cuda"

    # Fallback to CPU
    return "cpu"


def get_torch_device() -> torch.device:
    """Get the torch.device object for the best available device.

    Returns:
        torch.device for the optimal compute device
    """
    return torch.device(get_device())

