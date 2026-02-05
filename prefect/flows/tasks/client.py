"""Shared HTTP client for brain_api calls."""

import os

import httpx

BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "http://localhost:8000")

# Timeout settings (inference can take a few minutes)
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=300.0,  # 5 minute read timeout for inference
    write=30.0,
    pool=30.0,
)


def get_client() -> httpx.Client:
    """Create an HTTP client with appropriate timeouts."""
    return httpx.Client(base_url=BRAIN_API_URL, timeout=DEFAULT_TIMEOUT)
