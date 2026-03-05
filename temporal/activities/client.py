"""Shared HTTP client for brain_api calls."""

import os

import httpx

BRAIN_API_URL = os.environ.get("BRAIN_API_URL", "http://localhost:8000")

INFERENCE_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=300.0,  # 5 minute read timeout for inference
    write=30.0,
    pool=30.0,
)

TRAINING_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=28800.0,  # 8 hour read timeout for long-running training
    write=30.0,
    pool=30.0,
)


def get_client() -> httpx.Client:
    """Create an HTTP client with inference timeouts."""
    return httpx.Client(base_url=BRAIN_API_URL, timeout=INFERENCE_TIMEOUT)


def get_training_client() -> httpx.Client:
    """Create an HTTP client with training timeouts (8h read)."""
    return httpx.Client(base_url=BRAIN_API_URL, timeout=TRAINING_TIMEOUT)
