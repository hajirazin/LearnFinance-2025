"""Pytest configuration and fixtures for all tests.

This module ensures tests run in isolation from production environment variables.
"""

import os

import pytest


# HuggingFace-related environment variables that should not affect tests
HF_ENV_VARS = [
    "HF_MODEL_REPO",
    "HF_DATASET_REPO",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "STORAGE_BACKEND",
]


@pytest.fixture(autouse=True)
def isolate_from_env():
    """Clear HuggingFace env vars before each test to prevent external API calls.

    This fixture runs automatically for every test (autouse=True).
    It saves original values, clears them for the test, then restores after.
    """
    # Save original values
    original_values = {}
    for var in HF_ENV_VARS:
        if var in os.environ:
            original_values[var] = os.environ.pop(var)

    yield

    # Restore original values after test
    for var, value in original_values.items():
        os.environ[var] = value

