"""Prefect flows for LearnFinance-2025 training pipeline."""

__all__ = ["weekly_training_flow"]


def __getattr__(name: str):
    """Lazy import to avoid circular import issues when running as __main__."""
    if name == "weekly_training_flow":
        from flows.weekly_training import weekly_training_flow

        return weekly_training_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
