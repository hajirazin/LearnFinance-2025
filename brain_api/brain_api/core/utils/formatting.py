"""Formatting utilities for display and logging."""


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable string like "5.2s", "3m 45s", or "2h 15m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_number(n: int) -> str:
    """Format large numbers with K/M suffix.

    Args:
        n: Integer to format

    Returns:
        Formatted string like "1.5M", "250K", or "999"
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
