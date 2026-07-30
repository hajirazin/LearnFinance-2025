"""SAC training readiness domain objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SACReadinessIssue:
    """One exact missing input or provider/storage error."""

    source: str
    detail: str
    symbol: str | None = None
    retryable: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return an API-safe representation."""
        return asdict(self)


@dataclass(frozen=True)
class SACTrainingReadiness:
    """Preflight result for one universe and training window."""

    universe: str
    symbols: tuple[str, ...]
    ready: bool
    missing: tuple[SACReadinessIssue, ...] = ()
    errors: tuple[SACReadinessIssue, ...] = ()

    @classmethod
    def from_issues(
        cls,
        *,
        universe: str,
        symbols: list[str],
        missing: list[SACReadinessIssue],
        errors: list[SACReadinessIssue],
    ) -> SACTrainingReadiness:
        """Create readiness whose boolean is derived only from issue emptiness."""
        return cls(
            universe=universe,
            symbols=tuple(symbols),
            ready=not missing and not errors,
            missing=tuple(missing),
            errors=tuple(errors),
        )
