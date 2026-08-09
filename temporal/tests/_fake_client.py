"""Shared httpx-style fakes used by activity-level test modules.

Module name starts with ``_`` so pytest does not try to collect it
during test discovery (and so its public surface is implicit). Imported
explicitly by sibling test files via ``from _fake_client import ...``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class FakeResponse:
    """Stand-in for ``httpx.Response`` with a queued JSON payload."""

    def __init__(self, json_payload: dict, status: int = 200) -> None:
        self._payload = json_payload
        self.status_code = status
        self.text = str(json_payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class FakeClient:
    """Records the path + json body of each POST/GET, returns a queued response."""

    def __init__(
        self,
        responses: dict[str, dict],
        statuses: dict[str, int] | None = None,
    ) -> None:
        self._responses = responses
        self._statuses = statuses or {}
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, json: dict | None = None) -> FakeResponse:
        self.calls.append({"method": "POST", "path": path, "json": json})
        if path not in self._responses:
            raise AssertionError(f"Unexpected POST {path}")
        return FakeResponse(self._responses[path], status=self._statuses.get(path, 200))

    def get(self, path: str) -> FakeResponse:
        self.calls.append({"method": "GET", "path": path, "json": None})
        if path not in self._responses:
            raise AssertionError(f"Unexpected GET {path}")
        return FakeResponse(self._responses[path], status=self._statuses.get(path, 200))

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@contextmanager
def patch_client(module, fake: FakeClient):
    """Swap ``module.get_client`` for one that yields the fake client.

    Use as ``with patch_client(activities_module, fake_client): ...``.
    Restores the original ``get_client`` on exit so tests stay isolated
    even if the body raises.
    """
    original = module.get_client
    module.get_client = lambda: fake
    try:
        yield fake
    finally:
        module.get_client = original
