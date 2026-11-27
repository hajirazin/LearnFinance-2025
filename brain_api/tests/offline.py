"""Default test isolation at external I/O and production artifact boundaries."""

import socket
import sys
from pathlib import Path

import pytest


class OfflineTestGuard:
    def __init__(self):
        self.attempts: list[str] = []

    def block(self, operation):
        def blocked(*args, **kwargs):
            self.attempts.append(operation)
            raise AssertionError(f"Unexpected external I/O: {operation}")

        return blocked

    def assert_clean(self):
        assert not self.attempts, (
            "Unexpected external I/O was attempted (possibly caught by application code): "
            + ", ".join(self.attempts)
        )

    def install(self, patches, temporary_root: Path, production_roots):
        import curl_cffi.requests
        import httpx
        import huggingface_hub
        import joblib
        import pandas as pd
        import requests
        import torch
        import transformers

        for name in ("connect", "connect_ex"):
            patches.setattr(socket.socket, name, self.block(f"socket.{name}"))
        patches.setattr(requests.Session, "send", self.block("requests.Session.send"))
        patches.setattr(
            curl_cffi.requests.Session, "request", self.block("curl.Session.request")
        )
        patches.setattr(
            httpx.HTTPTransport, "handle_request", self.block("httpx.HTTPTransport")
        )

        async def async_blocked(*args, **kwargs):
            return self.block("httpx.AsyncHTTPTransport")()

        patches.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_blocked)
        for name in ("hf_hub_download", "snapshot_download"):
            original = getattr(huggingface_hub, name)
            blocked = self.block(f"Hugging Face {name}")
            patches.setattr(huggingface_hub, name, blocked)
            # Modules bind these imports before test fixtures run.
            for module_name, module in list(sys.modules.items()):
                if module_name.startswith("brain_api.") and module is not None:
                    for attr, value in list(vars(module).items()):
                        if value is original:
                            patches.setattr(module, attr, blocked)

        roots = tuple(Path(root).resolve() for root in production_roots)

        def guarded_loader(original, operation):
            def load(source, *args, **kwargs):
                name = (
                    source
                    if isinstance(source, str | Path)
                    else getattr(source, "name", None)
                )
                if isinstance(name, str | Path):
                    path = Path(name).resolve()
                    if any(path.is_relative_to(root) for root in roots):
                        return self.block(operation)()
                return original(source, *args, **kwargs)

            return load

        patches.setattr(
            torch, "load", guarded_loader(torch.load, "production model torch.load")
        )
        patches.setattr(
            joblib, "load", guarded_loader(joblib.load, "production model joblib.load")
        )
        patches.setattr(
            pd,
            "read_parquet",
            guarded_loader(pd.read_parquet, "production data read_parquet"),
        )
        pretrained = transformers.PreTrainedModel.from_pretrained.__func__

        def local_pretrained(cls, source, *args, **kwargs):
            path = Path(str(source)).resolve()
            if path.is_relative_to(temporary_root.resolve()) and path.exists():
                return pretrained(cls, source, *args, **kwargs)
            return self.block("production pretrained model")()

        patches.setattr(
            transformers.PreTrainedModel,
            "from_pretrained",
            classmethod(local_pretrained),
        )


@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch, tmp_path_factory):
    guard = OfflineTestGuard()
    guard.install(
        monkeypatch,
        tmp_path_factory.getbasetemp(),
        (Path(__file__).resolve().parents[1] / "data", Path.cwd() / "data"),
    )
    yield guard
    guard.assert_clean()
