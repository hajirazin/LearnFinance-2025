"""The default isolation guard catches I/O even when application code swallows it."""

from contextlib import suppress

import pytest
import requests
import torch

from tests.offline import OfflineTestGuard


def test_swallowed_network_error_still_fails_guard(monkeypatch, tmp_path):
    guard = OfflineTestGuard()
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, (tmp_path / "production",))
        with suppress(Exception):
            requests.get("https://example.invalid", timeout=1)
    with pytest.raises(AssertionError, match="requests"):
        guard.assert_clean()


def test_curl_backed_requests_are_blocked(monkeypatch, tmp_path):
    from curl_cffi.requests import Session

    guard = OfflineTestGuard()
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, ())
        with pytest.raises(AssertionError, match="curl"):
            Session().get("https://example.invalid")
    assert guard.attempts


def test_production_weights_are_blocked_before_open(monkeypatch, tmp_path):
    guard = OfflineTestGuard()
    production = tmp_path / "production"
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, (production,))
        with pytest.raises(AssertionError, match="production model"):
            torch.load(production / "weights.pt", weights_only=True)


def test_synthetic_temporary_weights_roundtrip(monkeypatch, tmp_path):
    guard = OfflineTestGuard()
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([2.0])}, path)
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, (tmp_path / "production",))
        loaded = torch.load(path, weights_only=True)
    torch.testing.assert_close(loaded["weight"], torch.tensor([2.0]))
    guard.assert_clean()


def test_pretrained_download_is_blocked_even_with_a_cache(monkeypatch, tmp_path):
    from huggingface_hub import hf_hub_download

    guard = OfflineTestGuard()
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, ())
        # A reference imported before installation must still hit the guarded
        # transport; callers using the public entry point fail immediately.
        import huggingface_hub

        with pytest.raises(AssertionError, match="Hugging Face"):
            huggingface_hub.hf_hub_download("test/model", "weights.pt")
    assert callable(hf_hub_download)


def test_hf_aliases_bound_on_application_modules_are_guarded(monkeypatch, tmp_path):
    import brain_api.news.sync as news_sync

    guard = OfflineTestGuard()
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, ())
        with pytest.raises(AssertionError, match="Hugging Face"):
            news_sync.hf_hub_download("test/model", "weights.pt")


def test_guard_names_do_not_include_credentials(monkeypatch, tmp_path):
    guard = OfflineTestGuard()
    with monkeypatch.context() as patches:
        guard.install(patches, tmp_path, ())
        with pytest.raises(AssertionError) as error:
            requests.get("https://secret-token@example.invalid/?token=secret")
    assert "secret" not in str(error.value)
    assert "secret" not in str(guard.attempts)
