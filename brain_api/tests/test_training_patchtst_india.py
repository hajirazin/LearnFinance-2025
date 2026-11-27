"""India-specific validation and isolation alongside the shared weekly API tests."""

from brain_api.storage.patchtst.local import PatchTSTHalalNewModelStorage
from tests.patchtst_cases import make_training_case, submit


def test_india_rejects_symbols_without_ns_suffix(monkeypatch, tmp_path):
    case = make_training_case(monkeypatch, tmp_path, "india")
    case.symbols = ["INFY"]
    response = case.client.post(case.endpoint, json={"universe": case.universe})
    assert response.status_code == 422
    assert not case.fit_calls


def test_india_preserves_suffix_and_does_not_promote_us(monkeypatch, tmp_path):
    case = make_training_case(monkeypatch, tmp_path / "india", "india")
    us = PatchTSTHalalNewModelStorage(tmp_path / "us")
    _, status = submit(case)
    assert status["status"] == "completed", status
    metadata = case.storage.read_metadata(status["result"]["version"])
    assert all(symbol.endswith(".NS") for symbol in metadata["symbols"])
    assert "news_sentiment" not in status["result"]["signals_used"]
    assert us.read_current_version() is None
