"""Production weekly HTTP paths with synthetic data and temporary artifacts."""

import pytest

from brain_api.core.patchtst.config import INDIA_WEEKLY_CONFIG, US_WEEKLY_CONFIG
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.main import app
from brain_api.routes.training.dependencies import get_weekly_patchtst_trainer
from tests.patchtst_cases import make_training_case, submit


@pytest.fixture(params=["us", "india"])
def training_case(request, monkeypatch, tmp_path):
    return make_training_case(monkeypatch, tmp_path, request.param)


def test_default_weekly_endpoint_persists_and_promotes(training_case):
    case = training_case
    _, status = submit(case)
    assert status["status"] == "completed", status
    result = status["result"]
    config = US_WEEKLY_CONFIG if case.market == "us" else INDIA_WEEKLY_CONFIG
    assert result["num_input_channels"] == config.num_input_channels
    assert result["signals_used"] == config.feature_names
    assert case.fit_calls[0][1] == config
    assert case.storage.read_current_version() == result["version"]
    assert status["progress"]["phase"] == "completed"
    metadata = case.storage.read_metadata(result["version"])
    assert metadata["metrics"]["best_epoch"] == 1
    assert metadata["training_evidence"]["latest_label_end"] == (
        "2024-11-22" if case.market == "us" else "2024-11-29"
    )
    assert not metadata["snapshots_pending"]
    assert not metadata["publication_pending"]


def test_weekly_endpoint_is_idempotent(training_case):
    case = training_case
    _, first = submit(case)
    calls = len(case.fit_calls)
    again = case.client.post(
        case.endpoint + "?skip_snapshot=true", json={"universe": case.universe}
    )
    assert again.status_code == 200, again.text
    assert again.json()["version"] == first["result"]["version"]
    assert len(case.fit_calls) == calls


def test_weekly_endpoint_defaults_to_its_market_universe(training_case):
    _, status = submit(training_case, body={})
    assert status["status"] == "completed", status


def test_unknown_universe_has_no_training_side_effects(training_case):
    case = training_case
    response = case.client.post(case.endpoint, json={"universe": "unknown"})
    assert response.status_code == 422
    assert not case.fit_calls
    assert not case.price_calls
    assert case.storage.read_current_version() is None


def test_unhealthy_fit_does_not_replace_current(training_case):
    case = training_case
    case.loss = float("nan")
    _, status = submit(case)
    assert status["status"] == "completed", status
    assert status["result"]["promoted"] is False
    assert status["result"]["failure_reasons"]
    assert case.storage.read_current_version() is None


def test_publication_failure_preserves_fit_for_retry(training_case):
    case = training_case
    case.repo = "test/model"
    case.upload.side_effect = ConnectionError("Hub unavailable")
    _, failed = submit(case)
    assert failed["status"] == "failed"
    assert case.storage.read_current_version() is None
    # The versioned fit is available even though publication failed.
    version = next(
        path.name for path in case.artifact_root.iterdir() if path.name.startswith("v")
    )
    assert case.storage.read_metadata(version)["publication_pending"]
    fit_count = len(case.fit_calls)
    case.upload.side_effect = None
    _, retried = submit(case)
    assert retried["status"] == "completed", retried
    assert len(case.fit_calls) == fit_count
    assert case.storage.read_current_version() == version
    assert not case.storage.read_metadata(version)["publication_pending"]
    assert case.upload.call_args.kwargs["make_current"] is True


def test_unhealthy_inaugural_upload_never_sets_hf_current(training_case):
    case = training_case
    case.repo, case.loss = "test/model", float("nan")
    _, status = submit(case)
    assert status["status"] == "completed", status
    assert case.upload.call_args.kwargs["make_current"] is False
    assert case.storage.read_current_version() is None


def test_shutdown_during_successful_publication_finishes_local_promotion(training_case):
    from brain_api.main import shutdown_event

    case = training_case
    case.repo = "test/model"
    published = case.upload.return_value

    def publish(**kwargs):
        shutdown_event.set()
        return published

    case.upload.side_effect = publish
    _, status = submit(case)
    assert status["status"] == "completed", status
    assert case.storage.read_current_version() == status["result"]["version"]


def test_default_snapshot_path_uses_weekly_fits_and_year_progress(
    training_case, monkeypatch
):
    from brain_api.routes.training.job_registry import update_progress

    case = training_case
    events = []

    def record(job_id, details):
        events.append(details)
        update_progress(job_id, details)

    monkeypatch.setattr(
        "brain_api.routes.training.patchtst_jobs.update_progress", record
    )
    _, status = submit(case, snapshots=True)
    assert status["status"] == "completed", status
    assert len(case.fit_calls) > 1
    snapshots = list(case.artifact_root.glob("snapshot-*"))
    assert snapshots
    assert all(
        data.X.shape[-1] == (7 if case.market == "us" else 6)
        for data, _ in case.fit_calls
    )
    metadata = case.storage.read_metadata(status["result"]["version"])
    assert metadata["training_evidence"]["first_snapshot_cutoff"]
    assert metadata["snapshots_pending"] is False
    assert any(event["phase"] == "loading_prices" for event in events)
    assert any(event.get("epoch") == 1 for event in events)
    fitting = [
        event
        for event in events
        if event.get("snapshot_phase") == "training_weekly_regression"
    ]
    assert fitting
    assert all(event["snapshot_cutoff"] and event["epoch"] == 1 for event in fitting)
    assert all("main_fit" in event for event in fitting)


def test_training_cancellation_is_reported_as_cancelled(training_case, monkeypatch):
    def cancelled(*args, **kwargs):
        raise TrainingCancelledError("cancelled")

    monkeypatch.setitem(
        app.dependency_overrides, get_weekly_patchtst_trainer, lambda: cancelled
    )
    _, status = submit(training_case)
    assert status["status"] == "cancelled"
    assert training_case.storage.read_current_version() is None


def test_snapshot_cancellation_does_not_promote_incomplete_run(
    training_case, monkeypatch
):
    case = training_case
    fit = case.fit
    count = 0

    def cancel_snapshot(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            raise TrainingCancelledError("snapshot cancelled")
        return fit(*args, **kwargs)

    monkeypatch.setitem(
        app.dependency_overrides, get_weekly_patchtst_trainer, lambda: cancel_snapshot
    )
    _, status = submit(case, snapshots=True)
    assert status["status"] == "cancelled", status
    assert case.storage.read_current_version() is None


def test_duplicate_training_request_does_not_enqueue_again(training_case, monkeypatch):
    from datetime import date

    from brain_api.core.patchtst.version import compute_version
    from brain_api.routes.training.job_registry import get_or_create_job

    case = training_case
    config = US_WEEKLY_CONFIG if case.market == "us" else INDIA_WEEKLY_CONFIG
    version = compute_version(
        date(2023, 1, 1), date(2024, 11, 29), case.symbols, config
    )
    kind = "patchtst" if case.market == "us" else "patchtst_india"
    get_or_create_job(kind, version)
    response = case.client.post(case.endpoint, json={"universe": case.universe})
    assert response.status_code == 202
    assert "already in progress" in response.json()["message"]
    assert not case.fit_calls
