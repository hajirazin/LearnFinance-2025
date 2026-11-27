"""Offline cohort, realized-return, and comparison-family checks."""

import numpy as np
import pandas as pd
import report_mad_patchtst as report
import torch
import train_mad_patchtst as trainer
from mad_experiment_data import cohort_audit
from transformers import PatchTSTConfig, PatchTSTForRegression


def test_listing_audit_does_not_call_short_legrand_history_a_new_ipo():
    manifest = {
        "universe": {"symbols": ["A", "AI", "LGRDY", "UBER", "MISSING"]},
        "symbols_with_data": {
            "A": {"first_session": "2016-01-04"},
            "AI": {"first_session": "2020-12-09"},
            "LGRDY": {"first_session": "2018-05-08"},
            "UBER": {"first_session": "2019-05-10"},
        },
    }
    audit = cohort_audit(manifest).set_index("symbol")
    assert audit.loc["LGRDY", "status"] == "retain"
    assert audit.loc["AI", "status"] == "identity_exclusion"
    assert audit.loc["UBER", "status"] == "later_series_exclusion"
    assert audit.loc["MISSING", "status"] == "missing_price_exclusion"


def test_missing_holding_outcome_excludes_week_instead_of_averaging_other_names(
    monkeypatch,
):
    dates = pd.date_range("2026-01-02", periods=22, freq="W-FRI")
    symbols = [f"S{i}" for i in range(15)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["friday", "symbol"])
    cohort = pd.DataFrame(
        0.01, index=index, columns=[report.MAD, report.RAW, report.V1]
    )
    full = pd.DataFrame(0.01, index=index, columns=[report.V1_FULL, report.PRIOR])
    actual = pd.Series(0.02, index=index, name="actual").drop((dates[-1], symbols[-1]))
    choices = {(day, model): symbols for day in dates for model in report.MODELS}
    monkeypatch.setattr(
        report,
        "market_returns",
        lambda days: pd.DataFrame(0.0, index=days, columns=["SPY", "QQQ"]),
    )
    weekly, returns, omissions = report.weekly_results(cohort, full, actual, choices)
    assert len(returns) == 21
    assert omissions == [str(dates[-1].date())]
    assert weekly.loc[weekly.friday.eq(dates[-1]), "actual"].isna().all()
    assert np.allclose(returns[report.MAD], 0.02)


def test_four_prespecified_contrasts_and_holm_family():
    dates = pd.date_range("2026-01-02", periods=21, freq="W-FRI")
    returns = pd.DataFrame(
        {name: 0.002 for name in (*report.MODELS, "SPY", "QQQ")}, index=dates
    )
    returns[report.MAD] = 0.012
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["friday", "symbol"])
    panel = pd.DataFrame(
        0.0,
        index=index,
        columns=[report.MAD, report.RAW, report.V1, report.PRIOR, "Zero"],
    )
    panel["actual"] = np.tile([0.0, 0.02], len(dates))
    choices = {
        (day, model): [f"S{i}" for i in range(15)]
        for day in dates
        for model in report.MODELS
    }
    _, forecast, tests, turnover = report.statistics(returns, panel, choices)
    primary = tests.loc[tests.Block.eq(4)]
    assert set(primary.Reference) == set(report.REFERENCES)
    assert len(tests) == 12
    assert np.allclose(primary["MAD minus reference (pp)"], 1.0)
    assert primary["Holm p"].ge(primary["Two-sided p"]).all()
    assert primary["Holm p"].lt(0.05).all()
    assert np.isclose(
        forecast.loc[forecast.Model.eq("Zero"), "Stock RMSE (pp)"].iloc[0], np.sqrt(2)
    )
    assert turnover["Mean names replaced"].eq(0).all()


def test_trainer_runs_real_weekly_regression_and_reloads_owned_artifacts(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 16, 7)).astype(np.float32)
    y = rng.normal(0, 0.01, size=16).astype(np.float32)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2021-01-01", periods=4, freq="W-FRI"), list("ABCD")],
        names=["friday", "symbol"],
    )
    config = PatchTSTConfig(
        context_length=16,
        num_input_channels=7,
        patch_length=4,
        patch_stride=2,
        d_model=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        ffn_dim=16,
        num_targets=1,
        scaling=None,
    )
    monkeypatch.setattr(trainer, "OUTPUT", tmp_path)
    monkeypatch.setattr(trainer, "make_patchtst", lambda: PatchTSTForRegression(config))
    monkeypatch.setattr(trainer, "load_arrays", lambda split, arm: (x, y, index))
    monkeypatch.setattr(trainer, "EPOCHS", 2)
    monkeypatch.setattr(trainer, "BATCH", 8)
    (tmp_path / "dataset.json").write_text("{}\n")
    original = x.copy()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        trainer.fit("mad", {"target_std": 0.01}, torch.device("cpu"))
        stored = np.load(tmp_path / "mad_predictions.npy")
        model = PatchTSTForRegression(config)
        model.load_state_dict(torch.load(tmp_path / "mad.pt", weights_only=True))
        replay = trainer.patch_predictions(model, x, torch.device("cpu")) * 0.01
        np.testing.assert_allclose(stored, replay, rtol=0, atol=0)
        np.testing.assert_array_equal(x, original)
        assert stored.shape == (16,)
        assert np.isfinite(stored).all()
        # Idempotent completion must reuse verified files rather than refitting.
        trainer.fit("mad", {"target_std": 0.01}, torch.device("cpu"))
    finally:
        torch.set_num_threads(old_threads)


def test_calendar_comparison_includes_four_session_week_with_observed_prices(
    tmp_path, monkeypatch
):
    dates = pd.date_range("2026-01-02", periods=22, freq="W-FRI")
    sessions = pd.bdate_range(dates[0], dates[-1] + pd.Timedelta(days=7)).difference(
        pd.DatetimeIndex(["2026-01-19"])
    )
    symbols = [f"S{i}" for i in range(15)]
    close = pd.DataFrame(
        100 * 1.01 ** np.arange(len(sessions))[:, None] * np.ones((1, 15)),
        index=sessions,
        columns=symbols,
    )
    logs = np.log(close).diff()
    index = pd.MultiIndex.from_product([dates, symbols], names=["friday", "symbol"])
    cohort = pd.DataFrame(0.0, index=index, columns=[report.MAD, report.RAW, report.V1])
    full = pd.DataFrame(0.0, index=index, columns=[report.V1_FULL, report.PRIOR])
    choices = {(day, model): symbols for day in dates for model in report.MODELS}
    monkeypatch.setattr(
        report, "session_data", lambda: (None, None, close, None, logs, None)
    )
    monkeypatch.setattr(report, "OUTPUT", tmp_path)
    monkeypatch.setattr(report, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(
        report,
        "market_returns",
        lambda days: pd.DataFrame(0.0, index=days, columns=["SPY", "QQQ"]),
    )
    report.FIGURES.mkdir()
    summary, _ = report.calendar_sensitivity(cohort, full, choices)
    assert summary["weeks"] == 22
    assert summary["four_session_decisions"] == ["2026-01-16"]
    returns = pd.read_csv(
        tmp_path / "calendar_friday/paired_returns.csv", index_col="friday"
    )
    assert np.isclose(returns.loc["2026-01-16", report.MAD], 1.01**4 - 1)
    assert np.isclose(returns.loc["2026-01-09", report.MAD], 1.01**5 - 1)
