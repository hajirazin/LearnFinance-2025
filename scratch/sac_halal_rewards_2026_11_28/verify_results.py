"""Verify the completed experiment's frozen inputs and paired trace integrity."""

import hashlib
import json

import numpy as np
import pandas as pd
from run_experiment import ROOT, SEEDS, VARIANTS, metrics


def main():
    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    panel_digest = hashlib.sha256(
        (ROOT / "evidence/panel.pkl").read_bytes()
    ).hexdigest()
    assert panel_digest == manifest["panel_sha256"]
    assert (
        hashlib.sha256(
            (ROOT / "evidence/adjusted_ohlcv.parquet").read_bytes()
        ).hexdigest()
        == manifest["price_sha256"]
    )
    output = ROOT / "results_50000"
    benchmark = pd.read_csv(output / "benchmarks.csv")
    assert len(benchmark) == 104 and benchmark.session.is_unique
    benchmark_metrics = json.loads((output / "benchmarks.json").read_text())
    for name in benchmark.columns.drop("session"):
        measured = metrics(benchmark[name])
        for key, value in measured.items():
            assert np.isclose(value, benchmark_metrics[name][key])
    for variant in VARIANTS:
        for seed in SEEDS:
            folder = output / variant / f"seed_{seed}"
            result = json.loads((folder / "metrics.json").read_text())
            trace = pd.read_csv(folder / "test_returns.csv")
            assert result["variant"] == variant and result["seed"] == seed
            assert result["config"]["seed"] == seed
            assert result["config"]["total_timesteps"] == 50_000
            assert result["panel_sha256"] == panel_digest
            assert trace.session.equals(benchmark.session)
            values = trace.drop(columns="session").to_numpy()
            assert np.isfinite(values).all()
            weights = trace.filter(like="weight_")
            assert weights.shape == (104, len(manifest["symbols"]))
            assert (weights.to_numpy() >= 0).all()
            assert (trace.cash >= 0.02 - 1e-10).all()
            assert np.allclose(weights.sum(axis=1) + trace.cash, 1)
            assert (trace.cost_usd >= 0).all()
            for key, value in metrics(trace.net_return).items():
                assert np.isclose(value, result["test"][key])
            for name in ("model.pt", "scaler.pkl", "regime_hmm.pkl"):
                assert (folder / name).stat().st_size > 0
    sensitivity = pd.read_csv(output / "execution_sensitivity.csv")
    assert len(sensitivity) == 48
    assert not sensitivity.duplicated(
        ["variant", "seed", "deadband", "slippage_bps"]
    ).any()
    assert np.isfinite(sensitivity.select_dtypes(include="number").to_numpy()).all()
    assert (sensitivity.weeks == 104).all()
    audit = {
        "matched_training_runs": 12,
        "steps_per_run": 50_000,
        "seeds": list(SEEDS),
        "test_periods_per_run": 104,
        "execution_sensitivity_evaluations": 48,
        "input_hashes_unchanged": True,
        "aligned_traces_and_recomputed_metrics": True,
        "finite_weights_and_cash_floor": True,
        "artifact_files_present": True,
    }
    (output / "verification.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
