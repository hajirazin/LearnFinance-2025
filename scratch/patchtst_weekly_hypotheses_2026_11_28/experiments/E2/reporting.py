"""Paired, week-level evaluation after both research checkpoints are locked."""

import numpy as np
import pandas as pd
import torch
from evidence import ROOT, read_json, sha256, write_json

from brain_api.core.patchtst.config import create_patchtst_model
from brain_api.core.patchtst.weekly_training import (
    PatchTSTTargetScale,
    predict_weekly_returns,
)


def weekly_metrics(index, predicted):
    frame = index.copy()
    frame["predicted"] = predicted
    records, previous = [], set()
    for anchor, group in frame.groupby("anchor", sort=True):
        ordered = group.sort_values(["predicted", "symbol"], ascending=[False, True])
        selected = set(ordered.head(15).symbol)
        if len(group) < 30 or group.predicted.std(ddof=0) < 1e-12:
            raise ValueError(f"Insufficient/degenerate forecast ranks at {anchor}")
        residual = group.predicted - group.actual
        records.append(
            {
                "anchor": str(anchor),
                "rank_ic": float(group.predicted.rank().corr(group.actual.rank())),
                "mse": float(np.mean(residual**2)),
                "mae": float(np.mean(abs(residual))),
                "bias": float(residual.mean()),
                "prediction_std": float(group.predicted.std(ddof=0)),
                "top15_excess": float(
                    ordered.head(15).actual.mean() - group.actual.mean()
                ),
                "top_bottom_spread": float(
                    ordered.head(15).actual.mean() - ordered.tail(15).actual.mean()
                ),
                "replacement_fraction": len(selected - previous) / 15
                if previous
                else None,
                "selected": ",".join(ordered.head(15).symbol),
                "eligible": len(group),
            }
        )
        previous = selected
    return pd.DataFrame(records)


def block_interval(differences, block):
    differences = np.asarray(differences, dtype=float)
    if not np.isfinite(differences).all() or len(differences) < block * 2:
        raise ValueError("Insufficient finite paired weeks")
    rng = np.random.default_rng(42)
    starts = rng.integers(
        0, len(differences), (20000, int(np.ceil(len(differences) / block)))
    )
    rows = (starts[..., None] + np.arange(block)) % len(differences)
    means = differences[rows.reshape(20000, -1)[:, : len(differences)]].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "block": block,
        "difference": float(differences.mean()),
        "low": float(low),
        "high": float(high),
    }


def decision(primary, rmse_ratio):
    if primary["low"] > 0 and primary["difference"] >= 0.005 and rmse_ratio <= 1.02:
        return "Promising screening result; needs independent seed/era confirmation."
    if primary["high"] < 0.005:
        return "Does not support the prespecified useful ranking improvement here."
    return "Inconclusive against the prespecified useful improvement."


def evaluate(panel, config, device, attempt, manifest):
    index = pd.DataFrame(
        {
            "anchor": panel.anchor_dates,
            "target_end": panel.target_dates,
            "symbol": panel.symbols,
            "actual": panel.y,
        }
    )
    weekly, summaries, fits = {}, {}, {}
    # Verify checkpoint integrity before evaluating either arm.
    for arm in ("mse", "huber"):
        fits[arm] = read_json(attempt / arm / "result.json")
        if sha256(attempt / arm / "model.pt") != fits[arm]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {arm}")
    for arm in ("mse", "huber"):
        fit = fits[arm]
        model = create_patchtst_model(config).to(device)
        model.load_state_dict(
            torch.load(
                attempt / arm / "model.pt", map_location=device, weights_only=True
            )
        )
        predicted = predict_weekly_returns(
            model, panel.X, PatchTSTTargetScale(fit["target_std"])
        )
        index[arm] = predicted
        week = weekly_metrics(index, predicted)
        week.to_csv(attempt / f"{arm}_weekly.csv", index=False)
        weekly[arm] = week
        mean = week.mean(numeric_only=True)
        squared = np.sort((predicted - panel.y) ** 2)
        tail = squared[-max(1, int(np.ceil(len(squared) * 0.01))) :]
        counts = index.groupby("anchor").symbol.transform("size").to_numpy()
        weights = 1.0 / counts
        summaries[arm] = {
            "mean_rank_ic": float(mean.rank_ic),
            "rmse_pp": float(100 * np.sqrt(mean.mse)),
            "mae_pp": float(100 * mean.mae),
            "bias_pp": float(100 * mean.bias),
            "mean_within_week_prediction_std_pp": float(100 * mean.prediction_std),
            "top15_excess_pp": float(100 * mean.top15_excess),
            "top_bottom_spread_pp": float(100 * mean.top_bottom_spread),
            "replacement_fraction": float(mean.replacement_fraction),
            "top1pct_squared_error_share": float(tail.sum() / squared.sum()),
            "zero_baseline_rmse_pp": float(
                100 * np.sqrt(np.average(panel.y**2, weights=weights))
            ),
            "mean_baseline_rmse_pp": float(
                100
                * np.sqrt(
                    np.average((panel.y - fit["training_mean"]) ** 2, weights=weights)
                )
            ),
        }
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    if not weekly["mse"].anchor.equals(weekly["huber"].anchor):
        raise ValueError("Evaluation weeks do not match")
    differences = weekly["huber"].rank_ic - weekly["mse"].rank_ic
    intervals = [block_interval(differences, block) for block in (4, 2, 8)]
    disposition = decision(
        intervals[0], summaries["huber"]["rmse_pp"] / summaries["mse"]["rmse_pp"]
    )
    acf = {str(lag): float(differences.autocorr(lag)) for lag in (1, 2, 4, 8)}
    index.to_csv(attempt / "predictions.csv", index=False)
    result = {
        "metrics": summaries,
        "paired_ic_intervals": intervals,
        "paired_ic_autocorrelations": acf,
        "decision": disposition,
        "weeks": len(differences),
        "outcomes": len(index),
    }
    write_json(attempt / "evaluation.json", result)
    lines = [
        "# E2 result: MSE versus Huber",
        "",
        disposition,
        "",
        f"Attempt: `{attempt.name}`. {len(differences)} paired weeks, {len(index):,} stock outcomes.",
        "",
        "| Arm | Best epoch | Stopped epoch | Validation MSE | Evaluation Rank IC | Evaluation RMSE (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("mse", "huber"):
        f, m = fits[arm], summaries[arm]
        lines.append(
            f"| {arm} | {f['best_epoch']} | {f['stopped_epoch']} | "
            f"{f['validation_mse']:.8f} | {m['mean_rank_ic']:.5f} | {m['rmse_pp']:.4f} |"
        )
    lines += [
        "",
        "Huber minus MSE mean weekly Rank IC, paired circular-week bootstrap:",
        "",
    ]
    for interval in intervals:
        lines.append(
            f"- Block {interval['block']}: {interval['difference']:+.5f}, "
            f"95% interval [{interval['low']:+.5f}, {interval['high']:+.5f}]."
        )
    lines += [
        "",
        f"Fold evidence: `{manifest['folds']}`.",
        "",
        "One seed, reused 2025 evaluation, current-roster survivorship bias and incomplete "
        "independent ticker/corporate-action audit limit the conclusion. Strict point-in-time "
        "news was used; missing cells were not filled with neutral scores. Both arms retain "
        "unmodified returns and use the same validation-MSE checkpoint policy. Top-15 marks "
        "are descriptive gross five-session returns, without transaction costs or HRP. "
        "More epochs alone do not establish useful learning. No production model was changed.",
        "",
        f"[Protocol](PROTOCOL.md) · [Full metrics](data/attempts/{attempt.name}/evaluation.json) · "
        f"[MSE epochs](data/attempts/{attempt.name}/mse/progress.json) · "
        f"[Huber epochs](data/attempts/{attempt.name}/huber/progress.json)",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result
