"""Read-only math probes and real five-task checkpoint comparison. No retraining."""

import copy
import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import PatchTSTForPrediction

from brain_api.core.features import compute_ohlcv_log_returns
from brain_api.core.patchtst.price_history import completed_context_dates, session_dates
from brain_api.storage.patchtst.local import PatchTSTModelStorage

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(90210)
storage = PatchTSTModelStorage(ROOT / "brain_api/data")
pointer = storage.read_current_version()
prices = pickle.loads((OUT / "full_price_evidence.pkl").read_bytes())
universe = json.loads(
    (ROOT / "brain_api/data/cache/universe/halal_new_2026-09.json").read_text()
)
symbols = sorted({s["symbol"] for s in universe["stocks"]})
versions = [
    d.name
    for d in sorted(storage._model_path.glob("v*"))
    if (d / "metadata.json").exists()
]
artifacts = {v: storage.load_version_artifacts(v) for v in versions}


def panel(monday):
    dates = completed_context_dates(monday, 60)
    target = session_dates(monday, monday + timedelta(days=12))[:5]
    xs, ys, calendar, ss = [], [], [], []
    for s in symbols:
        frame = prices.get(s)
        if (
            frame is None
            or len(dates.difference(frame.index))
            or len(target.difference(frame.index))
        ):
            continue
        all_dates = dates.append(target)
        features = compute_ohlcv_log_returns(frame.loc[all_dates])
        x = features.iloc[:60].to_numpy()
        y = features.iloc[60:].to_numpy()
        if (
            x.shape != (60, 5)
            or y.shape != (5, 5)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            continue
        xs.append(x)
        ys.append(y)
        ss.append(s)
        friday = monday + timedelta(days=4)
        end = target[target.date <= friday][-1]
        calendar.append(np.log(frame.loc[end, "close"] / frame.loc[dates[-1], "close"]))
    return (
        ss,
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(calendar),
    )


# setup-only boundary: math_training_probes execs the prefix above this line.
rows = []
saved = []
probe_data = None
for ts in pd.date_range("2026-05-11", "2026-09-28", freq="W-MON"):
    monday = ts.date()
    ss, x, y, cal = panel(monday)
    if len(ss) < 15:
        raise RuntimeError("panel too small")
    mu = x[:, :, 3].mean(axis=1)
    actual = y[:, :, 3].sum(axis=1)
    for version, art in artifacts.items():
        metadata = storage.read_metadata(version)
        trained = pd.Timestamp(metadata["training_timestamp"]).date()
        if trained >= monday:
            continue
        xx = x if art.config.num_input_channels == 5 else x[:, :, 3:4]
        with torch.no_grad():
            output = art.model(past_values=torch.from_numpy(xx))
        ci = art.config.feature_names.index("close_ret")
        prediction = output.prediction_outputs[:, :, ci].numpy().sum(axis=1)
        loc_component = output.loc[:, :, ci].numpy().reshape(-1) * 5
        residual = prediction - loc_component
        ranking = np.argsort(-prediction, kind="stable")
        valid = np.array(
            [not (s == "CTVA" and monday == date(2026, 9, 28)) for s in ss]
        )
        row = {
            "version": version,
            "week": str(monday),
            "n_common": len(ss),
            "channels": art.config.num_input_channels,
            "rank_ic_next5": float(spearmanr(prediction, actual).statistic),
            "rank_ic_next5_ex_corporate_action": float(
                spearmanr(prediction[valid], actual[valid]).statistic
            ),
            "rank_ic_calendar": float(spearmanr(prediction, cal).statistic),
            "corr_pred_mean60": float(spearmanr(prediction, mu).statistic),
            "std_pred_log": float(prediction.std()),
            "std_loc_component": float(loc_component.std()),
            "std_learned_residual": float(residual.std()),
            "top15_equal_calendar_pct": float(np.expm1(cal[ranking[:15]]).mean() * 100),
            "top15_equal_next5_pct": float(np.expm1(actual[ranking[:15]]).mean() * 100),
            "mae_next5_pp": float(
                np.abs(np.expm1(prediction) - np.expm1(actual)).mean() * 100
            ),
            "top15_symbols": [ss[i] for i in ranking[:15]],
            "median_prediction_pct": float(np.median(np.expm1(prediction)) * 100),
        }
        rows.append(row)
        if monday >= date(2026, 8, 24):
            saved.append(
                {
                    "version": version,
                    "week": str(monday),
                    "symbols": ss,
                    "predicted_weekly_logs": prediction.tolist(),
                    "actual_weekly_logs": actual.tolist(),
                    "mean60_daily_logs": mu.tolist(),
                    "loc_component": loc_component.tolist(),
                    "learned_residual": residual.tolist(),
                }
            )
    if monday == date(2026, 8, 24):
        probe_data = (x[:64], y[:64])
    print(str(monday), len(ss), "completed", flush=True)

summary = {}
for v in versions:
    frame = pd.DataFrame(
        [r for r in rows if r["version"] == v and r["week"] >= "2026-08-24"]
    )
    summary[v] = {
        "weeks": len(frame),
        "mean_rank_ic_next5": float(frame.rank_ic_next5.mean()),
        "mean_rank_ic_ex_corporate_action": float(
            frame.rank_ic_next5_ex_corporate_action.mean()
        ),
        "mean_corr_pred_mean60": float(frame.corr_pred_mean60.mean()),
        "top15_calendar_compounded_pct": float(
            (np.prod(1 + frame.top15_equal_calendar_pct / 100) - 1) * 100
        ),
    }

x, y = map(torch.from_numpy, probe_data)
probes = {}
for version in [
    "v2026-07-24-786e036ecc83",
    "v2026-08-14-e7b14b211a54",
    "v2026-08-21-0af698826abd",
]:
    art = artifacts[version]
    model = art.model
    ci = art.config.feature_names.index("close_ret")
    xx = x if ci == 3 else x[:, :, 3:4]
    yy = y if ci == 3 else y[:, :, 3:4]
    with torch.no_grad():
        out = model(past_values=xx, future_values=yy)
        shifted = xx.clone()
        shifted[:, :, ci] += 0.001
        shifted_out = model(past_values=shifted).prediction_outputs
        errors = (out.prediction_outputs - yy).square().mean(dim=(0, 1))
        normalized = ((out.prediction_outputs - yy) / out.scale).square().mean()
    gradients = {}
    for name, multitask in [("old_all_channel_loss", True), ("close_only_loss", False)]:
        model.zero_grad(set_to_none=True)
        outputs = model(past_values=xx, future_values=yy)
        loss = (
            outputs.loss
            if multitask
            else (outputs.prediction_outputs[:, :, ci] - yy[:, :, ci]).square().mean()
        )
        loss.backward()
        gg = torch.cat(
            [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
        )
        parameters = torch.cat(
            [p.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
        )
        gradients[name] = {
            "loss": float(loss.detach()),
            "grad_l2": float(gg.norm()),
            "fraction_abs_grad_below_adam_epsilon": float(
                (gg.abs() < 1e-8).float().mean()
            ),
            "decay_to_gradient_l2_ratio_at_1e4": float(
                1e-4 * parameters.norm() / gg.norm()
            ),
        }
    result = {
        "hf_loss": float(out.loss),
        "manual_raw_mse": float(errors.mean()),
        "manual_revin_normalized_mse": float(normalized),
        "channel_mse": errors.tolist(),
        "volume_share_of_total_squared_error": float(errors[4] / errors.sum())
        if ci == 3
        else None,
        "mean_shift_added_daily_log": 0.001,
        "max_error_from_expected_daily_shift": float(
            (shifted_out[:, :, ci] - out.prediction_outputs[:, :, ci] - 0.001)
            .abs()
            .max()
        ),
        "expected_weekly_score_multiplier": float(np.exp(5 * 0.001)),
        "gradients_eval_mode": gradients,
    }
    if ci == 3:
        oneconfig = copy.deepcopy(model.config)
        oneconfig.num_input_channels = 1
        one = PatchTSTForPrediction(oneconfig).eval()
        one.load_state_dict(model.state_dict())
        with torch.no_grad():
            one_output = one(past_values=x[:, :, 3:4]).prediction_outputs[:, :, 0]
        result["same_weights_geometry_5_vs_1_eval_close_delta"] = float(
            (one_output - out.prediction_outputs[:, :, 3]).abs().max()
        )
    probes[version] = result
assert storage.read_current_version() == pointer
result = {
    "rows": rows,
    "post_migration_same_panel_summary": summary,
    "math_probes": probes,
    "current_pointer_unchanged": True,
    "caveats": [
        "Frozen September universe applied historically: survivorship bias.",
        "All comparisons are checkpoint forward diagnostics after their actual training timestamp, not production portfolio NAV.",
        "Top15 diagnostics have no HRP/stickiness, costs or execution.",
        "CTVA September28 target includes stock-distribution discontinuity; both inclusive and excluded rank IC are reported.",
    ],
}
(OUT / "math_audit.json").write_text(json.dumps(result, indent=2))
(OUT / "math_prediction_decomposition.json").write_text(json.dumps(saved, indent=2))
pd.DataFrame(rows).drop(columns=["top15_symbols"]).to_csv(
    OUT / "math_weekly_comparison.csv", index=False
)
print(json.dumps({"summary": summary, "probes": probes}, indent=2))
