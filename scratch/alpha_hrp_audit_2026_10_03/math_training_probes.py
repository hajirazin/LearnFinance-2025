"""Controlled forward/gradient probes. Never trains or writes a production model."""

import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import PatchTSTForPrediction

# Reuse only the evidence-loading/panel functions, not the walk-forward execution.
source = (
    Path(__file__)
    .with_name("math_audit.py")
    .read_text()
    .split(
        "# setup-only boundary: math_training_probes execs the prefix above this line."
    )[0]
)
namespace: dict = {}
exec(compile(source, str(Path(__file__).with_name("math_audit.py")), "exec"), namespace)
panel = namespace["panel"]
artifacts = namespace["artifacts"]
OUT = namespace["OUT"]
versions = namespace["versions"]
storage = namespace["storage"]
pointer = namespace["pointer"]

ss, xa, ya, cal = panel(date(2026, 8, 24))
x = torch.from_numpy(xa[:64])
y = torch.from_numpy(ya[:64])
probes = {}
for version, art in artifacts.items():
    ci = art.config.feature_names.index("close_ret")
    xx = x if ci == 3 else x[:, :, 3:4]
    yy = y if ci == 3 else y[:, :, 3:4]
    model = art.model
    with torch.no_grad():
        out = model(past_values=xx, future_values=yy)
    entry = {
        "weight_decay_in_saved_config": art.config.weight_decay,
        "effective_patch_stride": model.config.patch_stride,
        "patch_count": model.model.patchifier.num_patches,
        "norm_type": model.config.norm_type,
        "raw_channel_mse": ((out.prediction_outputs - yy) ** 2).mean((0, 1)).tolist(),
    }
    normalized_close = ((out.prediction_outputs - out.loc) / out.scale)[:, :, ci]
    shuffled_close = xx.clone()
    shuffled_close[:, :, ci] = torch.flip(shuffled_close[:, :, ci], dims=[1])
    with torch.no_grad():
        reordered_pred = model(past_values=shuffled_close).prediction_outputs[:, :, ci]
    entry["normalized_close_prediction_max_cross_asset_std"] = float(
        normalized_close.std(dim=0).max()
    )
    entry["reverse_close_order_preserving_mean_variance_max_daily_change"] = float(
        (reordered_pred - out.prediction_outputs[:, :, ci]).abs().max()
    )
    if ci == 3:
        perturbed = xx.clone()
        for c in [0, 1, 2, 4]:
            perturbed[:, :, c] = torch.flip(perturbed[:, :, c], dims=[1])
        with torch.no_grad():
            other = model(past_values=perturbed).prediction_outputs[:, :, ci]
        entry["nonclose_reversal_eval_max_close_change"] = float(
            (other - out.prediction_outputs[:, :, ci]).abs().max()
        )
        # Isolate batch normalization from stochastic dropout. Train-mode probes
        # operate on disposable copies; no optimizer steps or persisted state.
        original = copy.deepcopy(model).train()
        changed = copy.deepcopy(model).train()
        for m in list(original.modules()) + list(changed.modules()):
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        inputs = xx.clone().requires_grad_()
        train_pred = original(past_values=inputs).prediction_outputs[:, :, ci]
        train_loss = (train_pred - yy[:, :, ci]).square().mean()
        train_loss.backward()
        nonclose_grad = inputs.grad[:, :, [0, 1, 2, 4]]
        with torch.no_grad():
            other_train = changed(past_values=perturbed).prediction_outputs[:, :, ci]
        entry["nonclose_reversal_train_max_close_change"] = float(
            (other_train - train_pred.detach()).abs().max()
        )
        entry["close_loss_nonclose_input_gradient_l2_train"] = float(
            nonclose_grad.norm()
        )
        one_config = copy.deepcopy(model.config)
        one_config.num_input_channels = 1
        one = PatchTSTForPrediction(one_config).train()
        one.load_state_dict(model.state_dict())
        for m in one.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        with torch.no_grad():
            one_train = one(past_values=x[:, :, 3:4]).prediction_outputs[:, :, 0]
        entry["same_weights_geometry_5_vs_1_train_max_close_change"] = float(
            (one_train - train_pred.detach()).abs().max()
        )
    probes[version] = entry

decomp = json.loads((OUT / "math_prediction_decomposition.json").read_text())
diagnostics = {}
for v in versions:
    r = [r for r in decomp if r["version"] == v]
    metrics = []
    for row in r:
        pred = np.array(row["predicted_weekly_logs"])
        actual = np.array(row["actual_weekly_logs"])
        loc = np.array(row["loc_component"])
        residual = np.array(row["learned_residual"])
        keep = np.array(
            [not (s == "CTVA" and row["week"] == "2026-09-28") for s in row["symbols"]]
        )
        full = set(np.argsort(-pred)[:15])
        mean_only = set(np.argsort(-loc)[:15])
        metrics.append(
            {
                "week": row["week"],
                "std_residual_over_std_loc": float(residual.std() / loc.std()),
                "top15_overlap_with_momentum_only": len(full & mean_only),
                "residual_rank_ic": float(
                    spearmanr(residual[keep], actual[keep]).statistic
                ),
                "momentum_only_rank_ic": float(
                    spearmanr(loc[keep], actual[keep]).statistic
                ),
                "network_rank_ic": float(spearmanr(pred[keep], actual[keep]).statistic),
            }
        )
    diagnostics[v] = {
        "weeks": metrics,
        "mean_top15_overlap": float(
            np.mean([m["top15_overlap_with_momentum_only"] for m in metrics])
        ),
        "mean_std_residual_over_std_loc": float(
            np.mean([m["std_residual_over_std_loc"] for m in metrics])
        ),
        "mean_residual_rank_ic": float(
            np.mean([m["residual_rank_ic"] for m in metrics])
        ),
    }
assert storage.read_current_version() == pointer
(OUT / "math_training_probes.json").write_text(
    json.dumps(
        {
            "probes": probes,
            "decomposition": diagnostics,
            "current_pointer_unchanged": True,
        },
        indent=2,
    )
)
print(
    json.dumps(
        {
            "probes": probes,
            "decomposition_summary": {
                v: {k: x for k, x in a.items() if k != "weeks"}
                for v, a in diagnostics.items()
            },
        },
        indent=2,
    )
)
