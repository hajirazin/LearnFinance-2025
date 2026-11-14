"""Train H26 daily and H27 weekly seven-target PatchTST models."""

import argparse
import json
import subprocess
from functools import partial

import numpy as np
import pandas as pd
import torch
from compare_feature_sets import SEVEN
from compare_patchtst import sha256, write_json
from patchtst_replay_models import fit_patchtst
from torch import nn
from train_v1_hypotheses import SevenFeaturePatchTST
from train_v1_loss_comparison import RESULT as LOSS_RESULT
from v1_hypothesis_data import (
    VALIDATION,
    daily_features,
    decision_dates,
    session_data,
    windows,
)

RESULT = LOSS_RESULT / "multitarget"
ARMS = {"h26": 5, "h27": 1}
TARGETS = (
    "close_log_return",
    *(
        ("log1p_volume" if name == "volume" else name)
        for name in SEVEN
        if name != "close"
    ),
)
FINAL_DATE = pd.Timestamp("2025-12-31")


class MultiTargetPatchTST(SevenFeaturePatchTST):
    def __init__(self, context, horizon):
        super().__init__(context)
        self.horizon = horizon
        self.head = nn.Linear(self.config.d_model * len(SEVEN), horizon * len(TARGETS))
        nn.init.normal_(self.head.weight, std=self.config.init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, mask):
        encoded = self.encoder(
            past_values=x[..., : len(SEVEN)], past_observed_mask=mask[..., : len(SEVEN)]
        ).last_hidden_state
        return self.head(encoded.mean(dim=2).flatten(1)).reshape(
            -1, self.horizon, len(TARGETS)
        )


def multichannel_output(model, x, mask, *, daily):
    return model(x, mask)


def close_forecast(output, *, daily, mean, scale):
    close_log = output[..., 0] * scale + mean
    return torch.expm1(close_log.sum(dim=1))


def future_targets(index, logs, features):
    """Build raw labels and keep missing values visible until the fold is selected."""
    dates = index.get_level_values("friday")
    start = logs.index.get_indexer(dates)
    symbols = logs.columns.get_indexer(index.get_level_values("symbol"))
    daily_pos = start[:, None] + np.arange(1, 6)
    end = logs.index.get_indexer(dates + pd.Timedelta(days=7))
    weekly = np.full((len(index), 1, len(TARGETS)), np.nan, dtype=np.float32)
    log_values = logs.to_numpy(dtype=np.float32)
    daily = np.empty((len(index), 5, len(TARGETS)), dtype=np.float32)
    daily[..., 0] = log_values[daily_pos, symbols[:, None]]
    aux = [i for i, name in enumerate(SEVEN) if name != "close"]
    daily[..., 1:] = features[daily_pos[..., None], symbols[:, None, None], aux]
    # A strict Friday pair may have a shortened week due to a midweek holiday.
    valid = (dates.day_of_week == 4) & (end > start) & (end - start <= 5)
    row = np.flatnonzero(valid)
    positions = end[row]
    weekly[row, 0, 1:] = features[positions[:, None], symbols[row, None], aux]
    for count in range(1, 6):
        use = row[end[row] - start[row] == count]
        weekly[use, 0, 0] = daily[use, :count, 0].sum(axis=1)
    daily[..., -1] = np.log1p(daily[..., -1])
    weekly[..., -1] = np.log1p(weekly[..., -1])
    exact_five = valid & (end - start == 5)
    return daily, weekly, exact_five


def train(device, arms):
    calendar, _, close, volume, logs, smaller = session_data()
    features = daily_features(calendar, close, volume, logs, evaluation=False)
    dates = decision_dates(close.index, evaluation=False)
    sessions = close.index
    dates = dates[sessions.get_indexer(dates) + 5 <= sessions.get_loc(FINAL_DATE)]
    folds = (
        dates[sessions[sessions.get_indexer(dates) + 5] < VALIDATION],
        dates[dates >= VALIDATION],
    )
    columns = logs.columns.isin(smaller)
    train_idx, train_x, train_mask, _ = windows(
        logs.loc[:, columns],
        features[:, columns],
        folds[0],
        use_features=True,
        year=False,
    )
    val_idx, val_x, val_mask, val_y = windows(
        logs, features, folds[1], use_features=True, year=False
    )
    wide_idx, _, _, wide_y = windows(
        logs, features, folds[1], use_features=True, year=True
    )
    common = val_idx.isin(wide_idx[np.isfinite(wide_y).all(axis=1)])
    index = train_idx.append(val_idx)
    x = np.concatenate([train_x, val_x])
    mask = np.concatenate([train_mask, val_mask])
    del train_x, train_mask, val_x, val_mask, val_y, wide_y
    daily, weekly, exact_five = future_targets(index, logs, features)
    train_rows = np.arange(len(index)) < len(train_idx)
    shared_validation = (
        np.r_[np.zeros(len(train_idx), dtype=bool), common]
        & exact_five
        & np.isfinite(daily).all(axis=(1, 2))
        & np.isfinite(weekly).all(axis=(1, 2))
    )
    for name in arms:
        weight_file = RESULT / f"{name}.pt"
        meta_file = RESULT / f"{name}.json"
        if (
            weight_file.exists()
            and meta_file.exists()
            and sha256(weight_file)
            == json.loads(meta_file.read_text())["weights_sha256"]
        ):
            print(f"{name}: reusing verified checkpoint", flush=True)
            continue
        horizon = ARMS[name]
        raw = daily if horizon == 5 else weekly
        fit = train_rows & np.isfinite(raw).all(axis=(1, 2))
        selected = fit | shared_validation
        fit_rows = fit[selected]
        target = raw[selected]
        mean = target[fit_rows].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        scale = target[fit_rows].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"{name}: nonpositive or invalid target scale")
        labels = pd.DataFrame(
            {"forward_return": np.expm1(target[..., 0].sum(axis=1, dtype=np.float64))},
            index=index[selected],
        )
        print(
            f"{name}: {fit.sum():,} fitting rows, {shared_validation.sum():,} validation rows",
            flush=True,
        )
        _, state, metadata = fit_patchtst(
            x[selected],
            (target - mean) / scale,
            mask[selected],
            fit_rows,
            ~fit_rows,
            ~fit_rows,
            labels,
            daily=True,
            device=device,
            epochs=60,
            batch_size=256,
            model_factory=lambda x, *, daily, horizon=horizon: MultiTargetPatchTST(
                x.shape[1], horizon
            ),
            output_fn=multichannel_output,
            return_fn=partial(
                close_forecast, mean=float(mean[0]), scale=float(scale[0])
            ),
        )
        torch.save(state, weight_file)
        write_json(
            meta_file,
            {
                **metadata,
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "weights_sha256": sha256(weight_file),
                "horizon": horizon,
                "targets": TARGETS,
                "means": mean.tolist(),
                "scales": scale.tolist(),
                "train_rows": int(fit.sum()),
                "train_weeks": len(index[fit].unique("friday")),
                "training_symbols": len(index[fit].unique("symbol")),
                "validation_rows": int(shared_validation.sum()),
                "validation_weeks": len(index[shared_validation].unique("friday")),
            },
        )
        if device.type == "mps":
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    RESULT.mkdir(exist_ok=True, parents=True)
    train(torch.device(args.device), (args.arm,) if args.arm else ARMS)


if __name__ == "__main__":
    main()
