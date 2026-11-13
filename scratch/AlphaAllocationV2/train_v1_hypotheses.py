"""Train the five frozen V1 factor arms."""

import argparse
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from brain_api.core.patchtst.config import DEFAULT_CONFIG
from compare_patchtst import sha256, write_json
from patchtst_replay_models import build_model, fit_patchtst
from torch import nn
from transformers import PatchTSTModel
from v1_hypothesis_data import (
    ARMS,
    OUTPUT,
    VALIDATION,
    daily_features,
    decision_dates,
    session_data,
    windows,
)


class SevenFeaturePatchTST(nn.Module):
    """Encode seven channels and retain close-return units in the output."""

    def __init__(self, context):
        super().__init__()
        self.config = replace(
            DEFAULT_CONFIG, context_length=context, num_input_channels=7
        ).to_hf_config()
        self.encoder = PatchTSTModel(self.config)
        self.head = nn.Linear(self.config.d_model * 7, 5)
        nn.init.normal_(self.head.weight, std=self.config.init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, past_values, past_observed_mask):
        features, close = past_values[..., :7], past_values[..., 7]
        encoded = self.encoder(
            past_values=features, past_observed_mask=past_observed_mask[..., :7]
        ).last_hidden_state
        mask = past_observed_mask[..., 7]
        count = mask.sum(dim=1, keepdim=True)
        loc = (close * mask).sum(dim=1, keepdim=True) / count
        scale = (
            (((close - loc) ** 2 * mask).sum(dim=1, keepdim=True) / count) + 1e-5
        ).sqrt()
        output = self.head(encoded.mean(dim=2).flatten(1)) * scale + loc
        return SimpleNamespace(prediction_outputs=output.unsqueeze(-1))


def model_factory(x, *, daily):
    if x.shape[-1] == 8:
        return SevenFeaturePatchTST(x.shape[1])
    return build_model(x, daily=daily)


def train(device, arms=ARMS):
    calendar, _prices, close, volume, logs, smaller = session_data()
    features = daily_features(calendar, close, volume, logs, evaluation=False)
    dates = decision_dates(close.index, evaluation=False)
    last_session = close.index.get_loc(pd.Timestamp("2025-12-31"))
    dates = dates[close.index.get_indexer(dates) + 5 <= last_session]
    validation_dates = dates[dates >= VALIDATION]
    val_index, val_x, val_mask, val_y = windows(
        logs, features, validation_dates, use_features=True, year=True
    )
    validation_keys = val_index[np.isfinite(val_y).all(axis=1)]
    del val_x, val_mask, val_y
    for name in arms:
        seven, year, restricted = ARMS[name]
        weights = OUTPUT / f"{name}.pt"
        metadata_path = OUTPUT / f"{name}.json"
        if (
            metadata_path.is_file()
            and weights.is_file()
            and pd.read_json(metadata_path, typ="series")["weights_sha256"]
            == sha256(weights)
        ):
            print(f"{name}: reusing verified checkpoint", flush=True)
            continue
        print(f"Building {name}", flush=True)
        # Only the restricted arms' fitting rows use the small roster. Validation
        # always ranks a common set from the full roster.
        train_dates = dates[
            close.index[close.index.get_indexer(dates) + 5] < VALIDATION
        ]
        train_logs, train_features = logs, features
        if restricted:
            columns = logs.columns.isin(smaller)
            train_logs = logs.loc[:, columns]
            train_features = features[:, columns] if seven else None
        train_idx, train_x, train_mask, train_y = windows(
            train_logs, train_features, train_dates, use_features=seven, year=year
        )
        train_ok = np.isfinite(train_y).all(axis=1)
        if not train_ok.all():
            train_idx, train_x, train_mask, train_y = (
                train_idx[train_ok],
                train_x[train_ok],
                train_mask[train_ok],
                train_y[train_ok],
            )
        val_idx, val_x, val_mask, val_y = windows(
            logs, features, validation_dates, use_features=seven, year=year
        )
        match = val_idx.get_indexer(validation_keys)
        if (match < 0).any():
            raise ValueError(f"Common validation rows missing from {name}")
        index = train_idx.append(validation_keys)
        x = np.concatenate([train_x, val_x[match]])
        mask = np.concatenate([train_mask, val_mask[match]])
        y = np.concatenate([train_y, val_y[match]])
        del train_x, train_mask, train_y, val_x, val_mask, val_y
        labels = pd.DataFrame(
            {"forward_return": np.expm1(y.sum(axis=1, dtype=np.float64))}, index=index
        )
        is_train = np.arange(len(index)) < len(train_idx)
        print(
            f"{name}: {is_train.sum():,} fitting rows, {(~is_train).sum():,} validation rows",
            flush=True,
        )
        _, state, metadata = fit_patchtst(
            x,
            y,
            mask,
            is_train,
            ~is_train,
            ~is_train,
            labels,
            daily=True,
            device=device,
            epochs=60,
            batch_size=256,
            model_factory=model_factory,
        )
        torch.save(state, weights)
        write_json(
            metadata_path,
            {
                **metadata,
                "weights_sha256": sha256(weights),
                "train_rows": len(train_idx),
                "training_symbols": len(train_idx.unique("symbol")),
                "validation_rows": len(validation_keys),
                "context_sessions_padded": x.shape[1],
                "features": seven,
            },
        )
        del x, mask, y
        if device.type == "mps":
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUTPUT.mkdir(exist_ok=True)
    train(torch.device(args.device), ARMS if args.arm is None else (args.arm,))


if __name__ == "__main__":
    main()
