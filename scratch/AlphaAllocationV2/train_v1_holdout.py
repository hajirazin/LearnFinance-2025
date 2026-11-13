"""Freeze the two V1 comparison checkpoints using the 2016–2025 archive."""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from build_v1_holdout_data import OUTPUT
from compare_patchtst import sha256, write_json
from patchtst_replay_data import load_replay
from patchtst_replay_models import fit_patchtst

MODELS = ("baseline_60d", "challenger_7f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    OUTPUT.mkdir(exist_ok=True)
    data = load_replay()
    friday = data.labels.index.get_level_values("friday")
    ends = data.labels["next_friday"]
    observed = data.labels["forward_return"].notna() & np.isfinite(data.daily_y).all(
        axis=1
    )
    train = (
        observed & (friday >= pd.Timestamp("2021-01-01")) & ends.lt("2025-01-01")
    ).to_numpy()
    validation = (
        observed & (friday >= pd.Timestamp("2025-01-01")) & ends.le("2025-12-31")
    ).to_numpy()
    print(
        f"Frozen fit: {train.sum():,} training rows, {validation.sum():,} validation rows",
        flush=True,
    )
    if not train.any() or not validation.any():
        raise ValueError("Empty fitting or validation data")
    for name, x, y, mask, daily in (
        (
            MODELS[0],
            data.daily_x[:, -60:],
            data.daily_y,
            data.daily_mask[:, -60:],
            True,
        ),
        (
            MODELS[1],
            data.weekly_x,
            data.labels["forward_return"].to_numpy()[:, None],
            np.zeros(len(friday), dtype=bool),
            False,
        ),
    ):
        print(name, flush=True)
        _, state, metadata = fit_patchtst(
            x,
            y,
            mask,
            train,
            validation,
            validation,
            data.labels,
            daily=daily,
            device=device,
            epochs=60,
            batch_size=256,
        )
        weights = OUTPUT / f"{name}.pt"
        torch.save(state, weights)
        write_json(
            OUTPUT / f"{name}.json",
            {
                **metadata,
                "weights_sha256": sha256(weights),
                "last_training_outcome": ends[train].max(),
                "last_validation_outcome": ends[validation].max(),
                "training_rows": int(train.sum()),
                "validation_rows": int(validation.sum()),
                "runtime": {"torch": torch.__version__, "device": args.device},
            },
        )
    print(json.dumps({"frozen": MODELS}), flush=True)


if __name__ == "__main__":
    main()
