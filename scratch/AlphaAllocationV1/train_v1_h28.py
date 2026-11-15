"""Fit the H5 architecture on complete-case inputs and labels."""

import argparse

import numpy as np
import pandas as pd
import torch
from compare_patchtst import sha256, write_json
from patchtst_replay_models import fit_patchtst
from train_v1_hypotheses import model_factory
from v1_hypothesis_data import (
    OUTPUT,
    SEVEN,
    VALIDATION,
    daily_features,
    decision_dates,
    session_data,
    windows,
)

RESULT = OUTPUT / "h28"
RETURN_RESULT = OUTPUT / "h28_close_return"


def replace_price_with_return(x, mask):
    """Use the existing past close returns as an encoder feature as well."""
    price = SEVEN.index("close")
    scaling_returns = len(SEVEN)
    if not np.array_equal(mask[..., price], mask[..., scaling_returns]):
        raise ValueError("Price and return histories must share the same coverage")
    features = x.copy()
    features[..., price] = x[..., scaling_returns]
    return features


def fit(device, policy, *, close_input="price"):
    result = RESULT if close_input == "price" else RETURN_RESULT
    calendar, _, close, volume, logs, _ = session_data()
    dates = decision_dates(close.index, evaluation=False)
    target_end = close.index.get_indexer(dates) + 5
    fit_dates = dates[target_end < close.index.searchsorted(VALIDATION)]
    val_dates = dates[
        (dates >= VALIDATION)
        & (target_end <= close.index.get_loc(pd.Timestamp("2025-12-31")))
    ]
    features = daily_features(calendar, close, volume, logs, evaluation=False)
    train_index, x, mask, y = windows(
        logs, features, fit_dates, use_features=True, year=False
    )
    has_label = np.isfinite(y).all(axis=1)
    complete = has_label & mask.all(axis=(1, 2))
    completeness = (
        pd.DataFrame(
            {"h1_eligible": has_label, "complete": complete}, index=train_index
        )
        .groupby(level="symbol")
        .sum()
        .reindex(close.columns, fill_value=0)
    )
    if policy == "whole-stock":
        included = completeness.complete.eq(len(fit_dates))
    else:
        included = completeness.complete.gt(0)
    selected = complete & train_index.get_level_values("symbol").isin(
        completeness.index[included]
    )
    train_index, x, mask, y = (
        train_index[selected],
        x[selected],
        mask[selected],
        y[selected],
    )
    completeness = completeness.assign(
        fit_anchors=len(fit_dates),
        included=included,
        selected_rows=completeness.complete.where(included, 0),
    )
    completeness = completeness.rename_axis("symbol")
    if close_input == "return":
        baseline = pd.read_csv(RESULT / "training_coverage.csv", index_col="symbol")
        expected = pd.read_json(RESULT / "h28.json", typ="series")
        if expected["policy"] != policy:
            raise ValueError("The return arm must use the saved H28 training policy")
        pd.testing.assert_frame_equal(completeness, baseline, check_index_type=False)
    completeness.to_csv(result / "training_coverage.csv")
    val_index, val_x, val_mask, val_y = windows(
        logs, features, val_dates, use_features=True, year=False
    )
    wide, _, _, wide_y = windows(
        logs, features, val_dates, use_features=True, year=True
    )
    validation_keys = wide[np.isfinite(wide_y).all(axis=1)]
    matched = val_index.get_indexer(validation_keys)
    if (matched < 0).any():
        raise ValueError("H5 validation rows missing")
    labels = np.concatenate([y, val_y[matched]])
    index = train_index.append(validation_keys)
    train_rows = np.arange(len(index)) < len(train_index)
    x = np.concatenate([x, val_x[matched]])
    mask = np.concatenate([mask, val_mask[matched]])
    if close_input == "return":
        if len(validation_keys) != expected["validation_rows"]:
            raise ValueError("H28 validation coverage changed")
        x = replace_price_with_return(x, mask)
    print(
        f"H28 {close_input}, {policy}: {included.sum()} training stocks, {len(train_index):,} training rows, "
        f"{len(validation_keys):,} H5 validation rows.",
        flush=True,
    )
    _, state, metadata = fit_patchtst(
        x,
        labels,
        mask,
        train_rows,
        ~train_rows,
        ~train_rows,
        pd.DataFrame(
            {"forward_return": np.expm1(labels.sum(axis=1, dtype=np.float64))},
            index=index,
        ),
        daily=True,
        device=device,
        epochs=60,
        batch_size=256,
        model_factory=model_factory,
    )
    weights = result / "h28.pt"
    torch.save(state, weights)
    write_json(
        result / "h28.json",
        {
            **metadata,
            "policy": policy,
            "close_input": close_input,
            "fit_anchors": len(fit_dates),
            "fit_start": fit_dates[0].date().isoformat(),
            "fit_end": fit_dates[-1].date().isoformat(),
            "training_symbols": int(included.sum()),
            "train_rows": int(train_rows.sum()),
            "validation_rows": len(validation_keys),
            "weights_sha256": sha256(weights),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", required=True, choices=("whole-stock", "complete-windows")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--close-input", choices=("price", "return"), default="price")
    args = parser.parse_args()
    result = RESULT if args.close_input == "price" else RETURN_RESULT
    result.mkdir(parents=True, exist_ok=True)
    if (result / "h28.json").exists() or (result / "h28.pt").exists():
        parser.error(f"H28 already has a saved checkpoint in {result}")
    torch.set_num_threads(4)
    fit(torch.device(args.device), args.policy, close_input=args.close_input)


if __name__ == "__main__":
    main()
