"""Fit the H21–H24 loss experiments with H5's architecture and training rows."""

import argparse
from functools import partial

import pandas as pd
import torch
from train_v1_hypotheses import train
from v1_hypothesis_data import ARMS, OUTPUT
from v1_ibkr_cost_loss import CostAwareWeeklyMSE
from v1_loss_objectives import pairwise_ranking, spy_relative, weekly_mse

RESULT = OUTPUT / "loss_comparison"
OBJECTIVES = {
    "h21": (weekly_mse, False, False),
    "h22": (pairwise_ranking, True, False),
    "h23": (spy_relative, True, True),
    "h24": (None, True, False),
    "h21_chronological": (None, True, False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--arm", choices=OBJECTIVES)
    args = parser.parse_args()
    torch.set_num_threads(4)
    RESULT.mkdir(exist_ok=True)
    names = OBJECTIVES if args.arm is None else (args.arm,)
    spy = None
    if "h23" in names:
        spy = pd.read_csv(
            RESULT / "spy_training_closes.csv", parse_dates=["date"]
        ).set_index("date")["close"]
    ordinary = {
        name: ARMS["h5_features_training_filter"]
        for name in names
        if name in ("h21", "h22", "h23")
    }
    if ordinary:
        train(
            torch.device(args.device),
            ordinary,
            output=RESULT,
            objectives=OBJECTIVES,
            spy_close=spy,
        )
    for name in ("h24", "h21_chronological"):
        if name not in names:
            continue
        output = RESULT / "h24_ibkr"
        output.mkdir(exist_ok=True)
        factory = (
            CostAwareWeeklyMSE
            if name == "h24"
            else partial(CostAwareWeeklyMSE, cost_weight=0)
        )
        train(
            torch.device(args.device),
            {name: ARMS["h5_features_training_filter"]},
            output=output,
            objectives=OBJECTIVES,
            sequence_factory=factory,
        )


if __name__ == "__main__":
    main()
