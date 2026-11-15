"""Train a rank-only PatchTST with previous selection and IBKR fees."""

import argparse

import numpy as np
import torch
from brain_api.core.patchtst.config import DEFAULT_CONFIG
from brain_api.core.patchtst.weekly_rank_ic import checkpoint_is_better
from patchtst_replay_models import SEED
from scipy.stats import spearmanr
from torch import nn
from train_v1_hypotheses import SevenFeaturePatchTST, train
from train_v1_loss_comparison import RESULT as LOSS_RESULT
from v1_hypothesis_data import ARMS
from v1_ibkr_cost_loss import COST_CONFIG, CostAwareWeeklyMSE
from v1_loss_objectives import pairwise_score_loss

RESULT = LOSS_RESULT / "h25_rank_ibkr"
COST_WEIGHT = 50.0
RANK_TEMPERATURE = 0.5


class RankPatchTST(SevenFeaturePatchTST):
    """Map seven feature histories and one membership flag to a score."""

    def __init__(self, context):
        super().__init__(context)
        self.head = nn.Linear(self.config.d_model * 7 + 1, 1)
        nn.init.normal_(self.head.weight, std=self.config.init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, mask, previous):
        encoded = self.encoder(
            past_values=x[..., :7], past_observed_mask=mask[..., :7]
        ).last_hidden_state
        pooled = encoded.mean(dim=2).flatten(1)
        return self.head(torch.cat((pooled, previous[:, None]), dim=1)).squeeze(-1)


def fit_rank(x, mask, labels, is_train, closes, device):
    torch.manual_seed(SEED)
    config = DEFAULT_CONFIG
    model = RankPatchTST(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    x = torch.tensor(x, dtype=torch.float32, device=device)
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    actual = torch.tensor(
        labels["forward_return"].to_numpy(copy=True), dtype=torch.float32, device=device
    )
    portfolio = CostAwareWeeklyMSE(labels.index, is_train, closes, device)
    dates = labels.index.get_level_values("friday")
    batches = {
        training: [
            torch.as_tensor(
                np.flatnonzero((is_train == training) & (dates == day)), device=device
            )
            for day in dates[is_train == training].unique().sort_values()
        ]
        for training in (True, False)
    }

    def epoch(training):
        portfolio.reset(training=training)
        model.train(training)
        losses, fees, ics = [], [], []
        with torch.set_grad_enabled(training):
            for number, rows in enumerate(batches[training]):
                previous = torch.tensor(
                    portfolio.membership(number), dtype=torch.float32, device=device
                )
                scores = model(x[rows], mask[rows], previous)
                target = actual[rows]
                fees_fraction = portfolio.fee_fraction(
                    scores, number, temperature=RANK_TEMPERATURE, sticky=False
                )
                loss = pairwise_score_loss(scores, target) + COST_WEIGHT * fees_fraction
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                else:
                    ics.append(
                        spearmanr(scores.cpu().numpy(), target.cpu().numpy()).statistic
                    )
                losses.append(loss.detach())
                fees.append(fees_fraction.detach())
        return (
            float(torch.stack(losses).mean()),
            float(torch.stack(fees).mean()),
            float(np.mean(ics)) if ics else None,
        )

    best_ic, best_loss, best_epoch = -np.inf, np.inf, 0
    state, history = None, []
    for step in range(1, config.epochs + 1):
        train_loss, train_fees, _ = epoch(True)
        validation_loss, validation_fees, ic = epoch(False)
        if not np.isfinite([train_loss, validation_loss, ic]).all():
            raise FloatingPointError(
                "Nonfinite H25 loss or validation rank correlation"
            )
        history.append(
            {
                "epoch": step,
                "train_loss": train_loss,
                "train_fee_fraction": train_fees,
                "validation_loss": validation_loss,
                "validation_fee_fraction": validation_fees,
                "validation_rank_ic": ic,
            }
        )
        scheduler.step(validation_loss)
        print(
            f"  H25 epoch {step}: val IC {ic:+.4f}, val fee {validation_fees * 10000:.2f} bps",
            flush=True,
        )
        if checkpoint_is_better(ic, validation_loss, best_ic, best_loss):
            best_ic, best_loss, best_epoch = ic, validation_loss, step
            state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if step - best_epoch >= config.early_stopping_patience:
            break
    return state, {
        "config": model.config.to_dict(),
        "seed": SEED,
        "best_epoch": best_epoch,
        "epochs": history,
        "cost_weight": COST_WEIGHT,
        "temperature": RANK_TEMPERATURE,
        "ibkr": COST_CONFIG.to_dict(),
        "objective": "pairwise_logistic_plus_ibkr",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    RESULT.mkdir(exist_ok=True)
    train(
        torch.device(args.device),
        {"h25": ARMS["h5_features_training_filter"]},
        output=RESULT,
        rank_fitter=fit_rank,
    )


if __name__ == "__main__":
    main()
