"""Weekly forecast, ranking, and benchmark-relative PatchTST objectives."""

import torch
import torch.nn.functional as F
from patchtst_replay_models import simple_returns

TOP_K = 15
RANK_TEMPERATURE = 0.02
BASKET_TEMPERATURE = 0.01
SHORTFALL_MULTIPLIER = 2.0


def weekly_mse(output, target, _benchmark=None):
    return F.mse_loss(
        simple_returns(output, daily=True), simple_returns(target, daily=True)
    )


def pairwise_ranking(output, target, _benchmark=None):
    predicted = simple_returns(output, daily=True)
    actual = simple_returns(target, daily=True)
    return pairwise_score_loss(predicted / RANK_TEMPERATURE, actual)


def pairwise_score_loss(scores, actual):
    gaps = scores[:, None] - scores[None, :]
    preferred = (actual[:, None] > actual[None, :]).to(gaps.dtype)
    return (F.softplus(-gaps) * preferred).sum() / preferred.sum().clamp_min(1)


def soft_top15(scores, *, temperature=BASKET_TEMPERATURE):
    """Approximate selecting 15 names via a smooth inclusion threshold."""
    if len(scores) < TOP_K:
        raise ValueError("A training week must have at least 15 candidates")
    if len(scores) == TOP_K:
        return scores * 0 + 1 / TOP_K
    with torch.no_grad():
        low = scores.min() - 30 * temperature
        high = scores.max() + 30 * temperature
        for _ in range(26):
            mid = (low + high) / 2
            count = torch.sigmoid((scores - mid) / temperature).sum()
            low, high = (
                torch.where(count > TOP_K, mid, low),
                torch.where(count > TOP_K, high, mid),
            )
        threshold = (low + high) / 2
    inclusion = torch.sigmoid((scores - threshold) / temperature)
    slope = (inclusion * (1 - inclusion)).sum().clamp_min(1e-8)
    # Newton correction gives the threshold its implicit gradient while
    # leaving the forward value at the solved top-k inclusion count.
    threshold = threshold + temperature * (inclusion.sum() - TOP_K) / slope
    inclusion = torch.sigmoid((scores - threshold) / temperature)
    return inclusion / inclusion.sum()


def excess_loss(excess):
    return -excess + (SHORTFALL_MULTIPLIER - 1) * F.relu(-excess)


def spy_relative(output, target, benchmark):
    weights = soft_top15(simple_returns(output, daily=True))
    realized = weights @ simple_returns(target, daily=True)
    return excess_loss(realized - benchmark)
