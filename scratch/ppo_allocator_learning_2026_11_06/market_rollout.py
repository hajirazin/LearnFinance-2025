"""Cached causal features with the production policy, reward and book drift."""

from copy import copy

import numpy as np
import torch

from brain_api.core.ppo_discovery.diagnostics import (
    compute_portfolio_transition_diagnostics,
)
from brain_api.core.ppo_discovery.environment import _post_rebalance_weights
from brain_api.core.ppo_discovery.rewards import ppo_discovery_reward
from brain_api.core.ppo_discovery.rollout import RolloutStep


def state_with_book(template, weights):
    state = copy(template)
    state.asset_features = template.asset_features.copy()
    state.globals = template.globals.copy()
    state.current_weights = dict(weights)
    state.held_symbols = tuple(
        sorted(s for s, w in weights.items() if s != "CASH" and w > 0)
    )
    eligible = {
        s for s, flag in zip(template.symbols, template.asset_mask, strict=True) if flag
    }
    missing = set(state.held_symbols) - eligible
    if missing:
        raise ValueError(f"Held symbols became ineligible: {sorted(missing)}")
    state.asset_features[:, 8] = [
        weights.get(s, 0.0) if flag else 0.0
        for s, flag in zip(template.symbols, template.asset_mask, strict=True)
    ]
    state.globals[2] = weights.get("CASH", 0.0)
    state.state_digest = ""  # Core-only ephemeral state, never sent to an API.
    return state


def equalize(target):
    stocks = [s for s, w in target.items() if s != "CASH" and w > 0]
    if not stocks:
        return {"CASH": 1.0}
    return {
        **dict.fromkeys(stocks, (1 - target["CASH"]) / len(stocks)),
        "CASH": target["CASH"],
    }


def execute(prior, target, row, config):
    active = {
        s
        for s, w in {**prior, **target}.items()
        if s != "CASH" and (prior.get(s, 0) > 0 or target.get(s, 0) > 0)
    }
    missing = active - set(row["returns"])
    if missing:
        raise ValueError(f"Missing next-open reward prices for {sorted(missing)}")
    returns = {s: row["returns"][s] for s in active}
    opens = {s: row["opens"][s] for s in active}
    reward, gross, cost, net = ppo_discovery_reward(
        prior_weights=prior,
        target_weights=target,
        symbol_returns=returns,
        symbol_prices=opens,
        nav_usd=config.training_nav_usd,
        config=config,
    )
    post = _post_rebalance_weights(target, returns, opens, cost)
    diag = compute_portfolio_transition_diagnostics(
        prior, target, transaction_cost_fraction=cost
    )
    return post, {"reward": reward, "gross": gross, "cost": cost, "net_log": net}, diag


def rollout(
    policy, rows, config, cache=None, deterministic=False, equal_selected=False
):
    prior = {"CASH": 1.0}
    steps = []
    for index, row in enumerate(rows):
        state = state_with_book(row["state"], prior)
        with torch.no_grad():
            if deterministic:
                from brain_api.core.ppo_discovery.environment import (
                    _action_from_weights,
                )

                target, _order, value = policy.infer_decision_value(state)
                if equal_selected:
                    target = equalize(target)
                action = _action_from_weights(state, target)
                logp = 0.0
            else:
                embeddings = None if cache is None else cache.get(state)
                action, value, logp = policy.sample_action_value_log_prob(
                    state, temporal_embeddings=embeddings
                )
        target = dict(action.percentage_weights)
        prior, outcome, diag = execute(prior, target, row, config)
        steps.append(
            RolloutStep(
                state=state,
                action=action,
                reward=outcome["reward"],
                value=value,
                log_p=logp,
                done=index == len(rows) - 1,
                realized_net_return=outcome["net_log"],
                diagnostics=diag,
            )
        )
    return steps


def selected_dispersion(weights):
    stock = np.array(
        [w for s, w in weights.items() if s != "CASH" and w > 0], dtype=float
    )
    if len(stock) < 2:
        return {"cv": None, "max_relative_deviation": None}
    return {
        "cv": float(stock.std() / stock.mean()),
        "max_relative_deviation": float(np.abs(stock / stock.mean() - 1).max()),
    }
