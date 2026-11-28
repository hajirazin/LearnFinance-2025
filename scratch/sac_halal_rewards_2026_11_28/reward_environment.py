"""Change only the scalar reward, retaining production transition accounting."""

from dataclasses import replace

import numpy as np


def benchmark_log_hurdle(
    variant, benchmark_returns, equal_weight_returns, *, benchmark="QQQ"
):
    if benchmark not in {"QQQ", "SPY"}:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    equal_weight = np.asarray(equal_weight_returns, dtype=float)
    if benchmark_returns.shape != equal_weight.shape:
        raise ValueError("Benchmarks must align")
    if not np.all(np.isfinite(benchmark_returns)) or not np.all(
        np.isfinite(equal_weight)
    ):
        raise ValueError("Benchmarks must be complete and finite")
    if np.any(benchmark_returns <= -1) or np.any(equal_weight <= -1):
        raise ValueError("Benchmark wealth must remain positive")
    if variant == "current":
        return np.zeros_like(benchmark_returns)
    if variant == f"{benchmark.lower()}_excess":
        return np.log1p(benchmark_returns)
    if variant == "equal_weight_excess":
        return np.log1p(equal_weight)
    if variant == f"{benchmark.lower()}_hybrid":
        return np.log1p(np.maximum(benchmark_returns, 0))
    raise ValueError(f"Unknown reward variant: {variant}")


class RewardOffsetEnv:
    def __init__(self, env, log_hurdles):
        self.env = env
        self.log_hurdles = np.asarray(log_hurdles, dtype=float)
        if self.log_hurdles.shape != (env.n_weeks,) or not np.all(
            np.isfinite(self.log_hurdles)
        ):
            raise ValueError(
                "Benchmark hurdle must be finite and aligned with every week"
            )

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def max_episode_weeks(self):
        return self.env.max_episode_weeks

    @max_episode_weeks.setter
    def max_episode_weeks(self, value):
        self.env.max_episode_weeks = value

    def reset(self, start_week=None):
        return self.env.reset(start_week=start_week)

    def step(self, action):
        week = self.env.current_week_idx
        result = self.env.step(action)
        return replace(
            result,
            reward=result.reward
            - self.env.config.reward_scale * self.log_hurdles[week],
        )
