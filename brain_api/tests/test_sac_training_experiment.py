from dataclasses import dataclass

from brain_api.core.sac import (
    DEFAULT_SAC_CONFIG,
    SAC_EXPERIMENT_SEEDS,
    SACTrainingExperiment,
)


@dataclass(frozen=True)
class _Result:
    seed: int
    eval_cagr: float


def test_training_experiment_runs_locked_seeds_and_selects_median_cagr():
    cagr_by_seed = {42: 0.31, 123: 0.12, 2026: 0.20}
    seen: list[int] = []

    def train(config):
        seen.append(config.seed)
        return _Result(config.seed, cagr_by_seed[config.seed])

    experiment = SACTrainingExperiment.run(
        config=DEFAULT_SAC_CONFIG,
        train_candidate=train,
        cagr_of=lambda result: result.eval_cagr,
    )

    assert tuple(seen) == SAC_EXPERIMENT_SEEDS
    assert experiment.selected.seed == 2026
    assert [candidate.seed for candidate in experiment.candidates] == [42, 123, 2026]


def test_training_experiment_breaks_equal_cagr_tie_by_seed_before_index_one():
    experiment = SACTrainingExperiment.run(
        config=DEFAULT_SAC_CONFIG,
        train_candidate=lambda config: _Result(config.seed, 0.20),
        cagr_of=lambda result: result.eval_cagr,
    )

    assert experiment.selected.seed == 123
