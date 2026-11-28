"""Explicit scratch scenario inputs; old command defaults remain unchanged."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSettings:
    root: Path
    universe: str
    benchmark: str

    def __post_init__(self):
        if self.universe not in {"halal", "halal_filtered"}:
            raise ValueError(f"Unsupported experiment universe: {self.universe}")
        if self.benchmark not in {"QQQ", "SPY"}:
            raise ValueError(f"Unsupported experiment benchmark: {self.benchmark}")

    @property
    def variants(self):
        symbol = self.benchmark.lower()
        return (
            "current",
            f"{symbol}_excess",
            "equal_weight_excess",
            f"{symbol}_hybrid",
        )


def load_settings(scenario=None):
    if scenario is None:
        return ExperimentSettings(Path(__file__).resolve().parent, "halal", "QQQ")
    source = Path(scenario).resolve(strict=True)
    settings = json.loads(source.read_text())
    return ExperimentSettings(
        source.parent, settings["universe"], settings["benchmark"]
    )


def command_settings():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", type=Path)
    args, _ = parser.parse_known_args()
    return load_settings(args.scenario)


SETTINGS = command_settings()
ROOT = SETTINGS.root
UNIVERSE = SETTINGS.universe
BENCHMARK = SETTINGS.benchmark
