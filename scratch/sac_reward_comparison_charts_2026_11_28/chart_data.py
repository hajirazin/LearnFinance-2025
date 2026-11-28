"""Read completed experiment traces and construct date-aligned wealth curves."""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import mean, stdev

SEEDS = (42, 123, 2026)
INITIAL_CAPITAL = 10_000.0


@dataclass(frozen=True)
class Curve:
    universe: str
    category: int
    label: str
    seed: int | None
    dates: tuple[date, ...]
    wealth: tuple[float, ...]
    cagr: float
    sharpe: float
    source: str
    source_sha256: str


def wealth_curve(sessions, final_date, returns, initial=INITIAL_CAPITAL):
    starts = tuple(date.fromisoformat(value) for value in sessions)
    end = date.fromisoformat(final_date)
    if not starts or len(starts) != len(returns):
        raise ValueError("Each weekly transition requires exactly one return")
    if tuple(sorted(set(starts))) != starts or end <= starts[-1]:
        raise ValueError("Session starts and final endpoint must be strictly ordered")
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("Initial capital must be finite and positive")
    values = [initial]
    for value in returns:
        if not math.isfinite(value) or value <= -1:
            raise ValueError("Each return must be finite and greater than -100%")
        values.append(values[-1] * (1 + value))
    return (*starts, end), tuple(values)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_curve(universe, category, label, seed, rows, column, final_date, source):
    returns = tuple(float(row[column]) for row in rows)
    dates, wealth = wealth_curve([row["session"] for row in rows], final_date, returns)
    cagr = (wealth[-1] / wealth[0]) ** (52 / len(returns)) - 1
    volatility = stdev(returns)
    sharpe = math.sqrt(52) * mean(returns) / volatility if volatility > 1e-10 else 0.0
    return Curve(
        universe,
        category,
        label,
        seed,
        dates,
        wealth,
        cagr,
        sharpe,
        str(source),
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def load_experiment(root: Path, universe: str, benchmark: str):
    manifest = json.loads((root / "evidence/manifest.json").read_text())
    output = root / "results_50000"
    benchmark_path = output / "benchmarks.csv"
    benchmark_rows = read_rows(benchmark_path)
    expected_sessions = [row["session"] for row in benchmark_rows]
    if len(expected_sessions) != 104:
        raise ValueError("Expected the completed 104-week evaluation window")
    curves = []
    variants = (
        ("current", "Current reward"),
        (f"{benchmark.lower()}_excess", f"Minus {benchmark} return"),
        ("equal_weight_excess", "Minus equal-weight return"),
        (f"{benchmark.lower()}_hybrid", f"{benchmark} hybrid"),
    )
    for category, (variant, label) in enumerate(variants):
        candidates = []
        for seed in SEEDS:
            folder = output / variant / f"seed_{seed}"
            record = json.loads((folder / "metrics.json").read_text())
            if record["seed"] != seed or record["config"]["total_timesteps"] != 50_000:
                raise ValueError("Candidate identity or training budget mismatch")
            candidates.append((record["test"]["cagr"], seed, folder, record))
        _, seed, folder, record = sorted(
            candidates, key=lambda item: (item[0], item[1])
        )[1]
        source = folder / "test_returns.csv"
        rows = read_rows(source)
        if [row["session"] for row in rows] != expected_sessions:
            raise ValueError("Candidate and benchmark sessions do not align")
        curve = make_curve(
            universe,
            category,
            label,
            seed,
            rows,
            "net_return",
            manifest["test_end"],
            source,
        )
        if not math.isclose(
            curve.cagr, record["test"]["cagr"], abs_tol=1e-10
        ) or not math.isclose(curve.sharpe, record["test"]["sharpe"], abs_tol=1e-10):
            raise ValueError("Chart curve disagrees with saved candidate metrics")
        curves.append(curve)
    for category, column, label in (
        (4, f"{benchmark}_buy_hold", f"{benchmark} buy-and-hold"),
        (5, f"{universe}_equal_weight_buy_hold", "Equal-weight buy-and-hold"),
    ):
        curves.append(
            make_curve(
                universe,
                category,
                label,
                None,
                benchmark_rows,
                column,
                manifest["test_end"],
                benchmark_path,
            )
        )
    return curves
