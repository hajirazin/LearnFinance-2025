"""Create seven-line copies, leaving the original six-line graphs untouched."""

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "scratch/sac_reward_comparison_charts_2026_11_28"))

from chart_data import load_experiment, make_curve, read_rows  # noqa: E402
from draw_charts import draw, set_style  # noqa: E402


def main():
    set_style()
    records = []
    for number, universe, benchmark, directory, stem, title in (
        (
            1,
            "halal",
            "QQQ",
            "sac_halal_rewards_2026_11_28",
            "01_sac_halal_with_hrp",
            "SAC halal: seven-scenario comparison",
        ),
        (
            2,
            "halal_filtered",
            "SPY",
            "sac_halal_filtered_spy_rewards_2026_11_28",
            "02_sac_halal_filtered_with_hrp",
            "SAC: seven-scenario comparison",
        ),
    ):
        curves = load_experiment(REPO / "scratch" / directory, universe, benchmark)
        metrics = json.loads((ROOT / f"{universe}_hrp_252_metrics.json").read_text())
        source = ROOT / f"{universe}_hrp_252_returns.csv"
        hrp = make_curve(
            universe,
            6,
            "HRP (252-session lookback)",
            None,
            read_rows(source),
            "net_return",
            metrics["test_end"],
            source,
        )
        if not math.isclose(
            hrp.cagr, metrics["cagr"], abs_tol=1e-10
        ) or not math.isclose(hrp.sharpe, metrics["sharpe"], abs_tol=1e-10):
            raise ValueError("HRP chart does not match HRP evaluation metrics")
        curves.append(hrp)
        if len(curves) != 7:
            raise ValueError("Expected exactly seven scenario lines")
        draw(
            curves,
            title,
            f"{len(metrics['symbols'])}-stock {universe} universe | Original six scenarios plus weekly one-stage HRP",
            stem,
            number,
            output_root=ROOT,
            extra_note="HRP: prior 252 daily returns, weekly rebalancing, 98% stocks and 2% cash",
        )
        for curve in curves:
            item = asdict(curve)
            item.pop("wealth")
            item["source"] = str(Path(curve.source).relative_to(REPO))
            item["dates"] = [day.isoformat() for day in curve.dates]
            item["final_value_usd"] = curve.wealth[-1]
            records.append(item)
    (ROOT / "series_manifest.json").write_text(json.dumps(records, indent=2))
    print(
        "Created two new seven-line graphs and exports; original graphs were not written."
    )


if __name__ == "__main__":
    main()
