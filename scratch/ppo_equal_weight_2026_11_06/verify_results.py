"""Independent checks on the exported experiment, without another model replay."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / "summary.json").read_text())
weeks = json.loads((ROOT / "weekly.json").read_text())
stocks = pd.read_csv(ROOT / "stock_weights.csv")
before = json.loads((ROOT / "protected_before.json").read_text())
after = json.loads((ROOT / "protected_after.json").read_text())
assert before == after and len(before) == summary["protected_file_count"]
assert len(weeks) == 69 and len(stocks) == 690
assert stocks.groupby("as_of").size().eq(10).all()
assert np.allclose(stocks.groupby("as_of").equity_weight.sum(), 1)
for week in weeks:
    stock = stocks[stocks.as_of == week["as_of"]]
    weights = stock.portfolio_weight.to_numpy()
    cv = weights.std() / weights.mean()
    assert abs(cv - week["cv"]) < 1e-12
    assert abs(sum(week["weights"].values()) - 1) < 1e-12
    assert (
        abs(len(stock) * week["equity_equal_portfolio_weight"] + week["cash"] - 1)
        < 1e-12
    )
for name in ["ppo", "equal_fixed"]:
    logs = np.array([week[name]["net_log"] for week in weeks])
    independent_cagr = float(np.exp(logs.sum() * 52 / len(logs)) - 1)
    assert abs(independent_cagr - summary["performance"][name]["cagr"]) < 1e-12
assert sum(w["max_relative_deviation"] > 0.10 for w in weeks) == 8
assert all(w["max_relative_deviation"] <= 0.15 for w in weeks)
assert summary["dispersion"]["mean_cv_bootstrap"]["4"]["upper95_one_sided"] < 0.05
print(
    "Verified: 69 weeks, 690 weights, dispersion, equal-weight mass, performance and 416 unchanged protected files."
)
