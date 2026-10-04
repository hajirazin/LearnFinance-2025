"""Offline real-checkpoint verification; no promotions or broker calls."""

import json
import pickle
from datetime import date
from pathlib import Path

import torch
from transformers import PatchTSTConfig as HFConfig, PatchTSTForPrediction

from brain_api.core.inference_utils import compute_week_boundaries
from brain_api.core.patchtst.inference import build_inference_features, run_inference
from brain_api.core.sticky_selection import select_with_rank_band
from brain_api.storage.patchtst.local import PatchTSTModelStorage

OUT = Path(__file__).parent
ROOT = OUT.resolve().parents[1]
torch.set_num_threads(4)
storage = PatchTSTModelStorage(ROOT / "brain_api/data")
pointer_before = storage.read_current_version()
current = storage.load_current_artifacts()
old = storage.load_version_artifacts("v2026-08-14-e7b14b211a54")
raw = json.loads((storage._version_path(old.version) / "config.json").read_text())
manual = HFConfig(
    **{
        k: raw[k]
        for k in [
            "num_input_channels",
            "context_length",
            "patch_length",
            "stride",
            "d_model",
            "num_attention_heads",
            "num_hidden_layers",
            "ffn_dim",
            "dropout",
            "prediction_length",
        ]
    },
    attention_dropout=raw["dropout"],
    positional_dropout=raw["dropout"],
    use_cls_token=False,
    pooling_type="mean",
)
original = PatchTSTForPrediction(manual)
original.load_state_dict(
    torch.load(
        storage._version_path(old.version) / "weights.pt",
        weights_only=True,
        map_location="cpu",
    )
)
original.eval()
x = torch.randn(4, 60, 5)
with torch.no_grad():
    delta = float(
        (
            old.model(past_values=x).prediction_outputs
            - original(past_values=x).prediction_outputs
        )
        .abs()
        .max()
    )
assert delta == 0.0
prices = pickle.loads((OUT / "prices.pkl").read_bytes())
audit = json.loads((OUT / "analysis.json").read_text())
rows = []
for d in ["2026-08-24", "2026-08-31", "2026-09-28"]:
    cutoff = date.fromisoformat(d)
    features = [
        build_inference_features(s, p, current.config, cutoff)
        for s, p in prices.items()
        if s not in {"SPY", "QQQ"}
    ]
    preds = run_inference(
        current.model,
        current.feature_scaler,
        features,
        compute_week_boundaries(cutoff),
        current.config,
    )
    scores = {
        p.symbol: p.predicted_weekly_return_pct
        for p in preds
        if p.predicted_weekly_return_pct is not None
    }
    saved = audit["weeks"][[r["week"] for r in audit["rows"]].index(d)][
        "full_recomputed_scores"
    ]
    common = scores.keys() & saved.keys()
    row = {
        "week": d,
        "valid_sessions_scores": len(scores),
        "common_symbols": len(common),
        "max_prior_audit_score_delta_pp": max(
            abs(scores[s] - saved[s]) for s in common
        ),
        "ATEYY": scores.get("ATEYY"),
        "SMCI": scores.get("SMCI"),
    }
    if d == "2026-08-31":
        prior = set(
            "PSX VLO CSLLY SSDOY CRL ZBRA STX PANW EXPE RCRUY PINS WDAY OKTA AXON TEAM".split()
        )
        original_set = set(
            "INCY ONC CSLLY CRL SSDOY PANW ZBRA DASH RCRUY EXPE MMYT AXON ATEYY TEAM SMCI".split()
        )
        selected = set(select_with_rank_band(scores, prior, 15, 20).selected)
        row.update(
            {
                "fresh_session_selected": sorted(selected),
                "added_vs_email": sorted(selected - original_set),
                "removed_vs_email": sorted(original_set - selected),
            }
        )
    elif row["max_prior_audit_score_delta_pp"] != 0:
        raise AssertionError("Complete context forecasts changed")
    rows.append(row)
assert storage.read_current_version() == pointer_before
result = {
    "active_version": current.version,
    "legacy_version": old.version,
    "legacy_effective_stride": old.model.config.patch_stride,
    "legacy_model_vs_original_adapter_output_delta": delta,
    "current_pointer_unchanged": True,
    "rows": rows,
}
(OUT / "correction_verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
