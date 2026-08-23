# Full-universe US PatchTST patch-geometry report

Experiment date: 2026-08-23. Preregistered design: [PLAN.md](PLAN.md). Attempt:
`20260823T053324Z` on repository commit
`c4725b7f0f49d72f80918a34c8dc3bfa50f9dc3e` (`main`).

## Decision

Reject all three patch geometries for research clearance and do not promote any
artifact.

The 10/5 geometry is the clear *within-PatchTST* winner on the untouched partial
2026 path: it had the highest rank IC, top-15 excess, top-minus-bottom spread,
and stability, and all three seeds had positive confirmatory rank IC. Its paired
rank-IC and spread advantages over both 8/4 and 16/8 excluded zero.

That positive result is not enough to clear the mandatory causal gates. Every
PatchTST ensemble had negative rank IC in both development years. In 2026, 10/5
beat causal mean and ridge on point rank IC and portfolio diagnostics, but the
four-week paired intervals were wide and included zero; it also had significantly
worse MAE than both gates. The preregistered rule therefore rejects 10/5 alongside
8/4 and 16/8.

Nothing was promoted, no production `current` pointer or cache was touched, no
trade was submitted, and no Temporal workflow was triggered.

## Frozen geometry-only design

The model received one channel: adjusted-close daily log return. Non-close OHLCV
fields had no effect on input values or eligibility. Every eligible symbol-week
required one pre-context close, all 60 context closes, and all five target closes
on exact XNYS sessions, with finite positive closes and no interpolation.

Only the following pair changed across arms:

| Arm | Patch length | Stride | Unpadded patches |
|---|---:|---:|---:|
| 8/4 | 8 | 4 | 14 |
| 10/5 | 10 | 5 | 11 |
| 16/8 | 16 | 8 | 6 |

No stride-1 arm was run. Every other setting was identical: context 60, horizon
5, mean pooling, two encoder layers, `d_model=64`, four heads, FFN 128, shared
embedding/projection, channel attention off, batch norm, pre-norm, GELU, fixed
sin/cos positions, no CLS token, HF standard scaling, attention and positional
dropout 0.20, all other dropout zero, daily close MSE, Adam at `3e-4`, zero
weight decay, batch 256, gradient clip 1.0, plateau scheduler, 60-epoch cap,
rank-IC checkpoint selection with validation-MSE tie-break, patience 8, and seeds
`20260823`, `20260824`, `20260825`.

Mean pooling was frozen because it keeps the forecast-head width constant when
the patch count changes. A close-only channel was frozen because the earlier
12-stock validation selected the close-only architecture and the prior
five-channel graph/objective introduced avoidable confounds.

## Data, folds, and label lock

- Universe: all 430 current `halal_new` symbols from the production-built August
  cache, fetched `2026-08-08T14:37:54.382306+00:00`; universe SHA-256
  `e2ed929ce96e21e61b46b135a36b5d6cd36874e016d38fccf6fdbbc31c2308b2`.
- Prices: yfinance 1.0, `auto_adjust=True`, requested `2015-01-01` through
  `2026-08-22` exclusive. All 430/430 symbols downloaded; observed files span
  `2015-01-02` through `2026-08-21`. The shortest newly listed history had 61
  rows and the longest had 2,926. Adjusted-price manifest SHA-256:
  `6b6801467e3f6fd8b4c739c1feb778b255dd9deae26facb526bb486e0c6ed4ed`.
- Development 2024: train through 2022, validate 2023, evaluate 2024. Evaluation
  had 21,536 rows over 51 weeks, 420–424 symbols/week (median 423).
- Development 2025: expanding train through 2023, validate 2024, evaluate 2025.
  Evaluation had 21,681 rows over 51 weeks, 424–426 symbols/week (median 425).
- Confirmatory 2026: expanding train through 2024, validate 2025, evaluate
  `2026-01-12..2026-08-17`. Evaluation had 13,688 rows over 32 weeks, 426–429
  symbols/week (median 428).
- Missing exact-session exclusions were 16,843, 17,114, and 17,207 in the three
  complete fold panels. There were no finite-positive-close exclusions.
- The confirmatory panel withheld target values while all nine 2026 checkpoints
  were trained. Only after their paths and SHA-256 values were verified did the
  runner write `confirmatory_unlock.json` at
  `2026-08-23T06:22:39.570529+00:00` and read 2026 evaluation targets.

Current-universe survivorship and Yahoo revision risk remain material. This is
not a point-in-time-membership strategy return study.

## Ensemble results

Returns, excesses, and spreads are arithmetic weekly averages. “Turnover” is the
fraction of top-15 names replaced from one decision week to the next.

### Development 2024

| Model | MAE | Direction | Rank IC | Top-15 excess | Top15-bottom15 | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST 8/4 | 3.428% | 50.66% | -0.00666 | 0.323% | 0.322% | 29.33% |
| PatchTST 10/5 | 3.436% | 50.93% | -0.01121 | 0.411% | 0.257% | 29.07% |
| PatchTST 16/8 | 3.423% | 50.64% | -0.01141 | 0.388% | 0.399% | 47.73% |
| Causal historical mean | 3.248% | 51.94% | 0.02165 | 0.855% | 0.547% | 5.33% |
| Close-only ridge | 3.249% | 51.96% | 0.01718 | 0.538% | 0.499% | 75.87% |

### Development 2025

| Model | MAE | Direction | Rank IC | Top-15 excess | Top15-bottom15 | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST 8/4 | 3.758% | 49.32% | -0.01428 | 0.341% | 0.272% | 37.20% |
| PatchTST 10/5 | 3.764% | 49.64% | -0.00079 | 0.465% | 0.502% | 33.47% |
| PatchTST 16/8 | 3.791% | 48.89% | -0.00470 | 0.681% | 0.661% | 42.80% |
| Causal historical mean | 3.588% | 53.15% | 0.00218 | 0.101% | -0.004% | 5.73% |
| Close-only ridge | 3.574% | 53.66% | 0.02281 | 0.481% | 0.619% | 73.20% |

### Untouched confirmatory 2026

| Model | MAE | Direction | Rank IC | Top-15 excess | Top15-bottom15 | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST 8/4 | 4.567% | 51.67% | 0.02940 | 1.525% | 1.310% | 31.40% |
| **PatchTST 10/5** | **4.558%** | **51.62%** | **0.04054** | **1.782%** | **2.019%** | **23.23%** |
| PatchTST 16/8 | 4.583% | 51.14% | 0.03022 | 1.226% | 1.143% | 31.18% |
| Causal historical mean | 4.390% | 51.94% | 0.00756 | 0.998% | 1.116% | 6.67% |
| Close-only ridge | 4.398% | 51.62% | 0.00632 | 0.941% | 0.456% | 81.08% |

Confirmatory rank IC was positive for every individual seed:

| Geometry | Seed 20260823 | Seed 20260824 | Seed 20260825 |
|---|---:|---:|---:|
| 8/4 | 0.03095 | 0.03054 | 0.02519 |
| 10/5 | 0.03480 | 0.04113 | 0.04155 |
| 16/8 | 0.03283 | 0.02763 | 0.02999 |

That seed agreement supports a genuine 2026 within-PatchTST ordering, but it does
not repair the negative development history or establish superiority to causal
models across regimes.

## Paired four-week block uncertainty

All intervals use 2,000 deterministic moving-block repetitions by decision week.

On confirmatory 2026, 10/5 minus 16/8 was:

- Rank IC: `+0.01032`, 95% interval `[+0.00215, +0.01776]`.
- Top-15 excess: `+0.556` percentage points/week, interval
  `[+0.015, +1.045]`.
- Top15-bottom15 spread: `+0.876` points/week, interval
  `[+0.093, +1.700]`.
- Turnover: `-7.96` percentage points, interval `[-13.33, -4.30]`.

On confirmatory 2026, 10/5 minus 8/4 was:

- Rank IC: `+0.01114`, interval `[+0.00419, +0.02043]`.
- Top-15 excess: `+0.257` points/week, interval `[-0.215, +0.797]`.
- Top15-bottom15 spread: `+0.708` points/week, interval
  `[+0.105, +1.453]`.
- Turnover: `-8.17` points, interval `[-14.59, -2.08]`.

The geometry ordering did not exist in pooled development evidence. For 10/5
minus 16/8, development rank IC was `+0.00206` with interval
`[-0.00901, +0.01213]`; for 10/5 minus 8/4 it was `+0.00447`, the sign-reversed
form of the stored 8/4-minus-10/5 comparison, with interval
`[-0.00468, +0.01507]`.

Against mandatory gates, confirmatory 10/5 had encouraging point estimates but
insufficient uncertainty:

- Versus causal mean: rank-IC delta `+0.03298`, interval
  `[-0.03591, +0.09411]`; top-15-excess delta `+0.784` points/week; spread delta
  `+0.902` points/week; MAE was worse by `+0.168` points with interval
  `[+0.065, +0.291]`.
- Versus ridge: rank-IC delta `+0.03422`, interval
  `[-0.08295, +0.14887]`; top-15-excess delta `+0.841` points/week; spread delta
  `+1.563` points/week; MAE was worse by `+0.160` points with interval
  `[+0.046, +0.292]`.

In pooled development, 10/5 rank IC lagged causal mean by `-0.01791` and ridge by
`-0.02599`; both intervals included zero, while MAE was consistently worse with
intervals wholly above zero.

## Hardware, serialization, and runtime

- Hardware: MacBook Pro, Apple M5 Pro, 20-core GPU, 48 GB unified memory.
- Runtime: Python 3.12.13, PyTorch 2.9.1, Transformers 4.57.3, Apple MPS.
- Deterministic smoke: exact repeated state hash
  `37e48f888be0b30a8d4778fb441ac2112497c601621d9be026abee30437e705e`
  and prediction hash
  `5fe3017ea66fee488fd96da4b5e15e360175925bc30ee0bd46f9013cf0671157`.
- 27 declared training jobs completed. The audit ledger recorded 36 model
  lifecycles—27 training/evaluation lifecycles plus nine post-unlock confirmatory
  checkpoint reloads—with `max_active_models=1`.
- End-to-end runtime: 2,808.78 seconds (46m 48.8s). Recorded model training time:
  2,668.30 seconds (44m 28.3s).
- Training runtime by geometry across nine fold/seed jobs: 8/4 769.07s, 10/5
  1,077.77s, 16/8 821.45s. Token count alone did not predict early-stopped
  runtime because selected/stopped epochs differed.

## Integrity identifiers

All four result artifacts and all 27 checkpoint weight hashes were recomputed
successfully after completion. Per-checkpoint hashes, per-symbol adjusted-price
hashes, full source hashes, and panel identity hashes are in
`results/manifest.json`.

| Artifact | SHA-256 |
|---|---|
| `results/predictions.csv` | `f2e39ec6f264d52ea85b594b9696a93182ed44dddb3b6924f57449e1a0da80aa` |
| `results/weekly_metrics.csv` | `2588f93f168d9664b13f68a16eb8bca63391f50ff2294948859fbb30fa795539` |
| `results/confirmatory_unlock.json` | `b42efb074ef61ab0f5583794cb29a04aabede06767eddcbf58c597d4f8bf6bd5` |
| `results/mps_smoke.json` | `be521d5c7e2c8154e2f0bc01c9f22674ad91d9e9bf04ba883d63459dd23134df` |

## Interpretation and next research step

This clean sweep answers the narrow geometry question: among these frozen
mean-pooled close-only models, 10/5 is the strongest geometry on the partial 2026
regime. It does **not** establish a robust forecaster. The development folds say
that geometry alone did not recover stable cross-sectional signal, and simple
causal models still provide better calibration and stronger multi-year evidence.

Do not promote 10/5 from this result. If research continues, preserve 10/5 as the
geometry prior and isolate exactly one next factor—pooling is the natural next
ablation already identified by the earlier audit—while keeping the causal gates,
annual expanding folds, exact-session panel, three seeds, and a new untouched
future confirmation period. Do not reopen 2026 as a selection set and then call it
confirmatory again.
