# Full-universe US PatchTST pooling-head report

Experiment date: 2026-08-23. Attempt: `20260823T071624Z` on repository
commit `3879345f75201497f677b319a97f1e7435a4d07b` (`main`). This is a
scratch research run, not a production source change.

## Decision

Hold **mean pooling** in the locked 10/5 close-only config. Do not switch
the forecast head to flatten, and do not promote either artifact.

On the locked confirmatory 2026 path, mean beat flatten on every headline
metric: rank IC `0.04054` vs `0.03442`, top-15 excess `1.782%` vs `1.537%`
per week, top-minus-bottom spread `2.019%` vs `1.776%`, MAE `4.558%` vs
`4.585%`, and top-15 turnover `23.23%` vs `38.71%`. All three seeds were
positive for both heads; mean was higher on every seed. The only paired
interval that excluded zero was turnover: mean replaced fewer names.

The rank-IC, excess, and spread advantages did **not** exclude zero under
the four-week paired bootstrap. Flatten is therefore not selected, but
mean is not a statistically isolated winner on ranking skill either. Both
heads still fail research clearance: negative rank IC in 2024 and 2025,
and confirmatory intervals versus causal mean and ridge include zero.

Nothing was promoted, no production `current` pointer or cache was
touched, no trade was submitted, and no Temporal workflow was triggered.

## Frozen pooling-only design

Geometry was held at the geometry-sweep winner: patch length 10, stride 5,
11 unpadded tokens, one adjusted-close log-return channel. Only the
Hugging Face forecast-head pooling changed:

| Arm | `pooling_type` | Head width | Parameters |
|---|---|---:|---:|
| mean | `"mean"` | `d_model` (64) | 68,677 |
| flatten | `None` (HF flatten) | `11 * d_model` (704) | 71,877 |

Flatten is the official Transformers 4.57.3 flatten head, not a custom
layer. The extra 3,200 parameters are the capacity difference being
tested.

Every other setting matched the geometry sweep: context 60, horizon 5,
two encoder layers, `d_model=64`, four heads, FFN 128, shared
embedding/projection, channel attention off, batch norm, pre-norm, GELU,
fixed sin/cos positions, no CLS token, HF standard scaling, attention and
positional dropout 0.20, all other dropout zero, daily close MSE, Adam at
`3e-4`, zero weight decay, batch 256, gradient clip 1.0, plateau
scheduler, 60-epoch cap, rank-IC checkpoint selection with
validation-MSE tie-break, patience 8, and seeds `20260823`, `20260824`,
`20260825`.

## Data, folds, and label lock

The runner reused the geometry-sweep adjusted-price cache by symlink
(identical request, dates, and file hashes).

- Universe: all 430 current `halal_new` symbols from the production-built
  August cache, fetched `2026-08-08T14:37:54.382306+00:00`; universe
  SHA-256
  `e2ed929ce96e21e61b46b135a36b5d6cd36874e016d38fccf6fdbbc31c2308b2`.
- Prices: yfinance 1.0, `auto_adjust=True`, requested `2015-01-01` through
  `2026-08-22` exclusive. All 430/430 symbols present; observed files span
  `2015-01-02` through `2026-08-21`. Shortest history 61 rows, longest
  2,926. Adjusted-price manifest SHA-256:
  `6b6801467e3f6fd8b4c739c1feb778b255dd9deae26facb526bb486e0c6ed4ed`.
- Development 2024: 21,536 evaluation rows over 51 weeks, 420–424
  symbols/week (median 423).
- Development 2025: 21,681 evaluation rows over 51 weeks, 424–426
  symbols/week (median 425).
- Confirmatory 2026: 13,688 evaluation rows over 32 weeks,
  `2026-01-12..2026-08-17`, 426–429 symbols/week (median 428).
- Missing exact-session exclusions were 16,843, 17,114, and 17,207.
  There were no finite-positive-close exclusions.
- The confirmatory panel withheld target values while all six 2026
  checkpoints were trained. Unlock wrote `confirmatory_unlock.json` at
  `2026-08-23T07:47:38.279386+00:00` (6 verified hashes) before 2026
  evaluation targets were read.

**Honesty about 2026.** The mean 10/5 confirmatory numbers are not new.
They match the geometry-sweep 10/5 ensemble and all three seeds to the
last reported digit, as expected from a deterministic retraining of the
same mean-pooled 10/5 contract on the same panel. 2026 is confirmatory
for the *pooling* hypothesis because flatten was not selected on 2026;
it is not an independent re-test of mean 10/5 itself. Development 2024
and 2025 remain previously opened evidence.

## Ensemble results

Returns, excesses, and spreads are arithmetic weekly averages. “Turnover”
is the fraction of top-15 names replaced from one decision week to the
next.

### Development 2024

| Model | MAE | Direction | Rank IC | Top-15 excess | Top15-bottom15 | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST mean | 3.436% | 50.93% | -0.01121 | 0.411% | 0.257% | 29.07% |
| PatchTST flatten | 3.445% | 50.58% | -0.00097 | 0.448% | 0.513% | 49.20% |
| Causal historical mean | 3.248% | 51.94% | 0.02165 | 0.855% | 0.547% | 5.33% |
| Close-only ridge | 3.249% | 51.96% | 0.01718 | 0.538% | 0.499% | 75.87% |

### Development 2025

| Model | MAE | Direction | Rank IC | Top-15 excess | Top15-bottom15 | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| PatchTST mean | 3.764% | 49.64% | -0.00079 | 0.465% | 0.502% | 33.47% |
| PatchTST flatten | 3.824% | 49.32% | -0.01437 | 0.128% | -0.055% | 52.00% |
| Causal historical mean | 3.588% | 53.15% | 0.00218 | 0.101% | -0.004% | 5.73% |
| Close-only ridge | 3.574% | 53.66% | 0.02281 | 0.481% | 0.619% | 73.20% |

### Confirmatory 2026 (new for flatten; previously seen for mean)

| Model | MAE | Direction | Rank IC | Top-15 excess | Top15-bottom15 | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| **PatchTST mean** | **4.558%** | **51.62%** | **0.04054** | **1.782%** | **2.019%** | **23.23%** |
| PatchTST flatten | 4.585% | 51.56% | 0.03442 | 1.537% | 1.776% | 38.71% |
| Causal historical mean | 4.390% | 51.94% | 0.00756 | 0.998% | 1.116% | 6.67% |
| Close-only ridge | 4.398% | 51.62% | 0.00632 | 0.941% | 0.456% | 81.08% |

Confirmatory rank IC was positive for every individual seed:

| Pooling | Seed 20260823 | Seed 20260824 | Seed 20260825 |
|---|---:|---:|---:|
| mean | 0.03480 | 0.04113 | 0.04155 |
| flatten | 0.03063 | 0.03488 | 0.03746 |

Flatten did not win any confirmatory seed. In development, flatten’s only
point-estimate win was 2024 rank IC (less negative); it was worse in 2025
on rank IC, excess, spread, MAE, and turnover, and never beat the causal
gates.

Flatten checkpoints also stopped earlier on the confirmatory fold (best
epochs 1–2 vs mean 3, 3, and 12), consistent with a wider head that did
not buy extra useful validation rank IC.

## Paired four-week block uncertainty

All intervals use 2,000 deterministic moving-block repetitions by
decision week. Stored comparisons are `mean_minus_flatten`.

On confirmatory 2026, mean minus flatten was:

- Rank IC: `+0.00612`, 95% interval `[-0.00298, +0.01545]`.
- Top-15 excess: `+0.245` percentage points/week, interval
  `[-0.886, +0.994]`.
- Top15-bottom15 spread: `+0.243` points/week, interval
  `[-1.154, +1.132]`.
- MAE: `-0.027` points (mean better), interval `[-0.067, +0.013]`.
- Turnover: `-15.48` percentage points, interval `[-19.38, -11.67]`.

On pooled development, mean minus flatten rank IC was `+0.00167` with
interval `[-0.01169, +0.01334]`. MAE favored mean with interval wholly
below zero (`[-0.00059, -0.00010]`). Turnover again favored mean
(`-19.33` points, interval `[-22.55, -15.97]`).

Against mandatory gates, confirmatory mean had the same encouraging point
estimates and insufficient uncertainty already reported in the geometry
sweep:

- Versus causal mean: rank-IC delta `+0.03298`, interval
  `[-0.03591, +0.09411]`; MAE worse by `+0.168` points with interval
  `[+0.065, +0.291]`.
- Versus ridge: rank-IC delta `+0.03422`, interval
  `[-0.08295, +0.14887]`; MAE worse by `+0.160` points with interval
  `[+0.046, +0.292]`.

Flatten’s confirmatory deltas versus the same gates were smaller and
still included zero on rank IC, with worse MAE intervals wholly above
zero.

## Hardware, serialization, and runtime

- Hardware: MacBook Pro, Apple M5 Pro, 20-core GPU, 48 GB unified memory.
- Runtime: Python 3.12.13, PyTorch 2.9.1, Transformers 4.57.3, Apple MPS.
- Deterministic smoke: exact repeated state hash
  `c66f6900263dcc7f70b47e832bc58ae97b2fff78ac4dea86b047a28c1d11936f`
  and prediction hash
  `5268bddeb2a168bdf8040029fdd2b0cb3de8efc88c3563c2601e1c51bdb2a269`.
- 18 declared training jobs completed. The audit ledger recorded 24 model
  lifecycles—18 training/evaluation lifecycles plus six post-unlock
  confirmatory checkpoint reloads—with `max_active_models=1`.
- End-to-end runtime: 1,915.59 seconds (31m 55.6s). Recorded model
  training time: 1,784.90 seconds (29m 44.9s).

## Integrity identifiers

All four result artifacts and all 18 checkpoint weight hashes were
written into `results/manifest.json`. Mean 10/5 confirmatory predictions
replicated the geometry-sweep 10/5 numbers exactly, which is the expected
deterministic-retraining check rather than new evidence for mean.

| Artifact | SHA-256 |
|---|---|
| `results/predictions.csv` | `850d40de815ef74c29a9e8a9af0af62229fab37f9ee316049ea490ccbb74cc61` |
| `results/weekly_metrics.csv` | `8894f724aca001e8e48690e787c015b02d9566b2018065509290ce2a52079598` |
| `results/confirmatory_unlock.json` | `ad00dcc81c25ee97ce3578b2fe7b937f9d4eecb4c1b91bb561b28cbf86ec2304` |
| `results/mps_smoke.json` | `e5c363a78116dcc7c90c33196f7f4b36ebf8fc71ed5346f54a3ca17da0b60de2` |

## Interpretation and locked config

This clean sweep answers the narrow pooling question: with 10/5 and
close-only held fixed, flatten does not improve the research model.
Mean pooling stays in the locked best-config stack:

- Geometry: 10/5 (11 tokens)
- Input: 1 close-return channel
- Channel attention: off
- Pooling: **mean**
- Objective: denormalized close MSE
- `weight_decay=0`
- Compact arch, Adam `3e-4`, batch 256, clip 1.0, attention and
  positional dropout 0.20
- Checkpoint on max validation weekly rank IC, MSE tie-break

Flatten’s extra head capacity bought higher turnover and no confirmatory
ranking gain. That is a config finding, not a promotion finding. Causal
gates still win calibration; research clearance remains failed for both
heads.

Do not reopen 2026 as a selection set. The next unused config factor, if
research continues, should be isolated the same way and confirmed on a
new untouched period. Production source, `current` pointers, and
promotion remain out of scope until that is a separate decision.
