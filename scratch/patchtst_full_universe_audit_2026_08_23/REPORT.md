# Full-universe US PatchTST experiment report

Experiment date: 2026-08-23. This report follows the setting-by-setting
[audit](AUDIT.md); the recommendation there is a defensible prior, not a result
chosen after seeing the test set.

## Decision

Reject the coherent 10/5 candidate for promotion. It did not beat the corrected
16/8 control or meaningful causal baselines on the locked 2024–2025 test period.
Both PatchTST ensembles had slightly negative rank IC, while the causal historical
mean and a five-feature ridge model had positive rank IC and lower point error.
Nothing was promoted, no `current` pointer was touched, and no production workflow
or trade was triggered.

## Predeclared configurations

Both arms used 60 adjusted daily OHLCV log-return observations to forecast five
daily outputs, pooled globally across symbols at weekly Monday decision anchors.
Common architecture/training settings were `d_model=64`, four heads, two layers,
FFN 128, channel independence, shared embeddings/projection, fixed sinusoidal
position encoding, no CLS token, HF per-series standard scaling, GELU, batch norm,
pre-norm, Adam at 3e-4, zero weight decay, batch 256, gradient clip 1.0, plateau
scheduler, 60-epoch cap, rank-IC checkpoint selection, patience 8, and seeds
20260823/20260824/20260825.

| Arm | Patches | Pool/head | Effective dropout | Training objective |
|---|---:|---|---|---|
| Corrected control 16/8 | 6, no accidental stride-1 overlap | mean pool | attention 0.20, positional 0.20; path/FFN/head 0 | close-channel daily MSE |
| Coherent candidate 10/5 | 11, 50% overlap | flatten all patch states | attention 0, positional/path/FFN 0.05; head 0 | equal-channel, train-scale-normalized OHLCV daily MSE |

The from-scratch India prior uses the same architecture because a ~200-name rather
than ~400-name universe changes the number of samples, not the 60-by-5 input
geometry. India-specific differences should be the full Nifty Shariah 500 universe,
unchanged `.NS` identifiers, exact XBOM sessions, and India-specific data-quality
checks. The failed US candidate is evidence against blindly adopting it in India;
India still needs its own locked evaluation.

## Data and protocol

- Universe: the complete current `halal_new` production cache, 430 unique symbols,
  fetched 2026-08-08T14:37:54Z from SPUS/SPTE/SPWO/HLAL/UMMA and already filtered
  for Alpaca tradability. Cache SHA-256:
  `e2ed929ce96e21e61b46b135a36b5d6cd36874e016d38fccf6fdbbc31c2308b2`.
- Prices: yfinance adjusted OHLCV, `auto_adjust=True`, requested 2015-01-01
  through 2026-01-05. 428/430 symbols downloaded. `FDXF` and `VGNT` had no Yahoo
  history after batch and individual retries.
- Eligibility was dynamic by symbol-week: every example required all exact XNYS
  context and target sessions and finite, positive raw OHLCV. There were 15,994
  missing-session exclusions and 4,723 nonpositive/nonfinite exclusions.
- Train: 2015-05-04 through 2022-12-19, 151,250 rows, 399 weeks, 415 unique
  symbols, 342–414 symbols/week (median 378).
- Validation: 2023-01-09 through 2023-12-18, 20,777 rows, 50 weeks, 418 unique
  symbols, 414–418/week (median 415).
- Locked test: 2024-01-08 through 2025-12-22, 43,512 rows, 103 weeks, 426 unique
  symbols, 419–425/week (median 422). Test labels were not loaded until all six
  validation-selected checkpoints were frozen.
- Uncertainty: paired 2,000-repetition moving-block bootstrap, four decision weeks
  per block. Top-15 returns are gross signal diagnostics, not sticky-selection,
  HRP, cost, or execution backtests.

## Hardware and execution

Training ran strictly one model/seed at a time on Apple MPS: MacBook Pro with an
Apple M5 Pro (18 CPU cores, 20 GPU cores), 48 GB unified memory, PyTorch 2.9.1,
and Transformers 4.57.3. PyTorch reported MPS available and a 40.20 GB recommended
working-set ceiling. A candidate batch of 256 used 2.6 MB of tensor allocation
(201.9 MB including Metal driver allocation). The six recorded training jobs took
583.82 seconds in total; the initial end-to-end run, including data-panel build,
scoring, and bootstrap, took 657.95 seconds (10m 58s). A later cached evaluation
rerun after the tie-handling fix took 74.21 seconds and did not retrain models.

## Locked test results

All return columns below are arithmetic weekly returns. Spread and excess values
are gross weekly averages.

| Model | MAE | Direction | Balanced direction | Weekly rank IC | Top-15 excess | Top15–bottom15 | Top-15 turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate 10/5, 3-seed ensemble | 3.624% | 49.51% | 48.78% | -0.01057 | 0.322% | 0.179% | 55.10% |
| Corrected control 16/8, ensemble | 3.612% | 49.61% | 48.97% | -0.01062 | 0.499% | 0.445% | 46.67% |
| Causal per-symbol historical mean | 3.408% | 52.51% | 49.98% | 0.01368 | 0.506% | 0.341% | 6.08% |
| Five-feature ridge | 3.401% | 52.79% | 50.59% | 0.01895 | 0.452% | 0.655% | 73.40% |
| One-week reversal | 4.894% | 50.22% | 50.45% | 0.01723 | 0.337% | 0.255% | 92.09% |
| Zero return | 3.412% | 46.71% | 50.00% | undefined | undefined | undefined | undefined |

Every candidate seed had negative test rank IC (-0.01059, -0.00972, -0.00835),
as did every candidate validation checkpoint. The control's three test rank ICs
were -0.00578, -0.01882, and -0.00681. The agreement across seeds is evidence that
the failure is not one unlucky initialization.

The candidate-minus-control rank-IC difference was only +0.00005 with a 95%
block-bootstrap interval of [-0.01319, +0.01294]. Candidate top-15 excess was
0.177 percentage points/week lower, interval [-0.487, +0.109], and its
top-minus-bottom spread was 0.266 points/week lower, interval [-0.795, +0.209].
The candidate's MAE was 0.012 percentage points/week higher, interval
[-0.014, +0.040]. None of those candidate/control differences is distinguishable
from zero on this path.

Against ridge, the candidate's MAE was worse by 0.223 percentage points/week,
95% interval [+0.172, +0.291]. Its rank IC was lower by 0.02952, interval
[-0.08252, +0.01438], and its spread was lower by 0.476 percentage points/week,
interval [-1.741, +0.654]. Thus the deep model clearly lost on point error and did
not establish a ranking advantage.

## Interpretation and precise next step

The stride correction is necessary, but it does not create signal. The candidate
bundle made the advertised OHLCV channels trainable through shared auxiliary
tasks, used a defensible 11-patch geometry, and aligned model selection to rank IC;
it still failed. That is evidence against promoting this bundle, not proof that
PatchTST can never work.

The next experiment should be a sequential expanding-window ablation with the
ridge model as a mandatory gate: corrected 16/8 close-only; 10/5 close-only with
mean pooling; 10/5 close-only with flatten pooling; and only then the 10/5
all-channel auxiliary objective. Use annual 2021–2025 test folds, the same three
seeds, exact-session eligibility, and paired block intervals. This isolates patch
geometry, pooling, and auxiliary-target effects that this coherent bundle cannot
attribute. Do not retrain or promote US or India production checkpoints until a
PatchTST arm has positive out-of-sample rank IC and credibly clears the causal
mean/ridge gates across more than one fold.
