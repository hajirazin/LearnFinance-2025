# PatchTST first-principles audit

Audit date: 2026-08-23. Library contract: Transformers 4.57.3, as locked in
`brain_api/uv.lock`. “Recommended” means a defensible prior for this application,
not an empirically proven optimum. The full-universe experiment tests the bundled
US recommendation against the corrected intended-production control; it is not a
hyperparameter sweep.

## Confirmed bugs and evaluation defects

1. **Confirmed and fixed before this audit — wrong Hugging Face stride keyword.**
   The domain adapter passed `stride=8`, but Transformers 4.57.3 defines
   `patch_stride`; unknown keywords are accepted by `PretrainedConfig`, so the
   model silently used the default stride 1. The old 60/16 model therefore had 45
   patches rather than 6. Training and inference now use `patch_stride`, and the
   60/10 default has 11 patches. The version-pinned constructor is visible in the
   [Transformers 4.57.3 configuration source](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/patchtst/configuration_patchtst.py).
2. **Confirmed and fixed in this audit — unused generic dropout keyword.** The
   Hugging Face PatchTST config has no model-level `dropout` field. Passing it only
   stored an unused attribute. The actual model used the explicitly supplied
   attention and positional dropout; FFN, residual-path, and head dropout remained
   zero. The unused keyword was removed, and training now calls the same adapter as
   inference. The regression test asserts full config equality and effective
   dropout sites.
3. **Confirmed, not changed in production — non-close OHLCV inputs cannot affect
   the close forecast under the present graph.** With `channel_attention=False`
   and close-channel-only loss, the independent open/high/low/volume branches are
   outside the close output's computational graph. The prior scratch sensitivity
   check measured exactly zero close-output change and zero non-close gradient.
   This is a semantic implementation defect relative to the advertised “OHLCV
   forecaster,” not proof that channel mixing will help. The experiment keeps
   channel independence but gives every channel a train-only scale-normalized
   auxiliary loss, matching PatchTST's global-univariate/shared-weight design.
4. **Confirmed, not changed in production — production training is not seeded.**
   Model initialization and shuffled batches vary across reruns even when the
   version ID is identical. The experiment uses three declared seeds and hashes
   every artifact. A production seed field should be added only after the research
   configuration is accepted, because it changes version identity.
5. **Confirmed, not changed in production — the persisted `StandardScaler` is fit
   on the whole dataset, including validation dates.** It is diagnostic only and
   is not consumed by training or inference, so it does not leak into predictions;
   any drift statistic derived from it is nevertheless validation-contaminated.
6. **Confirmed evaluation limitation — current-universe survivorship.** Historical
   samples use the halal ETF holdings and Alpaca-tradable roster fetched today,
   not point-in-time holdings. That is unsuitable for an unbiased historical
   strategy-return claim. CRSP explains why permanent identifiers and historical
   continuity matter for research data in its [research data overview](https://www.crsp.org/research/).
7. **Confirmed data-risk limitation — Yahoo provenance is not institutional
   point-in-time data.** `auto_adjust=True` adjusts OHLC for corporate actions, but
   volume remains a provider field and source revisions are possible. The ambiguity
   around `auto_adjust` is documented in [yfinance issue #687](https://github.com/ranaroussi/yfinance/issues/687).
8. **Confirmed and fixed in the research evaluator — constant forecasts were
   assigned arbitrary top-15 portfolios.** Zero-return and majority-sign baselines
   contain no cross-sectional ranking information, but deterministic symbol-order
   tie breaking made their top-15 metrics appear defined. Ranking, top-K, turnover,
   and paired-bootstrap results are now explicitly undefined for those baselines;
   regression tests also prevent all-NaN bootstrap warnings.

## Complete inventory

| Setting | Current production/effective value | Implicit HF 4.57.3 default | From-scratch US | From-scratch India | Rationale | Evidence strength | Expected risk |
|---|---|---|---|---|---|---|---|
| Universe | current `halal_new`, ~400 | N/A | same registered universe | full `nifty_shariah_500`, `.NS` intact | Preserve product and partition invariants; train the pooled forecaster broadly | High: repository contract | Current-roster survivorship biases historical economics |
| Price provider | yfinance | N/A | yfinance for this experiment, hash raw extracts | yfinance unless replaced consistently | Matches production and permits a controlled comparison | Medium | Revisions, delistings, rate limits, ticker identity |
| Adjustment | `auto_adjust=True` OHLCV download | N/A | explicit true; record provider/version | same | Split/dividend discontinuities in price levels are worse than adjusted returns | Medium | Volume is not adjusted like price; future corporate-action revisions can contaminate old bars |
| Provenance | logs only in production | N/A | manifest: request, fetch UTC, versions, rows, dates, hashes | same | Auditability and reproducibility | High | Re-download may differ despite identical code |
| Session calendar | per-symbol observed dates | N/A | exact XNYS sessions | exact XBOM sessions | Prevent missing bars from becoming multi-session “daily” returns | High | Some ADR/ETF calendars may not match perfectly |
| Missing bars | loader `dropna`; windows with nonfinite returns skipped | N/A | reject a symbol-week unless all exact context/target sessions exist | same | No interpolation or silent fallback | High | Dynamic cross-section; thin names excluded more often |
| Nonpositive OHLCV | return becomes NaN; affected windows skipped | N/A | same | same | Avoid `log(0)`/Inf and fabricated zeros | High: regression tests | Volume-zero names lose windows |
| Open transform | `log(Open_t/Open_{t-1})` | N/A | retain for comparability | same | Stationary but redundant with close; useful only as auxiliary shared task | Medium-low | Does not isolate overnight gap |
| High transform | `log(High_t/High_{t-1})` | N/A | retain | same | Stationary range-location proxy, but redundant | Medium-low | Microstructure/outlier sensitivity |
| Low transform | `log(Low_t/Low_{t-1})` | N/A | retain | same | Same argument as high | Medium-low | Microstructure/outlier sensitivity |
| Close transform | `log(Close_t/Close_{t-1})` | N/A | retain; primary output | same | Additive across five sessions and aligned with ranking score | High | Raw return predictability is weak and noisy |
| Volume transform | `log(Volume_t/Volume_{t-1})` | N/A | retain only as scale-normalized auxiliary task | same | May teach shared activity patterns without dominating raw MSE | Medium-low | Split/provider artifacts and fat tails |
| Cross-symbol pooling | one global panel; no symbol ID | N/A | retain | retain | More samples and regularization; learns common return dynamics | Medium | Large names/regimes dominate; no symbol-specific intercept |
| Symbol identity | none | N/A | none | none | Avoid memorizing current-roster identity and keep stateless arbitrary-symbol inference | Medium-high | Cannot model stable symbol heterogeneity |
| Same-date leakage | random sample batches after chronological split | N/A | keep whole decision dates in one split; never split a cross-section | same | A date's correlated labels must not straddle train/validation | High | Cross-sectional dependence reduces effective sample size |
| Sample anchor | last observed ISO-week session, min 3 days | N/A | Monday decision, context through prior XNYS session | Monday decision, prior XBOM session | Mirrors scheduled inference and gives one forecast per rebalance | High | Holiday weeks still have five-session targets spanning calendar weeks |
| Frequency | weekly samples | N/A | weekly | weekly | Matches ranking/Alpha-HRP decision cadence; avoids highly overlapping daily labels | High | Fewer independent time observations than row count suggests |
| Context length | 60 sessions | 32 | 60 | 60 | About one quarter; enough for 11 two-week patches without excessive stale history | Medium | Paper found longer context useful on nonfinancial benchmarks, but finance may decay faster |
| Horizon | five trading sessions | 24 | 5 | 5 | Exact downstream weekly-return contract | High | Not always calendar Mon–Fri in holiday weeks |
| Purge | conservative 7 calendar days before validation | N/A | interval-aware: no target session may reach next split; two-week embargo between blocks | same | Prevent overlapping labels and near-boundary contamination | High; finance literature supports purging overlapping outcomes | Reduces samples; embargo length remains a prior |
| Split design | one chronological 80/20 train/validation split | N/A | 2015–2022 train, 2023 validation, 2024–2025 locked test; dynamic eligible panel | same years when data permits, then India-specific walk-forward follow-up | Clear train/selection/test separation | High | Single regime path; not full walk-forward retraining |
| Walk-forward | yearly snapshots elsewhere; not model health evaluation | N/A | locked holdout now; expanding-window retrain is next experiment | same | This experiment answers configuration choice first | Medium | Static train fit understates model-refresh effects |
| Patch length | 10 at HEAD; old intended 16 | 1 | 10 | 10 | Two trading weeks per local token | Medium | Could blur fast reversal patterns |
| Patch stride | 5 corrected; old effective was 1 | 1 | 5 | 5 | One-week advance and 50% overlap | Medium-high | Overlap correlates adjacent tokens |
| Patch count | 11 for 60/10/5 | derived | 11 | 11 | Enough attention positions without the accidental 45-token redundancy | Medium | Small attention sequence may make Transformer unnecessary |
| End padding | none in HF implementation | none | none | none | 60/10/5 tiles exactly; no duplicated future edge | High | Different context lengths would discard oldest remainder |
| Scaling / RevIN | `scaling="std"` inherited | `"std"` | explicit `"std"` | same | Per-sample/channel normalization handles nonstationary scale; HF denormalizes output | High: paper and source | Near-zero variance requires epsilon handling |
| External scaler | diagnostic `StandardScaler`, unused by model | N/A | fit train only for diagnostics | same | Avoid contradictory double scaling and validation contamination | High | Stored legacy consumers may assume otherwise |
| Channel independence | `channel_attention=False` | false | false | false | Core PatchTST design; prior 12-stock mixing arm was worse, though inconclusive | Medium | At inference, non-close channels influence close only through shared training, not contemporaneously |
| Channel auxiliary learning | close-only raw MSE | HF `outputs.loss` is all-channel MSE | equal per-channel standardized daily MSE | same | Makes OHLCV branches train shared weights without volume dominance | Medium-low; candidate under test | Negative transfer from redundant/noisy channels |
| Shared patch embedding | true inherited | true | explicit true | true | Global-univariate inductive bias and parameter efficiency | High: PatchTST paper | Channel-specific transforms may deserve separate embeddings |
| Channel attention | false | false | false | false | Preserve paper design and prior negative diagnostic | Medium | No within-sample cross-channel fusion |
| Positional encoding | fixed sinusoidal inherited | `"sincos"` | explicit fixed sinusoidal | same | Stable prior for only 11 positions; avoids learning arbitrary positions | Medium | Original code often used learnable encodings |
| Positional dropout | 0.2 explicit | 0 | 0.05 | 0.05 | 0.2 is strong for a short 11-token sequence; 0.05 matches official-code scale | Medium-low | Less regularization may overfit |
| CLS token | false explicit | false | false | false | Forecasting benefits from all patch states; no evidence for a learned summary token | Medium-high | None material |
| Pooling/head input | mean pooling | mean | `None` (flatten all patches) | same | Official supervised code uses a flatten head; preserves recency/location information | Medium-high | More head parameters and overfit risk |
| Shared output projection | true inherited | true | explicit true | true | Same return mapping across OHLCV auxiliary series; compact model | Medium | Volume dynamics may not share output map with price returns |
| `d_model` | 64 | 128 | 64 | 64 | Weak-signal, short-context task does not justify larger capacity | Medium | Underfitting possible |
| Attention heads | 4 | 4 | 4 (16 dims/head) | 4 | Divides 64 cleanly and offers multiple temporal relations | Medium | Multiple heads may be redundant with 11 tokens |
| Encoder layers | 2 | 3 | 2 | 2 | Small receptive graph; deeper capacity raises variance | Medium | Underfitting nonlinear patterns |
| FFN size | 128 | 512 | 128 (2× model width) | 128 | Compact standard expansion for weak signal | Medium | Smaller than original-code experiments |
| Norm type | batch norm inherited | batchnorm | explicit batchnorm | batchnorm | Matches PatchTST paper/code and prior artifacts | Medium-high | Running stats depend on pooled training distribution |
| Norm epsilon | `1e-5` inherited | `1e-5` | explicit `1e-5` | same | Standard numerical safeguard | High | Negligible |
| Pre-norm | true inherited | true | true | true | Stable optimization in Transformers/HF implementation | Medium-high | Differs from some original-code defaults |
| Activation | GELU inherited | GELU | GELU | GELU | Smooth standard Transformer nonlinearity | Medium-high | No finance-specific evidence |
| Bias | true inherited | true | true | true | Standard small-network choice | Medium-high | Minor extra parameters |
| Attention dropout | 0.2 explicit | 0 | 0 | 0 | Official PatchTST code defaults to zero; 0.2 can erase scarce token relations | Medium | Less regularization |
| Residual/path dropout | 0 inherited | 0 | 0.05 | 0.05 | Light regularization at residual branches | Low-medium | Candidate bundle change lacks isolated attribution |
| FFN dropout | 0 inherited | 0 | 0.05 | 0.05 | Light regularization in highest-parameter sublayer | Low-medium | Same |
| Head dropout | 0 inherited | 0 | 0 | 0 | Do not randomly erase the already compact forecast representation | Medium | Flatten head may overfit |
| Init std | 0.02 inherited | 0.02 | 0.02 | 0.02 | HF-tested initialization | High | None material |
| Masking/pretraining | disabled (`do_mask_input=None`) | disabled | disabled | disabled | This is supervised comparison; adding pretraining is a separate experiment | High | Forgoes possible representation gains reported by paper |
| Distribution head | MSE point forecast | Student-t only matters with NLL | point/MSE head | same | Rank screen consumes one deterministic score | High | No predictive uncertainty output |
| Forecast head | five daily outputs for all channels | 24 steps | five daily outputs; compound close logs | same | Preserves API and additive return semantics | High | Daily path is more degrees of freedom than weekly scalar need |
| Reported return | `expm1(sum(close log returns))*100` | N/A | same | same | Exact compounding identity | High | Extreme forecasts need finite-value checks |
| Training objective | close daily MSE | all-channel MSE if labels supplied | train-scaled all-channel daily MSE; checkpoint selected on rank IC | same | Stable point loss plus aligned model selection; avoids failed ListNet magnitude behavior | Medium | Auxiliary tasks can hurt primary close ranking |
| Optimizer | `torch.optim.Adam` | N/A | Adam, betas .9/.999, eps 1e-8 | same | Preserve demonstrated zero-decay behavior | Medium-high | AdamW remains untested |
| Learning rate | `3e-4` | N/A | `3e-4` | `3e-4` | Existing stable value for compact model | Medium | Could be high after loss rescaling |
| Weight decay | 0 | N/A | 0 | 0 | Repo overfit-batch evidence found `>=1e-5` dominated tiny return gradients | Medium-high local evidence | No explicit parameter shrinkage |
| Batch size | 256 | N/A | 256 | 256 | Efficient pooled panel batches; same optimization noise across markets | Medium | Batch composition spans dates/symbols |
| Epoch cap | 100 | N/A | 60 in experiment; retain 100 production cap | same | Early stopping, not cap, should decide | Medium | Runtime; late overfit |
| Scheduler | ReduceLROnPlateau factor .5, patience 5 | N/A | same, driven by validation scaled MSE | same | Conservative adaptive reduction | Medium | Rank IC can peak before MSE |
| Early stopping | patience 15 on validation MSE | N/A | patience 8 on validation rank IC, MSE tie-break | same | Align selection with Alpha-HRP ranking and limit experiment cost | Medium | Noisy IC may stop prematurely |
| Gradient clip | 1.0 | N/A | 1.0 | 1.0 | Stable guardrail | Medium-high | Can mask badly scaled loss if always active |
| Seeds | unset | N/A | 20260823, 20260824, 20260825 | same minimum protocol | Quantifies initialization variance | High | Three seeds still give wide uncertainty |
| Primary selection metric | minimum validation MSE | N/A | mean weekly Spearman rank IC | same | Directly matches cross-sectional ranking use | High | IC ignores calibration/magnitude |
| Loss metrics | close MSE and mean-return baseline | N/A | daily/weekly MSE, MAE, zero and causal mean baselines | same | Calibration and error sanity checks | High | Low loss need not produce useful ranking |
| Direction metrics | basic direction in inference | N/A | accuracy, balanced accuracy, positive prevalence | same | Avoid majority-up baseline illusion | High | Ignores return magnitude |
| Ranking metrics | absent from production health | N/A | weekly rank IC, top-15 excess, top15–bottom15 spread | same | Matches rank-band selector and K=15 | High | Gross spreads omit costs and HRP weights |
| Stability metrics | absent | N/A | top-15 overlap, replacements, score-rank turnover | same | Sticky selection exists specifically because unstable ranks are costly | High | Does not simulate full sticky/HRP portfolio |
| Baselines | validation per-horizon mean | N/A | zero, causal mean, majority sign, 1w persistence/reversal, 4w momentum, ridge | same | Deep model must beat simple causal alternatives | High | Ridge features are deliberately limited |
| Statistical uncertainty | absent | N/A | paired four-week moving-block bootstrap by decision week | same | Respects serial and cross-sectional dependence better than row bootstrap | Medium-high | Block length and one historical path remain assumptions |
| US vs India architecture | identical default | N/A | values above | same values | Universe count changes sample volume, not the 60×5 observation geometry | Medium-high | India microstructure/regimes may later justify separate tuning |
| US vs India calendar/data | XNYS intended; current training inferred from bars | N/A | XNYS exact sessions | XBOM exact sessions; `.NS` preserved | This is the genuine market-specific difference | High | Yahoo India data quality and holidays differ |

## Evidence basis

The [PatchTST paper](https://arxiv.org/abs/2211.14730) establishes patching and
channel independence, and reports the benefit of longer receptive fields on its
benchmark datasets; it does not establish optimal settings for stock returns. The
[official supervised implementation](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py)
uses `unfold(..., step=stride)`, RevIN, shared channel-independent weights, and a
flatten forecast head. Its command-line defaults include patch 16, stride 8, GELU,
dropout 0.05, and MSE, but those values were chosen for long-horizon benchmark
datasets, not this 60-to-5 financial task. The exact inherited behavior above is
from the [Transformers 4.57.3 config](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/patchtst/configuration_patchtst.py)
and [model source](https://raw.githubusercontent.com/huggingface/transformers/v4.57.3/src/transformers/models/patchtst/modeling_patchtst.py).

For evaluation, [Gu, Kelly, and Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009)
evaluate stock-level predictions out of sample and adapt comparisons to the
cross-section rather than treating every stock row as independent. The need to
guard overlapping financial labels with purge/embargo logic is discussed in this
[peer-reviewed financial time-series study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9521884/).
The risk of selecting apparent winners from repeated backtests is the subject of
[Bailey et al.'s Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb).
These sources support the protocol, not a claim that the candidate will work.

Community reports were treated as warnings, not authority. In particular, a
[Hugging Face issue on PatchTST embeddings](https://github.com/huggingface/transformers/issues/29214)
confirms the channel-by-patch hidden-state layout, and the open
[official-repository channel-mixing question](https://github.com/yuqinie98/PatchTST/issues/139)
shows that disabling channel independence is not an established canonical recipe.
