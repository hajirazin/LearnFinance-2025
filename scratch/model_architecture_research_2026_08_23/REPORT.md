# PatchTSMixer, PatchTST and Time Series Transformer

Decision brief for LearnFinance-2025

August 23, 2026

## Recommendation

Keep PatchTST as the production baseline. Do not replace it with either model from documentation or published benchmark claims alone.

PatchTSMixer is the better challenger. It fits the current direct five-session forecasting contract, is smaller and can test whether MLP mixing generalizes better than attention on noisy returns. The evidence supports an experiment, not a migration.

Time Series Transformer is not the preferred challenger. Its value is native predictive uncertainty, but it changes the objective, input contract and inference semantics. Test it only if the portfolio logic will consume calibrated uncertainty rather than another scalar point estimate.

## What PatchTSMixer is

PatchTSMixer is the Hugging Face implementation of the TSMixer family. Like PatchTST, it splits a context window into patches and learns shared representations. The main difference is how it combines those patches:

- PatchTST uses Transformer self-attention across patch tokens.
- PatchTSMixer uses residual MLPs to mix information across patches and hidden features. It can optionally mix channels, apply gated attention or add a small self-attention layer.

The default `common_channel` mode mirrors PatchTST's channel-independent design. `mix_channel` explicitly combines channels and is the relevant mode if open, high, low and volume should influence the close forecast. Hugging Face also exposes `prediction_channel_indices`, so a model can consume all channels while predicting only the close channel. See the [Hugging Face PatchTSMixer v4.37 documentation](https://huggingface.co/docs/transformers/v4.37.0/model_doc/patchtsmixer) and the [KDD 2023 TSMixer paper](https://arxiv.org/abs/2306.09364).

It is a sibling architecture, not a newer PatchTST version or a layer placed on top of PatchTST.

## How it relates to the PatchTST in this repository

The current source contract is specific: 60 daily OHLCV log-return observations, a direct five-session forecast, RevIN-style standard scaling and a close-return-only MSE objective. Inference makes one forward pass, extracts five predicted daily close log returns and compounds them into the weekly-return score used by screening and SAC. The relevant implementation is [config.py](/Users/razin/personal/LearnFinance-2025/brain_api/brain_api/core/patchtst/config.py:10), [training.py](/Users/razin/personal/LearnFinance-2025/brain_api/brain_api/core/patchtst/training.py:1) and [inference.py](/Users/razin/personal/LearnFinance-2025/brain_api/brain_api/core/patchtst/inference.py:211).

There is a subtle but important consequence. Hugging Face PatchTST is channel-independent unless `channel_attention=True`. This repository does not enable channel attention and trains only the close output. Therefore the close forecast cannot use open, high, low or volume, even though five channels are passed to the model. The repository's own sensitivity study confirms this: perturbing non-close inputs produced zero forecast change and the non-close input gradient was zero for all three seeds. See [selection.json](/Users/razin/personal/LearnFinance-2025/scratch/patchtst_corrected_experiments_2026_08_23/results/selection.json:26).

This means the current PatchTST is operationally a shared-weight close-return forecaster, not an OHLCV interaction model. PatchTSMixer `common_channel` would preserve that behavior. PatchTSMixer `mix_channel` would test a genuinely different hypothesis: whether within-symbol OHLCV interactions help predict close returns.

## Comparison for this use case

| Property | Current PatchTST | PatchTSMixer | Time Series Transformer |
|---|---|---|---|
| Core mechanism | Patch self-attention | Patch and feature MLP mixing | Vanilla encoder-decoder attention |
| Forecast style | Direct five-step point forecast | Direct five-step point forecast; optional distribution head | Autoregressive probabilistic forecast |
| Current pipeline fit | Exact | High | Low to medium |
| OHLCV interaction | None in current configuration | Optional with `mix_channel` | Shared representation, diagonal output distributions in the documented implementation |
| Known future features | Not required | Not required | Future time/dynamic covariates required when configured and must be known at inference |
| Inference | One forward pass | One forward pass | Sequential sampling across the horizon |
| Native uncertainty | Optional distribution support in HF, not used here | Optional | Primary design goal |
| Best reason to test | Production baseline | Lower-capacity backbone and explicit channel mixing | Calibrated forecast distributions |
| Main risk | Current local evidence shows weak ranking signal | Published gains may not transfer; can lose content-adaptive attention | Objective mismatch, sampling latency and larger integration change |

Representative parameter counts, measured locally with Transformers 4.57.3 and a 60/5 context/horizon, are 68,677 for PatchTST, 47,511 for PatchTSMixer `common_channel`, 48,057 for `mix_channel` and 181,711 for a five-target Time Series Transformer. These are configuration-dependent capacity estimates, not performance benchmarks.

The compute case for PatchTSMixer is less decisive here than in the paper. With patch 10/stride 5, a 60-session context yields only 11 patch tokens; the active 16/8 artifacts yield six. Attention's quadratic term is small at those lengths. Mixer may still regularize better because it is smaller, but runtime savings are unlikely to be the main decision criterion.

## What the external evidence does and does not show

The [PatchTST paper](https://arxiv.org/abs/2211.14730) introduced patching and channel-independent shared Transformer weights for long-horizon forecasting. The [TSMixer paper](https://arxiv.org/abs/2306.09364) reports roughly 1-2% aggregate accuracy improvements over patch-based Transformers and substantial compute reductions.

Those headline results are not direct evidence for this portfolio system:

- The benchmarks are ETT, electricity, traffic and weather series, with long contexts and horizons commonly ranging from 96 to 720 steps.
- They do not measure equity returns, cross-sectional rank IC, top-15 stability, turnover or portfolio outcomes.
- PatchTSMixer is not uniformly better at every dataset and horizon.
- The paper's strongest variants use hierarchical and cross-channel reconciliation heads. Hugging Face v4.37 does not expose those heads, so the paper's best result cannot be assigned automatically to `PatchTSMixerForPrediction`.
- A five-session horizon gives much less room for hierarchical multi-horizon reconciliation to help.

The strongest evidence is local. A locked 2024-2025 full-universe study found weekly rank IC of -0.01057 for the 10/5 PatchTST ensemble and -0.01062 for the corrected 16/8 ensemble. A causal historical mean scored +0.01368 and ridge scored +0.01895, both with lower point error. See the [full-universe audit](/Users/razin/personal/LearnFinance-2025/scratch/patchtst_full_universe_audit_2026_08_23/REPORT.md:73).

The active artifacts tell the same cautionary story. US validation MSE is 0.0006348 against a 0.0006201 mean baseline; India is 0.0005531 against 0.0005425. Both were promoted because the current health check requires finite positive losses and complete files, not improvement over the baseline. See the [promotion health check](/Users/razin/personal/LearnFinance-2025/brain_api/brain_api/core/training_utils.py:53), [US metadata](/Users/razin/personal/LearnFinance-2025/brain_api/data/models/patchtst_halal_new/v2026-08-14-e7b14b211a54/metadata.json:470) and [India metadata](/Users/razin/personal/LearnFinance-2025/brain_api/data/models/patchtst_nifty_shariah_500/v2026-08-14-e89b75d03a12/metadata.json:238).

The practical conclusion is not that Mixer will fail. It is that architecture is not yet the demonstrated bottleneck. Objective alignment, validation selection and genuine out-of-sample rank signal matter more than benchmark MSE.

## Why Time Series Transformer is a poorer fit

Hugging Face's [Time Series Transformer](https://huggingface.co/docs/transformers/v4.37.0/model_doc/time_series_transformer) is a vanilla encoder-decoder model adapted to probabilistic forecasting. It learns Student-t, Normal or negative-binomial parameters with negative log-likelihood. At inference, `generate()` samples the next value, feeds that sample back into the decoder and repeats. The [Hugging Face forecasting guide](https://huggingface.co/blog/time-series-transformers) describes this workflow.

Adopting it would change several contracts:

- Direct close-only MSE becomes distributional NLL unless a custom head or loss is written.
- One-pass five-day prediction becomes autoregressive sampling, adding error propagation and latency.
- The data pipeline must create domain-appropriate lag sequences, observed masks and past/future time features.
- Unknown future OHLCV cannot be supplied as future dynamic covariates. A clean configuration would probably forecast close only and use calendar features, which discards the claimed OHLCV advantage.
- The scalar weekly score would need a documented reduction from a distribution, such as median return, downside probability or risk-adjusted expected return.

The model becomes attractive only if the downstream system will use uncertainty. For example, Alpha-HRP could reject names with high probability of a negative weekly return or SAC could consume forecast dispersion. That would require calibration tests such as CRPS, interval coverage and reliability, not only MSE and rank IC.

## Proposed research sequence

Do not add a Mixer arm to the geometry experiment now running in [the scratch study](/Users/razin/personal/LearnFinance-2025/scratch/patchtst_geometry_sweep_2026_08_23/PLAN.md:1). It deliberately isolates patch 8/4, 10/5 and 16/8. Adding a backbone would break attribution.

After that experiment freezes its result, run one matched architecture study:

1. PatchTST close-only with the selected geometry, as the control.
2. PatchTSMixer `common_channel`, close-only, as the pure backbone swap.
3. PatchTSMixer `mix_channel` with all five OHLCV inputs and `prediction_channel_indices=[3]`, as the explicit OHLCV-interaction test.
4. Keep Time Series Transformer out of this first study. Add a close-only Student-t arm later only if an uncertainty-aware decision rule is predeclared.

Freeze the same universe snapshots, exact exchange sessions, expanding folds, embargoes, three seeds, search budget and target construction. Select checkpoints on validation rank IC rather than daily MSE alone. Make causal mean and ridge mandatory gates.

The primary promotion condition should be positive out-of-sample weekly rank IC across multiple folds and a credible paired improvement over both controls. Secondary evidence should include top-15 excess and spread, top-15 turnover and stability, direction accuracy and MAE. A probabilistic arm must also clear calibration and proper-scoring-rule gates.

If PatchTSMixer wins, introduce it as a separate model bucket and artifact type for auditability. Do not silently store Mixer weights under a PatchTST identity. Regenerate walk-forward forecaster snapshots and retrain both SAC buckets because changing the forecast generator changes an input feature distribution.

## Final decision

- PatchTST: keep as the baseline while the current locked experiments finish.
- PatchTSMixer: research next; it is plausible but not proven better.
- Time Series Transformer: do not prioritize unless calibrated uncertainty becomes a first-class portfolio input.

The immediate model-selection problem is proving stable rank signal against simple causal baselines. Changing the backbone before that gate is met would add complexity without resolving the evidence gap.
