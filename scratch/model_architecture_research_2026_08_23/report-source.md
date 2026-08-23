# PatchTSMixer and Time Series Transformer assessment

Audience: LearnFinance-2025 maintainers

Date: August 23, 2026

Scope: Compare Hugging Face PatchTSMixer and Time Series Transformer with the repository's PatchTST forecaster for five-session equity-return ranking. This is architectural research, not an implementation or promotion decision.

## Direct answer

Keep PatchTST as the production baseline. PatchTSMixer is a credible research challenger because it preserves the patched, direct-forecast contract while replacing Transformer attention with lighter MLP mixing. Evidence does not establish that it is more accurate for equity returns. Time Series Transformer is a weaker fit because it changes the task to autoregressive probabilistic forecasting and requires a materially different data and inference contract.

## Consequential findings

1. The repository's PatchTST uses 60 daily observations, five OHLCV log-return channels and a direct five-session horizon. It trains only on close-return MSE and compounds the five predicted close log returns into the scalar weekly score.
2. Hugging Face PatchTST defaults to channel-independent processing when `channel_attention=False`. With a close-only loss, the non-close channels do not influence the close forecast. The repository's own sensitivity experiment confirms zero forecast change and zero non-close input gradient for the independent five-channel arm.
3. PatchTSMixer is a sibling of PatchTST, not an extension. It keeps patchification and channel-independent or explicit channel-mixing modes but replaces full patch self-attention with residual MLP mixing, optional gated attention and optional small self-attention.
4. The TSMixer paper reports modest aggregate accuracy gains over PatchTST and materially lower compute, but its evidence is from long-horizon electricity, traffic, weather and ETT benchmarks. The best paper variants also use reconciliation heads that Hugging Face v4.37 does not expose. The evidence does not cover finance, five-session horizons, cross-sectional rank IC or portfolio outcomes.
5. Time Series Transformer is a vanilla encoder-decoder Transformer with Student-t/Normal/negative-binomial emission heads, NLL training and autoregressive sampling. It needs lags, past/future time features, observed masks and any future dynamic covariates to be known at prediction time.
6. Local locked evidence is more relevant than generic benchmark claims. The 2024-2025 full-universe audit found negative rank IC for both PatchTST variants, while causal mean and ridge controls were positive. Current US and India PatchTST validation MSE also exceed their mean baselines, yet the artifact-health policy promotes any finite positive losses plus complete files. Architecture alone is therefore not the demonstrated bottleneck.

## Assumptions and limitations

- The decision target is the existing scalar weekly-return score used by Alpha-HRP and SAC, not a new uncertainty-aware product contract.
- No published primary-source benchmark was found for these models on OHLCV equity returns with cross-sectional ranking objectives.
- Parameter counts were measured locally under Transformers 4.57.3 for representative, not tuned, configurations. They indicate capacity, not accuracy or end-to-end runtime.
- The active artifacts still use patch 16/stride 8 while source defaults now specify patch 10/stride 5. A locked geometry sweep is currently running; it should finish before introducing another architecture axis.
- The Hugging Face v4.37.0 PatchTSMixer and Time Series Transformer documentation pages intermittently returned a cache miss through the research tool. Exact v4.37.0 tagged source and the materially identical v4.37.2 PatchTSMixer API were used to verify the contract.

## Claim-to-source ledger

| Claim family | Source | Publisher or author | Date | URL or local evidence | Access notes |
|---|---|---|---|---|---|
| PatchTST patching and channel independence | A Time Series is Worth 64 Words | Nie, Nguyen, Sinthong and Kalagnanam | 2023 | https://arxiv.org/abs/2211.14730 | ICLR 2023 primary paper |
| Hugging Face PatchTST v4.37 API | PatchTST | Hugging Face | Version-pinned | https://huggingface.co/docs/transformers/v4.37.0/model_doc/patchtst | Official documentation |
| PatchTSMixer architecture and API | PatchTSMixer | Hugging Face | Version-pinned | https://huggingface.co/docs/transformers/v4.37.0/model_doc/patchtsmixer | Official documentation; tagged source also checked |
| TSMixer benchmark and compute claims | TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting | Ekambaram, Jati, Nguyen, Sinthong and Kalagnanam | 2023 | https://arxiv.org/abs/2306.09364 | KDD 2023 primary paper |
| Time Series Transformer contract | Time Series Transformer | Hugging Face | Version-pinned | https://huggingface.co/docs/transformers/v4.37.0/model_doc/time_series_transformer | Official documentation and tagged source |
| Probabilistic generation workflow | Probabilistic Time Series Forecasting with Transformers | Rogge and Rasul, Hugging Face | 2022 | https://huggingface.co/blog/time-series-transformers | First-party technical tutorial |
| Vanilla Transformer basis | Attention Is All You Need | Vaswani et al. | 2017 | https://papers.neurips.cc/paper/7181-attention-is-all-you-need | NeurIPS primary paper |
| Repository PatchTST contract | `config.py`, `training.py`, `inference.py` | LearnFinance-2025 | Current workspace | `/Users/razin/personal/LearnFinance-2025/brain_api/brain_api/core/patchtst/` | Direct code inspection |
| Channel sensitivity | `selection.json` | LearnFinance-2025 research artifact | August 22, 2026 | `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_corrected_experiments_2026_08_23/results/selection.json` | Independent arm: zero non-close gradient and forecast delta |
| Locked downstream evidence | Full-universe US PatchTST experiment report | LearnFinance-2025 research artifact | August 23, 2026 | `/Users/razin/personal/LearnFinance-2025/scratch/patchtst_full_universe_audit_2026_08_23/REPORT.md` | Locked 2024-2025 labels, three-seed ensembles |
| Active artifact metrics | US and India `metadata.json` | LearnFinance-2025 | August 15, 2026 | `/Users/razin/personal/LearnFinance-2025/brain_api/data/models/patchtst_*` | Both active validation losses exceed stored baselines |

## Searches performed and stop reason

Research covered exact v4.37 Hugging Face documentation and tagged source, the PatchTST and TSMixer primary papers, the canonical Transformer paper, the Hugging Face probabilistic forecasting tutorial and direct repository code/artifacts. Targeted follow-up checked benchmark domains, implementation gaps, channel mixing, loss/output semantics, runtime version drift, active metrics and locked local experiments. Research stopped because the architecture/API claims have first-party support, the benchmark-applicability gap is explicit and further generic forecasting evidence would not resolve the finance-specific comparison without a matched local experiment.

