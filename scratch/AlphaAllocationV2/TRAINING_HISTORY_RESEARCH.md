# Training history for the weekly AlphaAllocationV2 screen

Keep the full 2016–2025 archive. Start the pooled screen with five years of labeled training data and compare it with a two-year rolling window and an expanding window. Five years is a starting choice for the experiment. The best window for the Friday top-15 objective still needs to be measured.

## What the research supports

Pesaran and Timmermann show that including older observations can improve forecasts even when relationships have changed. Older observations can reduce estimation noise while introducing bias if the old relationship differs from the current one. A shorter window is more responsive to change and has fewer examples of rare markets. How much history helps depends on the size of the change and how much recent evidence is available. [Selection of Estimation Window in the Presence of Breaks](https://www.sciencedirect.com/science/article/pii/S0304407606000418)

Gu, Kelly, and Xiu learn a common relationship between firm characteristics and future returns across stocks, with separate chronological samples for training, validation, and testing. Another study uses a ten-year rolling sample of monthly equity data, with seven years for fitting and three for validation. These show why pooling stocks and testing forward in time are useful. Their monthly targets, universes, and metrics differ from this project's weekly equal-weight top 15. Their window lengths are precedents for comparisons. [Gu, Kelly, and Xiu](https://doi.org/10.1093/rfs/hhaa009) · [Machine learning techniques for cross-sectional equity returns’ prediction](https://link.springer.com/article/10.1007/s00291-022-00693-w)

## How much evidence we have

These counts use the local price CSV, as of the start of 2025. A row has two closes on consecutive calendar Fridays and positive adjusted prices and reported volumes at both endpoints. Counts are before feature lookbacks, news checks, and corporate-action reconciliation. Multiple stocks in the same week share market shocks. The number of weeks is important for interpreting how much evidence we have about portfolio performance.

| Calendar training history available by January 2025 | Distinct Friday pairs | Stock-week rows |
|---|---:|---:|
| 2 years | 99 | 41,295 |
| 3 years | 149 | 61,939 |
| 5 years | 243 | 99,847 |
| 7 years | 343 | 138,186 |
| All available, 2016–2024 | 443 | 174,671 |

A full ten-year training window before a 2025 test needs earlier data. Longer rolling windows also require earlier feature warm-up data. The research archive is an August 2026 roster applied historically, so results remain conditional on that roster and are subject to survivorship and membership bias. [Dataset description](README.md#historical-datasets)

Our earlier full-universe PatchTST study reported rank IC near −0.011 for both ensembles on its 2024–2025 test, versus +0.019 for ridge. A separate geometry study found that the 10/5 PatchTST arm had negative rank IC in both development years and positive rank IC on its 32-week 2026 test. Those runs compared model configurations. They motivate simple baselines and comparisons across market periods, and provide no controlled estimate of the effect of training-window length. [Full-universe audit](../patchtst_full_universe_audit_2026_08_23/REPORT.md) · [Geometry sweep](../patchtst_geometry_sweep_2026_08_23/REPORT.md)

## Can a model score Uber without training on Uber?

Yes, if it learns a shared mapping from available features to a score. Pooled ridge, boosted trees, and shared sequence models can take the same input columns or return sequence for a stock absent from fitting. A model with a learned ticker-specific intercept, embedding, or scaler needs a way to handle a new ticker. The local global-forecasting question is whether unseen stocks score well out of sample. The PatchTST paper's transfer results concern its published forecasting benchmarks. [Montero-Manso and Hyndman on global forecasting](https://fpp.robjhyndman.com/publications/global-forecasting/) · [PatchTST paper](https://arxiv.org/abs/2211.14730)

The existing Alpha-HRP workflow requests PatchTST scores for the full `halal_new` roster. The score endpoint takes explicitly requested symbols and passes their feature tensors to one model. With the current default of 60 return observations, the US close-only path needs 61 exact completed exchange closes for each requested stock. The separate monthly `halal_filtered` builder applies a history screen before inference. That builder's roughly ten-year rule is distinct from PatchTST's input requirement. The inference path was inspected in code. A held-out Uber model was not trained or evaluated for this research note. [Alpha-HRP workflow](../../temporal/workflows/us_alpha_hrp.py) · [Score endpoint](../../brain_api/brain_api/routes/inference/patchtst.py) · [Model inputs](../../brain_api/brain_api/core/patchtst/inference.py) · [Completed-close requirement](../../brain_api/brain_api/core/patchtst/price_history.py) · [`halal_filtered` history screen](../../brain_api/brain_api/universe/halal_filtered.py)

Uber’s first price in our archive is May 10, 2019. Its 61st daily bar is August 6, 2019. A feature using a full year of daily prices would need more time. For example, 252 daily returns need 253 valid closes, and the 253rd Uber bar is May 11, 2020. In a live-like simulation, include Uber in training once its input features and realized labels exist. In the unseen-stock experiment, withhold Uber from fitting and use only its already available history to score it. [Uber’s first trading date](https://www.sec.gov/Archives/edgar/data/1666268/000183988222000538/ms3618_424b2-00342.htm)

## Comparison for this screen

1. Use two-year rolling, five-year rolling, and expanding-from-2016 histories on the same chronological evaluation Fridays. Keep the feature set, eligibility, estimator, and refit schedule fixed across arms. Use a pooled ridge baseline first. Test any more complex model with the same protocol. [Tashman on rolling-origin evaluation](https://www.sciencedirect.com/science/article/pii/S0169207000000650)
2. Fit transformations and model parameters using only information available at each refit. Allow each stock to contribute once it has enough history for the chosen features and a completed forward label. Require only the history its features actually use at inference. Compute realized Friday-to-Friday simple returns from verified price evidence. Preserve verified zero-article news as neutral and account for missing coverage independently.
3. Compare the equal-weight top-15 realized return and its excess over the same week's eligible equal-weight universe. Report weekly Spearman rank correlation, turnover, and coverage as diagnostics. Use paired blocks of weeks to estimate uncertainty and check performance across years. Use development folds to choose the window. The local 2025 period has appeared in prior studies and descriptive diagnostics, so a later unexamined period or prospective results provide a cleaner confirmation.
4. Run a stock holdout alongside the normal walk-forward test. Withhold Uber and a preselected cohort of other stocks entirely from model fitting and fitted preprocessing. Score them only when their inputs are available. Measure rank quality within the held-out cohort and the top-15 performance when held-out names are allowed to compete in the full weekly universe. Report results for seen stocks and recently listed stocks as well. One company can demonstrate that scoring works. A cohort is needed to estimate how well it generalizes.
