# H28 · Complete-case training and sensitivity to negative sentiment

H1 and H5 both feed seven feature histories to the same PatchTST encoder and joint head, predict five daily adjusted-close log returns, and train on close-only MSE. H1 starts with the full 428-symbol roster. H5 starts with the 356-symbol first-Friday cohort. Both require complete close-return history, an observed decision-date value in each feature, and at least 80% coverage per channel. They mean-pad masked gaps. The 15 zero-volume names were an observation on January 8, 2016, separate from the H1/H5 training filters.

H28 uses H1's roster and H5's architecture, with stricter fitting inputs. Every selected 60-session training window must have all seven features, complete close-return scaling history, and all five return labels. Retain each fully observed window and exclude a stock entirely only when it has no qualifying window. This lets newer listings contribute after they have enough history and avoids dropping complete windows because of a gap elsewhere in the stock's history. Counts refer to the frozen 428-symbol research roster. Verified no-news remains a valid zero. Use the existing 2021–2024 fitting cutoff, common broader 2025 validation, seed 42, validation weekly rank IC checkpoint criterion with MSE tie-break, and standard feature eligibility at 2026 inference. This keeps the complete-case change specific to the fitting fold.

## Sentiment intervention

Use each frozen checkpoint and the 40 archived decision Fridays in 2026. For one candidate at a time, rescore its input with the Friday sentiment feature set to 0 in the control and −0.50 in the intervention. Hold the other six feature histories, past sentiment cells, other stocks' scores, the candidate roster, and previously held names fixed. Each model starts with its actual-input prior selections from the same replay. Run the same deterministic 15/20 rank-band selector on both scores. A removal means the stock was selected under neutral sentiment and lost its slot after the intervention. Independent candidate interventions do not describe simultaneous shocks or accumulated changes across weeks.

The input is a seven-calendar-day rolling sentiment feature observed at each session close. A −0.50 feature value is a different intervention from one article scored −0.50. As a sensitivity check, compare adding one neutral article with adding one −0.50 article at Friday close, both with confidence 1. Use the archived confidence/recency weight denominator to recompute the feature. For a second sensitivity check apply the one-cell shock at the preceding session rather than at Friday close. When that earlier input is missing, keep that stock in the background universe and exclude it as an intervention subject.

## Statistical questions

- A literal universal exclusion is refuted by any stock that remains selected under the intervention. To give a statistical comparison with a nondegenerate boundary, also evaluate the practically near-universal null that the **mean weekly exit rate among neutral-selected candidates is at least 95%**.
- Separately evaluate whether negative sentiment reduces selection on balance. On each Friday compute the paired negative-minus-neutral selection rate across eligible candidates. Test the one-sided null that the expected weekly difference is zero or positive.
- Use a null-centered circular block bootstrap over weeks, with four-week blocks, 20,000 draws, and seed 42. Report mean differences and 95% intervals. The primary comparison is H28 with the decision-day feature shock. H1, H5, and alternative shocks are sensitivity analyses on reused 2026 history. The test measures frozen model behavior. Establish the effect on future forecast error or stock returns in a separate evaluation using observed data.

Also compare ordinary H28 forecasts with H1, H5, the V1 refit and zero on the exact same 33 dates, observed stock outcomes and fixed V1 baskets in the existing forecast accuracy report.

## Results

The full research roster contained 428 symbols. H1 contributed 83,583 fitting windows across 416 symbols. H28 kept 75,208 complete windows across 415 symbols, leaving 13 without a qualifying window. `VLTO` was the one name that contributed under H1 and lost its remaining two fitting windows under the stricter policy. H5 contributed 73,203 windows across 356 symbols. H28 used the same 20,938 validation examples as H1 and H5, selecting epoch 19 with mean validation weekly Spearman +0.0255.

| Checkpoint | Neutral picks | Exited at −0.50 | Entered at −0.50 | Mean weekly exit rate (95% CI) | p for lower net selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 594 | 22 | 7 | 3.67% (2.81%–4.65%) | <0.0001 |
| H5 | 598 | 3 | 25 | 0.50% (0.00%–1.00%) | 0.99995 |
| H28 | 600 | 22 | 18 | 3.66% (2.04%–5.29%) | 0.228 |

**H28 kept 578 of the 600 candidates that would have been selected under neutral sentiment.** The near-universal null of at least 95% exits was rejected with a one-sided bootstrap p-value below 0.0001. No null-centered resample was as extreme in 20,000 draws. The estimated net change across all candidates was −0.024 percentage points (95% CI −0.090 to +0.036). This test leaves a net decrease in selection unestablished.

The single −0.50 article caused 10 exits among 600 otherwise selected H28 cases and 15 entries. The preceding-session channel shock caused 5 exits among 605 otherwise selected cases and 48 entries. These sensitivity checks show that the response depends on timing and on how the news enters the rolling feature.

| Forecast | Basket MAE on the same 15 picks (pp) | Matched stock MAE (pp) |
| --- | ---: | ---: |
| H1 | 3.749 | 4.276 |
| H5 | 3.904 | 4.264 |
| H28 | 3.799 | 4.263 |
| V1 research refit | 4.830 | 4.393 |
| Zero forecast | 3.863 | 4.181 |

On the matched basket, H28 minus H1 was +0.050 pp (paired 95% CI −0.058 to +0.166). H28 minus H5 was −0.105 pp (−0.267 to +0.026). H28 minus zero was −0.064 pp (−0.377 to +0.295). The archive leaves the incremental forecast value of stricter fitting coverage inconclusive.

[Full statistical results and input hashes](data/v1_hypotheses/h28/summary.json)

## Reproduce

From the repository root, using the PatchTST scratch environment:

```sh
PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/train_v1_h28.py --policy complete-windows --device mps
PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/run_v1_h28.py --device cpu
```

Follow the regeneration command in the [H1–H27 forecast report](V1_FORECAST_ACCURACY.md) before the H28 replay if its forecast artifacts are absent. The raw [trial-level data](data/v1_hypotheses/h28/sentiment_trials.parquet), [weekly results](data/v1_hypotheses/h28/sentiment_weekly.csv), [forecast errors](data/v1_hypotheses/h28/forecast_weekly.csv), and [training coverage](data/v1_hypotheses/h28/training_coverage.csv) are stored locally under gitignored scratch data.
