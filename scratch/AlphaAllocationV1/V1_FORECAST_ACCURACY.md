# V1 forecast accuracy · H1–H27

**H2 had the lowest native return forecast error on the same 15 stocks each week (3.617 pp MAE).** H26 had the lowest native stock-level MAE (4.196 pp). The reference is the saved V1 research refit.

[Jump to the 27 charts](#individual-forecast-charts) · [All metrics and intervals](data/forecast_accuracy/model_metrics.csv) · [Weekly forecast detail](data/forecast_accuracy/weekly_forecasts.csv) · [Per-stock forecasts and actuals](data/forecast_accuracy/stock_forecasts.parquet) · [Inputs and run settings](data/forecast_accuracy/manifest.json)

The chart files are saved with this report. The linked CSV, Parquet, and JSON files are generated locally under `data/forecast_accuracy/`. See the [V1 README](README.md#archived-v1-experiments) for the reproduction command.

![Forecast error overview for the matched H1–H27 replay](figures/forecast_accuracy/overview.png)

## What was compared

- **Forecast target:** adjusted Friday close to the following Friday close, restricted to exactly five subsequent XNYS sessions. This matches the five-session forecasts and H27's weekly forecast. The common score panel had 33 eligible weeks (2026-01-09 to 2026-10-30, decision Fridays), with 13,712 observed stock outcomes. Exclude missing or unresolved archived actuals.
- **Primary metric:** mean absolute prediction error in percentage points on the same weekly equal-weight set of 15 stocks. Select using the V1 15/20 sticky rule on the common score universe before inspecting returns. Carry selections through all archived decisions, including weeks omitted from forecast scoring. All 15 realized returns must be available for a basket comparison. The stock metrics average each week's errors before averaging across weeks.
- **Own picks:** use each model's own 15/20 selections on that same shared universe. Also score the V1 forecast on exactly those names. Original archived portfolio and rank comparisons use their own eligibility and selection rules. H25's archived scores retain the prior-membership state from its original scoring run.
- **SPY:** show realized dividend-adjusted SPY on the same close-to-close horizon and split fixed-basket MAE by whether SPY fell. SPY is market context. Forecast errors are relative to the realized stocks or basket. The original trading panels use SPY adjusted opens over the archived executable intervals.
- **Calibration:** H22, H23, and H25 are selection scores. For a comparable secondary test, convert each model's score to a within-week percentile and fit a linear mapping with a nonnegative slope using only previously completed windows. Start once eight weeks of outcomes are known. Compare against zero and the expanding mean of past universe returns. This calibration is a separate forecasting rule.
- **Inference:** paired differences in same-basket absolute error, with 95% circular block-bootstrap intervals (4 adjacent evaluated weeks, 20,000 draws, seed 42). One-sided lower-error p-values are adjusted with Holm within each method and reference. Earlier experiments already used the evaluation period. Treat all results as exploratory.

## Native return forecasts

| Model | Same 15 MAE | Stock MAE | Stock RMSE | Stock bias | Own 15 MAE* | V1 on those same picks* | Zero on those same picks* | SPY down MAE | SPY up MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | 3.617 | 4.287 | 6.012 | 0.200 | 5.388 | 4.957 | 4.306 | 4.283 | 2.819 |
| H21 | 3.639 | 4.295 | 5.983 | 0.138 | 3.934 | 3.472 | 3.128 | 4.258 | 2.895 |
| H1 | 3.749 | 4.276 | 5.973 | -0.179 | 3.273 | 3.075 | 2.405 | 4.158 | 3.258 |
| H24 | 3.753 | 4.231 | 5.923 | 0.309 | 4.126 | 3.766 | 3.275 | 4.399 | 2.978 |
| H21 chronological | 3.853 | 4.226 | 5.912 | 0.111 | 4.144 | 3.539 | 3.095 | 4.364 | 3.240 |
| H26 | 3.862 | 4.196 | 5.875 | -0.071 | 1.884 | 1.763 | 1.707 | 4.068 | 3.614 |
| Zero | 3.863 | 4.181 | 5.869 | -0.110 | 3.863 | 4.830 | 3.863 | 4.078 | 3.605 |
| H27 | 3.863 | 4.269 | 5.945 | -0.294 | 2.194 | 2.092 | 2.120 | 3.955 | 3.753 |
| H5 | 3.904 | 4.264 | 5.965 | -0.182 | 3.147 | 2.712 | 2.447 | 4.195 | 3.554 |
| H4 | 4.113 | 4.578 | 6.330 | 1.928 | 6.840 | 5.473 | 5.369 | 5.208 | 2.798 |
| H3 | 4.696 | 4.398 | 6.152 | 0.096 | 5.107 | 4.853 | 4.110 | 5.947 | 3.195 |
| V1 | 4.830 | 4.393 | 6.145 | 0.041 | 4.830 | 4.830 | 3.863 | 6.230 | 3.150 |

Values are in percentage points. V1's fixed-basket MAE was 4.830 pp and the zero-return forecast was 3.863 pp. The best native model's paired MAE difference versus V1 was -1.213 pp (95% CI -1.940 to -0.534). Against zero it was -0.246 pp (95% CI -0.813 to +0.315). A negative difference favors the candidate.

Zero had a stock MAE of 4.181 pp, compared with 4.196 pp for the best trained model. H26 had the smallest native own-pick MAE (1.884 pp). On exactly those picks, V1 scored 1.763 pp and zero scored 1.707 pp. H1 had the lowest fixed-basket RMSE (4.697 pp), a metric that penalizes larger misses more heavily.

In the 18 weeks SPY fell, fixed-basket MAE was 4.283 pp for H2, 6.230 pp for V1, and 4.078 pp for zero. A correctly predicted negative basket return counts as an accurate forecast. The sign of SPY describes the market week.

## Past-only calibration

| Model | Same 15 MAE | Stock MAE | Stock RMSE | Stock bias | Own 15 MAE* | V1 on those same picks* | Zero on those same picks* | SPY down MAE | SPY up MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H4 | 3.749 | 4.122 | 5.852 | -0.381 | 5.860 | 5.860 | 5.824 | 3.999 | 3.553 |
| V1 | 3.750 | 4.119 | 5.853 | -0.381 | 3.750 | 3.750 | 3.808 | 4.043 | 3.519 |
| H3 | 3.756 | 4.120 | 5.853 | -0.381 | 4.119 | 4.122 | 4.047 | 4.031 | 3.540 |
| Zero | 3.808 | 4.085 | 5.813 | -0.363 | 3.808 | 3.750 | 3.808 | 3.799 | 3.815 |
| H2 | 3.827 | 4.113 | 5.847 | -0.381 | 4.432 | 4.437 | 4.382 | 3.929 | 3.747 |
| H21 | 3.880 | 4.110 | 5.845 | -0.381 | 3.309 | 3.299 | 3.239 | 3.904 | 3.861 |
| H21 chronological | 3.890 | 4.108 | 5.842 | -0.381 | 3.307 | 3.353 | 3.255 | 3.874 | 3.902 |
| H23 | 3.891 | 4.107 | 5.841 | -0.381 | 2.366 | 2.389 | 2.364 | 3.880 | 3.900 |
| H1 | 3.892 | 4.107 | 5.841 | -0.381 | 2.488 | 2.483 | 2.391 | 3.879 | 3.902 |
| H5 | 3.892 | 4.107 | 5.841 | -0.381 | 2.630 | 2.643 | 2.534 | 3.879 | 3.902 |
| Past universe mean | 3.892 | 4.107 | 5.841 | -0.381 | 3.892 | 3.750 | 3.808 | 3.879 | 3.902 |
| H24 | 3.892 | 4.107 | 5.841 | -0.381 | 3.802 | 3.812 | 3.730 | 3.881 | 3.901 |
| H22 | 3.893 | 4.108 | 5.842 | -0.381 | 3.278 | 3.345 | 3.261 | 3.881 | 3.902 |
| H26 | 3.893 | 4.107 | 5.841 | -0.381 | 1.662 | 1.607 | 1.646 | 3.877 | 3.906 |
| H25 | 3.893 | 4.107 | 5.842 | -0.381 | 1.495 | 1.499 | 1.490 | 3.879 | 3.904 |
| H27 | 3.903 | 4.109 | 5.842 | -0.381 | 2.280 | 2.263 | 2.226 | 3.895 | 3.909 |

H4 led the calibrated model forecasts on the fixed basket at 3.749 pp over 25 weeks, compared with 3.808 pp for zero and 3.892 pp for the past mean. Native and calibrated metrics cover different periods.

*Own-pick MAEs may differ in stocks and available weeks across models. Each model's comparison against V1 on its own picks uses the identical baskets and weeks. The [metrics file](data/forecast_accuracy/model_metrics.csv) also includes own-pick MAE on the intersection of observable weeks, counts, signed errors, directional accuracy, ranking diagnostics, and paired intervals.

## All 27 comparisons

Each chart shows the same-basket forecast and actual, weekly errors, each model's own picks, errors across individual stocks, and the original ranking or trading comparison. H6–H20 reuse the H1–H5 forecasts, so their forecast errors repeat by design. Historical trading outcomes are included for context.

| Chart | Model change | Forecast | Same 15 MAE | Stock MAE | Original metric | Model | V1 | SPY | Other original reference |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| [H01](#h01) | Seven features | Native | 3.749 | 4.276 | HRP compounded net % | -10.88 | 30.61 | +8.26 | — |
| [H02](#h02) | One-year input window | Native | 3.617 | 4.287 | HRP compounded net % | -10.22 | 31.62 | +8.26 | — |
| [H03](#h03) | Fit on 356 stocks | Native | 4.696 | 4.398 | HRP compounded net % | 30.80 | 35.71 | +8.26 | — |
| [H04](#h04) | Seven features, one year, and 356 stocks | Native | 4.113 | 4.578 | HRP compounded net % | 6.28 | 39.82 | +8.26 | — |
| [H05](#h05) | Seven features and 356 stocks | Native | 3.904 | 4.264 | HRP compounded net % | 1.63 | 30.61 | +8.26 | — |
| [H06](#h06) | Seven features | Native | 3.749 | 4.276 | equal weight compounded net % | -6.36 | 45.28 | +8.26 | — |
| [H07](#h07) | One-year input window | Native | 3.617 | 4.287 | equal weight compounded net % | -7.74 | 46.08 | +8.26 | — |
| [H08](#h08) | Fit on 356 stocks | Native | 4.696 | 4.398 | equal weight compounded net % | 41.59 | 50.77 | +8.26 | — |
| [H09](#h09) | Seven features, one year, and 356 stocks | Native | 4.113 | 4.578 | equal weight compounded net % | 12.51 | 46.13 | +8.26 | — |
| [H10](#h10) | Seven features and 356 stocks | Native | 3.904 | 4.264 | equal weight compounded net % | 3.21 | 45.28 | +8.26 | — |
| [H11](#h11) | Seven features | Native | 3.749 | 4.276 | mean top-15 hits | 0.81 | 1.65 | — | Random 0.54 hits |
| [H12](#h12) | One-year input window | Native | 3.617 | 4.287 | mean top-15 hits | 1.51 | 1.70 | — | Random 0.54 hits |
| [H13](#h13) | Fit on 356 stocks | Native | 4.696 | 4.398 | mean top-15 hits | 1.84 | 1.70 | — | Random 0.53 hits |
| [H14](#h14) | Seven features, one year, and 356 stocks | Native | 4.113 | 4.578 | mean top-15 hits | 1.97 | 1.65 | — | Random 0.54 hits |
| [H15](#h15) | Seven features and 356 stocks | Native | 3.904 | 4.264 | mean top-15 hits | 0.81 | 1.65 | — | Random 0.54 hits |
| [H16](#h16) | Seven features | Native | 3.749 | 4.276 | equal weight compounded net % | -6.36 | 45.28 | +8.26 | — |
| [H17](#h17) | One-year input window | Native | 3.617 | 4.287 | equal weight compounded net % | -7.74 | 46.08 | +8.26 | — |
| [H18](#h18) | Fit on 356 stocks | Native | 4.696 | 4.398 | equal weight compounded net % | 41.59 | 50.77 | +8.26 | — |
| [H19](#h19) | Seven features, one year, and 356 stocks | Native | 4.113 | 4.578 | equal weight compounded net % | 12.51 | 46.13 | +8.26 | — |
| [H20](#h20) | Seven features and 356 stocks | Native | 3.904 | 4.264 | equal weight compounded net % | 3.21 | 45.28 | +8.26 | — |
| [H21](#h21) | Five-session return MSE | Native | 3.639 | 4.295 | equal weight compounded net % | 19.95 | 45.28 | +8.26 | — |
| [H22](#h22) | Pairwise ranking loss | Calibrated | 3.893 | 4.108 | mean top-15 hits | 1.41 | 1.65 | — | Random 0.54 hits |
| [H23](#h23) | SPY-relative selection objective | Calibrated | 3.891 | 4.107 | equal weight compounded net % | 11.28 | 45.28 | +8.26 | — |
| [H24](#h24) | Five-session MSE with IBKR fees | Native | 3.753 | 4.231 | equal weight compounded net % | -10.90 | 42.94 | +8.26 | H21 +17.08%, H21 chronological -27.99% |
| [H25](#h25) | Ranking with prior membership and IBKR fees | Calibrated | 3.893 | 4.107 | HRP compounded net % | 6.13 | 27.95 | +8.26 | Hindsight top 15 +21504.99% |
| [H26](#h26) | Seven daily forecast targets | Native | 3.862 | 4.196 | HRP compounded net % | -9.00 | 27.95 | +8.26 | H5 -2.22% |
| [H27](#h27) | Seven weekly forecast targets | Native | 3.863 | 4.269 | HRP compounded net % | 12.56 | 27.95 | +8.26 | H21 +15.41% |

[Original comparison data and market-down counts](data/forecast_accuracy/hypothesis_metrics.csv) · [Calibration coefficients with label availability](data/forecast_accuracy/calibration.csv)

## Individual forecast charts

### H01

Seven features

![H01 — Seven features](figures/forecast_accuracy/H01.png)

### H02

One-year input window

![H02 — One-year input window](figures/forecast_accuracy/H02.png)

### H03

Fit on 356 stocks

![H03 — Fit on 356 stocks](figures/forecast_accuracy/H03.png)

### H04

Seven features, one year, and 356 stocks

![H04 — Seven features, one year, and 356 stocks](figures/forecast_accuracy/H04.png)

### H05

Seven features and 356 stocks

![H05 — Seven features and 356 stocks](figures/forecast_accuracy/H05.png)

### H06

Seven features

![H06 — Seven features](figures/forecast_accuracy/H06.png)

### H07

One-year input window

![H07 — One-year input window](figures/forecast_accuracy/H07.png)

### H08

Fit on 356 stocks

![H08 — Fit on 356 stocks](figures/forecast_accuracy/H08.png)

### H09

Seven features, one year, and 356 stocks

![H09 — Seven features, one year, and 356 stocks](figures/forecast_accuracy/H09.png)

### H10

Seven features and 356 stocks

![H10 — Seven features and 356 stocks](figures/forecast_accuracy/H10.png)

### H11

Seven features

![H11 — Seven features](figures/forecast_accuracy/H11.png)

### H12

One-year input window

![H12 — One-year input window](figures/forecast_accuracy/H12.png)

### H13

Fit on 356 stocks

![H13 — Fit on 356 stocks](figures/forecast_accuracy/H13.png)

### H14

Seven features, one year, and 356 stocks

![H14 — Seven features, one year, and 356 stocks](figures/forecast_accuracy/H14.png)

### H15

Seven features and 356 stocks

![H15 — Seven features and 356 stocks](figures/forecast_accuracy/H15.png)

### H16

Seven features

![H16 — Seven features](figures/forecast_accuracy/H16.png)

### H17

One-year input window

![H17 — One-year input window](figures/forecast_accuracy/H17.png)

### H18

Fit on 356 stocks

![H18 — Fit on 356 stocks](figures/forecast_accuracy/H18.png)

### H19

Seven features, one year, and 356 stocks

![H19 — Seven features, one year, and 356 stocks](figures/forecast_accuracy/H19.png)

### H20

Seven features and 356 stocks

![H20 — Seven features and 356 stocks](figures/forecast_accuracy/H20.png)

### H21

Five-session return MSE

![H21 — Five-session return MSE](figures/forecast_accuracy/H21.png)

### H22

Pairwise ranking loss

![H22 — Pairwise ranking loss](figures/forecast_accuracy/H22.png)

### H23

SPY-relative selection objective

![H23 — SPY-relative selection objective](figures/forecast_accuracy/H23.png)

### H24

Five-session MSE with IBKR fees

![H24 — Five-session MSE with IBKR fees](figures/forecast_accuracy/H24.png)

### H25

Ranking with prior membership and IBKR fees

![H25 — Ranking with prior membership and IBKR fees](figures/forecast_accuracy/H25.png)

### H26

Seven daily forecast targets

![H26 — Seven daily forecast targets](figures/forecast_accuracy/H26.png)

### H27

Seven weekly forecast targets

![H27 — Seven weekly forecast targets](figures/forecast_accuracy/H27.png)

## Final assessment

H2 and H21 had the two lowest native MAEs on the fixed basket. Their advantage over the V1 refit supports a fresh forecast evaluation. None of the native or calibrated models established lower same-basket MAE than zero in the paired tests at the adjusted 5% level. The own-pick results answer a separate question because the realized basket changes with the model. Use the same-basket and stock-level errors, with the zero baseline, to evaluate whether a candidate improves prediction.
