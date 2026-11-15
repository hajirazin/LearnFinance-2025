# H28 input ablation · adjusted close versus daily adjusted-close return

Compare the saved H28 with a new fit that replaces its adjusted-close price channel with the past one-session adjusted-close log return, `log(close[t] / close[t-1])`. The encoder still takes seven channels over 60 sessions. Both arms already use the same past close log returns to rescale their output and predict five future daily close log returns with daily MSE.

Keep H28’s 2021–2024 complete fitting windows, common 2025 validation rows, input mask, seed 42, PatchTST settings, and validation rank-IC checkpoint rule with MSE tie-break. Apply the channel replacement after building the shared input windows and verify that the two channels have identical coverage. Fit the challenger independently from initialization. Use the existing H28 checkpoint as the control.

Use the same 33 archived 2026 Fridays, observed stock outcomes, and fixed equal-weight V1 top-15 baskets in the [forecast accuracy report](data/forecast_accuracy/REPORT.md). The primary metric is paired absolute error on the same basket return, measured in percentage points. Test whether the challenger lowers mean weekly absolute error using a one-sided null-centered circular block bootstrap (four adjacent observed weeks, 20,000 draws, seed 42, 5% level). Report the two-sided 95% interval for challenger minus control. Treat same-stock MAE, RMSE, signed bias, and rank correlation as diagnostics. Report the zero-return forecast as a reference and show realized SPY on the matching close-to-close horizon. These dates have been studied before, so interpret the results as exploratory and specific to this seed.

From the repository root:

```sh
PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/train_v1_h28.py --policy complete-windows --close-input return --device mps
PYTHONPATH=brain_api:scratch/AlphaAllocationV2 python scratch/AlphaAllocationV1/compare_h28_close_input.py
```
