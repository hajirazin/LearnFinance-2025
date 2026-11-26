# Weekly PatchTST paper cutover

The release contains the US and India PatchTST regressors and both US SAC
policies. Each regressor emits one adjusted close-to-close simple return over the
next five modeled sessions. India uses weekdays excluding NSE cash holidays.
SAC's realized reward still follows its
rebalance-open clock.

## Required inputs

- Set `HF_NEWS_SENTIMENT_REPO` to a **private dataset** accessible to both
  laptops and the Pi. Give publishing machines a token with write access and
  restore-only machines a token with read access. An older public daily-sentiment
  dataset can remain separate. The news sync endpoint rejects it as a destination.
- Put the authoritative DuckDB news store or its consistent backup under the
  isolated Brain API working directory at `data/news/news_v1.duckdb`. Use a
  stopped service or a DuckDB-consistent export when taking a backup. Keep the
  production source intact.
- India ships a compact history of weekday closures and published NSE
  capital-market holidays. Brain API generates sessions from weekdays and
  excludes those dates. It ignores special weekend sessions and individual
  closing-time notices. When a required new year becomes available in the
  official holiday feed, Brain API caches it under `data/cache/nse_holidays`.
  If the required year is unpublished, scoring fails explicitly. An optional
  `NSE_CASH_CALENDAR_PATH` override must cover the fit, its price warm-up, and
  the last required forward session.

A minimal calendar file has this format. This fragment illustrates the format
only.

```json
{
  "start": "2026-11-20",
  "end": "2026-11-24",
  "sources": ["https://www.nseindia.com/api/holiday-master?type=trading"],
  "holidays": ["2026-11-24"]
}
```

Use the `CM` section of the [NSE trading holiday feed](https://www.nseindia.com/api/holiday-master?type=trading)
and the [NSE capital-market archives](https://www.nseindia.com/resources/historical-reports-capital-market-daily-monthly-archives)
for archived years.

## Stage and verify

Run the training server on a separate port and from a separate working
directory. Its `data/` directory owns the staged caches, sticky history, models,
and news. Use `STORAGE_BACKEND=local_first` and explicitly set the four
affected model repository variables to empty in the staging process so its
training endpoints cannot move production model pointers. The four variables are
`HF_PATCHTST_HALAL_NEW_MODEL_REPO`, `HF_PATCHTST_NIFTY_SHARIAH_500_MODEL_REPO`,
`HF_SAC_HALAL_FILTERED_MODEL_REPO`, and `HF_SAC_HALAL_MODEL_REPO`.
Keep the private news dataset setting in place. Seed only from approved existing news data.

1. Submit `POST /etl/news/sync` with `{"seed":true}` from the authoritative
   news store and poll `GET /etl/news/sync/{job_id}` until it completes. Record the
   result's `news_commit`. Submit the same route with `{}` from a blank
   second store and poll it. Run the same published news interval and check the
   backfill output and provider logs for zero redundant fetches and scores.
2. Use `POST /etl/news/gaps` with `symbols`, `start`, `end`, and
   `"purpose":"patchtst"` for the union of the broad US and legacy `halal`
   rosters. Request history from available 2015 coverage through the evaluation
   cutoff, including seven calendar days before the earliest required session.
   Existing coverage is restored first. This mode checks the same session-close
   sentiment reconstruction as PatchTST and fails with affected dates if fetched
   revisions cannot establish historical inputs. The default `"purpose":"monday"`
   remains the separate SAC/PPO window contract. Use
   `POST /etl/news/patchtst` with `{"session_date":"YYYY-MM-DD"}` for recent
   closed sessions, then poll the sync job.
3. Call `POST /train/patchtst` with `{"universe":"halal_new"}` and
   `POST /train/patchtst/india` with `{"universe":"nifty_shariah_500"}` serially.
   Poll `GET /train/status/{job_id}`. Check each main model and required annual
   inventory, including the US result's `training_evidence.first_snapshot_cutoff`
   in artifact metadata.
   A failed publication leaves the completed fit on disk. Repeat the same
   training request to finish missing snapshots and upload those saved weights;
   a compatible, current fit is not trained again. Stale label evidence requires
   repairing the source data and fitting again.
4. Remove the staged current-month cache files for just `halal_filtered` and
   `halal_india` after taking copies. Leave the sticky database in place. Call
   both universe endpoints and record their selected slates. Run
   `POST /train/sac/preflight` with `force=true` for each of `halal_filtered`
   and `halal`, then `POST /train/sac/full` with the same two respective
   bodies. Poll until both model jobs finish. Require compatible artifacts,
   at least 208 transitions, and existing SAC promotion eligibility.
5. Run both repository suites from their project directories with
   `uv run --frozen --all-extras pytest -q`. Check production-sized inference
   with paper accounts and observation-only or mocked broker actions. Verify
   the 10-minute Alpha-HRP budget, paper account routing, and exact correspondence
   between the staged `halal_filtered` cache and the new policy's symbol order.

From the Brain API directory, validate the staged artifact set and current-month
slates before publishing:

```sh
uv run --frozen python -m brain_api.scripts.weekly_patchtst_release --data-dir "$STAGED_DATA_DIR"
uv run --frozen python -m brain_api.scripts.weekly_patchtst_release --data-dir "$STAGED_DATA_DIR" --publish
```

The second command uses the configured model repos and publishes versioned
artifacts. Its HF main pointers remain untouched.

## Paper activation and rollback

Pause the two Alpha-HRP and both SAC allocation schedules, plus the four
affected training schedules. Let existing order attempts settle and capture the
current local pointers, HF heads, annual directories, screen caches, and
deployment revision. Run the release command with `--activate-hf` while those schedules remain paused.
Install the staged local artifact pointers and compatible snapshots. Verify the installed versions
and final monthly screen before resuming. An intervening change to the
`halal_filtered` slate requires a new staged fit. Register the weekday
`us-patchtst-news` schedule and verify one successful restore/refresh on the Pi.
Resume allocation at its next normal occurrence once the actual Alpaca and IBKR
paper destinations have been verified.

If activation fails, keep allocation paused, reconcile outstanding orders,
restore the prior code and four model heads with their compatible annual
snapshots, and restore or rebuild the two screen caches. Preserve the shared
news evidence, broker fills, idempotency records, experience, and committed
weekly selections.

After eight completed rebalances, review coverage, weekly forecast RMSE and
rank IC, selection changes, net return, drawdown, turnover, and fills. Compare
cash-flow-adjusted US paper NAV with dividend-adjusted SPY and India modeled
paper returns with the Nifty500 Shariah Total Return Index. Use the forecast
anchor and fifth-session end to join outcome labels. Use actual rebalance and
fill times for portfolio results.
