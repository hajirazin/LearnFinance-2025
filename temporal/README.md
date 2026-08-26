# LearnFinance Temporal Workflows

Temporal-based workflow orchestration for the LearnFinance-2025 trading pipeline.

## Quick Start

```bash
# Terminal 1: Temporal dev server (SQLite persistence, UI at localhost:8233)
devbox run temporal:server

# Terminal 2: brain-api
devbox run brain:run

# Terminal 3: Temporal worker
devbox run temporal:worker

# One-time: register cron schedules
devbox run temporal:schedule

# Optional: trigger a Double HRP run by hand (requires worker + brain_api)
devbox run temporal:run:india-double-hrp
devbox run temporal:run:us-double-hrp
```

## Workflows

Ten weekly-ish inference schedules plus five monthly training schedules are registered. Inference schedules land on the `learnfinance-inference`
queue (Pi worker, optional Mac inference backup); training schedules land on the
`learnfinance-training` queue (Mac trainer, `TEMPORAL_MAX_CONCURRENT_ACTIVITIES=1`
to serialise heavy GPU activities).

| Workflow | Schedule | Queue | Description |
|----------|----------|-------|-------------|
| USDoubleHRP | Monday 07:00 America/New_York | inference | Two-stage HRP (halal_new → sticky top 15) + dhrp orders + email |
| USAlphaHRP | Monday 07:30 America/New_York | inference | PatchTST alpha → rank-band sticky top 15 → HRP on the `hrp` Alpaca account + email |
| USWeeklyAllocation | Monday 09:00 America/New_York | inference | `halal_filtered` SAC allocation + sell-wait-buy + email |
| USSACHalalAllocation | Monday 09:05 America/New_York | inference | `halal` SAC allocation through the dedicated `sac_halal` Alpaca account + email |
| IndiaWeeklyAllocation | Monday 09:00 Asia/Kolkata | inference | India Alpha-HRP (PatchTST screen → rank-band sticky → HRP) + email (paper-only, no broker) |
| IndiaDoubleHRP | Monday 09:30 Asia/Kolkata | inference | Two-stage HRP (Nifty Shariah 500 → top 15) + email (paper-only, no broker) |
| USPPODiscoveryAllocation | Monday 09:10 America/New_York | inference | News-conditioned PPO on frozen `halal_new` via the `ppo_discovery` Alpaca account + email. Incomplete news / missing `current` skip with zero orders. |
| USForecastersTraining | First Sunday of month, 00:01 UTC | training | LSTM + PatchTST training (strictly serial) + email |
| USSACTraining | First Sunday of month, 06:01 UTC | training | SAC training on `halal_filtered` bucket + email |
| USSACHalalTraining | First Sunday of month, 12:01 UTC | training | SAC training on `halal` legacy yfinance bucket (parallel A/B sibling) + email |
| IndiaMonthlyTraining | First Sunday of month, 18:01 UTC | training | India PatchTST training + email |
| USPPODiscoveryTraining | Second Sunday of month, 00:01 UTC | training | PPO discovery train on frozen `halal_new` (candidate only; no auto-promote) + email |

To disable only PPO discovery, pause or delete `us-ppo-discovery-allocate` and `us-ppo-discovery-training` on the Temporal server. Do not edit other `SCHEDULES` entries or `first_sunday_of_month_at`.

## Schedule registration is idempotent

`schedules.py` creates missing schedules and updates existing definitions in
place while preserving their paused state. The docker-compose
`temporal-schedules-init` one-shot service can therefore safely run on every
`docker compose up -d --build`, including after a calendar or timezone change.

All twelve schedules above live in a single `SCHEDULES` list in `schedules.py`.
Training cadence cannot be expressed as a single cron string ("first Sunday of
month" requires AND-ing day-of-month and day-of-week, which Vixie cron OR's), so
training entries use `ScheduleCalendarSpec(day_of_month=[1..7], day_of_week=[0],
hour=H, minute=M)` via the `first_sunday_of_month_at` helper while inference
entries continue to use plain cron strings.

## Changing a schedule on the Pi

Because registration is create-if-not-exists, editing a schedule's cron in
`schedules.py` and redeploying will NOT update a running Pi -- the init service
sees the schedule already exists and logs `SKIP`. This is intentional (safety)
but means you need an explicit escape hatch to change a cron later:

```bash
# 1. Edit temporal/schedules.py on the Mac.

# 2. Delete the existing schedule on the Pi so init can recreate it:
docker --context razinpi compose exec temporal-server \
  temporal schedule delete --schedule-id us-weekly-allocate --address 127.0.0.1:7233

# 3. Redeploy from the Mac; the Docker CLI uses the razinpi context,
#    streams the local build context over SSH to the Pi's daemon, and
#    temporal-schedules-init creates the new version.
docker --context razinpi compose up -d --build
docker --context razinpi compose logs temporal-schedules-init
# Expect: "Created: us-weekly-allocate (<new cron>) - ..."
```

Note: no git checkout on the Pi is needed. The Pi only runs the Docker daemon;
the repo and compose file live on the Mac. Every deploy uses whatever code is
on the Mac at the moment you run `docker --context razinpi compose up -d --build`.

The same procedure applies on the laptop for local development (drop
`--context razinpi` and use `devbox run temporal:schedule` at step 3 instead).

## Operator notes

- `temporal-schedules-init` shows `Exited (0)` in `docker compose ps` after a
  successful run. That is the expected healthy state for a one-shot container.
- `temporal schedule describe --schedule-id <id>` is the source of truth for
  "did the last scheduled run fire, when, and with what status".
- The Temporal SQLite DB lives in a host bind mount on the Pi at
  `~/learnfinance/temporal-data/temporal.db` (owned by the Pi user, so the
  Temporal container's non-root user can write it). Do not delete that
  directory — it holds every schedule's run history.

## Future Ideas

- **Temporal Signals**: brain_api could send Signals to running workflows (e.g., "training done", "order filled") to eliminate polling. Requires adding a Temporal client to brain_api. Not needed now since the durable poll loop works well for a laptop setup.
- **Alpaca Webhooks**: If Alpaca adds webhook support, order fill events could signal the sell-wait-buy workflow directly instead of polling every 15 minutes.
