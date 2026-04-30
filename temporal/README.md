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

| Workflow | Schedule | Description |
|----------|----------|-------------|
| USWeeklyAllocation | Monday 11:00 UTC | SAC-only allocation + sell-wait-buy + email (naive HRP retired; `hrp` account now driven by USAlphaHRP) |
| IndiaWeeklyAllocation | Monday 03:30 UTC | India HRP allocation + email |
| IndiaDoubleHRP | Monday 04:00 UTC | Two-stage HRP (Nifty Shariah 500 → top 15) + email |
| USDoubleHRP | Monday 11:30 UTC | Two-stage HRP (halal_new → sticky top 15) + dhrp orders + email |
| USAlphaHRP | Monday 12:00 UTC | PatchTST alpha → rank-band sticky top 15 → HRP on the `hrp` Alpaca account + email |
| USWeeklyTraining | Sunday 11:00 UTC | Full US model training (not in `SCHEDULES` by default) |
| IndiaWeeklyTraining | Sunday 04:30 UTC | India PatchTST training (not in `SCHEDULES` by default) |

## Schedule registration is idempotent

`schedules.py` uses a **create-if-not-exists** pattern. On first run it creates
the schedules; on every subsequent run it logs `SKIP (already exists, not
updating)` and exits 0. This means the docker-compose `temporal-schedules-init`
one-shot service can safely run on every `docker compose up -d --build` without
side effects.

Five cron schedules are registered by default: US weekly allocation, India
weekly allocation, India Double HRP, US Double HRP, and US Alpha-HRP (see
`SCHEDULES` in `schedules.py`). Training schedules are preserved as a commented `SCHEDULES_MAC`
block for future use on a beefier host.

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
