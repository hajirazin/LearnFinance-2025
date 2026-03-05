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
```

## Workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| USWeeklyAllocation | Monday 11:00 UTC | Allocation + sell-wait-buy + email |
| IndiaWeeklyAllocation | Monday 03:30 UTC | India HRP allocation + email |
| USWeeklyTraining | Sunday 11:00 UTC | Full US model training |
| IndiaWeeklyTraining | Sunday 04:30 UTC | India PatchTST training |

## Future Ideas

- **Temporal Signals**: brain_api could send Signals to running workflows (e.g., "training done", "order filled") to eliminate polling. Requires adding a Temporal client to brain_api. Not needed now since the durable poll loop works well for a laptop setup.
- **Alpaca Webhooks**: If Alpaca adds webhook support, order fill events could signal the sell-wait-buy workflow directly instead of polling every 15 minutes.
