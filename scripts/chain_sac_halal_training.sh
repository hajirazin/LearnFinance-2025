#!/usr/bin/env bash
set -euo pipefail

BRAIN_API_URL="${BRAIN_API_URL:-http://localhost:8000}"
JOB_ID="sac_halal_filtered:v2026-11-27_879504e8"
NEXT_UNIVERSE="halal"

while true; do
    if ! RESPONSE=$(curl -sf "$BRAIN_API_URL/train/status/$JOB_ID"); then
        echo "$(date '+%Y-%m-%d %H:%M:%S') status check failed (is brain_api up?), retrying in 5m"
        sleep 300
        continue
    fi
    STATUS=$(printf '%s' "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    echo "$(date '+%Y-%m-%d %H:%M:%S') $JOB_ID: $STATUS"
    case "$STATUS" in
        pending|running)
            sleep 300
            ;;
        completed)
            printf '%s' "$RESPONSE" | python3 -c 'import json,sys; r = json.load(sys.stdin).get("result") or {}; print("promoted:", r.get("promoted"), "| failure_reasons:", r.get("failure_reasons"))'
            echo "queueing SAC training on universe '$NEXT_UNIVERSE'"
            TRIGGER_RESPONSE=$(curl -sS -X POST "$BRAIN_API_URL/train/sac/full" \
                -H "Content-Type: application/json" \
                -d "{\"universe\": \"$NEXT_UNIVERSE\"}")
            echo "$TRIGGER_RESPONSE"
            exit 0
            ;;
        *)
            echo "$JOB_ID ended with status '$STATUS'; not queueing next training"
            printf '%s' "$RESPONSE" | python3 -c 'import json,sys; print("error:", json.load(sys.stdin).get("error"))'
            exit 1
            ;;
    esac
done
