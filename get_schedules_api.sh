#!/bin/bash

# Target the Temporal Web UI API proxy. Change 8080 if your UI runs on a different port.
API_URL="http://razinpi.local:8233/api/v1/namespaces/default"

echo "Fetching schedules via Temporal REST API..."

# 1. Get all schedule IDs
SCHEDULES=$(curl -s "$API_URL/schedules" | jq -r '.schedules[].scheduleId // empty')

if [ -z "$SCHEDULES" ]; then
  echo "No schedules found or unable to connect to $API_URL"
  exit 0
fi

printf "%-30s | %-30s | %-25s | %-30s\n" "SCHEDULE ID" "NEXT UPCOMING RUN" "TASK QUEUE" "WORKFLOW NAME"
echo "--------------------------------------------------------------------------------------------------------------------------------"

# 2. Iterate and describe
for sid in $SCHEDULES; do
    curl -s "$API_URL/schedules/$sid" | jq -r '
      {
        id: "'"$sid"'",
        next_run: (.info.futureActionTimes[0] // "Paused/None"),
        queue: (.schedule.action.startWorkflow.taskQueue.name // "N/A"),
        workflow: (.schedule.action.startWorkflow.workflowType.name // "N/A")
      } | 
      "\(.id)\t\(.next_run)\t\(.queue)\t\(.workflow)"
    ' | while IFS=$'\t' read -r id next_run queue workflow; do
        printf "%-30s | %-30s | %-25s | %-30s\n" "$id" "$next_run" "$queue" "$workflow"
    done
done