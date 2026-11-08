"""Run unmodified production PPO inference activities on a Mac-only server."""

import argparse
import asyncio
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from temporalio.api.enums.v1 import EventType
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import ActivityInboundInterceptor, Interceptor, Worker

from inference_smoke_workflow import PPOInferenceOnlyDiagnostic

from activities.inference import get_monday_decision_window
from activities.ppo_discovery_inference import (
    build_ppo_discovery_state,
    get_ppo_discovery_portfolio,
    infer_ppo_discovery,
)

ROOT = Path(__file__).resolve().parent


class RecordResults(ActivityInboundInterceptor):
    async def execute_activity(self, input):
        result = await self.next.execute_activity(input)
        name = input.fn.__name__
        if name == "build_ppo_discovery_state":
            encoded = await pydantic_data_converter.encode([result])
            wire = encoded[0].data
            path = ROOT / "data" / "local_state_response.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wire)
            metrics = {
                "activity": name,
                "payload_data_bytes": len(wire),
                "protobuf_payload_bytes": encoded[0].ByteSize(),
                "gzip_bytes_diagnostic_only": len(gzip.compress(wire)),
                "eligible_assets": sum(result["asset_mask"]),
                "state_digest": result["state_digest"],
            }
            (ROOT / "local_state_size.json").write_text(json.dumps(metrics, indent=2))
            print(json.dumps(metrics), flush=True)
        return result


class RecordInterceptor(Interceptor):
    def intercept_activity(self, next):
        return RecordResults(next)


async def main(args):
    if args.address != "127.0.0.1:17233":
        raise ValueError(
            "This runner is restricted to the separate Mac Temporal server"
        )
    if urlparse(os.environ["BRAIN_API_URL"]).hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("This runner may only call a local Brain API")
    queue = "ppo-local-diagnostic-20261107"
    client = await Client.connect(args.address, data_converter=pydantic_data_converter)
    workflow_id = "ppo-local-inference-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "workflow_id": workflow_id,
        "address": args.address,
        "brain_api": os.environ["BRAIN_API_URL"],
        "task_queue": queue,
        "run_date": args.run_date,
        "registered_activities": [
            "get_monday_decision_window",
            "get_ppo_discovery_portfolio",
            "build_ppo_discovery_state",
            "infer_ppo_discovery",
        ],
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        async with Worker(
            client,
            task_queue=queue,
            workflows=[PPOInferenceOnlyDiagnostic],
            activities=[
                get_monday_decision_window,
                get_ppo_discovery_portfolio,
                build_ppo_discovery_state,
                infer_ppo_discovery,
            ],
            activity_executor=executor,
            max_concurrent_activities=1,
            interceptors=[RecordInterceptor()],
        ):
            handle = await client.start_workflow(
                PPOInferenceOnlyDiagnostic.run,
                args.run_date,
                id=workflow_id,
                task_queue=queue,
                execution_timeout=timedelta(hours=2, minutes=10),
            )
            print("STARTED", workflow_id, flush=True)
            try:
                report["result"] = await handle.result()
                report["success"] = True
            except Exception as exc:
                report["success"] = False
                report["errors"] = []
                current = exc
                while current is not None:
                    report["errors"].append(
                        {"type": type(current).__name__, "message": str(current)}
                    )
                    current = getattr(current, "cause", None)
            history = await handle.fetch_history()
            (ROOT / "data").mkdir(exist_ok=True)
            (ROOT / "data" / "local_workflow_history.json").write_text(
                history.to_json()
            )
            report["events"] = [
                EventType.Name(event.event_type) for event in history.events
            ]
    (ROOT / "local_inference_result.json").write_text(json.dumps(report, indent=2))
    print("RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="127.0.0.1:17233")
    parser.add_argument("--run-date", default="2026-11-02")
    asyncio.run(main(parser.parse_args()))
