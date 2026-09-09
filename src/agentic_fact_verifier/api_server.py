"""API server for the agentic fact verifier.

Run:
    uv run uvicorn agentic_fact_verifier.api_server:app --reload --port 8000 --app-dir src
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agentic_fact_verifier.graph import (
    MAX_ITERATIONS,
    MAX_TOKENS_PER_RUN,
    build_graph,
    run_or_resume,
    stream_or_resume,
)
from agentic_fact_verifier.mcp_client import mcp_judge_session, mcp_retrieval_session

DEV_JSON = Path(__file__).parent.parent.parent / "data" / "raw" / "dev.json"
_claims = json.loads(DEV_JSON.read_text())

CURATED_CLAIM_IDS = [0, 1, 6, 9, 10, 13]

CHECKPOINT_DB_URL = os.environ.get(
    "CHECKPOINT_DB_URL", "postgresql://afv:afv@localhost:5432/afv_checkpoints"
)

_checkpointer: AsyncPostgresSaver | None = None


@dataclass
class RunRecord:
    run_id: str
    claim_id: int
    created_at: float = field(default_factory=time.monotonic)
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    task: asyncio.Task | None = None
    done: bool = False

    def emit(self, event: str, data: dict[str, Any]) -> None:
        message = {"id": len(self.history) + 1, "event": event, "data": data}
        self.history.append(message)
        for queue in tuple(self.subscribers):
            queue.put_nowait(message)


_runs: dict[str, RunRecord] = {}
MAX_RETAINED_RUNS = 50


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer
    async with AsyncPostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer:
        await checkpointer.setup()
        _checkpointer = checkpointer
        try:
            yield
        finally:
            tasks = [record.task for record in _runs.values() if record.task and not record.task.done()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    _checkpointer = None


app = FastAPI(lifespan=lifespan)


@app.get("/api/claims")
async def list_claims():
    return JSONResponse(
        [
            {"claim_id": cid, "claim": _claims[cid]["claim"], "gold_label": _claims[cid]["label"]}
            for cid in CURATED_CLAIM_IDS
        ]
    )


def _verification_payload(entry: dict, claim_id: int, final_state: dict) -> dict:
    return {
        "claim": entry["claim"],
        "claim_id": str(claim_id),
        "verdict": final_state["verdict"],
        "all_evidence": final_state["all_evidence"],
        "rounds": final_state["rounds"],
        "threads": [
            {
                "thread_id": thread["thread_id"],
                "question": thread["question"],
                "resolved": thread["resolved"],
                "reasoning": thread["reasoning"],
                "evidence": thread["evidence"],
            }
            for thread in final_state["threads"]
        ],
        "gold_label": entry["label"],
    }


def _should_finish(update: dict) -> bool:
    return bool(
        update["is_sufficient"]
        or update["rounds"][-1]["round"] >= MAX_ITERATIONS
        or update["total_tokens_used"] >= MAX_TOKENS_PER_RUN
    )


async def _execute_run(record: RunRecord) -> None:
    entry = _claims[record.claim_id]
    config = {"configurable": {"thread_id": f"run:{record.run_id}"}}

    try:
        async with (
            mcp_retrieval_session() as retrieve_tool,
            mcp_judge_session() as judge_tool,
        ):
            graph_app = build_graph(retrieve_tool, judge_tool, checkpointer=_checkpointer)
            async for node, update in stream_or_resume(
                graph_app, entry["claim"], str(record.claim_id), config
            ):
                if node == "__custom__":
                    event_name = update["event"]
                    record.emit(event_name, {key: value for key, value in update.items() if key != "event"})
                elif node == "decompose":
                    record.emit(
                        "decomposition_completed",
                        {
                            "threads": update["threads"],
                            "total_tokens_used": update["total_tokens_used"],
                            "max_rounds": MAX_ITERATIONS,
                        },
                    )
                    record.emit("retrieval_started", {"round": 1, "max_rounds": MAX_ITERATIONS})
                elif node == "retrieve":
                    record.emit(
                        "retrieval_completed",
                        {
                            "round": update["iteration"],
                            "threads": update["threads"],
                            "details": update["last_round_detail"],
                        },
                    )
                elif node == "check_sufficiency":
                    latest_round = update["rounds"][-1]
                    record.emit(
                        "round_completed",
                        {
                            "round": latest_round,
                            "threads": update["threads"],
                            "is_sufficient": update["is_sufficient"],
                            "total_tokens_used": update["total_tokens_used"],
                        },
                    )
                    if _should_finish(update):
                        record.emit("verdict_started", {"rounds_completed": latest_round["round"]})
                    else:
                        record.emit(
                            "retrieval_started",
                            {"round": latest_round["round"] + 1, "max_rounds": MAX_ITERATIONS},
                        )
                elif node == "__complete__":
                    record.emit(
                        "completed",
                        _verification_payload(entry, record.claim_id, update),
                    )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        record.emit("failed", {"message": f"Pipeline run failed: {exc}"})
    finally:
        record.done = True


def _prune_finished_runs() -> None:
    overflow = len(_runs) - MAX_RETAINED_RUNS + 1
    if overflow <= 0:
        return
    finished = sorted(
        (record for record in _runs.values() if record.done), key=lambda record: record.created_at
    )
    for record in finished[:overflow]:
        _runs.pop(record.run_id, None)


@app.post("/api/runs/{claim_id}", status_code=202)
async def create_run(claim_id: int):
    if not (0 <= claim_id < len(_claims)):
        raise HTTPException(404, f"claim_id {claim_id} out of range (0-{len(_claims) - 1})")

    _prune_finished_runs()
    run_id = str(uuid.uuid4())
    record = RunRecord(run_id=run_id, claim_id=claim_id)
    _runs[run_id] = record
    entry = _claims[claim_id]
    record.emit(
        "run_started",
        {
            "run_id": run_id,
            "claim_id": str(claim_id),
            "claim": entry["claim"],
            "gold_label": entry["label"],
        },
    )
    record.task = asyncio.create_task(_execute_run(record), name=f"verification-{run_id}")
    return JSONResponse({"run_id": run_id}, status_code=202)


def _format_sse(message: dict[str, Any]) -> str:
    data = json.dumps(message["data"], ensure_ascii=False, separators=(",", ":"))
    return f'id: {message["id"]}\nevent: {message["event"]}\ndata: {data}\n\n'


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request):
    record = _runs.get(run_id)
    if record is None:
        raise HTTPException(404, "Unknown or expired verification run")

    try:
        last_event_id = int(request.headers.get("last-event-id", "0"))
    except ValueError:
        last_event_id = 0

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        record.subscribers.add(queue)
        try:
            snapshot = list(record.history)
            delivered_id = last_event_id
            for message in snapshot:
                if message["id"] > delivered_id:
                    yield _format_sse(message)
                    delivered_id = message["id"]

            while not record.done or delivered_id < len(record.history):
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if message["id"] > delivered_id:
                    yield _format_sse(message)
                    delivered_id = message["id"]
        finally:
            record.subscribers.discard(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/verify/{claim_id}")
async def verify(claim_id: int):
    if not (0 <= claim_id < len(_claims)):
        raise HTTPException(404, f"claim_id {claim_id} out of range (0-{len(_claims) - 1})")
    entry = _claims[claim_id]
    config = {"configurable": {"thread_id": str(claim_id)}}

    try:
        async with (
            mcp_retrieval_session() as retrieve_tool,
            mcp_judge_session() as judge_tool,
        ):
            graph_app = build_graph(retrieve_tool, judge_tool, checkpointer=_checkpointer)
            final_state = await run_or_resume(graph_app, entry["claim"], str(claim_id), config)
    except Exception as e:
        raise HTTPException(502, f"Pipeline run failed: {e}") from e

    return JSONResponse(_verification_payload(entry, claim_id, final_state))
