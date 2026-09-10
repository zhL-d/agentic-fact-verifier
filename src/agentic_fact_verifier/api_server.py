"""HTTP API and durable Server-Sent Event delivery."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from agentic_fact_verifier.api_models import (
    CancelRunResponse,
    ClaimSummary,
    HealthResponse,
    RunResponse,
    StartRunResponse,
)
from agentic_fact_verifier.dataset import CURATED_CLAIM_IDS, get_claim
from agentic_fact_verifier.run_store import TERMINAL_STATUSES, RunCapacityExceeded, RunStore
from agentic_fact_verifier.worker_tasks import verify_claim

MAX_ACTIVE_RUNS = int(os.environ.get("MAX_ACTIVE_RUNS", "10"))
SSE_POLL_INTERVAL_SECONDS = float(os.environ.get("SSE_POLL_INTERVAL_SECONDS", "0.5"))
SSE_KEEPALIVE_SECONDS = float(os.environ.get("SSE_KEEPALIVE_SECONDS", "15"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = RunStore()
    await store.open()
    await store.setup()
    app.state.run_store = store
    try:
        yield
    finally:
        await store.close()


app = FastAPI(
    title="Agentic Fact Verifier API",
    version="0.2.0",
    lifespan=lifespan,
)


def _store(request: Request) -> RunStore:
    return request.app.state.run_store


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _run_response(run: dict[str, Any]) -> dict[str, Any]:
    entry = get_claim(run["claim_id"])
    return {
        "run_id": str(run["run_id"]),
        "claim_id": run["claim_id"],
        "claim": entry["claim"],
        "gold_label": entry["label"],
        "status": run["status"],
        "cancel_requested": run["cancel_requested"],
        "last_event_id": run["last_event_id"],
        "result": run["result"],
        "error": run["error"],
        "created_at": _iso(run["created_at"]),
        "started_at": _iso(run["started_at"]),
        "finished_at": _iso(run["finished_at"]),
    }


@app.get("/api/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/claims", response_model=list[ClaimSummary])
async def list_claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for claim_id in CURATED_CLAIM_IDS:
        entry = get_claim(claim_id)
        claims.append({"claim_id": claim_id, "claim": entry["claim"], "gold_label": entry["label"]})
    return claims


@app.post("/api/runs/{claim_id}", response_model=StartRunResponse, status_code=202)
async def create_run(
    claim_id: int,
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=128),
) -> dict[str, str]:
    try:
        entry = get_claim(claim_id)
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc

    try:
        run, created = await _store(request).create_run(claim_id, idempotency_key, MAX_ACTIVE_RUNS)
    except RunCapacityExceeded as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from exc

    run_id = str(run["run_id"])
    if created:
        await _store(request).append_event(
            run_id,
            "run_started",
            {
                "run_id": run_id,
                "claim_id": str(claim_id),
                "claim": entry["claim"],
                "gold_label": entry["label"],
            },
            "run_started",
        )
        try:
            await asyncio.to_thread(verify_claim.apply_async, args=[run_id], task_id=run_id)
        except Exception as exc:
            await _store(request).finish_failed(run_id, f"Could not enqueue verification: {exc}")
            raise HTTPException(503, "Verification queue is temporarily unavailable") from exc
    else:
        response.headers["Idempotent-Replayed"] = "true"

    return {"run_id": run_id, "status": run["status"]}


@app.get("/api/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    run = await _store(request).get_run(run_id)
    if run is None:
        raise HTTPException(404, "Unknown verification run")
    return _run_response(run)


@app.post("/api/runs/{run_id}/cancel", response_model=CancelRunResponse)
async def cancel_run(run_id: str, request: Request) -> dict[str, str]:
    store = _store(request)
    current = await store.get_run(run_id)
    if current is None:
        raise HTTPException(404, "Unknown verification run")
    if current["status"] in TERMINAL_STATUSES:
        return {"run_id": run_id, "status": current["status"]}

    updated = await store.request_cancel(run_id)
    if updated is None:
        latest = await store.get_run(run_id)
        return {"run_id": run_id, "status": latest["status"]}
    if updated["status"] == "cancelled":
        await store.append_event(
            run_id,
            "cancelled",
            {"message": "Verification cancelled."},
            "cancelled",
        )
    return {"run_id": run_id, "status": updated["status"]}


def _format_sse(message: dict[str, Any]) -> str:
    data = json.dumps(message["data"], ensure_ascii=False, separators=(",", ":"))
    return f"id: {message['id']}\nevent: {message['event']}\ndata: {data}\n\n"


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
    store = _store(request)
    if await store.get_run(run_id) is None:
        raise HTTPException(404, "Unknown verification run")

    try:
        last_event_id = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        last_event_id = 0

    async def event_stream():
        delivered_id = last_event_id
        last_delivery_at = asyncio.get_running_loop().time()
        while True:
            if await request.is_disconnected():
                return

            events = await store.list_events(run_id, delivered_id)
            for message in events:
                yield _format_sse(message)
                delivered_id = message["id"]
                last_delivery_at = asyncio.get_running_loop().time()

            run = await store.get_run(run_id)
            if run is None:
                return
            if run["status"] in TERMINAL_STATUSES and delivered_id >= run["last_event_id"]:
                return

            now = asyncio.get_running_loop().time()
            if now - last_delivery_at >= SSE_KEEPALIVE_SECONDS:
                yield ": keep-alive\n\n"
                last_delivery_at = now
            await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
