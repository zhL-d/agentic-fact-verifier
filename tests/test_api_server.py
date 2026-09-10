"""Contract tests for the durable run API and SSE replay."""

from datetime import UTC, datetime
from unittest.mock import Mock

import httpx
import pytest

import agentic_fact_verifier.api_server as api
from agentic_fact_verifier.run_store import RunCapacityExceeded

RUN_ID = "00000000-0000-4000-8000-000000000123"


def _run(**overrides):
    now = datetime.now(UTC)
    value = {
        "run_id": RUN_ID,
        "claim_id": 0,
        "thread_id": f"run:{RUN_ID}",
        "idempotency_key": None,
        "celery_task_id": RUN_ID,
        "status": "queued",
        "cancel_requested": False,
        "last_event_id": 0,
        "result": None,
        "error": None,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "updated_at": now,
    }
    value.update(overrides)
    return value


class FakeRunStore:
    def __init__(self, run=None, created=True, events=None):
        self.run = run or _run()
        self.created = created
        self.events = events or []
        self.appended = []

    async def create_run(self, claim_id, idempotency_key, max_active_runs):
        return self.run, self.created

    async def append_event(self, run_id, event_type, data, dedupe_key=None):
        self.appended.append((event_type, data, dedupe_key))
        self.run["last_event_id"] += 1
        return self.run["last_event_id"]

    async def finish_failed(self, run_id, message):
        self.run.update(status="failed", error=message)

    async def get_run(self, run_id):
        return self.run if run_id == RUN_ID else None

    async def request_cancel(self, run_id):
        self.run.update(status="cancelled", cancel_requested=True)
        return self.run

    async def list_events(self, run_id, after):
        return [event for event in self.events if event["id"] > after]


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as value:
        yield value


async def test_create_run_is_idempotent_and_dispatches_once(client, monkeypatch):
    store = FakeRunStore()
    api.app.state.run_store = store
    dispatch = Mock()
    monkeypatch.setattr(api.verify_claim, "apply_async", dispatch)

    response = await client.post("/api/runs/0", headers={"Idempotency-Key": "browser-request-1"})

    assert response.status_code == 202
    assert response.json() == {"run_id": RUN_ID, "status": "queued"}
    dispatch.assert_called_once_with(args=[RUN_ID], task_id=RUN_ID)
    assert store.appended[0][0] == "run_started"

    store.created = False
    response = await client.post("/api/runs/0", headers={"Idempotency-Key": "browser-request-1"})
    assert response.headers["Idempotent-Replayed"] == "true"
    dispatch.assert_called_once()


async def test_create_run_returns_429_when_capacity_is_full(client, monkeypatch):
    class FullStore(FakeRunStore):
        async def create_run(self, claim_id, idempotency_key, max_active_runs):
            raise RunCapacityExceeded("queue full")

    api.app.state.run_store = FullStore()
    response = await client.post("/api/runs/0")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "10"


async def test_get_run_returns_validated_status(client):
    api.app.state.run_store = FakeRunStore(_run(status="running", last_event_id=4))
    response = await client.get(f"/api/runs/{RUN_ID}")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["last_event_id"] == 4


async def test_cancel_queued_run_emits_terminal_event(client):
    store = FakeRunStore()
    api.app.state.run_store = store
    response = await client.post(f"/api/runs/{RUN_ID}/cancel")
    assert response.json() == {"run_id": RUN_ID, "status": "cancelled"}
    assert store.appended[-1] == (
        "cancelled",
        {"message": "Verification cancelled."},
        "cancelled",
    )


async def test_sse_replays_only_events_after_last_event_id(client):
    events = [
        {"id": 1, "event": "run_started", "data": {"claim": "old"}},
        {"id": 2, "event": "failed", "data": {"message": "boom"}},
    ]
    api.app.state.run_store = FakeRunStore(
        _run(status="failed", last_event_id=2, error="boom"), events=events
    )

    response = await client.get(f"/api/runs/{RUN_ID}/events", headers={"Last-Event-ID": "1"})
    assert response.status_code == 200
    assert "id: 1" not in response.text
    assert "id: 2\nevent: failed" in response.text
    assert 'data: {"message":"boom"}' in response.text

