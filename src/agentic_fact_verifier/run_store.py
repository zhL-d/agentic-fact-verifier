"""Durable run and event storage backed by PostgreSQL."""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

CHECKPOINT_DB_URL = os.environ.get("CHECKPOINT_DB_URL", "postgresql://afv:afv@localhost:5432/afv_checkpoints")
MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"
TERMINAL_STATUSES = frozenset({"cancelled", "succeeded", "failed"})


class RunCapacityExceeded(RuntimeError):
    pass


class RunStore:
    def __init__(self, connection_string: str = CHECKPOINT_DB_URL) -> None:
        self.pool = AsyncConnectionPool(
            connection_string,
            min_size=1,
            max_size=int(os.environ.get("RUN_DB_POOL_SIZE", "10")),
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )

    async def open(self) -> None:
        await self.pool.open()
        await self.pool.wait()

    async def close(self) -> None:
        await self.pool.close()

    async def setup(self) -> None:
        async with self.pool.connection() as connection:
            for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
                await connection.execute(migration.read_text())

    async def create_run(
        self,
        claim_id: int,
        idempotency_key: str | None,
        max_active_runs: int,
    ) -> tuple[dict[str, Any], bool]:
        run_id = str(uuid.uuid4())
        thread_id = f"run:{run_id}"
        async with self.pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(8675309)")
            if idempotency_key:
                existing = await (
                    await connection.execute(
                        "SELECT * FROM verification_runs WHERE claim_id = %s AND idempotency_key = %s",
                        (claim_id, idempotency_key),
                    )
                ).fetchone()
                if existing:
                    return existing, False

            active = await (
                await connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM verification_runs
                    WHERE status IN ('queued', 'running', 'retrying', 'cancelling')
                    """
                )
            ).fetchone()
            if active["count"] >= max_active_runs:
                raise RunCapacityExceeded(f"Maximum of {max_active_runs} active runs reached")

            row = await (
                await connection.execute(
                    """
                    INSERT INTO verification_runs (
                        run_id, claim_id, thread_id, idempotency_key, celery_task_id, status
                    ) VALUES (%s, %s, %s, %s, %s, 'queued')
                    RETURNING *
                    """,
                    (run_id, claim_id, thread_id, idempotency_key, run_id),
                )
            ).fetchone()
            return row, True

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as connection:
            return await (
                await connection.execute("SELECT * FROM verification_runs WHERE run_id = %s", (run_id,))
            ).fetchone()

    async def list_events(self, run_id: str, after: int, limit: int = 100) -> list[dict[str, Any]]:
        async with self.pool.connection() as connection:
            return await (
                await connection.execute(
                    """
                    SELECT sequence AS id, event_type AS event, event_data AS data
                    FROM verification_run_events
                    WHERE run_id = %s AND sequence > %s
                    ORDER BY sequence
                    LIMIT %s
                    """,
                    (run_id, after, limit),
                )
            ).fetchall()

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        dedupe_key: str | None = None,
    ) -> int:
        async with self.pool.connection() as connection, connection.transaction():
            if dedupe_key:
                existing = await (
                    await connection.execute(
                        """
                        SELECT sequence FROM verification_run_events
                        WHERE run_id = %s AND dedupe_key = %s
                        """,
                        (run_id, dedupe_key),
                    )
                ).fetchone()
                if existing:
                    return existing["sequence"]

            run = await (
                await connection.execute(
                    """
                    UPDATE verification_runs
                    SET last_event_id = last_event_id + 1, updated_at = NOW()
                    WHERE run_id = %s
                    RETURNING last_event_id
                    """,
                    (run_id,),
                )
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown run {run_id}")
            sequence = run["last_event_id"]
            await connection.execute(
                """
                INSERT INTO verification_run_events (
                    run_id, sequence, event_type, event_data, dedupe_key
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, sequence, event_type, Jsonb(data), dedupe_key),
            )
            return sequence

    async def mark_running(self, run_id: str) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                UPDATE verification_runs
                SET status = 'running', started_at = COALESCE(started_at, NOW()), updated_at = NOW(), error = NULL
                WHERE run_id = %s AND status NOT IN ('cancelled', 'succeeded', 'failed')
                """,
                (run_id,),
            )

    async def mark_retrying(self, run_id: str, message: str) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                UPDATE verification_runs SET status = 'retrying', error = %s, updated_at = NOW()
                WHERE run_id = %s AND status IN ('running', 'retrying')
                """,
                (message, run_id),
            )

    async def finish_success(self, run_id: str, result: dict[str, Any]) -> None:
        await self._finish(run_id, "succeeded", "completed", result, result=result)

    async def finish_failed(self, run_id: str, message: str) -> None:
        await self._finish(run_id, "failed", "failed", {"message": message}, error=message)

    async def request_cancel(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as connection:
            return await (
                await connection.execute(
                    """
                    UPDATE verification_runs
                    SET cancel_requested = TRUE,
                        status = CASE WHEN status IN ('queued', 'retrying') THEN 'cancelled' ELSE 'cancelling' END,
                        finished_at = CASE WHEN status IN ('queued', 'retrying') THEN NOW() ELSE finished_at END,
                        updated_at = NOW()
                    WHERE run_id = %s AND status NOT IN ('cancelled', 'succeeded', 'failed')
                    RETURNING *
                    """,
                    (run_id,),
                )
            ).fetchone()

    async def finish_cancelled(self, run_id: str) -> None:
        await self._finish(
            run_id,
            "cancelled",
            "cancelled",
            {"message": "Verification cancelled."},
        )

    async def _finish(
        self,
        run_id: str,
        status: str,
        event_type: str,
        event_data: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Atomically persist the terminal event and status; a requested cancellation wins."""
        async with self.pool.connection() as connection, connection.transaction():
            run = await (
                await connection.execute(
                    """
                    SELECT status, cancel_requested, last_event_id
                    FROM verification_runs WHERE run_id = %s FOR UPDATE
                    """,
                    (run_id,),
                )
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown run {run_id}")
            if run["status"] in TERMINAL_STATUSES:
                return

            if run["cancel_requested"] and status != "cancelled":
                status = "cancelled"
                event_type = "cancelled"
                event_data = {"message": "Verification cancelled."}
                result = None
                error = None

            sequence = run["last_event_id"] + 1
            await connection.execute(
                """
                INSERT INTO verification_run_events (
                    run_id, sequence, event_type, event_data, dedupe_key
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, sequence, event_type, Jsonb(event_data), event_type),
            )
            await connection.execute(
                """
                UPDATE verification_runs
                SET status = %s, cancel_requested = cancel_requested OR %s,
                    last_event_id = %s, result = %s, error = %s,
                    finished_at = NOW(), updated_at = NOW()
                WHERE run_id = %s
                """,
                (
                    status,
                    status == "cancelled",
                    sequence,
                    Jsonb(result) if result is not None else None,
                    error,
                    run_id,
                ),
            )

    async def is_cancel_requested(self, run_id: str) -> bool:
        run = await self.get_run(run_id)
        return bool(run and run["cancel_requested"])

    @asynccontextmanager
    async def run_lock(self, run_id: str) -> AsyncIterator[bool]:
        async with self.pool.connection() as connection:
            row = await (
                await connection.execute("SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired", (run_id,))
            ).fetchone()
            acquired = bool(row["acquired"])
            try:
                yield acquired
            finally:
                if acquired:
                    await connection.execute("SELECT pg_advisory_unlock(hashtext(%s))", (run_id,))
