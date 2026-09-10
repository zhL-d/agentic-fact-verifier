"""Celery task entrypoints."""

import asyncio
import os

import httpx
import psycopg
from celery.exceptions import SoftTimeLimitExceeded
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from agentic_fact_verifier.celery_app import celery_app
from agentic_fact_verifier.run_executor import RunCancelled, execute_verification_run
from agentic_fact_verifier.run_store import RunStore

RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    httpx.TransportError,
    psycopg.OperationalError,
)


async def _execute(run_id: str) -> None:
    store = RunStore()
    await store.open()
    try:
        async with store.run_lock(run_id) as acquired:
            if acquired:
                await execute_verification_run(store, run_id)
    finally:
        await store.close()


async def _mark_retry(run_id: str, attempt: int, message: str) -> None:
    store = RunStore()
    await store.open()
    try:
        await store.mark_retrying(run_id, message)
        await store.append_event(
            run_id,
            "retrying",
            {"attempt": attempt, "message": message},
            f"retrying:{attempt}",
        )
    finally:
        await store.close()


async def _mark_failed(run_id: str, message: str) -> None:
    store = RunStore()
    await store.open()
    try:
        await store.finish_failed(run_id, message)
    finally:
        await store.close()


async def _mark_cancelled(run_id: str) -> None:
    store = RunStore()
    await store.open()
    try:
        await store.finish_cancelled(run_id)
    finally:
        await store.close()


def unwrap_exception(exc: BaseException) -> BaseException:
    """Flatten a single-leaf ExceptionGroup down to the exception inside it.

    A run executes inside the MCP session context managers, which are built on
    anyio task groups, and anyio 4 wraps *every* escaping exception in an
    ExceptionGroup, including a lone one. Without unwrapping, `isinstance`
    checks below never match: a RateLimitError arrives as
    `ExceptionGroup("unhandled errors in a TaskGroup", [RateLimitError(...)])`,
    silently defeating retry classification and reporting an opaque error to
    the user. Groups carrying several distinct failures are returned unchanged
    and fall through to the generic failure path.
    """
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


@celery_app.task(bind=True, max_retries=int(os.environ.get("RUN_MAX_RETRIES", "2")))
def verify_claim(self, run_id: str) -> None:
    try:
        asyncio.run(_execute(run_id))
    except Exception as raw:
        exc = unwrap_exception(raw)

        if isinstance(exc, RunCancelled):
            asyncio.run(_mark_cancelled(run_id))
            return

        if isinstance(exc, SoftTimeLimitExceeded):
            asyncio.run(_mark_failed(run_id, "Verification exceeded its time limit."))
            raise

        if isinstance(exc, RETRYABLE_ERRORS):
            message = f"Temporary dependency failure: {type(exc).__name__}"
            if self.request.retries >= self.max_retries:
                asyncio.run(_mark_failed(run_id, message))
                raise
            attempt = self.request.retries + 1
            asyncio.run(_mark_retry(run_id, attempt, message))
            raise self.retry(exc=exc, countdown=min(60, 2**attempt)) from exc

        asyncio.run(_mark_failed(run_id, f"Pipeline run failed: {type(exc).__name__}: {exc}"))
        raise
