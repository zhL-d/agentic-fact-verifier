"""Unit tests for worker_tasks.py's failure classification.

A run executes inside the MCP session context managers, which are anyio task
groups, and anyio 4 wraps every escaping exception in an ExceptionGroup — even
a single one. These tests pin the unwrapping that keeps `isinstance`-based
retry/cancel classification working through that wrapper.
"""

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded
from openai import RateLimitError

import agentic_fact_verifier.worker_tasks as worker_tasks
from agentic_fact_verifier.run_executor import RunCancelled
from agentic_fact_verifier.worker_tasks import unwrap_exception, verify_claim

RUN_ID = "11111111-1111-1111-1111-111111111111"


def _rate_limit_error() -> RateLimitError:
    """RateLimitError needs a response/body, it can't be constructed bare."""
    import httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


@pytest.fixture
def recorder(monkeypatch):
    """Replace the terminal-state writers with async recorders, and `retry`
    with a stub, so no Postgres or broker is needed."""
    calls: dict[str, list] = {"cancelled": [], "failed": [], "retrying": [], "retry": []}

    async def mark_cancelled(run_id):
        calls["cancelled"].append(run_id)

    async def mark_failed(run_id, message):
        calls["failed"].append((run_id, message))

    async def mark_retry(run_id, attempt, message):
        calls["retrying"].append((run_id, attempt, message))

    def fake_retry(exc=None, countdown=None, **kwargs):
        calls["retry"].append((type(exc).__name__, countdown))
        return Retry("retry scheduled")

    monkeypatch.setattr(worker_tasks, "_mark_cancelled", mark_cancelled)
    monkeypatch.setattr(worker_tasks, "_mark_failed", mark_failed)
    monkeypatch.setattr(worker_tasks, "_mark_retry", mark_retry)
    monkeypatch.setattr(verify_claim, "retry", fake_retry)
    return calls


def _run_raising(monkeypatch, exc: BaseException):
    async def execute(run_id):
        raise exc

    monkeypatch.setattr(worker_tasks, "_execute", execute)
    return verify_claim.apply(args=[RUN_ID])


def test_unwrap_flattens_a_single_leaf_through_nested_groups():
    leaf = ValueError("the real cause")
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [leaf])])
    assert unwrap_exception(nested) is leaf


def test_unwrap_leaves_a_plain_exception_untouched():
    leaf = ValueError("the real cause")
    assert unwrap_exception(leaf) is leaf


def test_unwrap_keeps_a_group_carrying_several_failures():
    group = ExceptionGroup("outer", [ValueError("a"), KeyError("b")])
    assert unwrap_exception(group) is group


def test_wrapped_retryable_error_still_schedules_a_retry(monkeypatch, recorder):
    """Before unwrapping, this arrived as an ExceptionGroup, missed
    RETRYABLE_ERRORS, and failed the run permanently instead of retrying."""
    group = ExceptionGroup("unhandled errors in a TaskGroup", [_rate_limit_error()])
    _run_raising(monkeypatch, group)

    assert recorder["retry"] == [("RateLimitError", 2)]
    assert recorder["retrying"] == [(RUN_ID, 1, "Temporary dependency failure: RateLimitError")]
    assert recorder["failed"] == []


def test_unwrapped_retryable_error_still_schedules_a_retry(monkeypatch, recorder):
    _run_raising(monkeypatch, _rate_limit_error())

    assert recorder["retry"] == [("RateLimitError", 2)]
    assert recorder["failed"] == []


def test_wrapped_cancellation_marks_the_run_cancelled(monkeypatch, recorder):
    group = ExceptionGroup("unhandled errors in a TaskGroup", [RunCancelled("cancelled by user")])
    _run_raising(monkeypatch, group)

    assert recorder["cancelled"] == [RUN_ID]
    assert recorder["failed"] == []


def test_wrapped_soft_time_limit_marks_the_run_failed(monkeypatch, recorder):
    group = ExceptionGroup("unhandled errors in a TaskGroup", [SoftTimeLimitExceeded()])
    _run_raising(monkeypatch, group)

    assert recorder["failed"] == [(RUN_ID, "Verification exceeded its time limit.")]
    assert recorder["retry"] == []


def test_wrapped_unexpected_error_reports_the_inner_type_not_the_group(monkeypatch, recorder):
    group = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("index is missing")])
    _run_raising(monkeypatch, group)

    assert recorder["failed"] == [(RUN_ID, "Pipeline run failed: RuntimeError: index is missing")]


def test_retry_budget_exhausted_fails_instead_of_retrying(monkeypatch, recorder):
    monkeypatch.setattr(verify_claim, "max_retries", 0)
    group = ExceptionGroup("unhandled errors in a TaskGroup", [_rate_limit_error()])
    _run_raising(monkeypatch, group)

    assert recorder["retry"] == []
    assert recorder["failed"] == [(RUN_ID, "Temporary dependency failure: RateLimitError")]
