"""Unit tests for mcp_auth.py's shared-secret ASGI middleware."""

import pytest

from agentic_fact_verifier.mcp_auth import SharedSecretAuthMiddleware

SECRET = "test-secret-123"


class RecordingApp:
    """Fake inner ASGI app: records whether it was called."""

    def __init__(self):
        self.called = False
        self.scope = None

    async def __call__(self, scope, receive, send):
        self.called = True
        self.scope = scope


def _http_scope(headers: dict[str, str] | None = None, path: str = "/mcp") -> dict:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return {"type": "http", "headers": raw_headers, "path": path}


async def _noop_receive():
    return {"type": "http.request"}


class _CapturingSend:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-secret"},
        {"Authorization": SECRET},
    ],
)
async def test_rejects_missing_or_wrong_credential(headers):
    inner = RecordingApp()
    middleware = SharedSecretAuthMiddleware(inner, SECRET)
    send = _CapturingSend()

    await middleware(_http_scope(headers), _noop_receive, send)

    assert not inner.called, "inner app must never run without a valid credential"
    assert send.messages[0]["status"] == 401


async def test_accepts_correct_bearer_token():
    inner = RecordingApp()
    middleware = SharedSecretAuthMiddleware(inner, SECRET)
    send = _CapturingSend()

    await middleware(_http_scope({"Authorization": f"Bearer {SECRET}"}), _noop_receive, send)

    assert inner.called, "inner app must run once the credential is valid"
    assert send.messages == [], "middleware itself sends nothing on the happy path"


async def test_lifespan_scope_always_passes_through_untouched():
    """Lifespan (session-manager startup/shutdown) must never be gated by auth."""
    inner = RecordingApp()
    middleware = SharedSecretAuthMiddleware(inner, SECRET)

    await middleware({"type": "lifespan"}, _noop_receive, _CapturingSend())

    assert inner.called
    assert inner.scope == {"type": "lifespan"}


async def test_health_endpoint_is_available_without_a_secret():
    inner = RecordingApp()
    middleware = SharedSecretAuthMiddleware(inner, SECRET)
    send = _CapturingSend()

    await middleware(_http_scope(path="/health"), _noop_receive, send)

    assert not inner.called
    assert send.messages[0]["status"] == 200
