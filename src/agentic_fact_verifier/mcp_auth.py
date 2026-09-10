"""Shared-secret auth for the MCP tool servers (mcp_server.py, judge_server.py)."""

from starlette.responses import JSONResponse


class SharedSecretAuthMiddleware:
    """Rejects any HTTP request missing `Authorization: Bearer <secret>`
    with 401. Only intercepts `scope["type"] == "http"`; other scope types
    (e.g. lifespan) pass through untouched."""

    def __init__(self, app, secret: str):
        self.app = app
        self.secret = secret

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path") == "/health":
            response = JSONResponse({"status": "ok"})
            await response(scope, receive, send)
            return

        headers = dict(scope["headers"])
        auth_header = headers.get(b"authorization", b"").decode()
        if auth_header != f"Bearer {self.secret}":
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
