"""Client helpers for connecting to mcp_server.py and judge_server.py."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

RETRIEVAL_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8100/mcp")
JUDGE_SERVER_URL = os.environ.get("JUDGE_SERVER_URL", "http://localhost:8101/mcp")

MCP_SHARED_SECRET = os.environ.get("MCP_SHARED_SECRET")
if not MCP_SHARED_SECRET:
    raise SystemExit("MCP_SHARED_SECRET not set, required to auth to the tool servers.")
_AUTH_HEADERS = {"Authorization": f"Bearer {MCP_SHARED_SECRET}"}


@asynccontextmanager
async def mcp_retrieval_session():
    connection = {"transport": "streamable_http", "url": RETRIEVAL_SERVER_URL, "headers": _AUTH_HEADERS}
    async with create_session(connection) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        yield tools_by_name["retrieve_evidence"]


@asynccontextmanager
async def mcp_judge_session():
    connection = {"transport": "streamable_http", "url": JUDGE_SERVER_URL, "headers": _AUTH_HEADERS}
    async with create_session(connection) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        yield tools_by_name["synthesize_verdict"]
