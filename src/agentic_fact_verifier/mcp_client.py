"""Client helpers for connecting to mcp_server.py and judge_server.py.
"""

import os
from contextlib import asynccontextmanager

from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

RETRIEVAL_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8100/mcp")
JUDGE_SERVER_URL = os.environ.get("JUDGE_SERVER_URL", "http://localhost:8101/mcp")


@asynccontextmanager
async def mcp_retrieval_session():
    connection = {"transport": "streamable_http", "url": RETRIEVAL_SERVER_URL}
    async with create_session(connection) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        yield tools_by_name["retrieve_evidence"]


@asynccontextmanager
async def mcp_judge_session():
    connection = {"transport": "streamable_http", "url": JUDGE_SERVER_URL}
    async with create_session(connection) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        tools_by_name = {t.name: t for t in tools}
        yield tools_by_name["synthesize_verdict"]
