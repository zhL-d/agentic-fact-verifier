import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"
_JUDGE_SERVER_SCRIPT = Path(__file__).parent / "judge_server.py"


@asynccontextmanager
async def mcp_retrieval_session():
    server_params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_SCRIPT)])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tools_by_name = {t.name: t for t in tools}
            yield tools_by_name["retrieve_evidence"]


@asynccontextmanager
async def mcp_judge_session():
    server_params = StdioServerParameters(
        command=sys.executable, args=[str(_JUDGE_SERVER_SCRIPT)]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tools_by_name = {t.name: t for t in tools}
            yield tools_by_name["synthesize_verdict"]
