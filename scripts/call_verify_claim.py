"""Manual test client for the verify_claim MCP tool
Usage:
    uv run scripts/call_verify_claim.py 1 9 10
"""

import asyncio
import json
import sys
from pathlib import Path

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"
_SERVER_SCRIPT = Path(__file__).parent.parent / "src" / "agentic_fact_verifier" / "verification_server.py"


async def main():
    claim_ids = [int(a) for a in sys.argv[1:]] or [0]
    claims = json.loads(DEV_JSON.read_text())

    server_params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_SCRIPT)])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            verify_tool = {t.name: t for t in tools}["verify_claim"]

            for cid in claim_ids:
                entry = claims[cid]
                content_blocks = await verify_tool.ainvoke(
                    {"claim": entry["claim"], "claim_id": str(cid)}
                )
                result = json.loads(content_blocks[0]["text"])
                verdict = result["verdict"]

                print(f"[{cid}] claim: {entry['claim'][:80]}...")
                print(f"    gold: {entry['label']!r}  predicted: {verdict['label']!r}  "
                      f"{'✓' if verdict['label'] == entry['label'] else '✗'}")
                for t in result["threads"]:
                    status = "resolved" if t["resolved"] else "UNRESOLVED"
                    print(f"    [{status}] {t['question'][:60]} (n_evidence={t['n_evidence']})")
                print(f"    justification: {verdict['justification']}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
