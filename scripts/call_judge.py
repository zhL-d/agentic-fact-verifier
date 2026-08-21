"""Manual test client for the synthesize_verdict MCP tool.

Usage:
    uv run scripts/call_judge.py 1 9 10
"""

import asyncio
import json
import sys
from pathlib import Path

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentic_fact_verifier.mcp_client import mcp_retrieval_session  # noqa: E402

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"
_JUDGE_SERVER_SCRIPT = (
    Path(__file__).parent.parent / "src" / "agentic_fact_verifier" / "judge_server.py"
)


async def main():
    claim_ids = [int(a) for a in sys.argv[1:]] or [0]
    claims = json.loads(DEV_JSON.read_text())

    judge_params = StdioServerParameters(command=sys.executable, args=[str(_JUDGE_SERVER_SCRIPT)])
    async with (
        mcp_retrieval_session() as retrieve_tool,
        stdio_client(judge_params) as (read, write),
        ClientSession(read, write) as judge_session,
    ):
        await judge_session.initialize()
        judge_tools = await load_mcp_tools(judge_session)
        synthesize_tool = {t.name: t for t in judge_tools}["synthesize_verdict"]

        for cid in claim_ids:
            entry = claims[cid]
            claim = entry["claim"]

            content_blocks = await retrieve_tool.ainvoke(
                {"claim_id": str(cid), "query": claim, "top_k": 5}
            )
            evidence = json.loads(content_blocks[0]["text"])

            if not evidence:
                print(f"[{cid}] SKIPPED, no evidence retrieved (claim not indexed yet?)\n")
                continue

            verdict_blocks = await synthesize_tool.ainvoke({"claim": claim, "evidence": evidence})
            verdict = json.loads(verdict_blocks[0]["text"])

            print(f"[{cid}] claim: {claim[:80]}...")
            print(
                f"    gold: {entry['label']!r}  predicted: {verdict['label']!r}  "
                f"{'✓' if verdict['label'] == entry['label'] else '✗'}"
            )
            print(f"    evidence used: {len(evidence)}  citations: {verdict['citations']}")
            print(f"    justification: {verdict['justification']}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
