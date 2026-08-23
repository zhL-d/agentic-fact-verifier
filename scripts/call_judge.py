"""Manual test client for the synthesize_verdict MCP tool (judge_server.py).


Usage:
    uv run scripts/call_judge.py 1 9 10
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentic_fact_verifier.mcp_client import mcp_judge_session, mcp_retrieval_session  # noqa: E402

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"


async def main():
    claim_ids = [int(a) for a in sys.argv[1:]] or [0]
    claims = json.loads(DEV_JSON.read_text())

    async with mcp_retrieval_session() as retrieve_tool, mcp_judge_session() as synthesize_tool:
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
