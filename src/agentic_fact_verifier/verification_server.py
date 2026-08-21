"""MCP server exposing the full agentic fact-verification pipeline as one
composed tool.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from agentic_fact_verifier.graph import build_graph, run_verification
from agentic_fact_verifier.mcp_client import mcp_retrieval_session

mcp = FastMCP("agentic-fact-verifier-verification")


@mcp.tool()
async def verify_claim(claim: str, claim_id: str) -> str:
    """Run the full agentic fact-verification pipeline (decompose ->
    retrieve -> check sufficiency -> retry/verdict) on a single claim,
    scoped to its pre-ingested per-claim knowledge store.

    Returns a JSON-encoded object with the verdict, the per-round audit
    trail.
    """
    async with mcp_retrieval_session() as retrieve_tool:
        app = build_graph(retrieve_tool)
        final_state = await run_verification(app, claim, claim_id)

    return json.dumps(
        {
            "claim": claim,
            "claim_id": claim_id,
            "verdict": final_state["verdict"],
            "all_evidence": final_state["all_evidence"],
            "rounds": final_state["rounds"],
            "threads": [
                {
                    "question": t["question"],
                    "resolved": t["resolved"],
                    "reasoning": t["reasoning"],
                    "evidence": t["evidence"],
                }
                for t in final_state["threads"]
            ],
        }
    )


if __name__ == "__main__":
    mcp.run()
