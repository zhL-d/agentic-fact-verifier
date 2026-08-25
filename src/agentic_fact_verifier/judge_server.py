"""MCP server exposing citation-grounded verdict synthesis as a standalone tool."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from agentic_fact_verifier.mcp_auth import SharedSecretAuthMiddleware
from agentic_fact_verifier.verdict import VERDICT_LABELS, make_verdict

HOST = os.environ.get("JUDGE_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("JUDGE_SERVER_PORT", "8101"))
MCP_SHARED_SECRET = os.environ.get("MCP_SHARED_SECRET")
if not MCP_SHARED_SECRET:
    raise SystemExit("MCP_SHARED_SECRET not set, required to auth this tool server.")

mcp = FastMCP("agentic-fact-verifier-judge", host=HOST, port=PORT)

_labels_env = os.environ.get("VERDICT_LABELS")
LABELS = [label.strip() for label in _labels_env.split(",")] if _labels_env else VERDICT_LABELS


@mcp.tool()
def synthesize_verdict(
    claim: str, evidence: list[dict], sub_findings: list[dict] | None = None
) -> str:
    """Given a claim, already-gathered evidence, and optionally per-sub-
    question findings, synthesize a citation-grounded verdict.

    Args:
        claim: the claim to fact-check.
        evidence: evidence chunks, each at least {"text": str, "url": str}.
            Citations in the justification are 1-indexed into this list.
        sub_findings: optional, one entry per sub-question already
            investigated elsewhere: {"question": str, "resolved": bool,
            "finding": str}. When provided, the verdict is synthesized FROM
            these findings (not re-derived independently from raw
            evidence); when omitted, the verdict reasons directly from
            `evidence`.

    Returns:
        JSON-encoded {"label", "justification", "citations",
        "invalid_citations", "n_evidence_available", "usage"}.
    """
    verdict = make_verdict(claim, evidence, sub_findings=sub_findings, labels=LABELS)
    return json.dumps(verdict)


if __name__ == "__main__":
    import uvicorn

    app = SharedSecretAuthMiddleware(mcp.streamable_http_app(), MCP_SHARED_SECRET)
    uvicorn.run(app, host=HOST, port=PORT)
