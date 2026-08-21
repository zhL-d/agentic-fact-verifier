"""MCP server exposing citation-grounded verdict synthesis as a standalone tool.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from agentic_fact_verifier.verdict import make_verdict  # noqa: E402

mcp = FastMCP("agentic-fact-verifier-judge")


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
        "invalid_citations", "n_evidence_available"}.
    """
    verdict = make_verdict(claim, evidence, sub_findings=sub_findings)
    return json.dumps(verdict)


if __name__ == "__main__":
    mcp.run()
