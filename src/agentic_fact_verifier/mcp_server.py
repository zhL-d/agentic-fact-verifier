"""MCP server exposing hybrid retrieval as a standard tool.
"""

import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_fact_verifier.retrieval import hybrid_search

mcp = FastMCP("agentic-fact-verifier-retrieval")
_es = Elasticsearch("http://localhost:9200")


@mcp.tool()
def retrieve_evidence(claim_id: str, query: str, top_k: int = 5) -> str:
    """Retrieve top_k evidence chunks for `query`.

    Returns a JSON-encoded list of {text, url, source_type, source_query}
    chunks.
    """
    chunks = hybrid_search(_es, claim_id, query, top_k=top_k)
    return json.dumps(chunks)


if __name__ == "__main__":
    mcp.run()
