"""MCP server exposing hybrid retrieval as a standard tool.

"""

import json
import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_fact_verifier.retrieval import INDEX_NAME, _get_model, hybrid_search  # noqa: E402

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_SERVER_PORT", "8100"))

mcp = FastMCP("agentic-fact-verifier-retrieval", host=HOST, port=PORT)
_es = Elasticsearch(ES_URL)


@mcp.tool()
def retrieve_evidence(claim_id: str, query: str, top_k: int = 5) -> str:
    """Retrieve top_k evidence chunks for `query`.

    Returns a JSON-encoded list of {text, url, source_type, source_query}
    chunks.
    """
    chunks = hybrid_search(_es, claim_id, query, top_k=top_k, index_name=INDEX_NAME)
    return json.dumps(chunks)


if __name__ == "__main__":
    _get_model()
    mcp.run(transport="streamable-http")
