"""MCP server exposing hybrid retrieval as a standard tool."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from mcp.server.fastmcp import FastMCP

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_fact_verifier.mcp_auth import SharedSecretAuthMiddleware
from agentic_fact_verifier.retrieval import INDEX_NAME, _get_model, hybrid_search

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_SERVER_PORT", "8100"))
MCP_SHARED_SECRET = os.environ.get("MCP_SHARED_SECRET")
if not MCP_SHARED_SECRET:
    raise SystemExit("MCP_SHARED_SECRET not set, required to auth this tool server.")

mcp = FastMCP("agentic-fact-verifier-retrieval", host=HOST, port=PORT)
_es = Elasticsearch(ES_URL)


@mcp.tool()
def retrieve_evidence(claim_id: str, query: str, top_k: int = 5) -> str:
    """Retrieve top_k evidence chunks for `query`.

    Returns a JSON-encoded list of {text, url, source_type, source_query,
    injection_markers} chunks.
    """
    chunks = hybrid_search(_es, claim_id, query, top_k=top_k, index_name=INDEX_NAME)
    return json.dumps(chunks)


if __name__ == "__main__":
    import uvicorn

    _get_model()
    app = SharedSecretAuthMiddleware(mcp.streamable_http_app(), MCP_SHARED_SECRET)
    uvicorn.run(app, host=HOST, port=PORT)
