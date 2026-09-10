"""MCP server exposing hybrid retrieval as a standard tool."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as FastMCPSettings

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_fact_verifier.config import read_secret  # noqa: E402
from agentic_fact_verifier.mcp_auth import SharedSecretAuthMiddleware  # noqa: E402
from agentic_fact_verifier.retrieval import INDEX_NAME, hybrid_search  # noqa: E402

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_SERVER_PORT", "8100"))
MCP_SHARED_SECRET = read_secret("MCP_SHARED_SECRET")
if not MCP_SHARED_SECRET:
    raise SystemExit("MCP_SHARED_SECRET not set, required to auth this tool server.")

FastMCPSettings.model_rebuild()
mcp = FastMCP("agentic-fact-verifier-retrieval", host=HOST, port=PORT)
_es = Elasticsearch(
    ES_URL,
    request_timeout=float(os.environ.get("ES_REQUEST_TIMEOUT_SECONDS", "30")),
    max_retries=3,
    retry_on_timeout=True,
)


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

    app = SharedSecretAuthMiddleware(mcp.streamable_http_app(), MCP_SHARED_SECRET)
    uvicorn.run(app, host=HOST, port=PORT)
