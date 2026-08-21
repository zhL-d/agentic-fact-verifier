"""Local web UI for the agentic fact verifier.

Run:
    uv run uvicorn web_app:app --reload
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent / "src"))

DEV_JSON = Path(__file__).parent / "data" / "raw" / "dev.json"
_claims = json.loads(DEV_JSON.read_text())

_SERVER_SCRIPT = Path(__file__).parent / "src" / "agentic_fact_verifier" / "verification_server.py"

CURATED_CLAIM_IDS = [0, 1, 6, 9, 10, 13]

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/api/claims")
async def list_claims():
    return JSONResponse(
        [
            {"claim_id": cid, "claim": _claims[cid]["claim"], "gold_label": _claims[cid]["label"]}
            for cid in CURATED_CLAIM_IDS
        ]
    )


@app.post("/api/verify/{claim_id}")
async def verify(claim_id: int):
    if not (0 <= claim_id < len(_claims)):
        raise HTTPException(404, f"claim_id {claim_id} out of range (0-{len(_claims) - 1})")
    entry = _claims[claim_id]

    try:
        server_params = StdioServerParameters(command=sys.executable, args=[str(_SERVER_SCRIPT)])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "verify_claim", {"claim": entry["claim"], "claim_id": str(claim_id)}
                )
                if result.isError:
                    detail = result.content[0].text if result.content else "unknown tool error"
                    raise HTTPException(502, f"verify_claim tool call failed: {detail}")
                payload = json.loads(result.content[0].text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Pipeline run failed: {e}") from e

    payload["gold_label"] = entry["label"]
    return JSONResponse(payload)
