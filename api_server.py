"""Local web UI for the agentic fact verifier.

Run:
    uv run uvicorn web_app:app --reload
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_fact_verifier.graph import build_graph, run_verification  # noqa: E402
from agentic_fact_verifier.mcp_client import mcp_judge_session, mcp_retrieval_session  # noqa: E402

DEV_JSON = Path(__file__).parent / "data" / "raw" / "dev.json"
_claims = json.loads(DEV_JSON.read_text())

CURATED_CLAIM_IDS = [0, 1, 6, 9, 10, 13]

WEB_DIST = Path(__file__).parent / "web" / "dist"

app = FastAPI()
app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")


@app.get("/")
async def index():
    return FileResponse(WEB_DIST / "index.html")


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
        async with mcp_retrieval_session() as retrieve_tool, mcp_judge_session() as judge_tool:
            graph_app = build_graph(retrieve_tool, judge_tool)
            final_state = await run_verification(graph_app, entry["claim"], str(claim_id))
    except Exception as e:
        raise HTTPException(502, f"Pipeline run failed: {e}") from e

    payload = {
        "claim": entry["claim"],
        "claim_id": str(claim_id),
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
        "gold_label": entry["label"],
    }
    return JSONResponse(payload)
