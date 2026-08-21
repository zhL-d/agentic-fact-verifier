"""The evidence sufficiency loop, via LangGraph. claim -> decompose
-> retrieve -> check sufficiency -> (loop back with refined queries, or
proceed) -> verdict.

Usage:
    uv run scripts/run_sufficiency_loop.py --limit 15
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentic_fact_verifier.graph import MAX_ITERATIONS, build_graph, run_verification  # noqa: E402
from agentic_fact_verifier.mcp_client import mcp_judge_session, mcp_retrieval_session  # noqa: E402

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    es = Elasticsearch("http://localhost:9200")
    if not es.ping():
        raise SystemExit("Can't reach Elasticsearch — is `docker compose up -d` running?")

    claims = json.loads(DEV_JSON.read_text())[: args.limit]

    async with mcp_retrieval_session() as retrieve_tool, mcp_judge_session() as judge_tool:
        app = build_graph(retrieve_tool, judge_tool)
        results = await _run_claims(app, claims)

    correct = sum(1 for r in results if r.get("correct"))
    n = sum(1 for r in results if "correct" in r)
    n_errors = sum(1 for r in results if "error" in r)
    if n:
        print(
            f"--- Phase 3 (sufficiency loop) accuracy: {correct}/{n} ({correct/n*100:.1f}%), "
            f"{n_errors} skipped due to errors ---"
        )
    else:
        print("No results.")

    out_path = Path(__file__).parent.parent / "eval" / "sufficiency_loop_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


async def _run_claims(app, claims) -> list[dict]:
    results = []
    for i, claim_entry in enumerate(claims):
        claim = claim_entry["claim"]
        gold_label = claim_entry["label"]
        claim_id = str(i)

        try:
            final_state = await run_verification(app, claim, claim_id)
        except Exception as e:
            print(f"[{i}] SKIPPED — pipeline failed: {e}\n")
            results.append(
                {"claim_id": claim_id, "claim": claim, "gold_label": gold_label, "error": str(e)}
            )
            continue

        verdict = final_state["verdict"]
        if verdict is None:
            print(f"[{i}] SKIPPED — no verdict produced (likely no evidence retrieved)\n")
            results.append(
                {"claim_id": claim_id, "claim": claim, "gold_label": gold_label, "error": "no verdict"}
            )
            continue

        is_correct = verdict["label"] == gold_label
        was_forced = (not final_state["is_sufficient"]) and final_state["iteration"] >= MAX_ITERATIONS

        all_evidence = _dedupe_for_report(
            [chunk for t in final_state["threads"] for chunk in t["evidence"]]
        )

        print(f"[{i}] claim: {claim[:80]}...")
        print(f"    rounds: {final_state['iteration']}  evidence used: {len(all_evidence)}")
        for round_entry in final_state["rounds"]:
            for t in round_entry["threads"]:
                status = "resolved" if t["resolved"] else "unresolved"
                print(f"    round {round_entry['round']} [{t['question'][:60]}]: {status} — {t['reasoning']}")
        print(f"    citations: {verdict['citations']}", end="")
        if verdict.get("invalid_citations"):
            print(f"  ⚠ INVALID (out of range): {verdict['invalid_citations']}", end="")
        print()
        print()

        results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "sub_questions": [t["question"] for t in final_state["threads"]],
                "n_rounds": final_state["iteration"],
                "rounds_detail": final_state["rounds"],  # full per-round audit trail
                "gold_label": gold_label,
                "predicted_label": verdict["label"],
                "justification": verdict["justification"],
                "citations": verdict["citations"],
                "cited_sources": verdict["cited_sources"],
                "invalid_citations": verdict.get("invalid_citations", []),
                "correct": is_correct,
                "n_evidence_used": len(all_evidence),
                "evidence": all_evidence,  # flattened
                "evidence_incomplete": was_forced,
                "unresolved_questions": [t["question"] for t in final_state["threads"] if not t["resolved"]],
            }
        )

    return results


def _dedupe_for_report(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for chunk in chunks:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            out.append(chunk)
    return out


if __name__ == "__main__":
    asyncio.run(main())
