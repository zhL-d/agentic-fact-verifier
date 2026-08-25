"""One arm of the evidence-compaction comparison (full pipeline, with vs.
without compaction), against the same claim IDs and model, so the two arms
differ only in whether MAX_EVIDENCE_CHARS_PER_THREAD is enforced.

Usage:
    uv run scripts/run_compaction_comparison.py --arm full_compact_on --claim-ids 0-9
    uv run scripts/run_compaction_comparison.py --arm full_compact_off --claim-ids 0-9
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=["full_compact_on", "full_compact_off"])
parser.add_argument("--claim-ids", required=True, help="e.g. '0-9' or '0,1,2,5,9'")
args = parser.parse_args()

if args.arm == "full_compact_off":
    os.environ["MAX_EVIDENCE_CHARS_PER_THREAD"] = "100000000"

import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"


def _parse_claim_ids(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def _dedupe_for_report(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for chunk in chunks:
        key = chunk["text"]
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out


async def run_full_pipeline(claim_ids, all_claims) -> list[dict]:
    from agentic_fact_verifier.graph import MAX_ITERATIONS, build_graph, run_verification
    from agentic_fact_verifier.mcp_client import mcp_judge_session, mcp_retrieval_session

    results = []
    async with mcp_retrieval_session() as retrieve_tool, mcp_judge_session() as judge_tool:
        app = build_graph(retrieve_tool, judge_tool)
        for i in claim_ids:
            claim_entry = all_claims[i]
            claim, gold_label, claim_id = claim_entry["claim"], claim_entry["label"], str(i)
            try:
                final_state = await run_verification(app, claim, claim_id)
            except Exception as e:
                print(f"[{claim_id}] SKIPPED, pipeline failed: {e}")
                results.append({"claim_id": claim_id, "claim": claim, "gold_label": gold_label, "error": str(e)})
                continue
            verdict = final_state["verdict"]
            if verdict is None:
                print(f"[{claim_id}] SKIPPED — no verdict produced")
                results.append({"claim_id": claim_id, "claim": claim, "gold_label": gold_label, "error": "no verdict"})
                continue
            is_correct = verdict["label"] == gold_label
            was_forced = (not final_state["is_sufficient"]) and final_state["iteration"] >= MAX_ITERATIONS
            all_evidence = _dedupe_for_report([chunk for t in final_state["threads"] for chunk in t["evidence"]])
            print(f"[{claim_id}] gold: {gold_label!r}  predicted: {verdict['label']!r}  {'✓' if is_correct else '✗'}"
                  f"  rounds: {final_state['iteration']}  evidence: {len(all_evidence)}")
            results.append(
                {
                    "claim_id": claim_id,
                    "claim": claim,
                    "sub_questions": [t["question"] for t in final_state["threads"]],
                    "n_rounds": final_state["iteration"],
                    "rounds_detail": final_state["rounds"],
                    "gold_label": gold_label,
                    "predicted_label": verdict["label"],
                    "justification": verdict["justification"],
                    "citations": verdict["citations"],
                    "cited_sources": verdict["cited_sources"],
                    "invalid_citations": verdict.get("invalid_citations", []),
                    "correct": is_correct,
                    "n_evidence_used": len(all_evidence),
                    "evidence": all_evidence,
                    "evidence_incomplete": was_forced,
                    "unresolved_questions": [t["question"] for t in final_state["threads"] if not t["resolved"]],
                    "total_tokens_used": final_state.get("total_tokens_used", 0),
                    "total_prompt_tokens": final_state.get("total_prompt_tokens", 0),
                    "total_completion_tokens": final_state.get("total_completion_tokens", 0),
                    "escalate": verdict.get("escalate", False),
                    "escalation_reasons": verdict.get("escalation_reasons", []),
                }
            )
    return results


def main():
    claim_ids = _parse_claim_ids(args.claim_ids)
    all_claims = json.loads(DEV_JSON.read_text())

    out_path = Path(__file__).parent.parent / "eval" / f"openai_terra_{args.arm}.json"
    results = asyncio.run(run_full_pipeline(claim_ids, all_claims))

    n = sum(1 for r in results if "correct" in r)
    correct = sum(1 for r in results if r.get("correct"))
    n_errors = sum(1 for r in results if "error" in r)
    if n:
        print(f"\n--- arm={args.arm}: {correct}/{n} correct ({correct/n*100:.1f}%), {n_errors} errors ---")
    else:
        print("No results.")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Written {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
