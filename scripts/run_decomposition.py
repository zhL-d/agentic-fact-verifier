"""Claim decomposition.

Usage:
    uv run scripts/run_decomposition.py --limit 3
"""

import argparse
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentic_fact_verifier.decomposition import decompose_claim  # noqa: E402
from agentic_fact_verifier.retrieval import hybrid_search  # noqa: E402
from agentic_fact_verifier.verdict import make_verdict  # noqa: E402

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for chunk in chunks:
        key = chunk["text"]
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top_k_per_subq", type=int, default=3)
    args = parser.parse_args()

    es = Elasticsearch("http://localhost:9200")
    if not es.ping():
        raise SystemExit("Can't reach Elasticsearch — is `docker compose up -d` running?")

    claims = json.loads(DEV_JSON.read_text())[: args.limit]
    correct = 0
    results = []

    for i, claim_entry in enumerate(claims):
        claim = claim_entry["claim"]
        gold_label = claim_entry["label"]
        claim_id = str(i)

        try:
            sub_questions = decompose_claim(claim)

            all_evidence = []
            for sq in sub_questions:
                all_evidence.extend(
                    hybrid_search(es, claim_id, sq.initial_query, top_k=args.top_k_per_subq)
                )
            all_evidence = dedupe_chunks(all_evidence)

            if not all_evidence:
                print(f"[{i}] WARNING: no evidence retrieved (claim not indexed yet?)")
                continue

            verdict = make_verdict(claim, all_evidence)
        except Exception as e:
            print(f"[{i}] SKIPPED — pipeline failed: {e}\n")
            results.append(
                {"claim_id": claim_id, "claim": claim, "gold_label": gold_label, "error": str(e)}
            )
            continue

        is_correct = verdict["label"] == gold_label
        correct += is_correct

        print(f"[{i}] claim: {claim[:80]}...")
        print(f"    sub-questions ({len(sub_questions)}): {[sq.question for sq in sub_questions]}")
        print(f"    gold: {gold_label!r}  predicted: {verdict['label']!r}  {'✓' if is_correct else '✗'}")
        print(f"    evidence chunks used: {len(all_evidence)}")
        print()

        results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "sub_questions": [sq.model_dump() for sq in sub_questions],
                "gold_label": gold_label,
                "predicted_label": verdict["label"],
                "justification": verdict["justification"],
                "correct": is_correct,
                "n_evidence_used": len(all_evidence),
            }
        )

    n = sum(1 for r in results if "correct" in r)
    n_errors = sum(1 for r in results if "error" in r)
    print(
        f"--- Phase 2 (decomposition) accuracy: {correct}/{n} ({correct/n*100:.1f}%), "
        f"{n_errors} skipped due to errors ---" if n else "No results."
    )

    out_path = Path(__file__).parent.parent / "eval" / "decomposition_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
