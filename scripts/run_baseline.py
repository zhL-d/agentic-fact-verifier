"""Phase 1 baseline: single-shot RAG, no agent loop. claim -> one hybrid
retrieval (using the raw claim text as the query) -> one verdict call.

This is the control group every later phase (decomposition, sufficiency
loop, ...) gets measured against.

Usage:
    export FIREWORKS_API_KEY=...
    uv run scripts/run_baseline.py --limit 3   # test against already-indexed claims
"""

import argparse
import json
import sys
from pathlib import Path

from elasticsearch import Elasticsearch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentic_fact_verifier.retrieval import hybrid_search  # noqa: E402
from agentic_fact_verifier.verdict import make_verdict  # noqa: E402

DEV_JSON = Path(__file__).parent.parent / "data" / "raw" / "dev.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=5)
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

        evidence = hybrid_search(es, claim_id, claim, top_k=args.top_k)
        if not evidence:
            print(f"[{i}] WARNING: no evidence retrieved (claim not indexed yet?)")
            continue

        verdict = make_verdict(claim, evidence)
        is_correct = verdict["label"] == gold_label
        correct += is_correct

        print(f"[{i}] claim: {claim[:80]}...")
        print(f"    gold: {gold_label!r}  predicted: {verdict['label']!r}  {'✓' if is_correct else '✗'}")
        print(f"    justification: {verdict['justification'][:150]}")
        print()

        results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label": verdict["label"],
                "justification": verdict["justification"],
                "correct": is_correct,
                "n_evidence_retrieved": len(evidence),
            }
        )

    n = len(results)
    print(f"--- Baseline accuracy: {correct}/{n} ({correct/n*100:.1f}%) ---" if n else "No results.")

    out_path = Path(__file__).parent.parent / "eval" / "baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
