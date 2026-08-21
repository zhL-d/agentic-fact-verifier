"""LangGraph orchestration the evidence sufficiency loop.

Agentic, the number of retrieval rounds isn't fixed in advance, 
it's decided dynamically by the sufficiency check,  
with the search strategy actually changing between rounds (refined queries, not identical retries).

Each sub-question is tracked as its own "thread", its own accumulated
evidence, its own resolved/reasoning state, its own pending queries. This
is what makes the audit trail.

Graph shape:
    decompose -> retrieve -> check_sufficiency -[any thread unresolved]-> retrieve (loop)
                                                \\-[all resolved, or max
                                                    iterations hit]-> verdict -> END
"""

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agentic_fact_verifier.decomposition import decompose_claim
from agentic_fact_verifier.sufficiency import check_sufficiency

MAX_ITERATIONS = 3  # hard cap, bounds cost/latency


class Thread(TypedDict):
    question: str  # the ORIGINAL sub-question, stable across rounds,
    # never rewritten (only queries_to_run changes round to round)
    queries_to_run: list[str]  # queries pending for the NEXT retrieve
    # round; empty once this thread is resolved (nothing left to search for)
    evidence: list[dict]  # accumulated evidence for THIS sub-question only
    resolved: bool
    reasoning: str


class VerificationState(TypedDict):
    claim: str
    claim_id: str
    threads: list[Thread]
    iteration: int
    rounds: list[dict]  # audit trail: one entry per round, with
    # per-thread detail (queries used, new evidence count, resolved, reasoning)
    last_round_detail: list[dict]  # handoff from retrieve_node to
    # sufficiency_node for this round's audit entry
    is_sufficient: bool  # True once every thread is resolved
    verdict: dict | None
    all_evidence: list[dict]  # the flattened, deduped list verdict_node
    # citations are numbered over


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for chunk in chunks:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            out.append(chunk)
    return out


def build_graph(retrieve_tool, judge_tool, top_k_per_query: int = 3):
    """`retrieve_tool` is the MCP-exposed `retrieve_evidence` tool."""

    def decompose_node(state: VerificationState) -> dict:
        sub_questions = decompose_claim(state["claim"])
        threads: list[Thread] = [
            {
                "question": sq.question,
                # round-1 search uses initial_query
                "queries_to_run": [sq.initial_query],
                "evidence": [],
                "resolved": False,
                "reasoning": "",
            }
            for sq in sub_questions
        ]
        return {"threads": threads, "iteration": 0}

    async def retrieve_node(state: VerificationState) -> dict:
        new_threads = []
        round_detail = []
        for thread in state["threads"]:
            queries = thread["queries_to_run"]
            new_evidence = []
            for query in queries:
                # LangChain's MCP tool wrapper returns a list of content
                # blocks (e.g. [{"type": "text", "text": "<our
                # json.dumps(...) string>"}])
                content_blocks = await retrieve_tool.ainvoke(
                    {"claim_id": state["claim_id"], "query": query, "top_k": top_k_per_query}
                )
                result_json = content_blocks[0]["text"]
                chunks = json.loads(result_json)
                for chunk in chunks:
                    chunk["retrieval_query"] = query
                new_evidence.extend(chunks)
            deduped_new = _dedupe(new_evidence)
            combined = _dedupe(thread["evidence"] + deduped_new)
            new_threads.append({**thread, "evidence": combined, "queries_to_run": []})
            round_detail.append(
                {
                    "question": thread["question"],
                    "queries_used": queries,
                    "new_evidence": deduped_new,
                    "n_new_evidence": len(deduped_new),
                }
            )
        return {
            "threads": new_threads,
            "iteration": state["iteration"] + 1,
            "last_round_detail": round_detail,
        }

    def sufficiency_node(state: VerificationState) -> dict:
        threads = state["threads"]

        unresolved_indices = [i for i, t in enumerate(threads) if not t["resolved"]]

        result_by_index = {}
        if unresolved_indices:
            check = check_sufficiency(
                state["claim"],
                [{"question": threads[i]["question"], "evidence": threads[i]["evidence"]} for i in unresolved_indices],
            )
            result_by_index = dict(zip(unresolved_indices, check.threads))

        last_detail = state.get("last_round_detail", [])
        new_threads = []
        round_threads_log = []
        for i, thread in enumerate(threads):
            if i in result_by_index:
                result = result_by_index[i]
                new_threads.append(
                    {
                        **thread,
                        "resolved": result.resolved,
                        "reasoning": result.reasoning,
                        "queries_to_run": [] if result.resolved else result.refined_queries,
                    }
                )
                resolved_this_round, reasoning_this_round = result.resolved, result.reasoning
            else:

                new_threads.append({**thread, "queries_to_run": []})
                resolved_this_round, reasoning_this_round = thread["resolved"], thread["reasoning"]

            detail = last_detail[i] if i < len(last_detail) else {
                "queries_used": [], "new_evidence": [], "n_new_evidence": 0
            }
            round_threads_log.append(
                {
                    "question": thread["question"],
                    "queries_used": detail["queries_used"],
                    "new_evidence": detail["new_evidence"],
                    "n_new_evidence": detail["n_new_evidence"],
                    "resolved": resolved_this_round,
                    "reasoning": reasoning_this_round,
                }
            )

        rounds_log = state["rounds"] + [{"round": state["iteration"], "threads": round_threads_log}]
        is_sufficient = all(t["resolved"] for t in new_threads)
        return {"threads": new_threads, "rounds": rounds_log, "is_sufficient": is_sufficient}

    def route_after_sufficiency(state: VerificationState) -> str:
        if state["is_sufficient"] or state["iteration"] >= MAX_ITERATIONS:
            return "verdict"
        return "retrieve"

    async def verdict_node(state: VerificationState) -> dict:
        sub_findings = [
            {"question": t["question"], "resolved": t["resolved"], "finding": t["reasoning"]}
            for t in state["threads"]
        ]

        all_evidence = _dedupe([chunk for t in state["threads"] for chunk in t["evidence"]])


        content_blocks = await judge_tool.ainvoke(
            {"claim": state["claim"], "evidence": all_evidence, "sub_findings": sub_findings}
        )
        verdict = json.loads(content_blocks[0]["text"])

        cited_sources = [
            {"citation_number": n, "url": all_evidence[n - 1]["url"]}
            for n in verdict["citations"]
            if 1 <= n <= len(all_evidence)
        ]
        verdict["cited_sources"] = cited_sources
        return {"verdict": verdict, "all_evidence": all_evidence}

    graph = StateGraph(VerificationState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_sufficiency", sufficiency_node)
    graph.add_node("verdict", verdict_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency", route_after_sufficiency, {"retrieve": "retrieve", "verdict": "verdict"}
    )
    graph.add_edge("verdict", END)

    return graph.compile()


async def run_verification(app, claim: str, claim_id: str) -> VerificationState:
    initial_state: VerificationState = {
        "claim": claim,
        "claim_id": claim_id,
        "threads": [],
        "iteration": 0,
        "rounds": [],
        "last_round_detail": [],
        "is_sufficient": False,
        "verdict": None,
        "all_evidence": [],
    }
    return await app.ainvoke(initial_state)
