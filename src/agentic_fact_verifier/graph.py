"""LangGraph orchestration of the evidence sufficiency loop.

Graph shape:
    decompose -> retrieve -> check_sufficiency -[any thread unresolved]-> retrieve (loop)
                                                \\-[all resolved, or max
                                                    iterations hit]-> verdict -> END
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from agentic_fact_verifier.decomposition import decompose_claim
from agentic_fact_verifier.sufficiency import check_sufficiency

MAX_ITERATIONS = 3
MAX_TOKENS_PER_RUN = int(os.environ.get("MAX_TOKENS_PER_RUN", "150000"))
MAX_EVIDENCE_CHARS_PER_THREAD = int(os.environ.get("MAX_EVIDENCE_CHARS_PER_THREAD", "8000"))

ESCALATION_LABELS = {
    label.strip()
    for label in os.environ.get(
        "ESCALATION_LABELS", "Not Enough Evidence,Conflicting Evidence/Cherrypicking"
    ).split(",")
    if label.strip()
}


class Thread(TypedDict):
    thread_id: str
    question: str
    queries_to_run: list[str]
    evidence: list[dict]
    resolved: bool
    reasoning: str


class VerificationState(TypedDict):
    claim: str
    claim_id: str
    threads: list[Thread]
    iteration: int
    rounds: list[dict]
    last_round_detail: list[dict]
    is_sufficient: bool
    verdict: dict | None
    all_evidence: list[dict]
    total_tokens_used: int
    total_prompt_tokens: int
    total_completion_tokens: int


def _dedupe(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for chunk in chunks:
        normalized_text = " ".join(chunk["text"].split()).casefold()
        if normalized_text not in seen:
            seen.add(normalized_text)
            out.append(chunk)
    return out


def _compact_evidence(chunks: list[dict], max_chars: int) -> list[dict]:
    """Keep the strongest, source-diverse evidence within the text budget.
    """
    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (float(item[1].get("retrieval_score", 0.0)), item[0]),
        reverse=True,
    )
    distinct_sources: list[tuple[int, dict]] = []
    repeated_sources: list[tuple[int, dict]] = []
    seen_urls: set[str] = set()
    for item in ranked:
        url = item[1].get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            distinct_sources.append(item)
        else:
            repeated_sources.append(item)

    kept: list[dict] = []
    total = 0
    for _, chunk in distinct_sources + repeated_sources:
        length = len(chunk["text"])
        if kept and total + length > max_chars:
            continue
        kept.append(chunk)
        total += length
        if total >= max_chars:
            break
    return kept


def build_graph(retrieve_tool, judge_tool, top_k_per_query: int = 3, checkpointer=None):
    """`retrieve_tool` is the MCP-exposed `retrieve_evidence` tool.
    `checkpointer`, when given, makes the compiled graph resumable."""

    def decompose_node(state: VerificationState) -> dict:
        sub_questions, usage = decompose_claim(state["claim"])
        threads: list[Thread] = [
            {
                "thread_id": f"q{index + 1}",
                "question": sq.question,
                "queries_to_run": [sq.initial_query],
                "evidence": [],
                "resolved": False,
                "reasoning": "",
            }
            for index, sq in enumerate(sub_questions)
        ]
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        return {
            "threads": threads,
            "iteration": 0,
            "total_tokens_used": prompt_tokens + completion_tokens,
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
        }

    async def retrieve_node(state: VerificationState) -> dict:
        writer = get_stream_writer()
        new_threads = []
        round_detail = []
        round_number = state["iteration"] + 1
        for thread in state["threads"]:
            queries = thread["queries_to_run"]
            new_evidence = []
            if queries:
                writer(
                    {
                        "event": "thread_retrieval_started",
                        "thread_id": thread["thread_id"],
                        "question": thread["question"],
                        "round": round_number,
                        "queries": queries,
                    }
                )
            for query in queries:
                content_blocks = await retrieve_tool.ainvoke(
                    {"claim_id": state["claim_id"], "query": query, "top_k": top_k_per_query}
                )
                result_json = content_blocks[0]["text"]
                chunks = json.loads(result_json)
                for chunk in chunks:
                    chunk["retrieval_query"] = query
                new_evidence.extend(chunks)
            deduped_new = _dedupe(new_evidence)
            combined = _compact_evidence(_dedupe(thread["evidence"] + deduped_new), MAX_EVIDENCE_CHARS_PER_THREAD)
            new_threads.append({**thread, "evidence": combined, "queries_to_run": []})
            if queries:
                writer(
                    {
                        "event": "thread_retrieval_completed",
                        "thread_id": thread["thread_id"],
                        "question": thread["question"],
                        "round": round_number,
                        "new_hits": len(deduped_new),
                        "unique_retained": len(combined),
                    }
                )
            round_detail.append(
                {
                    "thread_id": thread["thread_id"],
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
        writer = get_stream_writer()
        threads = state["threads"]

        unresolved_indices = [i for i, t in enumerate(threads) if not t["resolved"]]

        result_by_index = {}
        call_prompt_tokens, call_completion_tokens = 0, 0
        if unresolved_indices:
            for index in unresolved_indices:
                writer(
                    {
                        "event": "thread_sufficiency_started",
                        "thread_id": threads[index]["thread_id"],
                        "question": threads[index]["question"],
                        "round": state["iteration"],
                    }
                )
            check, usage = check_sufficiency(
                state["claim"],
                [{"question": threads[i]["question"], "evidence": threads[i]["evidence"]} for i in unresolved_indices],
            )
            result_by_index = dict(zip(unresolved_indices, check.threads, strict=True))
            call_prompt_tokens = usage.get("prompt_tokens", 0) or 0
            call_completion_tokens = usage.get("completion_tokens", 0) or 0

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
                writer(
                    {
                        "event": "thread_sufficiency_completed",
                        "thread_id": thread["thread_id"],
                        "question": thread["question"],
                        "round": state["iteration"],
                        "resolved": result.resolved,
                        "reasoning": result.reasoning,
                        "refined_queries": result.refined_queries,
                    }
                )
            else:
                new_threads.append({**thread, "queries_to_run": []})
                resolved_this_round, reasoning_this_round = thread["resolved"], thread["reasoning"]

            detail = last_detail[i] if i < len(last_detail) else {
                "queries_used": [], "new_evidence": [], "n_new_evidence": 0
            }
            round_threads_log.append(
                {
                    "thread_id": thread["thread_id"],
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
        return {
            "threads": new_threads,
            "rounds": rounds_log,
            "is_sufficient": is_sufficient,
            "total_tokens_used": state["total_tokens_used"] + call_prompt_tokens + call_completion_tokens,
            "total_prompt_tokens": state["total_prompt_tokens"] + call_prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + call_completion_tokens,
        }

    def route_after_sufficiency(state: VerificationState) -> str:
        if (
            state["is_sufficient"]
            or state["iteration"] >= MAX_ITERATIONS
            or state["total_tokens_used"] >= MAX_TOKENS_PER_RUN
        ):
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

        verdict_usage = verdict.get("usage", {})
        total_prompt_tokens = state["total_prompt_tokens"] + (verdict_usage.get("prompt_tokens", 0) or 0)
        total_completion_tokens = state["total_completion_tokens"] + (verdict_usage.get("completion_tokens", 0) or 0)
        total_tokens_used = total_prompt_tokens + total_completion_tokens
        verdict["total_tokens_used"] = total_tokens_used
        verdict["total_prompt_tokens"] = total_prompt_tokens
        verdict["total_completion_tokens"] = total_completion_tokens
        budget_exceeded = total_tokens_used >= MAX_TOKENS_PER_RUN

        unresolved_questions = [t["question"] for t in state["threads"] if not t["resolved"]]
        reasons = []
        if unresolved_questions:
            reasons.append(
                f"{len(unresolved_questions)} of {len(state['threads'])} sub-question(s) "
                f"remained unresolved after the maximum retrieval rounds: "
                + "; ".join(unresolved_questions)
            )
        if verdict["label"] in ESCALATION_LABELS:
            reasons.append(
                f"Verdict label ({verdict['label']!r}) is inherently ambiguous by "
                f"design, not a confident Supported/Refuted call."
            )
        if verdict.get("invalid_citations"):
            reasons.append(
                f"Citation(s) {verdict['invalid_citations']} could not be resolved to "
                f"valid evidence even after self-correction."
            )
        if budget_exceeded:
            reasons.append(
                f"Run stopped early: token budget of {MAX_TOKENS_PER_RUN} exceeded "
                f"({total_tokens_used} tokens used across the run), this run may not "
                f"have finished as much retrieval/refinement as a normal one would."
            )
        verdict["escalate"] = bool(reasons)
        verdict["escalation_reasons"] = reasons

        return {
            "verdict": verdict,
            "all_evidence": all_evidence,
            "total_tokens_used": total_tokens_used,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        }

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

    return graph.compile(checkpointer=checkpointer)


async def run_verification(app, claim: str, claim_id: str, config: dict | None = None) -> VerificationState:
    """Starts a fresh run. `config` only matters if `app` was built with a
    checkpointer, pass the same `config` to resume an existing thread
    instead of starting fresh."""
    return await app.ainvoke(_initial_state(claim, claim_id), config=config)


def _initial_state(claim: str, claim_id: str) -> VerificationState:
    return {
        "claim": claim,
        "claim_id": claim_id,
        "threads": [],
        "iteration": 0,
        "rounds": [],
        "last_round_detail": [],
        "is_sufficient": False,
        "verdict": None,
        "all_evidence": [],
        "total_tokens_used": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }


async def run_or_resume(app, claim: str, claim_id: str, config: dict) -> VerificationState:
    """Resumes a run already in progress under `config`'s thread_id if one
    exists and hasn't reached END, otherwise starts a fresh run."""
    existing = await app.aget_state(config)
    if existing.next:
        return await app.ainvoke(None, config=config)
    return await run_verification(app, claim, claim_id, config=config)


async def stream_or_resume(
    app, claim: str, claim_id: str, config: dict
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield each completed LangGraph node and then the final state.

    ``stream_mode="updates"`` is the graph's actual execution stream: events
    are emitted only after a node has committed its state update. With a
    checkpointer, an interrupted run resumes from its pending node rather than
    replaying completed work.
    """
    existing = await app.aget_state(config)
    graph_input = None if existing.next else _initial_state(claim, claim_id)

    async for part in app.astream(
        graph_input,
        config=config,
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        if part["type"] == "updates":
            for node_name, node_update in part["data"].items():
                yield node_name, node_update
        elif part["type"] == "custom":
            yield "__custom__", part["data"]

    snapshot = await app.aget_state(config)
    yield "__complete__", dict(snapshot.values)
