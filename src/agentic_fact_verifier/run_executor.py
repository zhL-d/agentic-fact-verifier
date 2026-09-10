"""Execute one persisted verification run and emit durable trace events."""

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agentic_fact_verifier.dataset import get_claim
from agentic_fact_verifier.graph import MAX_ITERATIONS, MAX_TOKENS_PER_RUN, build_graph, stream_or_resume
from agentic_fact_verifier.mcp_client import mcp_judge_session, mcp_retrieval_session
from agentic_fact_verifier.run_store import CHECKPOINT_DB_URL, TERMINAL_STATUSES, RunStore


class RunCancelled(RuntimeError):
    pass


def verification_payload(entry: dict, claim_id: int, final_state: dict) -> dict[str, Any]:
    return {
        "claim": entry["claim"],
        "claim_id": str(claim_id),
        "verdict": final_state["verdict"],
        "all_evidence": final_state["all_evidence"],
        "rounds": final_state["rounds"],
        "threads": [
            {
                "thread_id": thread["thread_id"],
                "question": thread["question"],
                "resolved": thread["resolved"],
                "reasoning": thread["reasoning"],
                "evidence": thread["evidence"],
            }
            for thread in final_state["threads"]
        ],
        "gold_label": entry["label"],
    }


def _should_finish(update: dict) -> bool:
    return bool(
        update["is_sufficient"]
        or update["rounds"][-1]["round"] >= MAX_ITERATIONS
        or update["total_tokens_used"] >= MAX_TOKENS_PER_RUN
    )


def _event_key(event: str, data: dict[str, Any]) -> str:
    parts: list[str | int] = [event]
    if "round" in data:
        round_value = data["round"]
        parts.append(round_value.get("round", 0) if isinstance(round_value, dict) else round_value)
    if "thread_id" in data:
        parts.append(data["thread_id"])
    return ":".join(str(part) for part in parts)


async def _emit(store: RunStore, run_id: str, event: str, data: dict[str, Any]) -> None:
    await store.append_event(run_id, event, data, _event_key(event, data))


async def execute_verification_run(store: RunStore, run_id: str) -> None:
    run = await store.get_run(run_id)
    if run is None:
        raise KeyError(f"Unknown verification run {run_id}")
    if run["status"] in TERMINAL_STATUSES:
        if run["status"] == "cancelled" and run["last_event_id"] == 0:
            await store.finish_cancelled(run_id)
        return
    if run["cancel_requested"]:
        await store.finish_cancelled(run_id)
        return

    claim_id = run["claim_id"]
    entry = get_claim(claim_id)
    config = {"configurable": {"thread_id": run["thread_id"]}}
    await store.mark_running(run_id)

    async with AsyncPostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer:
        await checkpointer.setup()
        async with (
            mcp_retrieval_session() as retrieve_tool,
            mcp_judge_session() as judge_tool,
        ):
            graph_app = build_graph(retrieve_tool, judge_tool, checkpointer=checkpointer)
            async for node, update in stream_or_resume(graph_app, entry["claim"], str(claim_id), config):
                if await store.is_cancel_requested(run_id):
                    raise RunCancelled("Verification cancelled by user")

                if node == "__custom__":
                    event_name = update["event"]
                    await _emit(
                        store,
                        run_id,
                        event_name,
                        {key: value for key, value in update.items() if key != "event"},
                    )
                elif node == "decompose":
                    await _emit(
                        store,
                        run_id,
                        "decomposition_completed",
                        {
                            "threads": update["threads"],
                            "total_tokens_used": update["total_tokens_used"],
                            "max_rounds": MAX_ITERATIONS,
                        },
                    )
                    await _emit(
                        store,
                        run_id,
                        "retrieval_started",
                        {"round": 1, "max_rounds": MAX_ITERATIONS},
                    )
                elif node == "retrieve":
                    await _emit(
                        store,
                        run_id,
                        "retrieval_completed",
                        {
                            "round": update["iteration"],
                            "threads": update["threads"],
                            "details": update["last_round_detail"],
                        },
                    )
                elif node == "check_sufficiency":
                    latest_round = update["rounds"][-1]
                    await _emit(
                        store,
                        run_id,
                        "round_completed",
                        {
                            "round": latest_round,
                            "threads": update["threads"],
                            "is_sufficient": update["is_sufficient"],
                            "total_tokens_used": update["total_tokens_used"],
                        },
                    )
                    if _should_finish(update):
                        await _emit(
                            store,
                            run_id,
                            "verdict_started",
                            {"rounds_completed": latest_round["round"]},
                        )
                    else:
                        await _emit(
                            store,
                            run_id,
                            "retrieval_started",
                            {"round": latest_round["round"] + 1, "max_rounds": MAX_ITERATIONS},
                        )
                elif node == "__complete__":
                    await store.finish_success(run_id, verification_payload(entry, claim_id, update))
