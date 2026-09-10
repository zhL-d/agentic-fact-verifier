"""Unit tests for graph.py's orchestration logic: routing, token-budget
accumulation, and the four human-in-the-loop escalation triggers."""

import pytest

import agentic_fact_verifier.graph as graph_mod
from agentic_fact_verifier.decomposition import SubQuestion
from agentic_fact_verifier.sufficiency import SufficiencyCheck, ThreadSufficiency

CLAIM = "The Eiffel Tower was completed in 1889 and is located in Paris."


def _two_subquestions(usage_tokens=100):
    def fake_decompose(claim):
        return (
            [
                SubQuestion(question="When was it completed?", initial_query="Eiffel Tower completed year"),
                SubQuestion(question="Where is it located?", initial_query="Eiffel Tower location"),
            ],
            {"prompt_tokens": usage_tokens, "completion_tokens": 0},
        )

    return fake_decompose


def _always_resolved(usage_tokens=50):
    def fake_sufficiency(claim, threads):
        return (
            SufficiencyCheck(
                threads=[
                    ThreadSufficiency(index=i, resolved=True, reasoning="enough evidence", refined_queries=[])
                    for i in range(len(threads))
                ]
            ),
            {"prompt_tokens": usage_tokens, "completion_tokens": 0},
        )

    return fake_sufficiency


def _never_resolved(usage_tokens=50):
    def fake_sufficiency(claim, threads):
        return (
            SufficiencyCheck(
                threads=[
                    ThreadSufficiency(
                        index=i, resolved=False, reasoning="still missing something", refined_queries=["refined query"]
                    )
                    for i in range(len(threads))
                ]
            ),
            {"prompt_tokens": usage_tokens, "completion_tokens": 0},
        )

    return fake_sufficiency


def _judge_returning(**overrides):
    def judge(args):
        verdict = {
            "label": "Supported",
            "justification": "See [1].",
            "citations": [1],
            "invalid_citations": [],
            "n_evidence_available": len(args["evidence"]),
            "usage": {"prompt_tokens": 30, "completion_tokens": 0},
        }
        verdict.update(overrides)
        return verdict

    return judge


def _retrieve_returning_evidence(args):
    return [{"text": f"evidence for {args['query']}", "url": "http://a"}]


async def test_evidence_compaction_bounds_growth_and_keeps_most_recent(fake_tool, monkeypatch):
    """Confirms each thread's final evidence stays within budget under a
    cap, and specifically keeps the chunk from its LAST round, not an earlier one."""
    monkeypatch.setattr(graph_mod, "MAX_EVIDENCE_CHARS_PER_THREAD", 100)

    call_n = {"n": 0}

    def big_retrieve(args):
        call_n["n"] += 1
        return [{"text": f"chunk#{call_n['n']}-" + "x" * 60, "url": "http://a"}]

    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(),
        _never_resolved(),
        _judge_returning(),
        retrieve_fn=big_retrieve,
    )

    threads = final_state["threads"]
    assert len(threads) == 2
    for thread in threads:
        total_chars = sum(len(c["text"]) for c in thread["evidence"])
        assert total_chars <= 100, "must stay within budget, not grow across all 3 rounds unbounded"
        assert len(thread["evidence"]) == 1, "budget only fits one of these chunks, not several"

    assert call_n["n"] == 6
    assert threads[0]["evidence"][0]["text"].startswith("chunk#5-")
    assert threads[1]["evidence"][0]["text"].startswith("chunk#6-")


def test_evidence_compaction_prefers_relevance_and_source_diversity():
    chunks = [
        {"text": "a" * 40, "url": "https://a.example/1", "retrieval_score": 0.9},
        {"text": "b" * 40, "url": "https://a.example/1", "retrieval_score": 0.8},
        {"text": "c" * 40, "url": "https://b.example/2", "retrieval_score": 0.7},
    ]

    compacted = graph_mod._compact_evidence(chunks, 90)

    assert [chunk["url"] for chunk in compacted] == ["https://a.example/1", "https://b.example/2"]


def test_evidence_deduplication_normalizes_whitespace_and_case():
    chunks = [
        {"text": "Same   Evidence", "url": "https://a.example"},
        {"text": "same evidence", "url": "https://b.example"},
    ]

    assert graph_mod._dedupe(chunks) == [chunks[0]]


async def _build_and_run(
    monkeypatch, fake_tool, decompose_fn, sufficiency_fn, judge_fn, retrieve_fn=_retrieve_returning_evidence
):
    monkeypatch.setattr(graph_mod, "decompose_claim", decompose_fn)
    monkeypatch.setattr(graph_mod, "check_sufficiency", sufficiency_fn)
    retrieve_tool = fake_tool(retrieve_fn)
    judge_tool = fake_tool(judge_fn)
    app = graph_mod.build_graph(retrieve_tool, judge_tool)
    return await graph_mod.run_verification(app, CLAIM, "test-claim-id")


async def test_happy_path_resolves_in_one_round_and_sums_token_usage(fake_tool, monkeypatch):
    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(usage_tokens=100),
        _always_resolved(usage_tokens=50),
        _judge_returning(),
    )

    assert len(final_state["rounds"]) == 1
    assert final_state["verdict"]["label"] == "Supported"
    assert final_state["total_tokens_used"] == 100 + 50 + 30
    assert final_state["verdict"]["escalate"] is False
    assert final_state["verdict"]["escalation_reasons"] == []


async def test_unresolved_threads_hit_max_iterations_and_escalate(fake_tool, monkeypatch):
    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(),
        _never_resolved(),
        _judge_returning(),
    )

    assert len(final_state["rounds"]) == graph_mod.MAX_ITERATIONS
    assert final_state["verdict"]["escalate"] is True
    assert any("remained unresolved" in r for r in final_state["verdict"]["escalation_reasons"])


async def test_ambiguous_label_triggers_escalation(fake_tool, monkeypatch):
    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(),
        _always_resolved(),
        _judge_returning(label="Conflicting Evidence/Cherrypicking"),
    )

    assert final_state["verdict"]["escalate"] is True
    assert any("inherently ambiguous" in r for r in final_state["verdict"]["escalation_reasons"])


async def test_invalid_citations_trigger_escalation(fake_tool, monkeypatch):
    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(),
        _always_resolved(),
        _judge_returning(citations=[1, 7], invalid_citations=[7]),
    )

    assert final_state["verdict"]["escalate"] is True
    assert any("Citation(s) [7]" in r for r in final_state["verdict"]["escalation_reasons"])


async def test_normal_budget_does_not_trigger_budget_escalation(fake_tool, monkeypatch):
    monkeypatch.setattr(graph_mod, "MAX_TOKENS_PER_RUN", 150_000)
    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(usage_tokens=100),
        _never_resolved(usage_tokens=50),
        _judge_returning(),
    )

    assert len(final_state["rounds"]) == graph_mod.MAX_ITERATIONS
    assert not any("token budget" in r for r in final_state["verdict"]["escalation_reasons"])


async def test_tiny_budget_stops_the_loop_early(fake_tool, monkeypatch):
    monkeypatch.setattr(graph_mod, "MAX_TOKENS_PER_RUN", 120)
    final_state = await _build_and_run(
        monkeypatch,
        fake_tool,
        _two_subquestions(usage_tokens=100),
        _never_resolved(usage_tokens=50),
        _judge_returning(),
    )

    assert len(final_state["rounds"]) == 1, "should stop after round 1, not run all 3"
    assert final_state["verdict"]["escalate"] is True
    assert any("token budget" in r for r in final_state["verdict"]["escalation_reasons"])


async def test_resume_continues_interrupted_run_instead_of_restarting(fake_tool, monkeypatch):
    """Whether decompose ran once (resumed) or twice (restarted) is the
    direct signal that a crashed run resumed rather than restarted."""
    from langgraph.checkpoint.memory import InMemorySaver

    decompose_calls = {"n": 0}

    def counting_decompose(claim):
        decompose_calls["n"] += 1
        return _two_subquestions()(claim)

    monkeypatch.setattr(graph_mod, "decompose_claim", counting_decompose)
    monkeypatch.setattr(graph_mod, "check_sufficiency", _never_resolved())

    call_count = {"n": 0}

    def flaky_retrieve(args):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated ES timeout")
        return [{"text": f"evidence for {args['query']}", "url": "http://a"}]

    retrieve_tool = fake_tool(flaky_retrieve)
    judge_tool = fake_tool(_judge_returning())
    app = graph_mod.build_graph(retrieve_tool, judge_tool, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume-test"}}

    with pytest.raises(RuntimeError, match="simulated ES timeout"):
        await graph_mod.run_or_resume(app, CLAIM, "resume-test", config)

    assert decompose_calls["n"] == 1, "sanity check on the crash itself"
    calls_before_crash = call_count["n"]
    assert calls_before_crash == 3

    state = await app.aget_state(config)
    assert state.next, "run should be incomplete (not finished) right after the crash"

    final_state = await graph_mod.run_or_resume(app, CLAIM, "resume-test", config)

    assert decompose_calls["n"] == 1, (
        "decompose must NOT run again, a second call means this restarted "
        "from scratch instead of resuming from round 2"
    )
    assert call_count["n"] > calls_before_crash, "resuming must continue the run, not restart from scratch"
    assert final_state["verdict"] is not None


async def test_fresh_thread_id_has_no_leftover_state(fake_tool, monkeypatch):
    """A never-used thread_id must start fresh, isolation between
    different claims' runs, not just resumability for a shared one."""
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(graph_mod, "decompose_claim", _two_subquestions())
    monkeypatch.setattr(graph_mod, "check_sufficiency", _always_resolved())

    app = graph_mod.build_graph(
        fake_tool(_retrieve_returning_evidence), fake_tool(_judge_returning()), checkpointer=InMemorySaver()
    )

    state = await app.aget_state({"configurable": {"thread_id": "never-used"}})
    assert not state.next


async def test_stream_or_resume_yields_real_node_updates_and_final_state(fake_tool, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(graph_mod, "decompose_claim", _two_subquestions())
    monkeypatch.setattr(graph_mod, "check_sufficiency", _always_resolved())
    app = graph_mod.build_graph(
        fake_tool(_retrieve_returning_evidence),
        fake_tool(_judge_returning()),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "stream-test"}}

    events = [
        (node, update)
        async for node, update in graph_mod.stream_or_resume(app, CLAIM, "stream-test", config)
    ]

    node_updates = [(node, update) for node, update in events if node != "__custom__"]
    custom_events = [update for node, update in events if node == "__custom__"]

    assert [node for node, _ in node_updates] == [
        "decompose",
        "retrieve",
        "check_sufficiency",
        "verdict",
        "__complete__",
    ]
    assert node_updates[1][1]["iteration"] == 1
    assert node_updates[2][1]["rounds"][0]["round"] == 1
    assert node_updates[-1][1]["verdict"]["label"] == "Supported"
    assert [event["event"] for event in custom_events] == [
        "thread_retrieval_started",
        "thread_retrieval_completed",
        "thread_retrieval_started",
        "thread_retrieval_completed",
        "thread_sufficiency_started",
        "thread_sufficiency_started",
        "thread_sufficiency_completed",
        "thread_sufficiency_completed",
    ]
    assert custom_events[0]["thread_id"] == "q1"
    assert custom_events[1]["new_hits"] == 1
