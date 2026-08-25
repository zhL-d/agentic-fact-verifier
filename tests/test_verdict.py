"""Unit tests for verdict.py's citation self-correction retry loop."""

import agentic_fact_verifier.verdict as verdict_mod
from agentic_fact_verifier.llm_client import LLMResult
from agentic_fact_verifier.prompt_guard import ANTI_INJECTION_NOTICE
from agentic_fact_verifier.verdict import MAX_CITATION_ATTEMPTS, make_verdict

EVIDENCE = [{"text": "some evidence", "url": "http://a"}]


def _capture_prompt(monkeypatch):
    """Returns (captured dict) — after a make_verdict call, captured["prompt"]
    holds the first message's content actually sent to the model."""
    captured = {}

    def fake_call_llm(messages, response_format=None, max_tokens=6000):
        captured.setdefault("prompt", messages[0]["content"])
        content = '{"label": "Supported", "justification": "See [1].", "citations": [1]}'
        return LLMResult(content=content, usage={"prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr(verdict_mod, "call_llm", fake_call_llm)
    return captured


def test_prompt_includes_anti_injection_notice_and_wrapped_evidence_no_sub_findings(monkeypatch):
    captured = _capture_prompt(monkeypatch)

    make_verdict("some claim", EVIDENCE)

    prompt = captured["prompt"]
    assert ANTI_INJECTION_NOTICE.strip() in prompt, (
        "must reach the actual prompt sent to the model, not just be an "
        "importable constant nothing uses"
    )
    assert '<evidence index="1" source="http://a">' in prompt


def test_prompt_includes_anti_injection_notice_and_wrapped_evidence_with_sub_findings(monkeypatch):
    captured = _capture_prompt(monkeypatch)

    make_verdict(
        "some claim",
        EVIDENCE,
        sub_findings=[{"question": "q?", "resolved": True, "finding": "found it"}],
    )

    prompt = captured["prompt"]
    assert ANTI_INJECTION_NOTICE.strip() in prompt
    assert '<evidence index="1" source="http://a">' in prompt


def test_valid_first_attempt_makes_no_retry_call(monkeypatch):
    calls = []

    def fake_call_llm(messages, response_format=None, max_tokens=6000):
        calls.append(messages)
        content = '{"label": "Supported", "justification": "See [1].", "citations": [1]}'
        return LLMResult(content=content, usage={"prompt_tokens": 30, "completion_tokens": 12})

    monkeypatch.setattr(verdict_mod, "call_llm", fake_call_llm)

    result = make_verdict("some claim", EVIDENCE)

    assert len(calls) == 1, "a valid first response should never trigger a retry call"
    assert result["citations"] == [1]
    assert result["invalid_citations"] == []
    assert result["usage"]["prompt_tokens"] == 30
    assert result["usage"]["completion_tokens"] == 12
    assert result["usage"]["total_tokens"] == 42


def test_invalid_citation_triggers_one_self_correction_retry(monkeypatch):
    calls = []

    def fake_call_llm(messages, response_format=None, max_tokens=6000):
        calls.append(messages)
        if len(calls) == 1:
            content = '{"label": "Supported", "justification": "See [1] and [5].", "citations": [1, 5]}'
        else:
            content = '{"label": "Supported", "justification": "See [1].", "citations": [1]}'
        return LLMResult(content=content, usage={"prompt_tokens": 7, "completion_tokens": 3})

    monkeypatch.setattr(verdict_mod, "call_llm", fake_call_llm)

    result = make_verdict("some claim", EVIDENCE)

    assert len(calls) == 2, "an invalid citation should trigger exactly one retry"
    retry_prompt = calls[1][-1]["content"]
    assert "5" in retry_prompt
    assert "don't correspond" in retry_prompt
    assert result["citations"] == [1]
    assert result["invalid_citations"] == []
    assert result["usage"]["prompt_tokens"] == 14, "usage should accumulate across both attempts"
    assert result["usage"]["completion_tokens"] == 6
    assert result["usage"]["total_tokens"] == 20


def test_retry_is_bounded_when_model_never_fixes_it(monkeypatch):
    calls = []

    def fake_call_llm(messages, response_format=None, max_tokens=6000):
        calls.append(messages)
        content = '{"label": "Supported", "justification": "See [9].", "citations": [9]}'
        return LLMResult(content=content, usage={"prompt_tokens": 3, "completion_tokens": 2})

    monkeypatch.setattr(verdict_mod, "call_llm", fake_call_llm)

    result = make_verdict("some claim", EVIDENCE)

    assert len(calls) == MAX_CITATION_ATTEMPTS, "must stop retrying at the bound, not loop forever"
    assert result["invalid_citations"] == [9]
