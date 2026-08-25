"""Unit test confirming the prompt-injection defense is actually wired into
check_sufficiency's constructed prompt."""

import agentic_fact_verifier.sufficiency as sufficiency_mod
from agentic_fact_verifier.llm_client import LLMResult
from agentic_fact_verifier.prompt_guard import ANTI_INJECTION_NOTICE
from agentic_fact_verifier.sufficiency import check_sufficiency


def test_prompt_includes_anti_injection_notice_and_wrapped_evidence(monkeypatch):
    captured = {}

    def fake_call_llm(prompt, max_tokens=6000, response_format=None):
        captured["prompt"] = prompt
        content = '{"threads": [{"index": 0, "resolved": true, "reasoning": "ok", "refined_queries": []}]}'
        return LLMResult(content=content, usage={"prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr(sufficiency_mod, "call_llm", fake_call_llm)

    check_sufficiency(
        "some claim",
        [{"question": "some question?", "evidence": [{"text": "evidence text", "url": "http://a"}]}],
    )

    prompt = captured["prompt"]
    assert ANTI_INJECTION_NOTICE.strip() in prompt, (
        "the anti-injection notice must actually reach the prompt sent to the "
        "model, not just exist as an importable constant nothing uses"
    )
    assert '<evidence index="1" source="http://a">' in prompt, (
        "evidence must actually be wrapped in delimiter tags, not spliced in raw"
    )
