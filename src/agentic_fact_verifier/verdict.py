"""Single-shot verdict: given a claim and retrieved evidence, one LLM call
produces a label + justification. This is the Phase 1 baseline, no
decomposition, no sufficiency loop, no citation-required schema yet. Every
later phase is measured against this as the control group."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/nemotron-3-ultra-nvfp4"

VERDICT_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

PROMPT_TEMPLATE = """You are fact-checking a claim using retrieved evidence.

Claim: {claim}

Retrieved evidence:
{evidence}

Based only on the evidence above, decide the claim's veracity. Choose exactly \
one label from: {labels}.

Respond with a single JSON object only, in this exact shape:
{{"label": "<one of the labels above>", "justification": "<brief reasoning>"}}
"""


def _get_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def make_verdict(claim: str, evidence_chunks: list[dict]) -> dict:
    evidence_text = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(evidence_chunks)
    )
    prompt = PROMPT_TEMPLATE.format(
        claim=claim, evidence=evidence_text, labels=", ".join(VERDICT_LABELS)
    )

    client = _get_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3000,  # Nemotron-3-Ultra reasons internally before the
        # final answer (confirmed in the Ev2R cost probe: 632-1070
        # completion tokens for a much simpler task) — 1000 was too low
        # and truncated responses mid-reasoning, before any JSON appeared.
    )
    raw = completion.choices[0].message.content
    finish_reason = completion.choices[0].finish_reason

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found in model response (finish_reason="
            f"{finish_reason!r}, len={len(raw)}). Raw response:\n{raw}"
        )
    return json.loads(raw[start : end + 1])
