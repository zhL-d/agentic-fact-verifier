"""Verdict: given a claim and retrieved evidence, one LLM call
produces a label + justification + citations. Citation-required by design
, every claimed fact in the justification must point at a
specific numbered piece of evidence, not a bare assertion. This is what
makes the audit trail real: a verdict is traceable to the exact evidence
it came from, not just a claim that it is."""

import json

from pydantic import BaseModel, Field

from agentic_fact_verifier.llm_client import call_llm

VERDICT_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

PROMPT_TEMPLATE = """You are fact-checking a claim using retrieved evidence.

Claim: {claim}

Retrieved evidence (numbered):
{evidence}
{incomplete_note}
Based only on the evidence above, decide the claim's veracity. Choose exactly \
one label from: {labels}.

Every claim you make in your justification must cite the specific evidence \
number(s) it's based on. Do not state a fact without a citation.

Respond with a single JSON object only, in this exact shape:
{{"label": "<one of the labels above>", "justification": "<brief reasoning \
with inline citations like [2]>", "citations": [<evidence numbers used, as \
integers>]}}

Keep the justification to at most 2 sentences.
"""

SUB_FINDINGS_PROMPT_TEMPLATE = """You are fact-checking a claim. The claim was broken into \
independent sub-questions, each investigated separately with its own evidence. Build your verdict \
FROM these sub-question findings, do not bypass them and re-derive your own conclusions straight \
from the raw evidence pool below; the evidence is provided only so you can cite specific sources, \
not as a substitute for the findings.

Claim: {claim}

Sub-question findings:
{sub_findings}

Full evidence (numbered, for citation only, cite whichever numbers support each point you make):
{evidence}

Based on the sub-question findings above, decide the claim's veracity. Choose exactly one label \
from: {labels}. If any UNRESOLVED sub-question is actually load-bearing for the claim's truth, \
prefer "Not Enough Evidence" over a confident guess — even if some RESOLVED sub-question looks \
supportive on its own.

Every claim you make in your justification must cite the specific evidence number(s) it's based \
on. Do not state a fact without a citation. Your justification should read as synthesizing the \
sub-question findings, not as an independent re-analysis of the evidence.

Respond with a single JSON object only, in this exact shape:
{{"label": "<one of the labels above>", "justification": "<brief reasoning \
with inline citations like [2]>", "citations": [<evidence numbers used, as \
integers>]}}

Keep the justification to at most 3 sentences.
"""

INCOMPLETE_NOTE_SPECIFIC = (
    "\nNote: evidence-gathering was stopped after repeated search attempts, and "
    "the following specific sub-questions are STILL UNRESOLVED, no evidence "
    "above answers them:\n{unresolved}\n"
    "Do not guess to fill those specific gaps. If the claim's veracity actually "
    "depends on any of the unresolved sub-questions above, prefer \"Not Enough "
    "Evidence\" over a confident guess, even if some OTHER part of the claim "
    "looks well-supported on its own.\n"
)

INCOMPLETE_NOTE_GENERIC = (
    "\nNote: evidence-gathering was stopped after repeated search attempts still "
    "left real gaps in the evidence, this is not a complete picture. Do not "
    "guess to fill those gaps. If the evidence doesn't clearly support or refute "
    "the claim on its own, prefer \"Not Enough Evidence\" over a confident "
    "guess.\n"
)


class Verdict(BaseModel):
    label: str
    justification: str
    citations: list[int] = Field(default_factory=list)


def make_verdict(
    claim: str,
    evidence_chunks: list[dict],
    evidence_incomplete: bool = False,
    unresolved_questions: list[str] | None = None,
    sub_findings: list[dict] | None = None,
) -> dict:
    """`sub_findings`, when provided, is the preferred path: a list of
    {"question": str, "resolved": bool, "finding": str}, one per
    sub-question thread, and the verdict is synthesized from these."""
    evidence_text = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(evidence_chunks)
    )

    if sub_findings:
        findings_text = "\n\n".join(
            f"{i+1}. [{'RESOLVED' if f['resolved'] else 'UNRESOLVED'}] {f['question']}\n"
            f"   Finding: {f['finding']}"
            for i, f in enumerate(sub_findings)
        )
        prompt = SUB_FINDINGS_PROMPT_TEMPLATE.format(
            claim=claim,
            sub_findings=findings_text,
            evidence=evidence_text,
            labels=", ".join(VERDICT_LABELS),
        )
    else:
        if evidence_incomplete and unresolved_questions:
            unresolved_text = "\n".join(f"- {q}" for q in unresolved_questions)
            incomplete_note = INCOMPLETE_NOTE_SPECIFIC.format(unresolved=unresolved_text)
        elif evidence_incomplete:
            incomplete_note = INCOMPLETE_NOTE_GENERIC
        else:
            incomplete_note = ""

        prompt = PROMPT_TEMPLATE.format(
            claim=claim,
            evidence=evidence_text,
            incomplete_note=incomplete_note,
            labels=", ".join(VERDICT_LABELS),
        )

    raw = call_llm(prompt)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response (len={len(raw)}). Raw:\n{raw}")
    parsed = json.loads(raw[start : end + 1])
    validated = Verdict.model_validate(parsed)

    n_evidence = len(evidence_chunks)
    invalid_citations = [c for c in validated.citations if c < 1 or c > n_evidence]

    result = validated.model_dump()
    result["invalid_citations"] = invalid_citations
    result["n_evidence_available"] = n_evidence
    return result
