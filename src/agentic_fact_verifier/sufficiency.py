"""Evidence sufficiency check: given each sub-question's own accumulated
evidence, judges per sub-question whether it's sufficient, and if not,
proposes refined search queries. One LLM call handles every thread per
round."""

import json

from pydantic import BaseModel, Field

from agentic_fact_verifier.llm_client import call_llm, schema_response_format
from agentic_fact_verifier.prompt_guard import ANTI_INJECTION_NOTICE, wrap_evidence

PROMPT_TEMPLATE = """You are checking whether enough evidence has been gathered to \
answer each of several independent fact-checking sub-questions, all derived from \
the same claim below. Each sub-question has its OWN evidence gathered so far, \
evidence for one sub-question must not be used to judge a different sub-question.

""" + ANTI_INJECTION_NOTICE + """
Original claim: {claim}

{threads}

For EACH sub-question above, judge whether ITS OWN evidence is sufficient to \
answer it. If not, propose a small set of refined search queries, different from \
previous queries for that sub-question (broader, narrower, or rephrased), that \
would help fill the gap for THAT sub-question specifically.

Every refined query must be a concrete search-engine query, not a description of \
what information is missing. It must keep the claim's actual specific entities \
names, numbers, dates, places, from the claim above. Do NOT write generic \
meta-instructions like "identify the source of the claim" or "find the \
publication date", those keep no searchable content once separated from the \
claim they came from, so they won't match anything in the evidence being \
searched. Instead, rephrase the same concrete facts differently: try synonyms, \
narrower or broader phrasing, or a different angle on the same specific details.

Respond with a single JSON object only, in this exact shape:
{{"threads": [{{"index": <int, matching the sub-question number above>, \
"resolved": <true or false>, "reasoning": "<brief reasoning>", \
"refined_queries": ["<query>", ...]}}, ...]}}

Include exactly one entry per sub-question, with its correct index. If resolved \
is true, refined_queries should be an empty list. Keep each reasoning to at most \
1 sentence.
"""


class ThreadSufficiency(BaseModel):
    index: int
    resolved: bool
    reasoning: str
    refined_queries: list[str] = Field(default_factory=list)


class SufficiencyCheck(BaseModel):
    threads: list[ThreadSufficiency]


_RESPONSE_FORMAT = schema_response_format(SufficiencyCheck, "SufficiencyCheck")


def check_sufficiency(claim: str, threads: list[dict]) -> tuple[SufficiencyCheck, dict]:
    """`threads` is a list of {"question": str, "evidence": list[dict]}, 
    one per sub-question, each carrying only its own accumulated evidence.

    Returns (check, usage), usage is this call's token usage."""
    blocks = []
    for i, t in enumerate(threads):
        evidence_text = "\n\n".join(
            wrap_evidence(j + 1, chunk.get("url", ""), chunk["text"])
            for j, chunk in enumerate(t["evidence"])
        ) or "  (none yet)"
        blocks.append(
            f"Sub-question {i}: {t['question']}\nEvidence for sub-question {i}:\n{evidence_text}"
        )
    threads_text = "\n\n".join(blocks)

    prompt = PROMPT_TEMPLATE.format(claim=claim, threads=threads_text)
    result = call_llm(prompt, max_tokens=10000, response_format=_RESPONSE_FORMAT)
    check = SufficiencyCheck.model_validate(json.loads(result.content))

    by_index = {t.index: t for t in check.threads}
    normalized = []
    for i in range(len(threads)):
        if i in by_index:
            normalized.append(by_index[i])
        else:
            normalized.append(
                ThreadSufficiency(
                    index=i,
                    resolved=False,
                    reasoning="(model did not return a result for this sub-question)",
                    refined_queries=[],
                )
            )
    return SufficiencyCheck(threads=normalized), result.usage
