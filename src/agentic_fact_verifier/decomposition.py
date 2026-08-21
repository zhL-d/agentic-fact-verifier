"""Claim decomposition: break a claim into independently-verifiable
sub-questions, so retrieval happens per sub-question instead of against
the raw claim text.

Each sub-question now carries TWO separate strings: `question` (a
readable question, for the audit trail / UI) and `initial_query` (a
concrete search-engine query, kept anchored to the claim's actual
entities/numbers/dates, not a generic restatement of the question). These
used to be conflated into one string, reused verbatim as the literal
round-1 Elasticsearch query."""

import json

from pydantic import BaseModel, Field

from agentic_fact_verifier.llm_client import call_llm

PROMPT_TEMPLATE = """Break the following fact-checking claim into a small set of \
independently-verifiable sub-questions. Each sub-question should be answerable \
on its own, using evidence retrieved from the web, and together they should \
cover everything needed to verify the claim.

Claim: {claim}

For each sub-question, ALSO provide an initial search query to find evidence for \
it. The query must be a concrete search-engine query, not a restatement of the \
question — keep the claim's actual specific entities (names, numbers, dates, \
places) in it. Do not write generic phrasing like "who made this claim" or "when \
was this stated" with nothing else — a query like that has no claim-specific \
content to match against and won't find anything useful.

Respond with a single JSON object only, in this exact shape:
{{"sub_questions": [{{"question": "<question 1>", "initial_query": "<search query \
1>"}}, ...]}}

Use 2-5 sub-questions. Do not include the claim itself as a sub-question.
"""


class SubQuestion(BaseModel):
    question: str
    initial_query: str


class SubQuestions(BaseModel):
    sub_questions: list[SubQuestion] = Field(min_length=1)


def decompose_claim(claim: str) -> list[SubQuestion]:
    prompt = PROMPT_TEMPLATE.format(claim=claim)
    raw = call_llm(prompt)

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in decomposition response (len={len(raw)}). Raw:\n{raw}")
    parsed = json.loads(raw[start : end + 1])
    return SubQuestions.model_validate(parsed).sub_questions
