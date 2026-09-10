"""Validated public API contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["queued", "running", "retrying", "cancelling", "cancelled", "succeeded", "failed"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class EvidenceChunk(ApiModel):
    text: str
    url: str
    retrieval_query: str = ""
    injection_markers: list[str] = Field(default_factory=list)


class RoundThread(ApiModel):
    thread_id: str | None = None
    question: str
    queries_used: list[str]
    new_evidence: list[EvidenceChunk]
    n_new_evidence: int
    resolved: bool
    reasoning: str


class Round(ApiModel):
    round: int
    threads: list[RoundThread]


class VerificationThread(ApiModel):
    thread_id: str
    question: str
    resolved: bool
    reasoning: str
    evidence: list[EvidenceChunk]


class CitedSource(ApiModel):
    citation_number: int
    url: str


class Verdict(ApiModel):
    label: str
    justification: str
    citations: list[int]
    invalid_citations: list[int]
    n_evidence_available: int
    cited_sources: list[CitedSource]
    escalate: bool
    escalation_reasons: list[str]
    total_tokens_used: int
    total_prompt_tokens: int
    total_completion_tokens: int


class VerificationResponse(ApiModel):
    claim: str
    claim_id: str
    verdict: Verdict
    all_evidence: list[EvidenceChunk]
    rounds: list[Round]
    threads: list[VerificationThread]
    gold_label: str


class ClaimSummary(ApiModel):
    claim_id: int
    claim: str
    gold_label: str


class StartRunResponse(ApiModel):
    run_id: str
    status: RunStatus


class RunResponse(ApiModel):
    run_id: str
    claim_id: int
    claim: str
    gold_label: str
    status: RunStatus
    cancel_requested: bool
    last_event_id: int
    result: VerificationResponse | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class CancelRunResponse(ApiModel):
    run_id: str
    status: RunStatus


class HealthResponse(ApiModel):
    status: Literal["ok"]


class EventEnvelope(ApiModel):
    id: int
    event: str
    data: dict[str, Any]
