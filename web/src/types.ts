// API response shapes shared across the frontend.

export interface EvidenceChunk {
  text: string;
  url: string;
  retrieval_query: string;
  injection_markers: string[];
}

export interface RoundThread {
  thread_id?: string;
  question: string;
  queries_used: string[];
  new_evidence: EvidenceChunk[];
  n_new_evidence: number;
  resolved: boolean;
  reasoning: string;
}

export interface Round {
  round: number;
  threads: RoundThread[];
}

export interface Thread {
  thread_id?: string;
  question: string;
  resolved: boolean;
  reasoning: string;
  evidence: EvidenceChunk[];
}

export interface CitedSource {
  citation_number: number;
  url: string;
}

export interface Verdict {
  label: string;
  justification: string;
  citations: number[];
  invalid_citations: number[];
  n_evidence_available: number;
  cited_sources: CitedSource[];
  escalate: boolean;
  escalation_reasons: string[];
  total_tokens_used: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
}

export interface VerifyResponse {
  claim: string;
  claim_id: string;
  verdict: Verdict;
  all_evidence: EvidenceChunk[];
  rounds: Round[];
  threads: Thread[];
  gold_label: string;
}

export interface ClaimSummary {
  claim_id: number;
  claim: string;
  gold_label: string;
}

export interface LiveThread extends Thread {
  thread_id: string;
  queries_to_run: string[];
}

export interface RetrievalDetail {
  thread_id?: string;
  question: string;
  queries_used: string[];
  new_evidence: EvidenceChunk[];
  n_new_evidence: number;
}

export interface StartRunResponse {
  run_id: string;
}

export interface RunStartedEvent {
  run_id: string;
  claim_id: string;
  claim: string;
  gold_label: string;
}

export interface DecompositionCompletedEvent {
  threads: LiveThread[];
  total_tokens_used: number;
  max_rounds: number;
}

export interface RetrievalStartedEvent {
  round: number;
  max_rounds: number;
}

export interface RetrievalCompletedEvent {
  round: number;
  threads: LiveThread[];
  details: RetrievalDetail[];
}

export interface RoundCompletedEvent {
  round: Round;
  threads: LiveThread[];
  is_sufficient: boolean;
  total_tokens_used: number;
}

export interface ThreadRetrievalStartedEvent {
  thread_id: string;
  question: string;
  round: number;
  queries: string[];
}

export interface ThreadRetrievalCompletedEvent {
  thread_id: string;
  question: string;
  round: number;
  new_hits: number;
  unique_retained: number;
}

export interface ThreadSufficiencyStartedEvent {
  thread_id: string;
  question: string;
  round: number;
}

export interface ThreadSufficiencyCompletedEvent {
  thread_id: string;
  question: string;
  round: number;
  resolved: boolean;
  reasoning: string;
  refined_queries: string[];
}
