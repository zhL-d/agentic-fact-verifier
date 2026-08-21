
export interface EvidenceChunk {
  text: string;
  url: string;
  retrieval_query: string;
}

export interface RoundThread {
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
