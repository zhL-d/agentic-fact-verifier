import { z } from "zod";

export const evidenceChunkSchema = z.object({
  text: z.string(),
  url: z.string(),
  retrieval_query: z.string().default(""),
  injection_markers: z.array(z.string()).default([]),
});

export const roundThreadSchema = z.object({
  thread_id: z.string().optional(),
  question: z.string(),
  queries_used: z.array(z.string()),
  new_evidence: z.array(evidenceChunkSchema),
  n_new_evidence: z.number().int().nonnegative(),
  resolved: z.boolean(),
  reasoning: z.string(),
});

export const roundSchema = z.object({
  round: z.number().int().positive(),
  threads: z.array(roundThreadSchema),
});

export const threadSchema = z.object({
  thread_id: z.string().optional(),
  question: z.string(),
  resolved: z.boolean(),
  reasoning: z.string(),
  evidence: z.array(evidenceChunkSchema),
});

export const liveThreadSchema = threadSchema.extend({
  thread_id: z.string(),
  queries_to_run: z.array(z.string()),
});

export const verdictSchema = z.object({
  label: z.string(),
  justification: z.string(),
  citations: z.array(z.number().int().positive()),
  invalid_citations: z.array(z.number().int().positive()),
  n_evidence_available: z.number().int().nonnegative(),
  cited_sources: z.array(z.object({ citation_number: z.number().int().positive(), url: z.string() })),
  escalate: z.boolean(),
  escalation_reasons: z.array(z.string()),
  total_tokens_used: z.number().int().nonnegative(),
  total_prompt_tokens: z.number().int().nonnegative(),
  total_completion_tokens: z.number().int().nonnegative(),
});

export const verifyResponseSchema = z.object({
  claim: z.string(),
  claim_id: z.string(),
  verdict: verdictSchema,
  all_evidence: z.array(evidenceChunkSchema),
  rounds: z.array(roundSchema),
  threads: z.array(threadSchema),
  gold_label: z.string(),
});

export const claimSummarySchema = z.object({
  claim_id: z.number().int().nonnegative(),
  claim: z.string(),
  gold_label: z.string(),
});

export const runStatusSchema = z.enum([
  "queued",
  "running",
  "retrying",
  "cancelling",
  "cancelled",
  "succeeded",
  "failed",
]);

export const startRunResponseSchema = z.object({ run_id: z.string().uuid(), status: runStatusSchema });

export const runResponseSchema = z.object({
  run_id: z.string().uuid(),
  claim_id: z.number().int().nonnegative(),
  claim: z.string(),
  gold_label: z.string(),
  status: runStatusSchema,
  cancel_requested: z.boolean(),
  last_event_id: z.number().int().nonnegative(),
  result: verifyResponseSchema.nullable(),
  error: z.string().nullable(),
  created_at: z.string(),
  started_at: z.string().nullable(),
  finished_at: z.string().nullable(),
});

export const runStartedEventSchema = z.object({
  run_id: z.string().uuid(),
  claim_id: z.string(),
  claim: z.string(),
  gold_label: z.string(),
});

export const decompositionCompletedEventSchema = z.object({
  threads: z.array(liveThreadSchema),
  total_tokens_used: z.number().int().nonnegative(),
  max_rounds: z.number().int().positive(),
});

export const retrievalStartedEventSchema = z.object({
  round: z.number().int().positive(),
  max_rounds: z.number().int().positive(),
});

export const retrievalDetailSchema = z.object({
  thread_id: z.string().optional(),
  question: z.string(),
  queries_used: z.array(z.string()),
  new_evidence: z.array(evidenceChunkSchema),
  n_new_evidence: z.number().int().nonnegative(),
});

export const retrievalCompletedEventSchema = z.object({
  round: z.number().int().positive(),
  threads: z.array(liveThreadSchema),
  details: z.array(retrievalDetailSchema),
});

export const roundCompletedEventSchema = z.object({
  round: roundSchema,
  threads: z.array(liveThreadSchema),
  is_sufficient: z.boolean(),
  total_tokens_used: z.number().int().nonnegative(),
});

export const threadRetrievalStartedEventSchema = z.object({
  thread_id: z.string(),
  question: z.string(),
  round: z.number().int().positive(),
  queries: z.array(z.string()),
});

export const threadRetrievalCompletedEventSchema = z.object({
  thread_id: z.string(),
  question: z.string(),
  round: z.number().int().positive(),
  new_hits: z.number().int().nonnegative(),
  unique_retained: z.number().int().nonnegative(),
});

export const threadSufficiencyStartedEventSchema = z.object({
  thread_id: z.string(),
  question: z.string(),
  round: z.number().int().positive(),
});

export const threadSufficiencyCompletedEventSchema = z.object({
  thread_id: z.string(),
  question: z.string(),
  round: z.number().int().positive(),
  resolved: z.boolean(),
  reasoning: z.string(),
  refined_queries: z.array(z.string()),
});

export const retryingEventSchema = z.object({ attempt: z.number().int().positive(), message: z.string() });
export const verdictStartedEventSchema = z.object({ rounds_completed: z.number().int().positive() });
export const terminalMessageEventSchema = z.object({ message: z.string() });

export type EvidenceChunk = z.infer<typeof evidenceChunkSchema>;
export type RoundThread = z.infer<typeof roundThreadSchema>;
export type Round = z.infer<typeof roundSchema>;
export type Thread = z.infer<typeof threadSchema>;
export type LiveThread = z.infer<typeof liveThreadSchema>;
export type Verdict = z.infer<typeof verdictSchema>;
export type VerifyResponse = z.infer<typeof verifyResponseSchema>;
export type ClaimSummary = z.infer<typeof claimSummarySchema>;
export type RunStatus = z.infer<typeof runStatusSchema>;
export type StartRunResponse = z.infer<typeof startRunResponseSchema>;
export type RunResponse = z.infer<typeof runResponseSchema>;
export type RunStartedEvent = z.infer<typeof runStartedEventSchema>;
export type DecompositionCompletedEvent = z.infer<typeof decompositionCompletedEventSchema>;
export type RetrievalDetail = z.infer<typeof retrievalDetailSchema>;
export type RetrievalStartedEvent = z.infer<typeof retrievalStartedEventSchema>;
export type RetrievalCompletedEvent = z.infer<typeof retrievalCompletedEventSchema>;
export type RoundCompletedEvent = z.infer<typeof roundCompletedEventSchema>;
export type ThreadRetrievalStartedEvent = z.infer<typeof threadRetrievalStartedEventSchema>;
export type ThreadRetrievalCompletedEvent = z.infer<typeof threadRetrievalCompletedEventSchema>;
export type ThreadSufficiencyStartedEvent = z.infer<typeof threadSufficiencyStartedEventSchema>;
export type ThreadSufficiencyCompletedEvent = z.infer<typeof threadSufficiencyCompletedEventSchema>;
