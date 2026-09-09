import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { startVerificationRun } from "../api";
import { LABEL_CLASS } from "../labels";
import type {
  DecompositionCompletedEvent,
  LiveThread,
  RetrievalCompletedEvent,
  RetrievalStartedEvent,
  Round,
  RoundCompletedEvent,
  RunStartedEvent,
  ThreadRetrievalCompletedEvent,
  ThreadRetrievalStartedEvent,
  ThreadSufficiencyCompletedEvent,
  ThreadSufficiencyStartedEvent,
  VerifyResponse,
} from "../types";
import { CitationText } from "./CitationText";
import { EvidenceTrace, type EvidenceRegistry, type EvidenceRowEntry } from "./EvidenceTrace";
import { Masthead } from "./Masthead";

interface VerificationPageProps {
  claimId: number;
  onBack: () => void;
}

type LivePhase = "decompose" | "research" | "verdict";
type ThreadActivityStatus = "queued" | "retrieving" | "retrieved" | "evaluating" | "resolved" | "open";

interface ThreadActivity {
  round: number;
  status: ThreadActivityStatus;
  queries?: string[];
  newHits?: number;
  uniqueRetained?: number;
  reasoning?: string;
}

interface LiveState {
  claim: string;
  status: string;
  threads: LiveThread[];
  rounds: Round[];
  tokens: number;
  phase: LivePhase;
  currentRound: number;
  maxRounds: number;
  activities: Record<string, ThreadActivity>;
}

function initialLiveState(): LiveState {
  return {
    claim: "",
    status: "Decomposing the claim into research questions…",
    threads: [],
    rounds: [],
    tokens: 0,
    phase: "decompose",
    currentRound: 0,
    maxRounds: 3,
    activities: {},
  };
}

export function VerificationPage({ claimId, onBack }: VerificationPageProps) {
  const [data, setData] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState<LiveState>(initialLiveState);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let source: EventSource | null = null;
    let finished = false;
    setData(null);
    setError(null);
    setLive(initialLiveState());

    startVerificationRun(claimId, controller.signal).then(({ run_id: runId }) => {
      if (controller.signal.aborted) return;
      source = new EventSource(`/api/runs/${runId}/events`);

      const listen = <T,>(eventName: string, handler: (payload: T) => void) => {
        source?.addEventListener(eventName, (event) => {
          try {
            handler(JSON.parse((event as MessageEvent<string>).data) as T);
          } catch {
            source?.close();
            setError("The live trace returned an unreadable event.");
          }
        });
      };

      listen<RunStartedEvent>("run_started", (event) => {
        setLive((current) => ({ ...current, claim: event.claim }));
      });
      listen<DecompositionCompletedEvent>("decomposition_completed", (event) => {
        setLive((current) => ({
          ...current,
          status: `Created ${event.threads.length} research question${event.threads.length === 1 ? "" : "s"}.`,
          threads: event.threads,
          tokens: event.total_tokens_used,
          phase: "research",
          maxRounds: event.max_rounds,
          activities: Object.fromEntries(
            event.threads.map((thread) => [thread.thread_id, { round: 0, status: "queued" }]),
          ),
        }));
      });
      listen<RetrievalStartedEvent>("retrieval_started", (event) => {
        setLive((current) => ({
          ...current,
          status: `Searching the evidence store — round ${event.round}…`,
          phase: "research",
          currentRound: event.round,
          maxRounds: event.max_rounds,
        }));
      });
      listen<ThreadRetrievalStartedEvent>("thread_retrieval_started", (event) => {
        setLive((current) => ({
          ...current,
          activities: {
            ...current.activities,
            [event.thread_id]: {
              round: event.round,
              status: "retrieving",
              queries: event.queries,
            },
          },
        }));
      });
      listen<ThreadRetrievalCompletedEvent>("thread_retrieval_completed", (event) => {
        setLive((current) => ({
          ...current,
          activities: {
            ...current.activities,
            [event.thread_id]: {
              ...current.activities[event.thread_id],
              round: event.round,
              status: "retrieved",
              newHits: event.new_hits,
              uniqueRetained: event.unique_retained,
            },
          },
        }));
      });
      listen<RetrievalCompletedEvent>("retrieval_completed", (event) => {
        setLive((current) => ({
          ...current,
          status: `Checking whether round ${event.round} evidence is sufficient…`,
          threads: event.threads,
        }));
      });
      listen<ThreadSufficiencyStartedEvent>("thread_sufficiency_started", (event) => {
        setLive((current) => ({
          ...current,
          activities: {
            ...current.activities,
            [event.thread_id]: {
              ...current.activities[event.thread_id],
              round: event.round,
              status: "evaluating",
            },
          },
        }));
      });
      listen<ThreadSufficiencyCompletedEvent>("thread_sufficiency_completed", (event) => {
        setLive((current) => ({
          ...current,
          threads: current.threads.map((thread) => thread.thread_id === event.thread_id
            ? { ...thread, resolved: event.resolved, reasoning: event.reasoning }
            : thread),
          activities: {
            ...current.activities,
            [event.thread_id]: {
              ...current.activities[event.thread_id],
              round: event.round,
              status: event.resolved ? "resolved" : "open",
              reasoning: event.reasoning,
            },
          },
        }));
      });
      listen<RoundCompletedEvent>("round_completed", (event) => {
        setLive((current) => ({
          ...current,
          status: event.is_sufficient
            ? "Evidence is sufficient; preparing the verdict…"
            : `Round ${event.round.round} complete; refining unresolved questions…`,
          threads: event.threads,
          rounds: [...current.rounds.filter((round) => round.round !== event.round.round), event.round],
          tokens: event.total_tokens_used,
        }));
      });
      listen<{ rounds_completed: number }>("verdict_started", (event) => {
        setLive((current) => ({
          ...current,
          status: "Synthesizing the final verdict and validating citations…",
          phase: "verdict",
          currentRound: event.rounds_completed,
        }));
      });
      listen<VerifyResponse>("completed", (event) => {
        finished = true;
        source?.close();
        setData(event);
      });
      listen<{ message: string }>("failed", (event) => {
        finished = true;
        source?.close();
        setError(event.message);
      });
      source.onerror = () => {
        if (!finished && source?.readyState === EventSource.CLOSED) {
          setError("The live trace connection closed before the run completed.");
        }
      };
    }).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setError(reason instanceof Error ? reason.message : String(reason));
    });

    return () => {
      controller.abort();
      source?.close();
    };
  }, [claimId, retry]);

  return (
    <>
      <Masthead onBack={onBack} />
      {error ? (
        <div className="findings">
          <span className="section-title">Run failed</span>
          <p>{error}</p>
          <p style={{ marginTop: 14 }}>
            <button className="cite" onClick={() => setRetry((value) => value + 1)}>Retry</button>
          </p>
        </div>
      ) : data ? (
        <VerificationDocket data={data} />
      ) : (
        <LiveVerificationTrace live={live} />
      )}
    </>
  );
}

function LiveVerificationTrace({ live }: { live: LiveState }) {
  const phaseOrder: LivePhase[] = ["decompose", "research", "verdict"];
  const activePhaseIndex = phaseOrder.indexOf(live.phase);

  return (
    <div className="live-trace">
      <div className="live-trace-head">
        <div className="spinner" aria-hidden="true" />
        <div className="live-trace-copy">
          <span className="section-title">Live agent trace</span>
          <strong role="status" aria-live="polite">{live.status}</strong>
        </div>
        <span className="live-badge"><span />Live</span>
      </div>

      {live.claim && <p className="live-claim">“{live.claim}”</p>}

      <ol className="live-phases" aria-label="Verification stages">
        {phaseOrder.map((phase, index) => {
          const state = index < activePhaseIndex ? "complete" : index === activePhaseIndex ? "active" : "upcoming";
          const label = phase === "decompose" ? "Decompose" : phase === "research" ? "Research" : "Verdict";
          const detail = phase === "decompose"
            ? (live.threads.length ? `${live.threads.length} questions` : "Planning")
            : phase === "research"
              ? (live.currentRound ? `Round ${live.currentRound}/${live.maxRounds}` : "Pending")
              : (live.phase === "verdict" ? "Generating" : "Pending");
          return (
            <li className={state} key={phase}>
              <span className="live-phase-marker" aria-hidden="true" />
              <span><strong>{label}</strong><small>{detail}</small></span>
            </li>
          );
        })}
      </ol>

      {live.threads.length > 0 && (
        <div className="live-questions">
          <div className="live-questions-meta">
            <span>{live.threads.length} research questions</span>
            <span>{live.currentRound ? `Round ${live.currentRound}/${live.maxRounds}` : "Research pending"}</span>
            <span>{live.tokens.toLocaleString()} tokens</span>
          </div>
          <div className="live-question-list">
          {live.threads.map((thread) => {
            const activity = live.activities[thread.thread_id];
            const threadRounds = live.rounds.flatMap((round) => {
              const result = round.threads.find((item) =>
                item.thread_id === thread.thread_id || item.question === thread.question,
              );
              return result && result.queries_used.length > 0 ? [{ round: round.round, result }] : [];
            });
            const resolvedRound = threadRounds.find(({ result }) => result.resolved)?.round;
            const evidenceCount = activity?.uniqueRetained ?? thread.evidence.length;
            const activeRoundCommitted = activity
              ? threadRounds.some(({ round }) => round === activity.round)
              : false;
            const showActiveRound = Boolean(activity && activity.round > 0 && !activeRoundCommitted);
            const statusLabel = thread.resolved
              ? `Resolved${resolvedRound ? ` · R${resolvedRound}` : ""}`
              : activity?.status === "retrieving"
                ? "Retrieving"
                : activity?.status === "evaluating"
                  ? "Evaluating"
                  : threadRounds.length > 0
                    ? "Open"
                    : "Queued";
            const statusClass = thread.resolved
              ? "resolved"
              : activity?.status === "retrieving" || activity?.status === "evaluating"
                ? "active"
                : threadRounds.length > 0
                  ? "unresolved"
                  : "pending";
            return (
              <section className="live-question-card" key={thread.thread_id}>
                <div className="live-question-head">
                  <span className={`status-chip ${statusClass}`}>{statusLabel}</span>
                  <div>
                    <strong>{thread.question}</strong>
                    <small>{evidenceCount} unique evidence item{evidenceCount === 1 ? "" : "s"} retained</small>
                  </div>
                </div>
                <div className="live-question-rounds">
                  {threadRounds.map(({ round, result }) => (
                    <div className="live-question-round" key={round}>
                      <span className="round-tag">Round {round}</span>
                      <div>
                        <span className="live-round-query">
                          {result.queries_used.length > 0 ? result.queries_used.join(" · ") : "No additional query"}
                        </span>
                        {result.reasoning && <small>{result.reasoning}</small>}
                      </div>
                      <span className={`live-round-outcome ${result.resolved ? "resolved" : "open"}`}>
                        {result.n_new_evidence} hit{result.n_new_evidence === 1 ? "" : "s"} · {result.resolved ? "resolved" : "open"}
                      </span>
                    </div>
                  ))}
                  {showActiveRound && activity && (
                    <div className="live-question-round active">
                      <span className="round-tag">Round {activity.round}</span>
                      <div>
                        <span className="live-round-query">
                          {activity.queries?.join(" · ") || "Using retained evidence"}
                        </span>
                        {activity.reasoning && <small>{activity.reasoning}</small>}
                      </div>
                      <span className="live-round-outcome active">
                        {activity.status === "retrieving"
                          ? "retrieving…"
                          : activity.status === "retrieved"
                            ? `${activity.newHits ?? 0} retrieval hit${activity.newHits === 1 ? "" : "s"}`
                            : activity.status === "evaluating"
                              ? "evaluating…"
                              : activity.status === "resolved"
                                ? "resolved"
                                : activity.status === "open"
                                  ? "needs refinement"
                                  : "queued"}
                      </span>
                    </div>
                  )}
                  {threadRounds.length === 0 && !showActiveRound && (
                    <div className="live-question-round empty">Waiting for the first retrieval round…</div>
                  )}
                </div>
              </section>
            );
          })}
          </div>
        </div>
      )}
    </div>
  );
}

function VerificationDocket({ data }: { data: VerifyResponse }) {
  const [showTrace, setShowTrace] = useState(false);
  const entriesByThread = useRef<Map<string, EvidenceRowEntry>[]>([]);
  const globalEntries = useRef(new Map<string, EvidenceRowEntry>());

  const register = useCallback((threadIndex: number, text: string, entry: EvidenceRowEntry) => {
    const byThread = entriesByThread.current[threadIndex] ?? new Map<string, EvidenceRowEntry>();
    entriesByThread.current[threadIndex] = byThread;
    if (!byThread.has(text)) byThread.set(text, entry);
    if (!globalEntries.current.has(text)) globalEntries.current.set(text, entry);

    return () => {
      if (byThread.get(text) === entry) byThread.delete(text);
      if (globalEntries.current.get(text) === entry) globalEntries.current.delete(text);
    };
  }, []);

  const jumpTo = useCallback((text: string, scopeThreadIndex: number | null) => {
    const scopedEntry = scopeThreadIndex === null ? undefined : entriesByThread.current[scopeThreadIndex]?.get(text);
    const entry = scopedEntry ?? globalEntries.current.get(text);
    if (!entry) return;

    const roundButton = entry.getRoundButton();
    if (roundButton?.getAttribute("aria-expanded") !== "true") roundButton?.click();
    entry.row.scrollIntoView({ behavior: "smooth", block: "center" });
    if (entry.row.getAttribute("aria-expanded") !== "true") entry.row.click();
    entry.row.style.outline = "2px solid var(--accent)";
    window.setTimeout(() => {
      entry.row.style.outline = "";
    }, 1200);
  }, []);

  const registry = useMemo<EvidenceRegistry>(() => ({ register, jumpTo }), [jumpTo, register]);
  const citedTextToNumber = useMemo(() => {
    const map = new Map<string, number>();
    data.verdict.citations.forEach((citationNumber) => {
      const chunk = data.all_evidence[citationNumber - 1];
      if (chunk) map.set(chunk.text, citationNumber);
    });
    return map;
  }, [data]);

  const verdict = data.verdict;
  const labelClass = LABEL_CLASS[verdict.label] || "";

  return (
    <>
      <div className="exhibit-header">
        <span className="exhibit-label">Claim under review</span>
        <p className="claim-text">“{data.claim}”</p>
        <div className="claim-meta">
          <span>Rounds <b>{data.rounds.length}</b></span>
          <span>Evidence reviewed <b>{data.all_evidence.length}</b></span>
          <span>Sub-questions <b>{data.threads.length}</b></span>
          <span>Tokens used <b>{verdict.total_tokens_used.toLocaleString()}</b></span>
        </div>
        <div className={`stamp${labelClass ? ` ${labelClass}` : ""}`}>{verdict.label}</div>
      </div>

      {verdict.escalate && (
        <div className="escalation-banner">
          <span className="escalation-title">⚠ Needs human review before this verdict is treated as final</span>
          <ul>{verdict.escalation_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
        </div>
      )}

      <div className="findings">
        <span className="section-title">Findings</span>
        <p>
          <CitationText
            evidence={data.all_evidence}
            onCitationClick={registry.jumpTo}
            scopeThreadIndex={null}
            text={verdict.justification}
          />
        </p>
      </div>

      <div className="view-toggle">
        <button className={showTrace ? "" : "active"} onClick={() => setShowTrace(false)}>Result only</button>
        <button className={showTrace ? "active" : ""} onClick={() => setShowTrace(true)}>Show reasoning trace</button>
      </div>
      <div className="trace" hidden={!showTrace}>
        <EvidenceTrace
          citedTextToNumber={citedTextToNumber}
          registry={registry}
          rounds={data.rounds}
          threads={data.threads}
        />
      </div>

      <footer className="foot">Agentic Fact Verifier</footer>
    </>
  );
}
