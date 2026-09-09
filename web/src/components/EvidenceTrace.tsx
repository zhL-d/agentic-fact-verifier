import { useCallback, useEffect, useRef, useState } from "react";
import type { EvidenceChunk, Round, RoundThread, Thread } from "../types";
import { CitationText, domainOf } from "./CitationText";

export interface EvidenceRowEntry {
  getRoundButton: () => HTMLButtonElement | null;
  row: HTMLButtonElement;
}

export interface EvidenceRegistry {
  register: (threadIndex: number, text: string, entry: EvidenceRowEntry) => () => void;
  jumpTo: (text: string, scopeThreadIndex: number | null) => void;
}

interface EvidenceTraceProps {
  threads: Thread[];
  rounds: Round[];
  citedTextToNumber: Map<string, number>;
  registry: EvidenceRegistry;
}

export function EvidenceTrace({ threads, rounds, citedTextToNumber, registry }: EvidenceTraceProps) {
  return (
    <>
      {threads.map((thread, threadIndex) => (
        <ThreadCard
          citedTextToNumber={citedTextToNumber}
          key={`${threadIndex}-${thread.question}`}
          registry={registry}
          rounds={rounds}
          thread={thread}
          threadIndex={threadIndex}
        />
      ))}
    </>
  );
}

interface ThreadCardProps {
  thread: Thread;
  threadIndex: number;
  rounds: Round[];
  citedTextToNumber: Map<string, number>;
  registry: EvidenceRegistry;
}

function ThreadCard({ thread, threadIndex, rounds, citedTextToNumber, registry }: ThreadCardProps) {
  const statusLabel = thread.resolved
    ? "Resolved"
    : `Unresolved after ${rounds.length} round${rounds.length === 1 ? "" : "s"}`;

  return (
    <div className="exhibit-card">
      <div className="exhibit-card-head">
        <span className={`status-chip ${thread.resolved ? "resolved" : "unresolved"}`}>{statusLabel}</span>
        <span className="question">{thread.question}</span>
      </div>
      <div className="exhibit-card-body">
        <div className="round-history">
          {rounds.map((round) => {
            const roundThread = round.threads[threadIndex];
            if (!roundThread || roundThread.queries_used.length === 0) return null;
            return (
              <EvidenceRound
                citedTextToNumber={citedTextToNumber}
                isFinalRound={round.round === rounds.length}
                key={round.round}
                registry={registry}
                round={round}
                roundThread={roundThread}
                threadIndex={threadIndex}
                threadResolved={thread.resolved}
              />
            );
          })}
        </div>
        <p className="reasoning">
          <CitationText
            evidence={thread.evidence}
            onCitationClick={registry.jumpTo}
            scopeThreadIndex={threadIndex}
            text={thread.reasoning}
          />
        </p>
      </div>
    </div>
  );
}

interface EvidenceRoundProps {
  round: Round;
  roundThread: RoundThread;
  threadIndex: number;
  threadResolved: boolean;
  isFinalRound: boolean;
  citedTextToNumber: Map<string, number>;
  registry: EvidenceRegistry;
}

function EvidenceRound({
  round,
  roundThread,
  threadIndex,
  threadResolved,
  isFinalRound,
  citedTextToNumber,
  registry,
}: EvidenceRoundProps) {
  const [expanded, setExpanded] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const getRoundButton = useCallback(() => buttonRef.current, []);
  const outcomeClass = roundThread.resolved ? "resolved-now" : isFinalRound ? "still-open" : "gain";
  const outcomeText = roundThread.resolved
    ? `+${roundThread.n_new_evidence} · resolved`
    : `+${roundThread.n_new_evidence}${isFinalRound && !threadResolved ? " · still unresolved" : ""}`;

  return (
    <div className="round-entry-wrap">
      <button
        aria-expanded={expanded}
        className="round-entry"
        onClick={() => setExpanded((current) => !current)}
        ref={buttonRef}
      >
        <span className="round-tag">Round {round.round}</span>
        <span className="round-query">
          <span className="label">{round.round === 1 ? "query" : "refined"}</span>
          {`"${roundThread.queries_used.join('", "')}"`}
        </span>
        <span className={`round-outcome ${outcomeClass}`}>{outcomeText}</span>
        <span className="chevron">▶</span>
      </button>
      <div className="round-evidence" hidden={!expanded}>
        <div className="evidence-list">
          {roundThread.new_evidence.map((chunk, evidenceIndex) => (
            <EvidenceRow
              chunk={chunk}
              citationNumber={citedTextToNumber.get(chunk.text)}
              getRoundButton={getRoundButton}
              key={`${evidenceIndex}-${chunk.url}`}
              registry={registry}
              threadIndex={threadIndex}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

interface EvidenceRowProps {
  chunk: EvidenceChunk;
  citationNumber?: number;
  threadIndex: number;
  getRoundButton: () => HTMLButtonElement | null;
  registry: EvidenceRegistry;
}

function EvidenceRow({ chunk, citationNumber, threadIndex, getRoundButton, registry }: EvidenceRowProps) {
  const [expanded, setExpanded] = useState(false);
  const rowRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const row = rowRef.current;
    if (!row) return;
    return registry.register(threadIndex, chunk.text, { getRoundButton, row });
  }, [chunk.text, getRoundButton, registry, threadIndex]);

  return (
    <>
      <button
        aria-expanded={expanded}
        className="evidence-row"
        onClick={() => setExpanded((current) => !current)}
        ref={rowRef}
      >
        <span className="evidence-tag">{domainOf(chunk.url)}</span>
        <span className="url">{chunk.url}</span>
        {citationNumber !== undefined && <span className="cited-badge">Cited as [{citationNumber}]</span>}
        {chunk.injection_markers?.length > 0 && <span className="injection-flag">⚠ flagged</span>}
        <span className="chevron">▶</span>
      </button>
      <div className="evidence-detail" hidden={!expanded}>
        <span className="found-via">Found via: &quot;{chunk.retrieval_query}&quot;</span>
        <span className="excerpt">{chunk.text}</span>
      </div>
    </>
  );
}
