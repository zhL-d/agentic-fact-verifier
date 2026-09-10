import { useCallback, useEffect, useState } from "react";
import type { ZodType } from "zod";
import { cancelRun, fetchRun, startVerificationRun } from "./api";
import {
  decompositionCompletedEventSchema,
  retrievalCompletedEventSchema,
  retrievalStartedEventSchema,
  retryingEventSchema,
  roundCompletedEventSchema,
  runStartedEventSchema,
  terminalMessageEventSchema,
  threadRetrievalCompletedEventSchema,
  threadRetrievalStartedEventSchema,
  threadSufficiencyCompletedEventSchema,
  threadSufficiencyStartedEventSchema,
  verdictStartedEventSchema,
  verifyResponseSchema,
  type LiveThread,
  type Round,
  type VerifyResponse,
} from "./types";

export type LivePhase = "decompose" | "research" | "verdict";
type ThreadActivityStatus = "queued" | "retrieving" | "retrieved" | "evaluating" | "resolved" | "open";

interface ThreadActivity {
  round: number;
  status: ThreadActivityStatus;
  queries?: string[];
  newHits?: number;
  uniqueRetained?: number;
  reasoning?: string;
}

export interface LiveState {
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

export function initialLiveState(): LiveState {
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

interface UseVerificationRunOptions {
  claimId: number;
  initialRunId?: string;
  onRunStarted: (runId: string) => void;
}

export function useVerificationRun({ claimId, initialRunId, onRunStarted }: UseVerificationRunOptions) {
  const [data, setData] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorTitle, setErrorTitle] = useState("Run failed");
  const [live, setLive] = useState<LiveState>(initialLiveState);
  const [runId, setRunId] = useState<string | null>(initialRunId ?? null);
  const [attempt, setAttempt] = useState(0);
  const [startFresh, setStartFresh] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let source: EventSource | null = null;
    let finished = false;
    let reconnectTimer: number | undefined;

    setData(null);
    setError(null);
    setErrorTitle("Run failed");
    setLive(initialLiveState());

    const fail = (message: string, title = "Run failed") => {
      finished = true;
      source?.close();
      setErrorTitle(title);
      setError(message);
    };

    const connect = (id: string) => {
      if (controller.signal.aborted) return;
      source = new EventSource(`/api/runs/${id}/events`);

      const listen = <T,>(eventName: string, schema: ZodType<T>, handler: (payload: T) => void) => {
        source?.addEventListener(eventName, (event) => {
          try {
            const payload = schema.parse(JSON.parse((event as MessageEvent<string>).data));
            handler(payload);
          } catch {
            fail("The live trace returned an unreadable event.");
          }
        });
      };

      listen("run_started", runStartedEventSchema, (event) => {
        setLive((current) => ({ ...current, claim: event.claim }));
      });
      listen("decomposition_completed", decompositionCompletedEventSchema, (event) => {
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
      listen("retrieval_started", retrievalStartedEventSchema, (event) => {
        setLive((current) => ({
          ...current,
          status: `Searching the evidence store — round ${event.round}…`,
          phase: "research",
          currentRound: event.round,
          maxRounds: event.max_rounds,
        }));
      });
      listen("thread_retrieval_started", threadRetrievalStartedEventSchema, (event) => {
        setLive((current) => ({
          ...current,
          activities: {
            ...current.activities,
            [event.thread_id]: { round: event.round, status: "retrieving", queries: event.queries },
          },
        }));
      });
      listen("thread_retrieval_completed", threadRetrievalCompletedEventSchema, (event) => {
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
      listen("retrieval_completed", retrievalCompletedEventSchema, (event) => {
        setLive((current) => ({
          ...current,
          status: `Checking whether round ${event.round} evidence is sufficient…`,
          threads: event.threads,
        }));
      });
      listen("thread_sufficiency_started", threadSufficiencyStartedEventSchema, (event) => {
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
      listen("thread_sufficiency_completed", threadSufficiencyCompletedEventSchema, (event) => {
        setLive((current) => ({
          ...current,
          threads: current.threads.map((thread) =>
            thread.thread_id === event.thread_id
              ? { ...thread, resolved: event.resolved, reasoning: event.reasoning }
              : thread,
          ),
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
      listen("round_completed", roundCompletedEventSchema, (event) => {
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
      listen("verdict_started", verdictStartedEventSchema, (event) => {
        setLive((current) => ({
          ...current,
          status: "Synthesizing the final verdict and validating citations…",
          phase: "verdict",
          currentRound: event.rounds_completed,
        }));
      });
      listen("retrying", retryingEventSchema, (event) => {
        setLive((current) => ({
          ...current,
          status: `Temporary dependency failure; retrying (${event.attempt})…`,
        }));
      });
      listen("completed", verifyResponseSchema, (event) => {
        finished = true;
        source?.close();
        setData(event);
      });
      listen("failed", terminalMessageEventSchema, (event) => fail(event.message));
      listen("cancelled", terminalMessageEventSchema, (event) => fail(event.message, "Run stopped"));

      source.onopen = () => {
        if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
        reconnectTimer = undefined;
      };
      source.onerror = () => {
        if (finished || reconnectTimer !== undefined) return;
        setLive((current) => ({ ...current, status: "Connection interrupted; reconnecting…" }));
        reconnectTimer = window.setTimeout(() => {
          void fetchRun(id, controller.signal)
            .then((run) => {
              if (run.status === "succeeded" && run.result) setData(run.result);
              else if (run.status === "failed") fail(run.error ?? "Verification failed.");
              else if (run.status === "cancelled") fail("Verification cancelled.", "Run stopped");
              else fail("The run is still active, but the live trace could not reconnect.");
            })
            .catch((reason: unknown) => fail(reason instanceof Error ? reason.message : String(reason)));
        }, 30_000);
      };
    };

    const startOrAttach = async () => {
      try {
        const existingRunId = startFresh ? undefined : initialRunId;
        if (existingRunId) {
          const existing = await fetchRun(existingRunId, controller.signal);
          setRunId(existingRunId);
          if (existing.status === "succeeded" && existing.result) {
            finished = true;
            setData(existing.result);
            return;
          }
          if (existing.status === "failed") {
            fail(existing.error ?? "Verification failed.");
            return;
          }
          if (existing.status === "cancelled") {
            fail("Verification cancelled.", "Run stopped");
            return;
          }
          connect(existingRunId);
          return;
        }

        const idempotencyKey = crypto.randomUUID();
        const started = await startVerificationRun(claimId, idempotencyKey, controller.signal);
        setRunId(started.run_id);
        onRunStarted(started.run_id);
        connect(started.run_id);
      } catch (reason: unknown) {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        fail(reason instanceof Error ? reason.message : String(reason));
      }
    };

    void startOrAttach();
    return () => {
      controller.abort();
      source?.close();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
    };
  }, [attempt, claimId, initialRunId, onRunStarted, startFresh]);

  const retry = useCallback(() => {
    setStartFresh(true);
    setAttempt((value) => value + 1);
  }, []);

  const cancel = useCallback(async () => {
    if (!runId || data || error) return;
    setLive((current) => ({ ...current, status: "Cancelling verification…" }));
    await cancelRun(runId);
  }, [data, error, runId]);

  return { cancel, data, error, errorTitle, live, retry, runId };
}
