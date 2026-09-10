import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import type { VerifyResponse } from "./types";

const verification: VerifyResponse = {
  claim: "The test claim",
  claim_id: "1",
  gold_label: "Supported",
  all_evidence: [
    {
      text: "The supporting evidence",
      url: "https://example.com/evidence",
      retrieval_query: "test query",
      injection_markers: [],
    },
  ],
  rounds: [
    {
      round: 1,
      threads: [
        {
          question: "Was the claim reported?",
          queries_used: ["test query"],
          new_evidence: [
            {
              text: "The supporting evidence",
              url: "https://example.com/evidence",
              retrieval_query: "test query",
              injection_markers: [],
            },
          ],
          n_new_evidence: 1,
          resolved: true,
          reasoning: "It was reported [1].",
        },
      ],
    },
  ],
  threads: [
    {
      question: "Was the claim reported?",
      resolved: true,
      reasoning: "It was reported [1].",
      evidence: [
        {
          text: "The supporting evidence",
          url: "https://example.com/evidence",
          retrieval_query: "test query",
          injection_markers: [],
        },
      ],
    },
  ],
  verdict: {
    label: "Supported",
    justification: "The evidence supports the claim [1].",
    citations: [1],
    invalid_citations: [],
    n_evidence_available: 1,
    cited_sources: [{ citation_number: 1, url: "https://example.com/evidence" }],
    escalate: false,
    escalation_reasons: [],
    total_tokens_used: 1234,
    total_prompt_tokens: 1000,
    total_completion_tokens: 234,
  },
};

const RUN_ID = "00000000-0000-4000-8000-000000000123";

class FakeEventSource {
  static readonly CLOSED = 2;
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readyState = 1;
  onerror: (() => void) | null = null;
  private listeners = new Map<string, ((event: MessageEvent<string>) => void)[]>();

  constructor(url: string | URL) {
    this.url = String(url);
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = listener as (event: MessageEvent<string>) => void;
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]);
  }

  close() {
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(type: string, payload: unknown) {
    const event = new MessageEvent(type, { data: JSON.stringify(payload) });
    this.listeners.get(type)?.forEach((listener) => listener(event));
  }
}

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    FakeEventSource.instances = [];
    window.history.replaceState(null, "", "/");
  });

  it("renders live graph events before the final expandable evidence docket", async () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify([{ claim_id: 1, claim: "The test claim", gold_label: "Supported" }]), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ run_id: RUN_ID, status: "queued" }), {
          status: 202,
          headers: { "Content-Type": "application/json" },
        }),
      );

    const user = userEvent.setup();
    render(<App />);

    await user.click(await screen.findByRole("button", { name: /The test claim/ }));
    expect(screen.getByText("Live agent trace")).toBeInTheDocument();
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.instances[0].url).toBe(`/api/runs/${RUN_ID}/events`);

    act(() => {
      const source = FakeEventSource.instances[0];
      source.emit("run_started", {
        run_id: RUN_ID,
        claim_id: "1",
        claim: verification.claim,
        gold_label: verification.gold_label,
      });
      source.emit("decomposition_completed", {
        threads: [{ ...verification.threads[0], thread_id: "q1", queries_to_run: ["test query"] }],
        total_tokens_used: 100,
        max_rounds: 3,
      });
      source.emit("retrieval_started", { round: 1, max_rounds: 3 });
      source.emit("thread_retrieval_started", {
        thread_id: "q1",
        question: verification.threads[0].question,
        round: 1,
        queries: ["test query"],
      });
      source.emit("thread_retrieval_completed", {
        thread_id: "q1",
        question: verification.threads[0].question,
        round: 1,
        new_hits: 1,
        unique_retained: 1,
      });
      source.emit("retrieval_completed", {
        round: 1,
        threads: [{ ...verification.threads[0], thread_id: "q1", queries_to_run: [] }],
        details: [{
          thread_id: "q1",
          question: verification.threads[0].question,
          queries_used: ["test query"],
          new_evidence: verification.all_evidence,
          n_new_evidence: 1,
        }],
      });
      source.emit("thread_sufficiency_started", {
        thread_id: "q1",
        question: verification.threads[0].question,
        round: 1,
      });
      source.emit("thread_sufficiency_completed", {
        thread_id: "q1",
        question: verification.threads[0].question,
        round: 1,
        resolved: true,
        reasoning: verification.threads[0].reasoning,
        refined_queries: [],
      });
      source.emit("round_completed", {
        round: verification.rounds[0],
        threads: [{ ...verification.threads[0], thread_id: "q1", queries_to_run: [] }],
        is_sufficient: true,
        total_tokens_used: 200,
      });
    });
    expect(screen.getByText("Research")).toBeInTheDocument();
    expect(screen.getAllByText("Round 1/3")).toHaveLength(2);
    expect(screen.getByText("Resolved · R1")).toBeInTheDocument();
    expect(screen.getByText("1 hit · resolved")).toBeInTheDocument();
    expect(screen.getByText("test query")).toBeInTheDocument();

    act(() => FakeEventSource.instances[0].emit("completed", verification));
    expect(await screen.findByText("Claim under review")).toBeInTheDocument();
    expect(screen.getByText("1,234")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show reasoning trace" }));
    expect(screen.getByText("Was the claim reported?")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /Round 1/ }));
    expect(screen.getByText("https://example.com/evidence")).toBeVisible();
    await user.click(screen.getByRole("button", { name: /EXAMPLE/ }));
    expect(screen.getByText("The supporting evidence")).toBeVisible();

    await user.click(screen.getAllByRole("button", { name: "1" })[0]);
    expect(screen.getByRole("button", { name: /EXAMPLE/ })).toHaveStyle({ outline: "2px solid var(--accent)" });
  });

  it("reattaches to a persisted active run from the URL", async () => {
    window.history.replaceState(null, "", `/?claim=1&run=${RUN_ID}`);
    vi.stubGlobal("EventSource", FakeEventSource);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({
        run_id: RUN_ID,
        claim_id: 1,
        claim: verification.claim,
        gold_label: verification.gold_label,
        status: "running",
        cancel_requested: false,
        last_event_id: 4,
        result: null,
        error: null,
        created_at: "2026-09-09T12:00:00+00:00",
        started_at: "2026-09-09T12:00:01+00:00",
        finished_at: null,
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    render(<App />);

    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(FakeEventSource.instances[0].url).toBe(`/api/runs/${RUN_ID}/events`);
    expect(fetchSpy).toHaveBeenCalledWith(`/api/runs/${RUN_ID}`, expect.any(Object));
  });

  it("restores a completed result without starting another run", async () => {
    window.history.replaceState(null, "", `/?claim=1&run=${RUN_ID}`);
    vi.stubGlobal("EventSource", FakeEventSource);
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({
        run_id: RUN_ID,
        claim_id: 1,
        claim: verification.claim,
        gold_label: verification.gold_label,
        status: "succeeded",
        cancel_requested: false,
        last_event_id: 12,
        result: verification,
        error: null,
        created_at: "2026-09-09T12:00:00+00:00",
        started_at: "2026-09-09T12:00:01+00:00",
        finished_at: "2026-09-09T12:01:00+00:00",
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    render(<App />);

    expect(await screen.findByText("Claim under review")).toBeInTheDocument();
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});
