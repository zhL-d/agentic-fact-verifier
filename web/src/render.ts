import { el } from "./dom";
import { fetchClaims, fetchVerification } from "./api";
import {
  resetEvidenceIndex,
  beginThread,
  registerEvidence,
  linkifyCitations,
  domainOf,
} from "./citations";
import type { ClaimSummary, VerifyResponse } from "./types";

const LABEL_CLASS: Record<string, string> = {
  Supported: "supported",
  Refuted: "refuted",
  "Conflicting Evidence/Cherrypicking": "conflicting",
  "Not Enough Evidence": "insufficient",
};

const page = document.getElementById("page") as HTMLDivElement;

function masthead(showBack: boolean): HTMLElement {
  const children: Node[] = [
    el("div", { class: "brand", text: "Agentic Fact Verifier · AVeriTeC docket" }),
  ];
  if (showBack) children.push(el("a", { onclick: () => loadPicker(), text: "← back to claims" }));
  return el("div", { class: "masthead" }, children);
}

export async function loadPicker(): Promise<void> {
  resetEvidenceIndex();
  page.innerHTML = "";
  page.appendChild(masthead(false));
  page.appendChild(
    el("p", {
      class: "intro",
      text:
        "Pick a claim below. This runs the real agentic pipeline (decompose → retrieve → " +
        "check sufficiency → retry/verdict) live against pre-ingested, per-claim evidence " +
        "not open web search, so only claims already in the AVeriTeC dev set can be verified.",
    })
  );
  const list = el("div", { class: "picker-list" });
  page.appendChild(list);

  const claims = await fetchClaims();
  claims.forEach((c: ClaimSummary) => {
    const badgeClass = LABEL_CLASS[c.gold_label] || "";
    const badge = el("span", {
      class: "picker-badge" + (badgeClass ? " " + badgeClass : ""),
      text: c.gold_label,
    });
    const card = el(
      "button",
      { class: "picker-card", onclick: () => loadDocket(c.claim_id) },
      [badge, el("span", { class: "picker-claim", text: c.claim })]
    );
    list.appendChild(card);
  });
}

export async function loadDocket(claimId: number): Promise<void> {
  resetEvidenceIndex();
  page.innerHTML = "";
  page.appendChild(masthead(true));
  page.appendChild(
    el("div", { class: "loading" }, [
      el("div", { class: "spinner" }),
      el("div", {
        text: "Running the agentic loop, this takes a while (multiple LLM calls per round)…",
      }),
    ])
  );

  try {
    const data = await fetchVerification(claimId);
    renderDocket(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    page.innerHTML = "";
    page.appendChild(masthead(true));
    page.appendChild(
      el("div", { class: "findings" }, [
        el("span", { class: "section-title", text: "Run failed" }),
        el("p", { text: message }),
        el("p", { style: "margin-top: 14px" }, [
          el("button", { class: "cite", onclick: () => loadDocket(claimId), text: "Retry" }),
        ]),
      ])
    );
  }
}

function renderDocket(data: VerifyResponse): void {
  page.innerHTML = "";
  page.appendChild(masthead(true));

  const verdict = data.verdict;
  const labelClass = LABEL_CLASS[verdict.label] || "";
  const header = el("div", { class: "exhibit-header" });
  header.appendChild(el("span", { class: "exhibit-label", text: "Claim under review" }));
  header.appendChild(el("p", { class: "claim-text", text: "“" + data.claim + "”" }));
  const meta = el("div", { class: "claim-meta" });
  meta.appendChild(
    el("span", {}, [document.createTextNode("Rounds "), el("b", { text: String(data.rounds.length) })])
  );
  meta.appendChild(
    el("span", {}, [
      document.createTextNode("Evidence reviewed "),
      el("b", { text: String(data.all_evidence.length) }),
    ])
  );
  meta.appendChild(
    el("span", {}, [document.createTextNode("Sub-questions "), el("b", { text: String(data.threads.length) })])
  );
  header.appendChild(meta);
  header.appendChild(el("div", { class: "stamp" + (labelClass ? " " + labelClass : ""), text: verdict.label }));

  const matches = verdict.label === data.gold_label;
  const goldNote = el("div", { class: "gold-note" });
  goldNote.appendChild(document.createTextNode("Reference label (AVeriTeC dev set): "));
  goldNote.appendChild(el("b", { text: data.gold_label }));
  goldNote.appendChild(document.createTextNode(" — "));
  goldNote.appendChild(el("span", { class: matches ? "match" : "diff", text: matches ? "matches" : "differs" }));
  header.appendChild(goldNote);
  page.appendChild(header);

  const findings = el("div", { class: "findings" });
  findings.appendChild(el("span", { class: "section-title", text: "Findings" }));
  const findingsP = el("p");
  findingsP.appendChild(linkifyCitations(verdict.justification, data.all_evidence, null));
  findings.appendChild(findingsP);
  page.appendChild(findings);

  // Result / trace toggle
  const trace = el("div", { class: "trace", hidden: "" });
  const btnOff = el("button", { class: "active", text: "Result only" });
  const btnOn = el("button", { text: "Show reasoning trace" });
  function setTrace(show: boolean): void {
    trace.hidden = !show;
    btnOn.classList.toggle("active", show);
    btnOff.classList.toggle("active", !show);
  }
  btnOff.addEventListener("click", () => setTrace(false));
  btnOn.addEventListener("click", () => setTrace(true));
  page.appendChild(el("div", { class: "view-toggle" }, [btnOff, btnOn]));
  page.appendChild(trace);
  trace.appendChild(
    el("span", {
      class: "section-title",
      text: "Sub-questions — each tracked as its own thread; click a round to see exactly what it found",
    })
  );

  const citedTextToNumber = new Map<string, number>();
  verdict.citations.forEach((n) => {
    const chunk = data.all_evidence[n - 1];
    if (chunk) citedTextToNumber.set(chunk.text, n);
  });

  data.threads.forEach((thread, ti) => {
    beginThread(ti);
    const card = el("div", { class: "exhibit-card" });
    const statusLabel = thread.resolved
      ? "Resolved"
      : "Unresolved after " + data.rounds.length + " round" + (data.rounds.length === 1 ? "" : "s");
    card.appendChild(
      el("div", { class: "exhibit-card-head" }, [
        el("span", {
          class: "status-chip " + (thread.resolved ? "resolved" : "unresolved"),
          text: statusLabel,
        }),
        el("span", { class: "question", text: thread.question }),
      ])
    );
    const body = el("div", { class: "exhibit-card-body" });
    const history = el("div", { class: "round-history" });

    data.rounds.forEach((round) => {
      const t = round.threads[ti];
      if (!t || t.queries_used.length === 0) return;

      const evidWrap = el("div", { class: "round-evidence", hidden: "" });
      const evidList = el("div", { class: "evidence-list" });
      t.new_evidence.forEach((chunk) => {
        const citeNumber = citedTextToNumber.get(chunk.text);
        const rowChildren: Node[] = [
          el("span", { class: "evidence-tag", text: domainOf(chunk.url) }),
          el("span", { class: "url", text: chunk.url }),
        ];
        if (citeNumber !== undefined) {
          rowChildren.push(el("span", { class: "cited-badge", text: "Cited as [" + citeNumber + "]" }));
        }
        rowChildren.push(el("span", { class: "chevron", text: "▶" }));
        const detail = el(
          "div",
          { class: "evidence-detail", hidden: "" },
          [
            el("span", { class: "found-via", text: 'Found via: "' + chunk.retrieval_query + '"' }),
            el("span", { class: "excerpt", text: chunk.text }),
          ]
        ) as HTMLDivElement;
        const row = el(
          "button",
          {
            class: "evidence-row",
            "aria-expanded": "false",
            onclick: () => {
              const expanded = row.getAttribute("aria-expanded") === "true";
              row.setAttribute("aria-expanded", String(!expanded));
              detail.hidden = expanded;
            },
          },
          rowChildren
        ) as HTMLButtonElement;
        evidList.appendChild(row);
        evidList.appendChild(detail);
        registerEvidence(ti, chunk.text, { getRoundBtn: () => roundBtn, row, detail });
      });
      evidWrap.appendChild(evidList);

      const outcomeClass = t.resolved
        ? "resolved-now"
        : round.round === data.rounds.length
          ? "still-open"
          : "gain";
      const outcomeText = t.resolved
        ? "+" + t.n_new_evidence + " · resolved"
        : "+" +
          t.n_new_evidence +
          (round.round === data.rounds.length && !thread.resolved ? " · still unresolved" : "");
      const roundBtn: HTMLButtonElement = el(
        "button",
        {
          class: "round-entry",
          "aria-expanded": "false",
          onclick: () => {
            const expanded = roundBtn.getAttribute("aria-expanded") === "true";
            roundBtn.setAttribute("aria-expanded", String(!expanded));
            evidWrap.hidden = expanded;
          },
        },
        [
          el("span", { class: "round-tag", text: "Round " + round.round }),
          el("span", { class: "round-query" }, [
            el("span", { class: "label", text: round.round === 1 ? "query" : "refined" }),
            document.createTextNode('"' + t.queries_used.join('", "') + '"'),
          ]),
          el("span", { class: "round-outcome " + outcomeClass, text: outcomeText }),
          el("span", { class: "chevron", text: "▶" }),
        ]
      ) as HTMLButtonElement;
      history.appendChild(el("div", { class: "round-entry-wrap" }, [roundBtn, evidWrap]));
    });

    body.appendChild(history);
    const reasoningP = el("p", { class: "reasoning" });
    reasoningP.appendChild(linkifyCitations(thread.reasoning, thread.evidence, ti));
    body.appendChild(reasoningP);
    card.appendChild(body);
    trace.appendChild(card);
  });

  page.appendChild(
    el("footer", {
      class: "foot",
      text: "Agentic Fact Verifier · built on the AVeriTeC benchmark · synthesize_verdict MCP tool",
    })
  );
}
