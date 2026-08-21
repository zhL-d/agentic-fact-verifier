// Citation linking + the per-thread evidence index.

import { el } from "./dom";
import type { EvidenceChunk } from "./types";

interface EvidenceRowEntry {
  getRoundBtn: () => HTMLButtonElement;
  row: HTMLButtonElement;
  detail: HTMLDivElement;
}

let evidenceIndexByThread: Record<string, EvidenceRowEntry>[] = [];
let globalFirstOccurrence: Record<string, EvidenceRowEntry> = {};

export function resetEvidenceIndex(): void {
  evidenceIndexByThread = [];
  globalFirstOccurrence = {};
}

export function beginThread(threadIndex: number): void {
  evidenceIndexByThread[threadIndex] = {};
}

export function registerEvidence(
  threadIndex: number,
  chunkText: string,
  entry: EvidenceRowEntry
): void {
  const byThread = evidenceIndexByThread[threadIndex];
  if (!(chunkText in byThread)) byThread[chunkText] = entry;
  if (!(chunkText in globalFirstOccurrence)) globalFirstOccurrence[chunkText] = entry;
}

export function linkifyCitations(
  text: string,
  evidenceList: EvidenceChunk[],
  scopeThreadIndex: number | null
): DocumentFragment {
  const frag = document.createDocumentFragment();
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  while ((match = re.exec(text)) !== null) {
    frag.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
    const numbers = match[1].split(",").map((s) => parseInt(s.trim(), 10));
    frag.appendChild(document.createTextNode("["));
    numbers.forEach((n, i) => {
      if (i > 0) frag.appendChild(document.createTextNode(", "));
      const chunk = evidenceList[n - 1];
      if (chunk) {
        frag.appendChild(
          el("button", {
            class: "cite",
            text: String(n),
            onclick: () => jumpToEvidence(chunk.text, scopeThreadIndex),
          })
        );
      } else {
        frag.appendChild(document.createTextNode(String(n)));
      }
    });
    frag.appendChild(document.createTextNode("]"));
    lastIndex = re.lastIndex;
  }
  frag.appendChild(document.createTextNode(text.slice(lastIndex)));
  return frag;
}

export function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "").split(".")[0].toUpperCase();
  } catch {
    return "SRC";
  }
}

export function jumpToEvidence(text: string, scopeThreadIndex: number | null): void {
  let entry: EvidenceRowEntry | undefined;
  if (scopeThreadIndex !== null && scopeThreadIndex !== undefined) {
    entry = evidenceIndexByThread[scopeThreadIndex]?.[text];
  }
  if (!entry) entry = globalFirstOccurrence[text];
  if (!entry) return;

  const roundBtn = entry.getRoundBtn();
  if (roundBtn.getAttribute("aria-expanded") !== "true") roundBtn.click();
  entry.row.scrollIntoView({ behavior: "smooth", block: "center" });
  if (entry.row.getAttribute("aria-expanded") !== "true") entry.row.click();
  entry.row.style.outline = "2px solid var(--accent)";
  const row = entry.row;
  setTimeout(() => {
    row.style.outline = "";
  }, 1200);
}
