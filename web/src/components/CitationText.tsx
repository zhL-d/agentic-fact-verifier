import type { ReactNode } from "react";
import type { EvidenceChunk } from "../types";

interface CitationTextProps {
  text: string;
  evidence: EvidenceChunk[];
  scopeThreadIndex: number | null;
  onCitationClick: (text: string, scopeThreadIndex: number | null) => void;
}

export function CitationText({ text, evidence, scopeThreadIndex, onCitationClick }: CitationTextProps) {
  const nodes: ReactNode[] = [];
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    nodes.push(text.slice(lastIndex, match.index), "[");
    match[1].split(",").forEach((rawNumber, index) => {
      if (index > 0) nodes.push(", ");
      const citationNumber = Number.parseInt(rawNumber.trim(), 10);
      const chunk = evidence[citationNumber - 1];
      nodes.push(
        chunk ? (
          <button
            className="cite"
            key={`${match?.index}-${index}`}
            onClick={() => onCitationClick(chunk.text, scopeThreadIndex)}
          >
            {citationNumber}
          </button>
        ) : (
          String(citationNumber)
        ),
      );
    });
    nodes.push("]");
    lastIndex = re.lastIndex;
  }
  nodes.push(text.slice(lastIndex));

  return <>{nodes}</>;
}

export function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "").split(".")[0].toUpperCase();
  } catch {
    return "SRC";
  }
}
