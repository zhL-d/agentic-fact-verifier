import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { EvidenceChunk } from "../types";
import { CitationText, domainOf } from "./CitationText";

const evidence: EvidenceChunk[] = [
  { text: "first source", url: "https://www.example.com/a", retrieval_query: "q", injection_markers: [] },
  { text: "second source", url: "https://news.test/b", retrieval_query: "q", injection_markers: [] },
];

describe("CitationText", () => {
  it("renders valid citations as buttons and preserves invalid citations as text", () => {
    const onCitationClick = vi.fn();
    render(
      <CitationText
        evidence={evidence}
        onCitationClick={onCitationClick}
        scopeThreadIndex={2}
        text="Finding [1, 2], invalid [3]."
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "2" }));
    expect(onCitationClick).toHaveBeenCalledWith("second source", 2);
    expect(screen.getByText(/invalid \[3\]/)).toBeInTheDocument();
  });
});

describe("domainOf", () => {
  it("creates the same compact source label as the original UI", () => {
    expect(domainOf("https://www.example.com/story")).toBe("EXAMPLE");
    expect(domainOf("not a url")).toBe("SRC");
  });
});
