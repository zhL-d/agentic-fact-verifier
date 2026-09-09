import { useEffect, useState } from "react";
import { fetchClaims } from "../api";
import { LABEL_CLASS } from "../labels";
import type { ClaimSummary } from "../types";
import { Masthead } from "./Masthead";

interface ClaimPickerProps {
  onSelect: (claimId: number) => void;
}

export function ClaimPicker({ onSelect }: ClaimPickerProps) {
  const [claims, setClaims] = useState<ClaimSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchClaims(controller.signal).then(setClaims).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => controller.abort();
  }, []);

  return (
    <>
      <Masthead />
      <p className="intro">Pick a claim below.</p>
      {error ? (
        <div className="findings">
          <span className="section-title">Failed to load claims</span>
          <p>{error}</p>
        </div>
      ) : (
        <div className="picker-list">
          {claims.map((claim) => {
            const badgeClass = LABEL_CLASS[claim.gold_label] || "";
            return (
              <button className="picker-card" key={claim.claim_id} onClick={() => onSelect(claim.claim_id)}>
                <span className={`picker-badge${badgeClass ? ` ${badgeClass}` : ""}`}>{claim.gold_label}</span>
                <span className="picker-claim">{claim.claim}</span>
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}
