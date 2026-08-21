import type { ClaimSummary, VerifyResponse } from "./types";

export async function fetchClaims(): Promise<ClaimSummary[]> {
  const res = await fetch("/api/claims");
  return res.json();
}

export async function fetchVerification(claimId: number): Promise<VerifyResponse> {
  const res = await fetch("/api/verify/" + claimId, { method: "POST" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
    
    }
    throw new Error(detail);
  }
  return res.json();
}
