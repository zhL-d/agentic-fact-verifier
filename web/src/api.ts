import type { ClaimSummary, StartRunResponse, VerifyResponse } from "./types";

async function errorDetail(res: Response): Promise<string> {
  let detail = res.statusText;
  try {
    const body = (await res.json()) as { detail?: string };
    detail = body.detail || detail;
  } catch {
    
  }
  return detail || `Request failed with status ${res.status}`;
}

export async function fetchClaims(signal?: AbortSignal): Promise<ClaimSummary[]> {
  const res = await fetch("/api/claims", { signal });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function fetchVerification(claimId: number, signal?: AbortSignal): Promise<VerifyResponse> {
  const res = await fetch("/api/verify/" + claimId, { method: "POST", signal });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}

export async function startVerificationRun(
  claimId: number,
  signal?: AbortSignal,
): Promise<StartRunResponse> {
  const res = await fetch("/api/runs/" + claimId, { method: "POST", signal });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
}
