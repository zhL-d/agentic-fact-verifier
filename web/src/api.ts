import {
  claimSummarySchema,
  runResponseSchema,
  startRunResponseSchema,
  type ClaimSummary,
  type RunResponse,
  type StartRunResponse,
} from "./types";

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
  return claimSummarySchema.array().parse(await res.json());
}

export async function startVerificationRun(
  claimId: number,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<StartRunResponse> {
  const res = await fetch("/api/runs/" + claimId, {
    method: "POST",
    headers: { "Idempotency-Key": idempotencyKey },
    signal,
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return startRunResponseSchema.parse(await res.json());
}

export async function fetchRun(runId: string, signal?: AbortSignal): Promise<RunResponse> {
  const res = await fetch(`/api/runs/${runId}`, { signal });
  if (!res.ok) throw new Error(await errorDetail(res));
  return runResponseSchema.parse(await res.json());
}

export async function cancelRun(runId: string): Promise<void> {
  const res = await fetch(`/api/runs/${runId}/cancel`, { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
}
