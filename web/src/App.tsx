import { useCallback, useEffect, useState } from "react";
import { ClaimPicker } from "./components/ClaimPicker";
import { VerificationPage } from "./components/VerificationPage";

type View = { kind: "picker" } | { kind: "verification"; claimId: number; runId?: string };

function viewFromLocation(): View {
  const params = new URLSearchParams(window.location.search);
  const claimValue = params.get("claim");
  if (claimValue === null) return { kind: "picker" };
  const claimId = Number(claimValue);
  if (!Number.isInteger(claimId) || claimId < 0) return { kind: "picker" };
  return { kind: "verification", claimId, runId: params.get("run") ?? undefined };
}

function urlForView(view: View): string {
  if (view.kind === "picker") return window.location.pathname;
  const params = new URLSearchParams({ claim: String(view.claimId) });
  if (view.runId) params.set("run", view.runId);
  return `${window.location.pathname}?${params}`;
}

export function App() {
  const [view, setView] = useState<View>(viewFromLocation);

  useEffect(() => {
    const handlePopState = () => setView(viewFromLocation());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigate = useCallback((next: View) => {
    window.history.pushState(null, "", urlForView(next));
    setView(next);
  }, []);

  if (view.kind === "verification") {
    return (
      <VerificationPage
        claimId={view.claimId}
        initialRunId={view.runId}
        onBack={() => navigate({ kind: "picker" })}
        onRunStarted={(runId) => {
          window.history.replaceState(
            null,
            "",
            urlForView({ kind: "verification", claimId: view.claimId, runId }),
          );
        }}
      />
    );
  }

  return <ClaimPicker onSelect={(claimId) => navigate({ kind: "verification", claimId })} />;
}
