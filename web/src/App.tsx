import { useState } from "react";
import { ClaimPicker } from "./components/ClaimPicker";
import { VerificationPage } from "./components/VerificationPage";

type View = { kind: "picker" } | { kind: "verification"; claimId: number };

export function App() {
  const [view, setView] = useState<View>({ kind: "picker" });

  if (view.kind === "verification") {
    return <VerificationPage claimId={view.claimId} onBack={() => setView({ kind: "picker" })} />;
  }

  return <ClaimPicker onSelect={(claimId) => setView({ kind: "verification", claimId })} />;
}
