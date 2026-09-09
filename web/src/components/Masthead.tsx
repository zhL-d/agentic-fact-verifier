interface MastheadProps {
  onBack?: () => void;
}

export function Masthead({ onBack }: MastheadProps) {
  return (
    <div className="masthead">
      <div className="brand">Agentic Fact Verifier</div>
      {onBack && <button className="back-link" onClick={onBack}>← back to claims</button>}
    </div>
  );
}
