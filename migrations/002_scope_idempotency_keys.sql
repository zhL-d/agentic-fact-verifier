ALTER TABLE verification_runs
    DROP CONSTRAINT IF EXISTS verification_runs_idempotency_key_key;

CREATE UNIQUE INDEX IF NOT EXISTS verification_runs_claim_id_idempotency_key_idx
    ON verification_runs (claim_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
