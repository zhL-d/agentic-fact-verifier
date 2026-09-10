CREATE TABLE IF NOT EXISTS verification_runs (
    run_id UUID PRIMARY KEY,
    claim_id INTEGER NOT NULL,
    thread_id TEXT NOT NULL UNIQUE,
    idempotency_key TEXT UNIQUE,
    celery_task_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'retrying', 'cancelling', 'cancelled', 'succeeded', 'failed')
    ),
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    last_event_id BIGINT NOT NULL DEFAULT 0,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS verification_runs_status_created_idx
    ON verification_runs (status, created_at);

CREATE TABLE IF NOT EXISTS verification_run_events (
    run_id UUID NOT NULL REFERENCES verification_runs(run_id) ON DELETE CASCADE,
    sequence BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    dedupe_key TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, sequence)
);

CREATE UNIQUE INDEX IF NOT EXISTS verification_run_events_dedupe_idx
    ON verification_run_events (run_id, dedupe_key)
    WHERE dedupe_key IS NOT NULL;

