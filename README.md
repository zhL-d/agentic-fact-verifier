# Agentic Fact Verifier

[![CI](https://github.com/zhL-d/agentic-fact-verifier/actions/workflows/ci.yml/badge.svg)](https://github.com/zhL-d/agentic-fact-verifier/actions/workflows/ci.yml)

A general-purpose agentic fact-verification system: given a claim, it
decomposes the claim into sub-questions, retrieves evidence per sub-question,
dynamically decides whether it has enough evidence (re-querying with refined
searches if not), and produces a verdict with citations traceable back to the
exact evidence and sub-question they came from, flagging low-confidence
verdicts for human review.

**Key features:**

- **Citation-grounded Verdicts**, every claimed fact in the justification
  cites a specific, traceable evidence.
- **Human-in-the-loop Escalation**, low-confidence verdicts are flagged
  for review instead of silently published.
- **Prompt-injection Defense**, retrieved web evidence is treated as 
  untrusted data to evaluate, rather than instructions to follow.
- **Context Compaction**, a capped, recency-based evidence budget per
  sub-question keeps prompts from growing unbounded across retrieval
  rounds.
- **Cost/budget Guardrail**, a hard per-run token ceiling stops a
  run from spending without bound.
- **MCP tool auth**, every retrieval/verdict tool call requires a shared
  secret, nothing is open to the network unauthenticated.
- **Checkpointed & Resumable**, a crashed run resumes from its last
  completed round instead of starting over.

![Demo: verifying a claim, then walking through the reasoning trace](demo.gif)

## Architecture

### Tech stack

- **LangGraph**, orchestration; the verification pipeline is an explicit state machine, so retrieval rounds and control flow are decided dynamically per sub-question.
- **RAG (Retrieval-Augmented Generation)**, the core pattern: every verdict is grounded in retrieved evidence with per-claim citations.
- **MCP (Model Context Protocol)**, exposes retrieval and verdict synthesis as standard tools, auth-gated with a shared secret between containers.
- **Elasticsearch**, hybrid BM25 + kNN retrieval powering the RAG layer.
- **PostgreSQL**, the checkpoint store behind crash-resumable runs (LangGraph's AsyncPostgresSaver).
- **LangSmith**, observability: traces every LLM call and graph step end-to-end.
- **FastAPI**, the API tier.
- **TypeScript + Vite**, the frontend.
- **Nginx**, reverse proxy; serves the built frontend and routes to the API tier.
- **Docker Compose**, local orchestration across all six services.
- **GitHub Actions**, CI: lint + tests for the backend, type-check + build for the frontend, on every push/PR to main.

### Deployment topology

```mermaid
flowchart LR
    Browser["Browser"]
    Nginx["nginx<br/>(:8000→80)<br/>serves web/dist/, proxies /api/*"]
    ApiServer["api_server.py<br/>(FastAPI container, :8000<br/>internally, :8001 direct debug)<br/>API only, runs the LangGraph<br/>pipeline"]
    RetrievalServer["mcp_server.py<br/>(:8100)<br/>retrieve_evidence tool"]
    JudgeServer["judge_server.py<br/>(:8101)<br/>synthesize_verdict tool"]
    ES[("Elasticsearch<br/>(:9200)<br/>BM25 + kNN, per-claim")]
    LLM["LLM<br/>"]
    Checkpoint[("Postgres<br/>(:5432)<br/>checkpoint state")]

    Browser -- "GET / (static assets)" --> Nginx
    Browser -- "POST /api/verify/:id" --> Nginx
    Nginx -- "proxy_pass /api/*" --> ApiServer
    ApiServer -- "MCP over streamable-http:<br/>retrieve_evidence(claim_id, query)<br/> shared secret auth" --> RetrievalServer
    RetrievalServer --> ES
    ApiServer -- "decompose / sufficiency calls<br/>" --> LLM
    ApiServer -- "MCP over streamable-http:<br/>synthesize_verdict(claim,<br/>evidence, sub_findings)<br/> shared secret auth" --> JudgeServer
    JudgeServer -- "verdict call" --> LLM
    ApiServer -- "read/write run state" --> Checkpoint
```

### Verification state machine

```mermaid
stateDiagram-v2
    [*] --> decompose
    decompose --> retrieve: split claim into sub-questions,<br/>each its own evidence "thread"
    retrieve --> check_sufficiency: accumulate evidence per thread
    check_sufficiency --> retrieve: any thread unresolved<br/>(refined queries)
    check_sufficiency --> verdict: all resolved, 3-round cap hit,<br/>or token budget exceeded
    verdict --> [*]: label + justification +<br/>citations, synthesized from<br/>per-thread findings
```

## Usage

### Prerequisites

- **Docker**
- **Python 3.12 + uv**, only needed to run the
  one-time indexing scripts below (`setup_es_index.py` / `ingest_knowledge_store.py`).

### Running against AVeriTeC (the default dataset)

**1. Get the AVeriTeC dev split.** Download `chenxwh/AVeriTeC` from
[Hugging Face](https://huggingface.co/chenxwh/AVeriTeC) (see that repo's own
instructions for cloning/downloading). You need two things
out of it:
- the dev-split claims JSON → save as `data/raw/dev.json` (500 claims + gold labels)
- the dev-split knowledge store → extract to `data/knowledge_store/dev/`
  (one JSON file per claim; the archive typically unpacks to a folder named
  `output_dev`, rename it to `dev`)

`docker-compose.yml` mounts `./data` into the app container read-only, so this
data lives on the host either way, no need to copy it anywhere else.

**2. Configure environment.** Create `.env` in the repo root:

```
OPENAI_API_KEY=...                             # default provider
# LLM_PROVIDER=openai                          # optional, "openai" is already the default
# OPENAI_MODEL=gpt-5.6-terra                    # optional, this is already the default
MCP_SHARED_SECRET=...                          # any random string; auths api_server to
                                                # mcp_server/judge_server (generate one with
                                                # `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`)
# POSTGRES_USER=afv
# POSTGRES_PASSWORD=afv
# POSTGRES_DB=afv_checkpoints

# optional, LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=                            # a Service Key
LANGSMITH_PROJECT=agentic-fact-verifier

# dataset config, defaults shown below already match AVeriTeC, so leave
# these as-is unless following "Using a different dataset" below
ES_INDEX_NAME=averitec_dev_chunks
VERDICT_LABELS=Supported,Refuted,Not Enough Evidence,Conflicting Evidence/Cherrypicking
```

**3. Start the stack:**

```bash
docker compose up -d --build
```

This builds and starts all six services: `elasticsearch`, `postgres`
(checkpoint state, port 5432), `mcp_server` (retrieval, port 8100),
`judge_server` (verdict synthesis, port 8101), `api_server` (API only,
port 8001 for direct debugging), and `nginx` (port 8000, the actual entry
point; serves the built frontend and proxies `/api/*` to
`api_server`). Optionally add `--build-arg BAKE_MODEL_WEIGHTS=true` to pre-fetch
the embedding model at build time instead of on `mcp_server`'s first request
(bigger image, no first-request download).

**4. Build the search index and ingest the knowledge store:**

```bash
uv run scripts/setup_es_index.py                     # creates the ES_INDEX_NAME index
uv run scripts/ingest_knowledge_store.py --limit 2    # smoke-test the mechanism first
uv run scripts/ingest_knowledge_store.py              # full run, all 500 claims
```

The **full run genuinely takes hours**, measured at ~235 chunks/sec on real
data, ingesting all 500 claims takes roughly **12–20
hours**.

The script skips already-indexed claims, so it's safe to interrupt and resume
(picks up where it left off, including after the `--limit 2` smoke test above).

**5. Verify a claim.** Open `http://localhost:8000`, pick a claim, watch it run
live, decomposition, retrieval rounds, sufficiency checks, and the final
verdict with clickable citations back to the exact evidence.

### Using a different dataset

1. **Write an ingestion script for your data's format**, adapted from
   `ingest_knowledge_store.py` (its parsing of AVeriTeC's own JSON schema is
   inherently dataset-specific, every dataset needs its own version of this
   step). It just needs to write documents shaped like
   `{claim_id, url, text, embedding, ...}` into Elasticsearch, one chunk per
   passage, `claim_id` matching whatever ID your claims use.
2. **Set `ES_INDEX_NAME`** in `.env` to a new index name for new dataset.
3. **Set `VERDICT_LABELS`** in `.env` to new dataset's own label taxonomy
   (comma-separated), if it differs from AVeriTeC's four-way one.
4. **Run `setup_es_index.py`** (it reads `ES_INDEX_NAME`, so this creates new
   new index) **then run ingestion script.**
5. **Point the claim picker at your data**: `api_server.py` currently reads
   `data/raw/dev.json` and a hardcoded `CURATED_CLAIM_IDS` list for the `/api/claims`
   picker, swap in new claims file and ID list.

Nothing else needs to change, `docker compose up -d --build` and the rest of
Setup above work identically once the two env vars point at your data.

### Running tests

```bash
uv run pytest tests/
```

Runs automatically on every push/PR to main via **GitHub Actions**
(`.github/workflows/ci.yml`), lint (`ruff`) and test suite for the
backend, a type-check + build for the frontend.

## Evaluation

Ran a small batch (n=10 AVeriTeC claims,same claims and same model, GPT-5.6 Terra throughout) 
aimed at validating mechanisms. Each finding states its own
sample size; none of these are proof, they're directional signal at small scale.

**Query-refinement effectiveness.** When a sub-question isn't resolved,
the system re-queries with a refined search rather than repeating the
same one. Of 44 refined-query events across the batch, all 44 (100%)
surfaced at least one evidence chunk the thread hadn't already seen.

**Escalation calibration.** A verdict is flagged for human review when a
sub-question never resolved, the point of the flag is
that it should track real uncertainty, not fire arbitrarily. In this
batch, flagged verdicts (n=7) scored 14.3% accuracy against gold labels;
unflagged verdicts (n=3) scored 100%. That's the intended pattern, 
directionally consistent with the flag meaning something, but n=7/n=3 is
too small to call this validated.

**Context compaction impact.** Capping each sub-question's accumulated
evidence at a budget (keeping the most recent chunks) cut
evidence volume by 57.4% (chunks) and 60.9% (characters) against an
uncapped run on the same 10 claims. Verdicts agreed between the capped and
uncapped runs on 7/10 claims and disagreed on 3, so compaction is not a
free no-op, it does occasionally change the final label by discarding
evidence the uncapped run still had.

Methodology: `scripts/run_compaction_comparison.py`.