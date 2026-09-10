# syntax=docker/dockerfile:1
#
# BAKE_MODEL_WEIGHTS build arg (default false): the SentenceTransformer
# embedding model is downloaded from Hugging Face on first use either
# way. `false` = smaller image `--build-arg BAKE_MODEL_WEIGHTS=true` for the baked variant.

FROM oven/bun:1.4.2-slim@sha256:cb3bbbb08e13a4a2ff400f24c7a2a1d5efa83f6ef8544d52d95a519631e2fc61 AS frontend
WORKDIR /app/web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src/ ./src/
RUN bun run build

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS python-base
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Non-root by default
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /usr/sbin/nologin appuser
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev


FROM python-base AS app
ENV UV_CACHE_DIR=/tmp/afv-uv-cache
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser migrations/ migrations/

RUN mkdir -p /app/checkpoints && chown appuser:appuser /app/checkpoints
USER appuser

EXPOSE 8000 8101

FROM python-base AS retrieval-dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group retrieval

ARG BAKE_MODEL_WEIGHTS=false

ARG EMBEDDING_MODEL=
ARG EMBEDDING_MODEL_REVISION=
RUN if [ "$BAKE_MODEL_WEIGHTS" = "true" ]; then \
      uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}' or 'nomic-ai/nomic-embed-text-v2-moe', revision=('${EMBEDDING_MODEL_REVISION}' or None), trust_remote_code=True)"; \
    fi

FROM retrieval-dependencies AS retrieval
ENV UV_CACHE_DIR=/tmp/afv-uv-cache
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser migrations/ migrations/
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache
USER appuser
EXPOSE 8100


FROM nginx:1.27-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10 AS nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend /app/web/dist /usr/share/nginx/html
EXPOSE 80
