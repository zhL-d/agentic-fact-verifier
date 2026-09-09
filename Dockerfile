# syntax=docker/dockerfile:1
#
# One image, three services (api_server, mcp_server, judge_server) selected
# via docker-compose.yml's `command:`
#
# BAKE_MODEL_WEIGHTS build arg (default false): the SentenceTransformer
# embedding model is downloaded from Hugging Face on first use either
# way. `false` = smaller image `--build-arg BAKE_MODEL_WEIGHTS=true` for the baked variant.

FROM oven/bun:1.4.2-slim AS frontend
WORKDIR /app/web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src/ ./src/
RUN bun run build

FROM python:3.12-slim AS app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Non-root by default
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /usr/sbin/nologin appuser
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY scripts/ scripts/

ARG BAKE_MODEL_WEIGHTS=false
RUN if [ "$BAKE_MODEL_WEIGHTS" = "true" ]; then \
      uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"; \
    fi

RUN mkdir -p /app/checkpoints && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8100 8101

# Frontend/API split: nginx serves web/dist/ directly and reverse-proxies
# /api/* to api_server.
FROM nginx:1.27-alpine AS nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend /app/web/dist /usr/share/nginx/html
EXPOSE 80
