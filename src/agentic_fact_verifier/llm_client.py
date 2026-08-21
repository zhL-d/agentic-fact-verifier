"""Shared LLM-calling helper with retry-on-rate-limit."""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/nemotron-3-ultra-nvfp4"

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise SystemExit("FIREWORKS_API_KEY not set.")
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def call_llm(prompt: str, max_tokens: int = 6000) -> str:
    """Returns the raw response text. Retries with exponential backoff on
    429s; other errors propagate immediately (caller's error handling
    already treats a failed claim as skip-and-log, not a crash)."""
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RATE_LIMIT" in str(e)
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_BACKOFF_SECONDS * (2**attempt)
            print(f"    (rate limited, retrying in {wait}s, attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError("unreachable")  # loop always returns or raises
