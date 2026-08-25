"""Shared LLM-calling helper with retry-on-rate-limit and schema-constrained
structured output."""

import os
import time
from typing import NamedTuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

_PROVIDER_CONFIG = {
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
        "api_key_env": "FIREWORKS_API_KEY",
        "max_tokens_param": "max_tokens",
        "temperature": 0,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"),
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens_param": "max_completion_tokens",
        "temperature": None,
    },
}

PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
if PROVIDER not in _PROVIDER_CONFIG:
    raise SystemExit(f"Unknown LLM_PROVIDER {PROVIDER!r}; expected one of {list(_PROVIDER_CONFIG)}")

_config = _PROVIDER_CONFIG[PROVIDER]
BASE_URL = _config["base_url"]
MODEL = _config["model"]
_API_KEY_ENV = _config["api_key_env"]
_MAX_TOKENS_PARAM = _config["max_tokens_param"]
_TEMPERATURE = _config["temperature"]

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2

_client: OpenAI | None = None


class LLMResult(NamedTuple):
    """`usage` mirrors the OpenAI-compatible `CompletionUsage` shape:
    prompt_tokens/completion_tokens/total_tokens."""

    content: str
    usage: dict


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise SystemExit(f"{_API_KEY_ENV} not set.")
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def _force_additional_properties_false(node: object) -> None:
    """Recursively patches every object node in a JSON schema to set
    additionalProperties: false and required = all property keys,
    pydantic's model_json_schema() doesn't set either on its own."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
        for value in node.values():
            _force_additional_properties_false(value)
    elif isinstance(node, list):
        for item in node:
            _force_additional_properties_false(item)


def schema_response_format(model: type[BaseModel], name: str) -> dict:
    """Builds an OpenAI-compatible `response_format` dict that constrains
    decoding to `model`'s JSON schema."""
    schema = model.model_json_schema()
    _force_additional_properties_false(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def call_llm(
    prompt_or_messages: str | list[dict],
    max_tokens: int = 6000,
    response_format: dict | None = None,
) -> LLMResult:
    """Returns an `LLMResult(content, usage)`. Retries with exponential
    backoff on 429s; other errors propagate immediately.

    `prompt_or_messages` is either a single user-turn string or a full
    messages list (for multi-turn self-correction). `response_format`,
    when given, constrains decoding to a JSON schema, build one with
    `schema_response_format`."""
    client = _get_client()
    messages = (
        [{"role": "user", "content": prompt_or_messages}]
        if isinstance(prompt_or_messages, str)
        else prompt_or_messages
    )

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                **{_MAX_TOKENS_PARAM: max_tokens},
                **({"temperature": _TEMPERATURE} if _TEMPERATURE is not None else {}),
                **({"response_format": response_format} if response_format else {}),
            )
            usage = completion.usage.model_dump() if completion.usage else {}
            return LLMResult(content=completion.choices[0].message.content, usage=usage)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RATE_LIMIT" in str(e)
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_BACKOFF_SECONDS * (2**attempt)
            print(f"    (rate limited, retrying in {wait}s, attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)

    raise RuntimeError("unreachable")
