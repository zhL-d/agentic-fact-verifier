"""Dataset access shared by the HTTP API and background worker."""

import json
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "dev.json"
CURATED_CLAIM_IDS = tuple(
    int(value) for value in os.environ.get("CURATED_CLAIM_IDS", "0,1,6,9,10,13").split(",") if value.strip()
)


@lru_cache(maxsize=1)
def load_claims() -> list[dict]:
    path = Path(os.environ.get("CLAIMS_DATASET_PATH", DEFAULT_DATASET_PATH))
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"Claims dataset not found at {path}") from exc


def get_claim(claim_id: int) -> dict:
    claims = load_claims()
    if not 0 <= claim_id < len(claims):
        raise IndexError(f"claim_id {claim_id} out of range (0-{len(claims) - 1})")
    return claims[claim_id]
