"""Embed and bulk-index the AVeriTeC dev knowledge store into Elasticsearch.

Usage:
    uv run scripts/ingest_knowledge_store.py             # full run, all 500 claims
    uv run scripts/ingest_knowledge_store.py --limit 5   # test on 5 claims first
"""

import argparse
import json
from pathlib import Path

import torch
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

KNOWLEDGE_STORE_DIR = Path(__file__).parent.parent / "data" / "knowledge_store" / "dev"
INDEX_NAME = "averitec_dev_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def claim_already_indexed(es: Elasticsearch, claim_id: str) -> bool:
    resp = es.count(index=INDEX_NAME, query={"term": {"claim_id": claim_id}})
    return resp["count"] > 0


TARGET_CHUNK_WORDS = 350  # approximates 256-512 tokens (~1.3 tokens/word for English)


def normalize(sentence: str) -> str:
    return sentence.strip().lower()


def group_into_chunks(sentences: list[str], target_words: int = TARGET_CHUNK_WORDS) -> list[str]:
    """Greedily group consecutive sentences (preserving document order)
    into ~target_words-sized passages."""
    chunks, current, current_words = [], [], 0
    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > target_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += n_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_chunks_for_claim(claim_file: Path) -> list[dict]:
    """Each line is one retrieved document. Sentences are deduplicated
    across the whole claim first (many documents overlap heavily), then
    grouped into ~350-word passages per document, preserving document
    order so each chunk stays textually coherent."""
    claim_id = claim_file.stem
    seen = set()
    chunks = []

    with claim_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            url = doc.get("url", "")
            source_type = doc.get("type", "")
            source_query = doc.get("query", "")

            fresh_sentences = []
            for sentence in doc.get("url2text", []):
                sentence = sentence.strip()
                if not sentence:
                    continue
                key = normalize(sentence)
                if key in seen:
                    continue
                seen.add(key)
                fresh_sentences.append(sentence)

            for passage in group_into_chunks(fresh_sentences):
                chunks.append(
                    {
                        "claim_id": claim_id,
                        "url": url,
                        "text": passage,
                        "source_type": source_type,
                        "source_query": source_query,
                    }
                )
    return chunks


def index_claim(es: Elasticsearch, model: SentenceTransformer, claim_file: Path) -> int:
    chunks = load_chunks_for_claim(claim_file)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True
    )

    actions = (
        {
            "_index": INDEX_NAME,
            "_source": {**chunk, "embedding": embedding.tolist()},
        }
        for chunk, embedding in zip(chunks, embeddings)
    )
    success, errors = helpers.bulk(es, actions, raise_on_error=False)
    if errors:
        print(f"  {len(errors)} indexing errors on {claim_file.name} (first: {errors[0]})")
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N claim files")
    args = parser.parse_args()

    es = Elasticsearch("http://localhost:9200")
    if not es.ping():
        raise SystemExit("Can't reach Elasticsearch at localhost:9200 — is `docker compose up -d` running?")

    device = get_device()
    print(f"Loading {EMBEDDING_MODEL} on device={device}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    claim_files = sorted(KNOWLEDGE_STORE_DIR.glob("*.json"), key=lambda p: int(p.stem))
    if args.limit:
        claim_files = claim_files[: args.limit]

    print(f"Ingesting {len(claim_files)} claim files...\n")

    total_chunks = 0
    skipped = 0
    for i, claim_file in enumerate(claim_files):
        claim_id = claim_file.stem
        if claim_already_indexed(es, claim_id):
            skipped += 1
            continue

        n_chunks = index_claim(es, model, claim_file)
        total_chunks += n_chunks
        print(f"[{i+1}/{len(claim_files)}] claim {claim_id}: indexed {n_chunks} chunks")

    print(f"\nDone. Indexed {total_chunks} new chunks, skipped {skipped} already-indexed claims.")


if __name__ == "__main__":
    main()
