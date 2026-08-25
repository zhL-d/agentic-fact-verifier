"""Create the Elasticsearch index for hybrid (BM25 + kNN) retrieval over the
AVeriTeC dev knowledge store or any other dataset, via ES_INDEX_NAME.

Usage:
    uv run scripts/setup_es_index.py
"""

import os

from elasticsearch import Elasticsearch

INDEX_NAME = os.environ.get("ES_INDEX_NAME", "averitec_dev_chunks")
EMBEDDING_DIMS = 384

MAPPING = {
    "properties": {
        "claim_id": {"type": "keyword"},
        "url": {"type": "keyword"},
        "text": {"type": "text"},
        "source_type": {"type": "keyword"},
        "source_query": {"type": "keyword"},
        "embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMS,
            "index": True,
            "similarity": "cosine",
        },
    }
}


def main():
    es = Elasticsearch("http://localhost:9200")

    if not es.ping():
        raise SystemExit(
            "Can't reach Elasticsearch at localhost:9200 "
            "is `docker compose up -d` running?"
        )

    if es.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists, leaving as-is.")
        return

    es.indices.create(index=INDEX_NAME, mappings=MAPPING)
    print(f"Created index '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
