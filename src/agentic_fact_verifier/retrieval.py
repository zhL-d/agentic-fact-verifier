"""Hybrid (BM25 + kNN) retrieval over the AVeriTeC knowledge base, scoped
to a single claim. Fuses two separate queries (BM25, kNN) with RRF,
implemented in Python."""

import os

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from agentic_fact_verifier.prompt_guard import scan_for_injection

INDEX_NAME = os.environ.get("ES_INDEX_NAME", "averitec_dev_chunks")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _rrf_fuse(*ranked_lists: list[str], k: int = RRF_K) -> list[str]:
    """Each ranked_list is a list of doc _ids in rank order. Returns doc
    _ids sorted by fused RRF score, descending."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(
    es: Elasticsearch,
    claim_id: str,
    query: str,
    top_k: int = 5,
    index_name: str = INDEX_NAME,
) -> list[dict]:
    """Retrieve top_k chunks for `query`, scoped to `claim_id`, combining
    BM25 keyword search and kNN semantic search via our own RRF fusion.

    `index_name` defaults to this module's INDEX_NAME (itself overridable
    via the ES_INDEX_NAME env var) but can be passed explicitly per call."""
    query_embedding = _get_model().encode(query, normalize_embeddings=True).tolist()
    candidate_pool = top_k * 10

    bm25_resp = es.search(
        index=index_name,
        query={
            "bool": {
                "must": {"match": {"text": query}},
                "filter": {"term": {"claim_id": claim_id}},
            }
        },
        size=candidate_pool,
    )
    knn_resp = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": candidate_pool,
            "num_candidates": candidate_pool * 2,
            "filter": {"term": {"claim_id": claim_id}},
        },
        size=candidate_pool,
    )

    bm25_ids = [hit["_id"] for hit in bm25_resp["hits"]["hits"]]
    knn_ids = [hit["_id"] for hit in knn_resp["hits"]["hits"]]
    by_id = {hit["_id"]: hit["_source"] for hit in bm25_resp["hits"]["hits"]}
    by_id.update({hit["_id"]: hit["_source"] for hit in knn_resp["hits"]["hits"]})

    fused_ids = _rrf_fuse(bm25_ids, knn_ids)[:top_k]
    chunks = []
    for doc_id in fused_ids:
        chunk = dict(by_id[doc_id])
        chunk.pop("embedding", None)
        chunk["injection_markers"] = scan_for_injection(chunk.get("text", ""))
        chunks.append(chunk)
    return chunks
