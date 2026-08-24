import os
import json
import pickle
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from .chunking import get_embedding
import re

def rrf(dense_ranked_ids, sparse_ranked_ids, k=60):
    scores = defaultdict(float)
    for rank, cid in enumerate(dense_ranked_ids):
        scores[cid] += 1 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ranked_ids):
        scores[cid] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])

_CACHE = {
    "bm25": None,
    "all_chunks": None,
    "defined_terms": None,
    "mtime": 0
}

def load_indices():
    data_dir = "/app/data" if os.path.exists("/app/data/chunks.json") else "./data"
    chunks_file = os.path.join(data_dir, "chunks.json")
    if not os.path.exists(chunks_file):
        return None, [], []
    
    current_mtime = os.path.getmtime(chunks_file)
    if _CACHE["all_chunks"] is not None and _CACHE["mtime"] == current_mtime:
        return _CACHE["bm25"], _CACHE["all_chunks"], _CACHE["defined_terms"]
        
    try:
        with open(os.path.join(data_dir, "bm25_index.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        with open(chunks_file, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        with open(os.path.join(data_dir, "defined_terms.json"), "r", encoding="utf-8") as f:
            defined_terms = json.load(f)
            
        _CACHE["bm25"] = bm25
        _CACHE["all_chunks"] = all_chunks
        _CACHE["defined_terms"] = defined_terms
        _CACHE["mtime"] = current_mtime
        return bm25, all_chunks, defined_terms
    except Exception as e:
        print(f"Error loading indices: {e}")
        return None, [], []

_QDRANT_CLIENT = None

def get_qdrant_client():
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        _QDRANT_CLIENT = QdrantClient(url=qdrant_url, check_compatibility=False, timeout=30.0)
    return _QDRANT_CLIENT

def hybrid_retrieve(question: str, doc_filter: str = None, top_k: int = 15):
    client = get_qdrant_client()
    collection_name = "contracts"
    
    bm25, all_chunks, defined_terms = load_indices()
    if not all_chunks:
        return []
            
    # Resolve doc_filter against known doc_ids for server-side Qdrant filtering
    query_filter = None
    matching_doc_ids = None
    if doc_filter:
        known_doc_ids = {c["doc_id"] for c in all_chunks}
        matching_doc_ids = [d for d in known_doc_ids if doc_filter.lower() in d.lower()]
        if matching_doc_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=matching_doc_ids)
                    )
                ]
            )
        else:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=["__non_existent_doc_id__"])
                    )
                ]
            )

    # 1. Dense Leg
    query_vector = get_embedding(question)
    dense_fetch_limit = 25 if doc_filter else 20
    
    try:
        dense_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=dense_fetch_limit
        ).points
    except AttributeError:
        # Fallback for older qdrant-client versions
        dense_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=dense_fetch_limit
        )
        
    dense_ranked_ids = [hit.payload["chunk_idx"] for hit in dense_results][:dense_fetch_limit]
    
    # 2. Sparse Leg
    tokenized_query = re.sub(r'[^\w\s]', '', question.lower()).split()
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Filter sparse scores if doc_filter
    if doc_filter:
        for i, c in enumerate(all_chunks):
            if doc_filter.lower() not in c["doc_id"].lower():
                sparse_scores[i] = -1.0
                
    sparse_ranked_ids = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:dense_fetch_limit]
    
    # 3. RRF
    effective_top_k = 20 if doc_filter else top_k
    fused = rrf(dense_ranked_ids, sparse_ranked_ids)
    top_chunk_indices = [cid for cid, score in fused[:effective_top_k]]
    
    # 4. Resolve cross references & defined terms
    retrieved_chunks = [all_chunks[i] for i in top_chunk_indices]
    combined_text = " ".join([c["text"] for c in retrieved_chunks]) + " " + question
    
    # Inject defined terms
    injected_cids = set()
    combined_text_lower = combined_text.lower()
    for dt in defined_terms:
        if dt["term"].lower() in combined_text_lower and dt["chunk_id"] not in top_chunk_indices:
            injected_cids.add(dt["chunk_id"])
            
    # Simple cross-ref: "Section X.Y"
    for match in re.finditer(r"Section\s+(\d+(\.\d+)*)", combined_text):
        ref_id = match.group(1)
        # Find chunk with this section_id
        for i, c in enumerate(all_chunks):
            if c.get("section_id") == ref_id and i not in top_chunk_indices:
                injected_cids.add(i)
                
    for cid in injected_cids:
        if cid < len(all_chunks):
            retrieved_chunks.append(all_chunks[cid])
            
    return retrieved_chunks
