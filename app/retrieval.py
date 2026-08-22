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

def hybrid_retrieve(question: str, doc_filter: str = None, top_k: int = 10):
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    collection_name = "contracts"
    
    # Load BM25 and chunks
    try:
        with open("/app/data/bm25_index.pkl", "rb") as f:
            bm25 = pickle.load(f)
        with open("/app/data/chunks.json", "r") as f:
            all_chunks = json.load(f)
        with open("/app/data/defined_terms.json", "r") as f:
            defined_terms = json.load(f)
    except FileNotFoundError:
        with open("./data/bm25_index.pkl", "rb") as f:
            bm25 = pickle.load(f)
        with open("./data/chunks.json", "r") as f:
            all_chunks = json.load(f)
        with open("./data/defined_terms.json", "r") as f:
            defined_terms = json.load(f)
            
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
    
    try:
        dense_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=20
        ).points
    except AttributeError:
        # Fallback for older qdrant-client versions
        dense_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=20
        )
        
    dense_ranked_ids = [hit.payload["chunk_idx"] for hit in dense_results][:20]
    
    # 2. Sparse Leg
    tokenized_query = re.sub(r'[^\w\s]', '', question.lower()).split()
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Filter sparse scores if doc_filter
    if doc_filter:
        for i, c in enumerate(all_chunks):
            if doc_filter.lower() not in c["doc_id"].lower():
                sparse_scores[i] = -1.0
                
    sparse_ranked_ids = sorted(range(len(sparse_scores)), key=lambda i: sparse_scores[i], reverse=True)[:20]
    
    # 3. RRF
    fused = rrf(dense_ranked_ids, sparse_ranked_ids)
    top_chunk_indices = [cid for cid, score in fused[:top_k]]
    
    # 4. Resolve cross references & defined terms
    retrieved_chunks = [all_chunks[i] for i in top_chunk_indices]
    combined_text = " ".join([c["text"] for c in retrieved_chunks]) + " " + question
    
    # Inject defined terms
    injected_cids = set()
    for dt in defined_terms:
        if dt["term"] in combined_text and dt["chunk_id"] not in top_chunk_indices:
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
