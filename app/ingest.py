import os
import json
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi
from .chunking import chunk_document, embed_batch
from .defined_terms import build_defined_terms_index
import glob

def ingest_all():
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    client = QdrantClient(url=qdrant_url)
    
    collection_name = "contracts"
    # Recreate collection
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except Exception as e:
        print(f"Failed to recreate Qdrant collection: {e}")
        return False
        
    all_chunks = []
    all_defined_terms = []
    
    # Process all PDFs in data/contracts
    import fnmatch
    pdf_paths = [f for f in glob.glob("/app/data/contracts/*") if f.lower().endswith(".pdf")]
    if not pdf_paths:
        # Fallback for local testing outside docker
        pdf_paths = [f for f in glob.glob("./data/contracts/*") if f.lower().endswith(".pdf")]
        
    for path in pdf_paths:
        doc_id = os.path.basename(path).replace(".pdf", "")
        print(f"Processing {doc_id}...")
        
        # Chunking handles parsing internally in our simplified structure
        chunks = chunk_document(path, doc_id)
        
        # Build defined terms
        full_text = " ".join([c["text"] for c in chunks])
        terms = build_defined_terms_index(doc_id, full_text, chunks)
        
        start_idx = len(all_chunks)
        for t in terms:
            t["chunk_id"] += start_idx
            
        all_defined_terms.extend(terms)
        
        all_chunks.extend(chunks)
        
    if not all_chunks:
        print("No chunks generated. Check PDFs.")
        return False
        
    # Embed chunks
    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    
    # Batch embed (simplistic)
    batch_size = 20
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings.extend(embed_batch(batch))
        
    # Upsert to Qdrant
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                "doc_id": c["doc_id"],
                "section_id": c["section_id"],
                "section_title": c["section_title"],
                "parent_section": c["parent_section"],
                "page_number": c["page_number"],
                "text": c["text"],
                "chunk_idx": i
            }
        )
        for i, c in enumerate(all_chunks)
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Build BM25
    import re
    tokenized_corpus = [re.sub(r'[^\w\s]', '', doc.lower()).split() for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 and metadata locally
    os.makedirs("/app/data", exist_ok=True)
    try:
        with open("/app/data/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        with open("/app/data/chunks.json", "w") as f:
            json.dump(all_chunks, f)
        with open("/app/data/defined_terms.json", "w") as f:
            json.dump(all_defined_terms, f)
    except FileNotFoundError:
        with open("./data/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        with open("./data/chunks.json", "w") as f:
            json.dump(all_chunks, f)
        with open("./data/defined_terms.json", "w") as f:
            json.dump(all_defined_terms, f)
            
    print("Ingestion complete.")
    return True
