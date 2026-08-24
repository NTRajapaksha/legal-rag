import os
import json
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi
from .chunking import chunk_document, embed_batch
from .defined_terms import build_defined_terms_index
import glob

import httpx
from rapidfuzz import fuzz

def verify_summary_grounding(summary: str, source_text: str, doc_id: str) -> bool:
    """
    Verifies that company/entity names mentioned in the summary actually exist in the source text or doc_id.
    """
    full_source = (source_text + " " + doc_id).lower()
    common_stops = {
        "the", "this", "a", "an", "in", "under", "for", "between", "contract", 
        "agreement", "joint", "venture", "license", "manufacturing", "transportation", 
        "hosting", "section", "article", "exhibit", "schedule", "party", "parties"
    }
    
    entity_candidates = re.findall(r'\b[A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)*(?:,\s*(?:Inc\.|LLC|L\.L\.C\.|Corp\.|PLC|Ltd\.))?', summary)
    for ent in entity_candidates:
        words = [w.lower() for w in re.split(r'\s+', ent) if w.lower() not in common_stops and len(w) > 2]
        if not words:
            continue
        clean_ent = " ".join(words)
        
        # Check if the named entity phrase is present or fuzzy matches
        score = fuzz.partial_ratio(clean_ent, full_source)
        if score < 80 and clean_ent not in full_source:
            # Check individual key words
            unmatched_words = [w for w in words if len(w) >= 4 and w not in full_source and fuzz.partial_ratio(w, full_source) < 85]
            if unmatched_words:
                print(f"Summary grounding failed for ungrounded entity: '{ent}' (unmatched: {unmatched_words}) in doc: {doc_id}")
                return False
    return True

def summarize_contract(doc_id: str, sample_text: str) -> str:
    gateway_url = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    key = os.getenv("AI_GATEWAY_KEY", "")
    model = os.getenv("GENERATION_MODEL", "openai/gpt-4o-mini")
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # 1. First attempt: Ask for specific parties if literally present in text
    primary_prompt = (
        f"Contract Document ID: {doc_id}\n\n"
        f"Introductory Excerpt:\n{sample_text[:3000]}\n\n"
        "Task: Summarize the contracting parties and the core commercial purpose of this contract in exactly one clear, factual sentence.\n"
        "Strict Rules:\n"
        "1. ONLY state specific company or individual names if they appear literally in the excerpt above. NEVER guess, expand acronyms, or infer corporate relationships.\n"
        "2. If exact entity names are not fully stated or are defined by role (e.g. 'Licensor', 'Transporter'), use those exact defined roles or the agreement title."
    )
    
    candidate_summary = None
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a legal analyst. Output only the one-sentence summary without preamble."},
                {"role": "user", "content": primary_prompt}
            ],
            "temperature": 0.0
        }
        res = httpx.post(f"{gateway_url}/chat/completions", headers=headers, json=payload, timeout=20.0)
        if res.status_code == 200:
            candidate_summary = res.json()["choices"][0]["message"]["content"].strip()
            # 2. Verify summary entity grounding
            if verify_summary_grounding(candidate_summary, sample_text, doc_id):
                return candidate_summary
            print(f"Grounding failed for candidate summary in {doc_id}. Falling back to safe role-based summary.")
    except Exception as e:
        print(f"Primary summary generation failed for {doc_id}: {e}")
    
    # 3. Fallback: Generate safe role-based summary without naming unverified entities
    fallback_prompt = (
        f"Contract Document ID: {doc_id}\n\n"
        f"Introductory Excerpt:\n{sample_text[:2500]}\n\n"
        "Task: Summarize the commercial purpose of this agreement in exactly one factual sentence using only the agreement title and generic defined roles (such as 'between Licensor and Licensee' or 'between Transporter and Customer') without asserting unverified company names."
    )
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a legal analyst. Output only the one-sentence summary without preamble."},
                {"role": "user", "content": fallback_prompt}
            ],
            "temperature": 0.0
        }
        res = httpx.post(f"{gateway_url}/chat/completions", headers=headers, json=payload, timeout=20.0)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Fallback summary generation failed for {doc_id}: {e}")
    
    return f"Legal contract concerning {doc_id}."

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
    doc_catalog = []
    
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
        
        # Generate 1-sentence contract description from preamble chunks
        intro_text = " ".join([c["text"] for c in chunks[:3]])
        description = summarize_contract(doc_id, intro_text)
        doc_catalog.append({"doc_id": doc_id, "description": description})
        
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
        with open("/app/data/documents.json", "w") as f:
            json.dump(doc_catalog, f)
    except FileNotFoundError:
        with open("./data/bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        with open("./data/chunks.json", "w") as f:
            json.dump(all_chunks, f)
        with open("./data/defined_terms.json", "w") as f:
            json.dump(all_defined_terms, f)
        with open("./data/documents.json", "w") as f:
            json.dump(doc_catalog, f)
            
    print("Ingestion complete.")
    return True
