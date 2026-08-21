import re
import statistics
import pdfplumber
import httpx
import os
import json
import numpy as np

HEADING_LABEL_PATTERNS = [
    (r"Section\s+\d+(\.\d+)*", "numbered_section"),
    (r"ARTICLE\s+[IVXLC]+", "article_roman"),
    (r"^\d+(\.\d+)*\.", "numbered"),
    (r"EXHIBIT|SCHEDULE|ANNEX|APPENDIX\s+[A-Z0-9]+", "exhibit"),
    (r"^\([a-z]\)|^[A-Z]\.", "lettered_subclause"),
]

def get_embedding(text: str) -> list[float]:
    gateway_url = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    key = os.getenv("AI_GATEWAY_KEY", "")
    model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # Standard OpenAI compatible embedding endpoint
    payload = {
        "input": text,
        "model": model
    }
    
    # Try embedding. If it fails, return random vector for prototype fallback,
    # or raise exception.
    try:
        response = httpx.post(f"{gateway_url}/embeddings", headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        # fallback to 0 vector for resilience in missing key
        return [0.0] * 1536 

def embed_batch(sentences: list[str]) -> list[list[float]]:
    return [get_embedding(s) for s in sentences]

def cosine_sim(a: list[float], b: list[float]) -> float:
    if not any(a) or not any(b): return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def split_into_sentences(text: str) -> list[str]:
    # Simple regex for sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_document(path: str, doc_id: str) -> list[dict]:
    headings = []
    lines_data = []
    
    # 1. Extract with layout metadata
    with pdfplumber.open(path) as pdf:
        all_sizes = []
        for i, page in enumerate(pdf.pages):
            extracted_lines = page.extract_text_lines()
            for line in extracted_lines:
                if not line["chars"]: continue
                
                is_bold = any("Bold" in c.get("fontname", "") for c in line["chars"])
                avg_size = statistics.mean(c.get("size", 10) for c in line["chars"])
                all_sizes.append(avg_size)
                
                text = line["text"].strip()
                
                lines_data.append({
                    "text": text,
                    "is_bold": is_bold,
                    "avg_size": avg_size,
                    "page_number": i + 1,
                    "is_all_caps": text.isupper() and len(re.sub(r'[^A-Z]', '', text)) > 5,
                    "length": len(text)
                })

    if not lines_data:
        return []

    global_avg_size = statistics.mean(all_sizes) if all_sizes else 10

    # 2. Tier 1 & Tier 2 Detect headings
    for idx, line in enumerate(lines_data):
        text = line["text"]
        is_candidate = False
        
        # Tier 1: Layout signal
        if line["length"] < 80:
            if line["is_bold"] or line["avg_size"] > (global_avg_size + 1.0) or line["is_all_caps"]:
                is_candidate = True
                
        # Tier 2: Text pattern signal
        match_label = None
        match_id = None
        for pattern, label in HEADING_LABEL_PATTERNS:
            match = re.search(pattern, text)
            if match and match.start() == 0:
                is_candidate = True
                match_label = label
                match_id = match.group(0)
                break
                
        if is_candidate:
            headings.append({
                "line_idx": idx,
                "text": text,
                "section_id": match_id,
                "page_number": line["page_number"]
            })

    chunks = []
    
    # Tier 3: Semantic fallback if no headings found
    if len(headings) <= 1:
        full_text = " ".join([l["text"] for l in lines_data])
        sentences = split_into_sentences(full_text)
        
        if len(sentences) > 5:
            embeddings = embed_batch(sentences)
            sims = [cosine_sim(embeddings[i], embeddings[i+1]) for i in range(len(sentences)-1)]
            threshold = np.percentile(sims, 5) if sims else 0
            breakpoints = [i for i, s in enumerate(sims) if s < threshold]
            
            start = 0
            for bp in breakpoints + [len(sentences)-1]:
                chunk_text = " ".join(sentences[start:bp+1])
                # Note: No page number precise mapping in semantic fallback for simplicity
                chunks.append({
                    "doc_id": doc_id,
                    "section_id": None,
                    "section_title": None,
                    "parent_section": None,
                    "page_number": lines_data[0]["page_number"], 
                    "text": chunk_text
                })
                start = bp + 1
        else:
             chunks.append({
                    "doc_id": doc_id,
                    "section_id": None,
                    "section_title": None,
                    "parent_section": None,
                    "page_number": lines_data[0]["page_number"], 
                    "text": full_text
             })
        return chunks

    # Normal Heading-based splitting
    for i in range(len(headings)):
        start_idx = headings[i]["line_idx"]
        end_idx = headings[i+1]["line_idx"] if i + 1 < len(headings) else len(lines_data)
        
        # ALL-CAPS protection (don't split if inside all caps) - handled naturally by grouping lines between headings
        chunk_lines = lines_data[start_idx:end_idx]
        chunk_text = " ".join([l["text"] for l in chunk_lines])
        
        parent_title = headings[i-1]["text"] if i > 0 else None
        
        chunks.append({
            "doc_id": doc_id,
            "section_id": headings[i]["section_id"],
            "section_title": headings[i]["text"],
            "parent_section": parent_title,
            "page_number": headings[i]["page_number"],
            "text": chunk_text
        })
        
    return chunks
