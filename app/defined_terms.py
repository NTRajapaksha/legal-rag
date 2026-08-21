import re

def build_defined_terms_index(doc_id: str, full_text: str, chunks: list[dict]) -> list[dict]:
    """
    Scans full text for capitalized defined terms and maintains a lookup index.
    Pattern: capitalized phrase in quotes followed by means, shall mean, or refers to.
    """
    defined_terms = []
    
    # Regex to find: "Term" means / "Term" shall mean / "Term" refers to
    pattern = re.compile(r'["“”]([A-Z][a-zA-Z\s]+)["“”]\s+(means|shall mean|refers to)', re.IGNORECASE)
    
    matches = pattern.finditer(full_text)
    
    for match in matches:
        term = match.group(1).strip()
        # Find which chunk contains this match based on substring index or simple text search
        # To simplify, we just find the first chunk that contains the definition text around this term.
        
        # A simple heuristic: find the chunk where this exact term declaration appears.
        declaration = match.group(0)
        found_chunk_id = None
        for i, chunk in enumerate(chunks):
            if declaration in chunk["text"]:
                found_chunk_id = i
                break
                
        if found_chunk_id is not None:
            defined_terms.append({
                "doc_id": doc_id,
                "term": term,
                "chunk_id": found_chunk_id
            })
            
    return defined_terms
