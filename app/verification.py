from rapidfuzz import fuzz
import re
from .generation import GenerationResponse

def verify_citations(response: GenerationResponse, context_chunks: list[dict]) -> tuple[GenerationResponse, list[dict]]:
    """
    Verifies quotes exist in chunks and numeric consistency.
    """
    if response.insufficient_context:
         return response, []
         
    verified_citations = []
    failed_citations = []
    
    for cit in response.citations:
        # Find chunk
        chunk_text = ""
        for c in context_chunks:
            if c["doc_id"] == cit.doc_id and (cit.section_id is None or c.get("section_id") == cit.section_id):
                chunk_text += c["text"] + " "
                
        if not chunk_text:
            failed_citations.append({"citation": cit.dict(), "reason": "Chunk not found"})
            continue
            
        # 1. Fuzzy Quote Match
        score = fuzz.partial_ratio(cit.quote.lower(), chunk_text.lower())
        if score < 85:
             failed_citations.append({"citation": cit.dict(), "reason": f"Quote mismatch (score: {score})"})
             continue
             
        # 2. Exact numeric guard
        num_pattern = re.compile(r'\b\d+(\.\d+)?\b')
        ans_nums = set(m.group() for m in num_pattern.finditer(cit.quote))
        chunk_nums = set(m.group() for m in num_pattern.finditer(chunk_text))
        
        numeric_drift = False
        for num in ans_nums:
            if num not in chunk_nums:
                numeric_drift = True
                failed_citations.append({"citation": cit.dict(), "reason": f"Numeric drift on {num}"})
                break
                
        if not numeric_drift:
            verified_citations.append(cit)
            
    # Update response to only include verified citations
    response.citations = verified_citations
    return response, failed_citations
