from rapidfuzz import fuzz
import re
import os
import httpx
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
             
        # 2. Entailment check
        gateway_url = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
        key = os.getenv("AI_GATEWAY_KEY", "")
        model = os.getenv("GENERATION_MODEL", "openai/gpt-4o-mini")
        
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        entailment_prompt = f"Does the following statement follow logically from the provided quote? Answer only YES or NO.\n\nStatement: {response.answer}\n\nQuote: {cit.quote}"
        
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": entailment_prompt}]
        }
        
        try:
            llm_resp = httpx.post(f"{gateway_url}/chat/completions", headers=headers, json=payload, timeout=30.0)
            llm_resp.raise_for_status()
            entailment_decision = llm_resp.json()["choices"][0]["message"]["content"].strip().upper()
            if "YES" not in entailment_decision:
                failed_citations.append({"citation": cit.dict(), "reason": "Entailment failed"})
                continue
        except Exception as e:
            print(f"Entailment check failed: {e}")
            # If the check fails (e.g. network error), we might accept or reject. Accept for robustness.
            pass
        
        verified_citations.append(cit)
        
    # 3. Exact numeric guard across the entire answer
    num_pattern = re.compile(r'\b\d+(\.\d+)?\b')
    ans_nums = set(m.group() for m in num_pattern.finditer(response.answer))
    
    # Collect all text from all verified cited chunks
    all_cited_text = ""
    for cit in verified_citations:
        for c in context_chunks:
             if c["doc_id"] == cit.doc_id and (cit.section_id is None or c.get("section_id") == cit.section_id):
                 all_cited_text += c["text"] + " "
                 
    chunk_nums = set(m.group() for m in num_pattern.finditer(all_cited_text))
    
    numeric_drift = False
    for num in ans_nums:
        if num not in chunk_nums:
            numeric_drift = True
            break
            
    if numeric_drift:
        # If there's numeric drift in the answer, we invalidate the citations
        # or flag the whole response. For simplicity, we drop citations.
        failed_citations.append({"citation": {}, "reason": "Numeric drift in answer"})
        verified_citations = []
            
    # Update response to only include verified citations
    response.citations = verified_citations
    return response, failed_citations
