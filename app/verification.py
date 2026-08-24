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
         
    gateway_url = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    key = os.getenv("AI_GATEWAY_KEY", "")
    model = os.getenv("GENERATION_MODEL", "openai/gpt-4o-mini")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    verified_citations = []
    failed_citations = []
    
    for cit in response.citations:
        # Find chunk for the entire document
        doc_chunk_text = ""
        section_chunk_text = ""
        
        for c in context_chunks:
            if c["doc_id"] == cit.doc_id:
                doc_chunk_text += c["text"] + " "
                if cit.section_id in ("", "None") or c.get("section_id") == cit.section_id:
                    section_chunk_text += c["text"] + " "
                
        if not doc_chunk_text:
            failed_citations.append({"citation": cit.dict(), "reason": "Document not found in context"})
            continue
            
        # 1. Fuzzy Quote Match at document level (support ellipsis-separated segments)
        quote_segments = [seg.strip() for seg in re.split(r'\s*(?:\.\.\.|\…)\s*', cit.quote) if seg.strip()]
        if not quote_segments:
            failed_citations.append({"citation": cit.dict(), "reason": "Empty quote"})
            continue
            
        doc_matches = []
        for seg in quote_segments:
            if len(seg) < 4:
                continue
            seg_score = fuzz.partial_ratio(seg.lower(), doc_chunk_text.lower())
            doc_matches.append(seg_score >= 85)
            
        if not doc_matches or not all(doc_matches):
            failed_citations.append({"citation": cit.dict(), "reason": "Quote not found in source document"})
            continue
              
        # 1b. Secondary section-level check
        if section_chunk_text:
            sec_matches = []
            for seg in quote_segments:
                if len(seg) < 4:
                    continue
                sec_score = fuzz.partial_ratio(seg.lower(), section_chunk_text.lower())
                sec_matches.append(sec_score >= 85)
            if not sec_matches or not all(sec_matches):
                cit.section_unconfirmed = True
        else:
            cit.section_unconfirmed = True
              
        # 2. Entailment check
        entailment_prompt = f"Does the provided quote support at least one specific claim made in the following statement? Answer only YES or NO.\n\nStatement: {response.answer}\n\nQuote: {cit.quote}"
        
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
            print(f"Entailment check failed (network/API error): {e}")
            # Fail closed for zero-hallucination guarantee
            failed_citations.append({"citation": cit.dict(), "reason": "Entailment API failure"})
            continue
        
        verified_citations.append(cit)
        
    # 3. Exact numeric guard across the entire answer (excluding list enumerations and section markers)
    num_pattern = re.compile(r'\b\d+(\.\d+)?\b')
    clean_ans = re.sub(r'(?m)^\s*\d+[\.\)]\s*', ' ', response.answer)
    clean_ans = re.sub(r'Section\s+\d+(\.\d+)*', ' ', clean_ans, flags=re.IGNORECASE)
    ans_nums = set(m.group() for m in num_pattern.finditer(clean_ans))
    
    # Collect all text and section_ids from all verified cited chunks
    all_cited_text = ""
    for cit in verified_citations:
        if cit.section_id:
            all_cited_text += cit.section_id + " "
        for c in context_chunks:
             if c["doc_id"] == cit.doc_id:
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
