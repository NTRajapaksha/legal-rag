import os
import httpx
from pydantic import BaseModel, Field

class Citation(BaseModel):
    doc_id: str
    section_id: str | None
    quote: str

class GenerationResponse(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool

def generate_answer(question: str, context_chunks: list[dict]) -> GenerationResponse:
    if not context_chunks:
        return GenerationResponse(
            answer="Not enough information to confirm.",
            citations=[],
            insufficient_context=True
        )
        
    gateway_url = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    key = os.getenv("AI_GATEWAY_KEY", "")
    model = os.getenv("GENERATION_MODEL", "openai/gpt-4o-mini")
    
    # Build context string
    doc_ids = set()
    context_str = ""
    for c in context_chunks:
        doc_ids.add(c["doc_id"])
        sec = c.get("section_id", "None")
        context_str += f"[{c['doc_id']} | Section {sec}] {c['text']}\n\n"
        
    multi_doc = len(doc_ids) > 1
    
    system_prompt = """You are a legal assistant. Answer ONLY from the provided context chunks.
If the answer isn't in the context, set insufficient_context to true and output EXACTLY: "Not enough information to confirm." Do not attempt to guess or hallucinate an answer.
Keep a professional, objective tone. For every claim, include the exact section number and a short supporting quote."""

    if multi_doc:
        system_prompt += "\nAnswer per-document (e.g. 'Contract A: ... Contract B: ...') unless all sources agree."
        
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # Use OpenAI function calling / structured output format
    schema = GenerationResponse.model_json_schema()
    
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "GenerationResponse",
                "schema": schema,
                "strict": True
            }
        }
    }
    
    try:
        response = httpx.post(f"{gateway_url}/chat/completions", headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        import json
        parsed = json.loads(content)
        return GenerationResponse(**parsed)
    except Exception as e:
        print(f"Generation failed: {e}")
        if 'response' in locals():
            print(response.text)
        return GenerationResponse(
            answer="Error during generation.",
            citations=[],
            insufficient_context=True
        )
