import os
import json
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from .ingest import ingest_all
from .retrieval import hybrid_retrieve
from .generation import generate_answer, GenerationResponse
from .verification import verify_citations
from .evaluate import evaluate_pipeline

app = FastAPI(title="Legal Contract RAG")

class QueryRequest(BaseModel):
    question: str
    doc_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    insufficient_context: bool
    failed_citations: list[dict]
    is_unverified: bool

@app.get("/documents")
def list_documents():
    docs_path = "/app/data/documents.json" if os.path.exists("/app/data/documents.json") else "./data/documents.json"
    if os.path.exists(docs_path):
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                return {"documents": json.load(f)}
        except Exception:
            pass
            
    # Fallback to chunk store if documents.json not generated yet
    chunks_path = "/app/data/chunks.json" if os.path.exists("/app/data/chunks.json") else "./data/chunks.json"
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            doc_ids = sorted(list({c["doc_id"] for c in all_chunks if "doc_id" in c}))
            return {"documents": [{"doc_id": d, "description": f"Legal agreement ({d})."} for d in doc_ids]}
        except Exception as e:
            return {"documents": [], "error": str(e)}
            
    return {"documents": []}

@app.post("/ingest")
def ingest():
    success = ingest_all()
    if success:
        return {"status": "success"}
    return {"status": "error", "message": "Ingestion failed"}

@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    # 1. Hybrid Retrieval
    chunks = hybrid_retrieve(body.question, body.doc_filter)
    
    # 2. Generate Answer
    raw_response = generate_answer(body.question, chunks)
    
    # 3. Verify
    verified_resp, failed_cits = verify_citations(raw_response, chunks)
    
    answer_text = verified_resp.answer
    is_unverified = False
    
    # Force canonical string on refusal to prevent LLM chatty behavior
    negative_phrases = ["no information provided", "not enough information", "does not contain", "no mention of", "cannot find"]
    is_pure_negative = len(verified_resp.citations) == 0 and any(p in answer_text.lower() for p in negative_phrases)

    if verified_resp.insufficient_context or is_pure_negative:
        answer_text = "Not enough information to confirm."
        verified_resp.citations = []
        verified_resp.insufficient_context = True
        is_unverified = False
    elif len(verified_resp.citations) == 0:
        is_unverified = True
        answer_text = "[WARNING: This answer could not be verified against the source text and may contain hallucinations.]\n\n" + answer_text
    
    return QueryResponse(
        answer=answer_text,
        citations=[c.dict() for c in verified_resp.citations],
        insufficient_context=verified_resp.insufficient_context,
        failed_citations=failed_cits,
        is_unverified=is_unverified
    )

@app.post("/evaluate")
def evaluate():
    questions = evaluate_pipeline()
    if "error" in questions:
        return questions
        
    results = []
    total = len(questions)
    passed = 0
    flagged = 0
    
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    for q in questions:
        # Run through our pipeline
        chunks = hybrid_retrieve(q["question"], q.get("doc_filter"))
        raw_response = generate_answer(q["question"], chunks)
        verified_resp, failed_cits = verify_citations(raw_response, chunks)
        
        answer_text = verified_resp.answer
        is_pure_negative = len(verified_resp.citations) == 0 and any(p in answer_text.lower() for p in negative_phrases)
        if is_pure_negative:
            verified_resp.insufficient_context = True

        status = "wrong"
        if q.get("expected_none"):
            if verified_resp.insufficient_context:
                status = "correct"
        else:
            if not verified_resp.insufficient_context:
                if len(verified_resp.citations) > 0:
                    status = "correct"
                else:
                    status = "flagged_unverified"
            
        if verified_resp.insufficient_context:
            answer_text = "Not enough information to confirm."
            verified_resp.citations = []
        elif len(verified_resp.citations) == 0:
            answer_text = "[WARNING: This answer could not be verified against the source text and may contain hallucinations.]\n\n" + answer_text
            
        if status == "correct":
            passed += 1
        elif status == "flagged_unverified":
            flagged += 1
            
        results.append({
            "question": q["question"],
            "status": status,
            "answer": answer_text,
            "failed_citations": len(failed_cits)
        })
        
        ragas_data["question"].append(q["question"])
        ragas_data["answer"].append(answer_text)
        ragas_data["contexts"].append([c["text"] for c in chunks])
        
    ragas_scores = {}
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness
        
        # Setting up Ragas to use our AI gateway keys
        os.environ["OPENAI_API_KEY"] = os.getenv("AI_GATEWAY_KEY", "")
        os.environ["OPENAI_API_BASE"] = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
        
        ds = Dataset.from_dict(ragas_data)
        ragas_result = ragas_evaluate(
            ds,
            metrics=[faithfulness]
        )
        
        # Convert Result object and numpy floats to standard Python dict/floats for FastAPI serialization
        for k, v in dict(ragas_result).items():
            try:
                ragas_scores[k] = float(v)
            except (TypeError, ValueError):
                ragas_scores[k] = str(v)
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        ragas_scores = {"error": str(e)}
        
    return {
        "accuracy": passed / total if total > 0 else 0,
        "guardrail_success_rate": (passed + flagged) / total if total > 0 else 0,
        "total": total,
        "ragas_scores": ragas_scores,
        "details": results
    }

@app.get("/health")
def health():
    return {"status": "ok"}
