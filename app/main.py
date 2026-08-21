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
    
    return QueryResponse(
        answer=verified_resp.answer,
        citations=[c.dict() for c in verified_resp.citations],
        insufficient_context=verified_resp.insufficient_context,
        failed_citations=failed_cits
    )

@app.post("/evaluate")
def evaluate():
    questions = evaluate_pipeline()
    if "error" in questions:
        return questions
        
    results = []
    total = len(questions)
    passed = 0
    
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    for q in questions:
        # Run through our pipeline
        chunks = hybrid_retrieve(q["question"])
        raw_response = generate_answer(q["question"], chunks)
        verified_resp, failed_cits = verify_citations(raw_response, chunks)
        
        is_correct = False
        if q.get("expected_none"):
            is_correct = verified_resp.insufficient_context
        else:
            is_correct = len(verified_resp.citations) > 0 and not verified_resp.insufficient_context
            
        if is_correct:
            passed += 1
            
        results.append({
            "question": q["question"],
            "correct": is_correct,
            "answer": verified_resp.answer,
            "failed_citations": len(failed_cits)
        })
        
        ragas_data["question"].append(q["question"])
        ragas_data["answer"].append(verified_resp.answer)
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
        ragas_scores = ragas_result
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
        ragas_scores = {"error": str(e)}
        
    return {
        "accuracy": passed / total if total > 0 else 0,
        "total": total,
        "ragas_scores": ragas_scores,
        "details": results
    }

@app.get("/health")
def health():
    return {"status": "ok"}
