# Legal Contract RAG

A robust, containerized Legal Contract RAG (Retrieval-Augmented Generation) prototype designed to accurately extract and cite information from complex legal documents while aggressively guarding against LLM hallucinations.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion Pipeline"]
        PDF["PDF Contracts<br><i>data/contracts/</i>"] --> Parser["Multi-Tier Parser<br>(pdfplumber + OCR)"]
        Parser --> Chunking["Structural Chunking<br>(~400 words + Headings)"]
        Chunking --> Terms["Defined-Terms Extractor"]
        Chunking --> DenseEmbed["Dense Embeddings<br>(OpenAI text-embedding-3-small)"]
        Chunking --> SparseIndex["Sparse Indexer<br>(Okapi BM25)"]
        DenseEmbed --> Qdrant[("Qdrant Vector Store<br>(Dense ANN)")]
        SparseIndex --> BM25File[("BM25 Index File<br>(on-disk)")]
        Terms --> TermsFile[("Defined Terms Index<br>(JSON lookup)")]
    end

    subgraph Retrieval ["2. Hybrid Retrieval Engine"]
        UserQuery["User Query<br>(POST /query)"] --> Scoper["Server-Side Scoper<br>(MatchAny doc_id)"]
        Scoper --> DenseSearch["Dense Vector Search"]
        Scoper --> SparseSearch["Sparse Lexical Search"]
        DenseSearch --> RRF["Reciprocal Rank Fusion<br>(RRF Merging)"]
        SparseSearch --> RRF
        RRF --> ContextInjection["Context Expansion<br>(Cross-Ref & Defined-Term Injection)"]
    end

    subgraph Generation ["3. Generation & Verification Guard"]
        ContextInjection --> LLMGen["LLM Answer Generator<br>(OpenAI Strict JSON Schema)"]
        LLMGen --> V1["Layer 1: Segmented Fuzzy Quote Matcher"]
        V1 --> V2["Layer 2: LLM Claim Entailment Verifier"]
        V2 --> V3["Layer 3: Sanitized Numeric Drift Guard"]
        V3 --> FinalResponse["Audited JSON Output<br>(Answer + Verified Citations)"]
    end

    Qdrant -.-> DenseSearch
    BM25File -.-> SparseSearch
    TermsFile -.-> ContextInjection

    classDef stage fill:#f8f9fa,stroke:#495057,stroke-width:1px,color:#212529;
    classDef storage fill:#e9ecef,stroke:#6c757d,stroke-width:1.5px,color:#212529;
    classDef accent fill:#e7f5ff,stroke:#1c7ed6,stroke-width:1.5px,color:#1864ab;
    class Scoper,RRF,LLMGen,FinalResponse accent;
    class Qdrant,BM25File,TermsFile storage;
```

---

## Core Capabilities

| Capability | Technical Mechanism | Benefit |
| :--- | :--- | :--- |
| **Convention-Agnostic Parsing** | Multi-tier detection (Layout $\rightarrow$ Regex capture groups $\rightarrow$ Semantic drop) | Parses Roman articles (`Article XV`), decimal sections, and subclauses across any contract. |
| **Server-Side Hybrid Retrieval** | Dense vector search (Qdrant) + Sparse lexical (BM25) fused via RRF | Captures semantic intent without losing exact legal jargon, party names, or section numbers. |
| **Cross-Reference Resolution** | Automated scanning for defined terms and `Section X.Y` pointers | Eliminates semantic gaps by auto-injecting referenced definitions into the context window. |
| **Triple-Layer Guardrails** | Segmented fuzzy quote check + LLM claim entailment + numeric drift filter | Guarantees zero hallucinations; validates quotes with `...` and rejects invented figures. |
| **Compound Query Handling** | Multi-intent prompt routing with partial-grounding priority rules | Answers supported questions with citations and declares absence for unmentioned topics. |

---

## Project Structure

```
.
├── app/
│   ├── main.py           # REST API endpoints (/ingest, /query, /evaluate, /health)
│   ├── ingest.py         # PDF parsing, chunking, and dual-index building
│   ├── chunking.py       # Multi-tier layout and regex heading detection, sub-chunking
│   ├── defined_terms.py  # Defined terms index builder and lookup scanner
│   ├── retrieval.py      # Server-side hybrid retrieval (Qdrant + BM25 + RRF)
│   ├── generation.py     # Strict structured generation and system prompts
│   ├── verification.py   # Triple-layer citation grounding, entailment, and numeric guards
│   └── evaluate.py       # Evaluation dataset loader and pipeline runner
├── data/                 # Persistent storage volume
│   ├── contracts/        # Target directory: place PDF contracts here
│   └── eval_set/         # Benchmark evaluation dataset
├── docker-compose.yml    # Container orchestration (FastAPI + Qdrant)
├── Dockerfile            # Application container specification
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## PDF Storage Instructions

> All target PDF contracts must be placed inside the **`data/contracts/`** directory.

* **Supported Input:** Standard text-based `.pdf` / `.PDF` documents.
* **OCR Fallback:** Automatic `pytesseract` image extraction on scanned/image-only pages.
* **Data Persistence:** The `./data` host directory is mounted as a persistent Docker volume (`./data:/app/data`).

---

## Quickstart Guide

### 1. Environment Setup
Copy the environment template and provide your API credentials:
```bash
cp .env.example .env
```
Ensure `.env` contains your Vercel AI Gateway key:
```env
AI_GATEWAY_KEY=your_key_here
AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1
```

### 2. Launch Services
Start the application and Qdrant vector database:
```bash
docker compose up --build
```
* **API Endpoint:** `http://localhost:8000`
* **Swagger Documentation:** `http://localhost:8000/docs`

---

## API Reference

### Ingest Documents
Parses and builds the dense and sparse indices from `data/contracts/`.
```bash
curl -X POST http://localhost:8000/ingest
```

---

### Query Contracts

#### Scoped Single-Document Query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the governing law of the Joint Venture Agreement?",
    "doc_filter": "JOINT VENTURE"
  }'
```

#### Multi-Document Cross-Comparison Query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the governing law specified across all agreements?"
  }'
```

#### Response Structure:
```json
{
  "answer": "The governing law of the Joint Venture Agreement is the laws of the Commonwealth of Pennsylvania, as stated in Section 14...",
  "citations": [
    {
      "doc_id": "ACCELERATEDTECHNOLOGIESHOLDINGCORP..._JOINT VENTURE AGREEMENT.PDF",
      "section_id": "14",
      "quote": "...construction and interpretation... shall be interpreted and construed under the laws of the Commonwealth of Pennsylvania...",
      "section_unconfirmed": false
    }
  ],
  "insufficient_context": false,
  "failed_citations": [],
  "is_unverified": false
}
```

---

### Batch Faithfulness Evaluation
Executes the gold-standard 20-question legal benchmark and reports `ragas` faithfulness:
```bash
curl -X POST http://localhost:8000/evaluate
```

---

## Bonus Tasks Mapping

| Assessment Item | Implementation Component | File Location |
| :--- | :--- | :--- |
| **Bonus 1: Hybrid Search** | Dense Qdrant + Okapi BM25 with Reciprocal Rank Fusion & Server-Side Filtering | `app/retrieval.py` |
| **Bonus 2: Anti-Hallucination** | Segmented quote grounding, LLM entailment verification & numeric drift guard | `app/verification.py` |
| **Bonus 3: Faithfulness Evaluation** | Benchmark execution with `ragas` faithfulness & guardrail containment metrics | `app/main.py` |
| **Bonus 4: Docker Compose** | Multi-container orchestration (FastAPI + Qdrant) with persistent volume mounts | `docker-compose.yml` |
