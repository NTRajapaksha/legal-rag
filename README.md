# Legal Contract RAG

A robust, containerized Legal Contract RAG (Retrieval-Augmented Generation) prototype designed to accurately extract and cite information from complex legal documents while aggressively guarding against LLM hallucinations.

---

## 🏛 System Architecture

```
                                 ┌────────────────────────────────────────────────────────┐
                                 │                   FastAPI Service                      │
                                 │                                                        │
    PDFs ──── /ingest ──────────▶│  • Multi-Tier Heading & Section Parser                 │
(data/contracts)                 │  • Chunking (~400 words) & Defined-Terms Extraction    │
                                 │  • Qdrant Dense Vector Embeddings + BM25 Sparse Index  │
                                 │                                                        │
    User ───── /query ──────────▶│  • Pre-Resolved Native Server-Side Scoped Search       │
   Query                         │  • Hybrid Retrieval (Dense + BM25 via RRF)             │
                                 │  • Cross-Reference & Defined-Term Injection            │
                                 │  • Strict Structured Output Generation (OpenAI)        │
                                 │  • Triple-Layer Post-Generation Verification Guard     │
                                 │                                                        │
    Eval ──── /evaluate ────────▶│  • Ragas Faithfulness & Guardrail Evaluation           │
                                 └──────────────────────────┬─────────────────────────────┘
                                                            │
                                  ┌─────────────────────────┼─────────────────────────┐
                                  ▼                         ▼                         ▼
                          ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
                          │    Qdrant     │         │  BM25 Index   │         │   Vercel AI   │
                          │ Vector Store  │         │  & Payloads   │         │    Gateway    │
                          │ (Dense ANN)   │         │   (on-disk)   │         │ (LLM / Embed) │
                          └───────────────┘         └───────────────┘         └───────────────┘
```

---

## 📂 Repository Folder Structure

```
.
├── app/                               # FastAPI Application Core
│   ├── __init__.py
│   ├── main.py                        # REST API routes (/ingest, /query, /evaluate, /health)
│   ├── ingest.py                      # PDF batch parsing, chunking, and dual-index building
│   ├── chunking.py                    # Multi-tier layout & regex heading detection, sub-chunking
│   ├── defined_terms.py               # Defined terms index builder and scanner
│   ├── retrieval.py                   # Server-side hybrid retrieval (Qdrant + BM25 + RRF + cross-refs)
│   ├── generation.py                  # Strict structured generation & system prompts
│   ├── verification.py                # Triple-layer citation grounding, entailment & numeric guards
│   └── evaluate.py                    # Evaluation dataset loader and pipeline runner
├── data/                              # Persistent Data & Volume Mount
│   ├── contracts/                     # ◀◀ TARGET FOLDER: Place all input PDF contracts here
│   │   ├── ACCELERATEDTECHNOLOGIES...pdf
│   │   ├── BellringBrandsInc...pdf
│   │   ├── Freecook...pdf
│   │   ├── MorganStanleyDirectLendingFund...pdf
│   │   └── PenntexMidstreamPartnersLp...pdf
│   ├── eval_set/                      # Gold-standard evaluation Q&A pairs
│   │   └── questions.json             # 20 curated benchmark legal questions
│   ├── bm25_index.pkl                 # Serialized Okapi BM25 sparse index (generated on ingest)
│   ├── chunks.json                    # Chunk metadata and text store (generated on ingest)
│   └── defined_terms.json             # Defined terms lookup table (generated on ingest)
├── approach_report.md                 # Detailed report on architectural choices and methodology
├── architecture.md                    # Technical architecture design and decision documentation
├── production_readiness_report.md     # Production gap analysis & scaling recommendations
├── docker-compose.yml                 # Multi-container orchestration (FastAPI + Qdrant)
├── Dockerfile                         # Application container build specification
├── requirements.txt                   # Python package dependencies
└── README.md                          # Project overview and usage guide
```

---

## 📄 PDF Contract Storage Instructions

> [!IMPORTANT]
> All target PDF contracts must be placed inside the **`data/contracts/`** directory.

- **Supported Format:** Standard text-based `.pdf` / `.PDF` files.
- **OCR Support:** If a scanned page with no extractable text layer is encountered, the pipeline automatically invokes `pytesseract` OCR as a fallback.
- **Persistence:** The `./data` folder is mounted into the container as a persistent Docker volume (`./data:/app/data`). When you add or replace PDFs in `data/contracts/`, call `POST /ingest` to re-parse and re-index the documents.

---

## ✨ Key Features

- **Convention-Agnostic Chunking:** Smart multi-tier heading detection (Layout → Regex Patterns with Capture Groups → Semantic Fallback) that parses any legal structure (Roman-numeral Articles, decimal sections, lettered subclauses, exhibits) without hardcoded templates.
- **Hierarchy Preservation:** Sub-chunks oversized clauses at ~400 words while maintaining exact `parent_section` and `page_number` metadata.
- **Server-Side Hybrid Search:** Combines dense embeddings in Qdrant with an Okapi BM25 sparse index merged via Reciprocal Rank Fusion (RRF). Human-friendly `doc_filter` values are pre-resolved to canonical IDs and enforced server-side using native Qdrant `MatchAny` filters.
- **Cross-Reference & Defined-Term Injection:** Scans retrieved text for capitalized defined terms and "Section X.Y" pointers, dynamically pulling definition chunks into context.
- **Strict Hallucination Prevention:** 
  - **OpenAI Strict Structured Outputs:** Enforces exact JSON schemas for generation.
  - **Fail-Closed Guardrails:** Refuses unsupported queries with `"Not enough information to confirm."`
  - **Compound Query Support:** Answers supported components with exact citations while declaring absence for unmentioned topics without triggering all-or-nothing refusals.
  - **Triple-Layer Post-Generation Verification:**
    1. *Segmented Fuzzy Quote Grounding:* Validates quotes against the source text, supporting ellipsis-shortened quotes (`...`).
    2. *Claim Entailment Verification:* LLM validates that the quote logically entails the stated claim.
    3. *Sanitized Numeric Drift Guard:* Strictly matches all numbers, dates, and amounts against the source text while filtering out markdown list markers (`1.`) and section labels (`Section X.Y`).
- **Methodical Faithfulness Evaluation:** Built-in `POST /evaluate` endpoint calculating `ragas` faithfulness metrics across curated gold-standard questions.

---

## 🚀 Setup & Configuration

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed.

### Configuration
1. Place your target PDF contracts into `data/contracts/`.
2. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Update `.env` with your Vercel AI Gateway credentials:
   ```env
   AI_GATEWAY_KEY=your_vercel_ai_gateway_key
   AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1
   ```

### Running the Application
Spin up the FastAPI app and Qdrant database:
```bash
docker compose up --build
```
- **API Base URL:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`

---

## 📡 API Endpoints

### 1. Ingest Documents (`POST /ingest`)
Parses, chunks, embeds, and indexes all PDFs currently located in `data/contracts/`.
```bash
curl -X POST http://localhost:8000/ingest
```
**Response:**
```json
{
  "status": "success"
}
```

---

### 2. Query Contract (`POST /query`)
Executes hybrid retrieval, structured generation, and triple-layer verification.

#### Single-Contract Query with `doc_filter`:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the governing law of the Joint Venture Agreement?",
    "doc_filter": "JOINT VENTURE"
  }'
```

#### Multi-Document Cross-Comparison Query (No filter):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the governing law specified in each of the contracts?"
  }'
```

**Sample Response:**
```json
{
  "answer": "The governing law of the Joint Venture Agreement is the laws of the Commonwealth of Pennsylvania, as stated in Section 14: \"the construction and interpretation of the terms and provisions of this Agreement shall be interpreted and construed under the laws of the Commonwealth of Pennsylvania.\"",
  "citations": [
    {
      "doc_id": "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.PDF",
      "section_id": "14",
      "quote": "the construction and interpretation of the terms and provisions of this Agreement shall be interpreted and construed under the laws of the Commonwealth of Pennsylvania.",
      "section_unconfirmed": false
    }
  ],
  "insufficient_context": false,
  "failed_citations": [],
  "is_unverified": false
}
```

---

### 3. Evaluate Pipeline (`POST /evaluate`)
Runs the 20 gold-standard legal questions through the live pipeline and scores accuracy, guardrail interception rates, and `ragas` faithfulness.
```bash
curl -X POST http://localhost:8000/evaluate
```

---

### 4. Health Check (`GET /health`)
```bash
curl http://localhost:8000/health
```

---

## 🏆 Bonus Tasks Implementation Mapping

| Bonus Task | Implementation Details | Location in Codebase |
| :--- | :--- | :--- |
| **Bonus 1: Hybrid Search & Retrieval** | Dense vector search via Qdrant combined with sparse Okapi BM25 via Reciprocal Rank Fusion (RRF), native server-side `MatchAny` filtering, and cross-reference/defined-term injection. | [app/retrieval.py](app/retrieval.py) |
| **Bonus 2: Hallucination Prevention** | Closed-book strict structured prompt, segmented fuzzy quote matcher, LLM claim entailment check, and sanitized numeric drift detector. | [app/verification.py](app/verification.py) & [app/generation.py](app/generation.py) |
| **Bonus 3: Faithfulness Evaluation** | Batch benchmark evaluator calculating `ragas` faithfulness and guardrail containment rate over curated questions. | [app/main.py](app/main.py) & [app/evaluate.py](app/evaluate.py) |
| **Bonus 4: Docker Compose** | Multi-container setup with automated service networking and persistent volume mounts. | [docker-compose.yml](docker-compose.yml) & [Dockerfile](Dockerfile) |
