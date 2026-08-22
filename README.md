# Legal Contract RAG

A robust, containerized Legal Contract RAG (Retrieval-Augmented Generation) prototype designed to accurately extract and cite information from legal documents while guarding against LLM hallucinations.

## Features
- **Convention-Agnostic Chunking:** Smart multi-tier heading detection (Layout → Text Patterns with Capture Groups → Semantic Fallback) that parses any legal document (Articles with Roman numerals, numbered sections, lettered subclauses, exhibits) without hardcoding specific numbering styles.
- **Robust Sub-Chunking:** Automatically caps excessively large semantic sections at ~400 words to maintain high retrieval precision while accurately preserving nested `parent_section` metadata.
- **Server-Side Hybrid Search Retrieval:** Combines Qdrant dense vector embeddings with an Okapi BM25 sparse index (via Reciprocal Rank Fusion) to ensure exact legal terminology isn't lost. When scoped via `doc_filter`, resolves substrings to canonical document IDs and enforces native Qdrant `MatchAny` server-side ANN filtering.
- **Cross-Reference & Defined Term Resolution:** Identifies capitalized legal terms and "Section X.Y" references in retrieved chunks and dynamically pulls those exact definition chunks into the context window to prevent semantic gaps.
- **Strict Hallucination Prevention:** 
  - Uses OpenAI's **Strict Structured Outputs** to enforce exact JSON schema adherence.
  - Generates closed-book answers, heavily penalized against hallucination. Fully out-of-context queries are rejected with a strict `"Not enough information to confirm."` response.
  - **Compound / Multi-Part Query Support:** Answers supported components with exact section citations while explicitly stating absence of information for unsupported parts without triggering all-or-nothing refusals.
  - **Defense-in-Depth Verification:** Citations are verified through **segmented fuzzy substring quote verification** (supporting ellipsis-shortened quotes `...`), **LLM-based claim entailment checks**, and **sanitized numeric drift detection** (preventing list enumerations or section labels from triggering false alarms).
  - Quote verification searches across the full document context, preventing false rejects from minor section formatting differences.
- **Methodical Evaluation:** Built-in endpoint to evaluate pipeline faithfulness using `ragas` metrics against a curated Q&A dataset of real legal queries.

## Setup & Configuration

### Prerequisites
- Docker and Docker Compose installed.

### Configuration
1. Place your target PDF contracts into the `data/contracts/` folder.
2. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Update the `.env` file with your actual Vercel AI Gateway key (`AI_GATEWAY_KEY=your_key`).

### Running the Application
Spin up the FastAPI and Qdrant containers using Docker Compose:
```bash
docker compose up --build
```
The API will be available at `http://localhost:8000`. You can access the auto-generated Swagger documentation at `http://localhost:8000/docs`.

## API Endpoints

### 1. Ingestion (`POST /ingest`)
Parses, chunks, and indexes all PDFs found in `data/contracts/`. This process populates the Qdrant database and builds the BM25 sparse index.
```bash
curl -X POST http://localhost:8000/ingest
```

### 2. Query (`POST /query`)
Submit a natural language question. Returns a structured JSON containing the answer, exact citations (doc_id, section_id, quote), and checks if the provided documents are insufficient.
```bash
curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the governing law?"}' http://localhost:8000/query
```
*Optional: Provide a `doc_filter` (doc_id string) to scope the search to a specific contract.*

### 3. Evaluate (`POST /evaluate`)
Runs the gold-standard questions defined in `data/eval_set/questions.json` through the pipeline and calculates `ragas` faithfulness, as well as our internal verification metrics.
```bash
curl -X POST http://localhost:8000/evaluate
```

## Bonus Tasks Implementation
- **Bonus 1 (Hybrid Search):** Found in `app/retrieval.py` (`hybrid_retrieve` and `rrf` functions).
- **Bonus 2 (Hallucination Prevention):** Found in `app/verification.py` (quote verification, entailment LLM check, and numeric drift matching).
- **Bonus 3 (Faithfulness Evaluation):** Found in `app/main.py` (`/evaluate` route utilizing the `ragas` library).
- **Bonus 4 (Docker Compose):** Located in the `docker-compose.yml` file configuring the `app` and `qdrant` services.
