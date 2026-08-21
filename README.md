# Legal Contract RAG

A robust, containerized Legal Contract RAG (Retrieval-Augmented Generation) prototype designed to accurately extract and cite information from legal documents while guarding against LLM hallucinations.

## Features
- **Convention-Agnostic Chunking:** Smart multi-tier heading detection (Layout → Text Patterns → Semantic Fallback) that parses any legal document without hardcoding specific numbering styles.
- **Hybrid Search Retrieval:** Combines Qdrant dense vector embeddings with an Okapi BM25 sparse index (via Reciprocal Rank Fusion) to ensure exact legal terminology isn't lost. Includes cross-reference resolution.
- **Strict Hallucination Prevention:** Enforces closed-book JSON generation, fuzzy quote verification against original document chunks, numeric drift detection, and claim entailment checking.
- **Methodical Evaluation:** Built-in endpoint to evaluate pipeline faithfulness using `ragas` metrics against a curated Q&A dataset.

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
