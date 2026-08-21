# Legal Contract RAG — Implementation Guide (Start to End)

Companion to `architecture.md`. Follow in order — each step assumes the previous one is done.

---

## Step 0 — Project Setup

```bash
mkdir legal-rag && cd legal-rag
mkdir -p app data/contracts data/eval_set
python -m venv venv && source venv/bin/activate
git init
```

`requirements.txt`:
```
fastapi
uvicorn[standard]
pydantic
pdfplumber
pytesseract
pdf2image
qdrant-client
rank-bm25
rapidfuzz
numpy
nltk
httpx
python-dotenv
ragas
datasets
```

```bash
pip install -r requirements.txt
```

`.env`:
```
AI_GATEWAY_KEY=your_vercel_ai_gateway_key_here
AI_GATEWAY_BASE_URL=https://ai-gateway.vercel.sh/v1
QDRANT_URL=http://localhost:6333
GENERATION_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
```
Place the 5 provided PDFs in `data/contracts/`.

---

## Step 1 — Parsing (`app/parsing.py`)

1. Load each PDF with `pdfplumber` (not `pypdf`), extract text page by page with layout metadata (font, position), keep `page_number` alongside text.
2. **Strip repeated footer noise** — regex out lines matching `^Source:.*\d{1,2}/\d{1,2}/\d{4}$` and bare page-number lines.
3. **Detect multi-column layout before extracting reading order.** Signature blocks and fee schedules are often laid out in two columns; naive left-to-right extraction interleaves them into nonsense. Use `page.extract_words()` and cluster by x-coordinate — if two distinct x-clusters exist for a page, extract each column separately (top-to-bottom) before concatenating, rather than trusting default reading order. `pdfplumber.extract_tables()` handles genuine tables directly.
4. **Detect near-empty pages** (below a character-count threshold) — this indicates a scanned image with no real text layer. Either OCR it with `pytesseract` or mark it `{"page_number": n, "unextracted": true}` in metadata so the gap is visible rather than silently missing from the index.
5. Concatenate cleaned pages into one full-text string per document, keeping a `page_number` offset map so any later text position can be traced back to a page.

```python
def extract_pdf_text(path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = extract_columns_aware(page)  # falls back to page.extract_text() if single-column
            text = strip_footer_noise(text)
            if len(text.strip()) < MIN_PAGE_CHARS:
                text = ocr_fallback(page) or ""
                unextracted = not text
            else:
                unextracted = False
            pages.append({"page_number": i + 1, "text": text, "unextracted": unextracted})
    return pages
```

---

## Step 1.5 — Defined-Terms Index (`app/defined_terms.py`)

1. After parsing, scan each document's full text for the defined-term pattern: a capitalized phrase in quotes followed by `means`, `shall mean`, or `refers to` (e.g. `"Brand"` ... `means the trademark...`).
2. Store `{doc_id, term, definition_text, chunk_id}` in a lookup table (simple JSON or a small SQLite table is enough).
3. This index is consulted at query time (Step 4) — if a question or a retrieved chunk contains a term found here, the definition chunk is force-included in the context, whether or not it scored high enough via search.

---

## Step 2 — Clause-Aware Chunking (`app/chunking.py`)

Design goal: work on **any** legal document's numbering convention, not just the ones seen so far. Use a layered detector — layout signal first, regex only to *classify* what layout already found, recursive splitting as the universal fallback.

1. **Extract with layout metadata, not just raw text.** Use `pdfplumber` instead of `pypdf` so each line carries font size, bold/weight, and position:
   ```python
   for page in pdf.pages:
       for line in page.extract_text_lines():
           # line has: text, top, chars (with fontname, size)
           is_bold = any("Bold" in c["fontname"] for c in line["chars"])
           avg_size = mean(c["size"] for c in line["chars"])
   ```
2. **Tier 1 — detect heading candidates by layout.** A line is a heading candidate if it's short (<80 chars), and either bold, above-average font size for the doc, or entirely uppercase, and stands alone on its own line (not wrapped mid-paragraph). This is convention-agnostic — it fires the same way on `Section 1.1`, `ARTICLE I`, `1.`, or an unnumbered bolded topic line.
3. **Tier 2 — detect heading candidates by text pattern, independent of formatting.** Some genuinely typed documents number their clauses (`Section 1.1`, `ARTICLE I`, `1.`) as plain inline text with no bold, no size change, nothing visually distinct — layout detection alone would miss these entirely and wrongly treat the whole document as unstructured. Run a second pass over every line (regardless of font/weight) matching it against the same heading-label regex library used for classification (below). Any line matching, at the start of a line, on its own, is promoted to a heading candidate even with zero layout signal. Merge Tier 1 and Tier 2 candidates (dedup by line position) — a heading only needs one of the two signals to be captured.
4. **Classify each candidate with a regex *library*, purely to extract an ID/label — not to find the heading in the first place.** Keep this as an extensible config table, not hardcoded logic:
   ```python
   HEADING_LABEL_PATTERNS = [
       (r"Section\s+\d+(\.\d+)*", "numbered_section"),
       (r"ARTICLE\s+[IVXLC]+", "article_roman"),
       (r"^\d+(\.\d+)*\.", "numbered"),
       (r"EXHIBIT|SCHEDULE|ANNEX|APPENDIX\s+[A-Z0-9]+", "exhibit"),
       (r"^\([a-z]\)|^[A-Z]\.", "lettered_subclause"),
   ]
   ```
   If a heading candidate (from either tier) matches none of these, it still becomes a heading — just with `section_id = None` and the raw heading text as `section_title`. Nothing is silently dropped for not matching a known convention.
5. Walk the document splitting at each detected heading (merged Tier 1 + Tier 2). Track `section_id`, `section_title`, and `parent_section` (nearest preceding heading at a shallower nesting level, inferred from indentation/numbering depth rather than a fixed hierarchy).
6. **Tier 3 — semantic chunking as the fallback for documents with no discoverable structure at all** (genuinely unnumbered prose, poor OCR, scanned images with weak text layer). Only reached if *both* Tier 1 and Tier 2 find nothing — meaning the document truly has no structure, not merely no formatting. Rather than blind fixed-size splitting, use semantic similarity to find natural topic boundaries even without any heading markup:
   ```python
   sentences = split_into_sentences(text)
   embeddings = embed_batch(sentences)          # same embedding model as indexing
   sims = [cosine_sim(embeddings[i], embeddings[i+1]) for i in range(len(sentences)-1)]
   # a breakpoint is where similarity drops sharply relative to the doc's own distribution
   threshold = percentile(sims, 5)              # e.g. bottom 5% of similarity scores
   breakpoints = [i for i, s in enumerate(sims) if s < threshold]
   ```
   Group sentences between breakpoints into chunks, still capping at ~500 tokens (split further if a semantic segment runs long, merge if segments are too short to be useful). This catches boundaries that blind token-splitting misses — e.g. "Either party may terminate..." and "Upon termination, all licenses shall cease..." are different topics even with zero formatting, and similarity-drop detection can separate them where a fixed token count would cut mid-clause. These chunks get `section_id = None` and rely on `page_number` + a computed offset for citation instead — degraded precision (no section number), but with cleaner topic boundaries than blind splitting.
   
   **Cost tradeoff, and why it's fine:** this needs one extra embedding call per sentence, on top of the per-chunk embedding already done at indexing time. Because it's only invoked when Tiers 1 and 2 both find nothing, this extra cost only applies to genuinely unstructured documents — well-formatted contracts (the common case) never pay for it.
7. **ALL-CAPS block protection** (content-based, already general): never split inside a paragraph that's fully uppercase, since these are almost always disclaimer/liability/waiver clauses in any legal doc, and cutting them mid-block risks a misleading partial citation.
8. Emit chunk objects:
   ```python
   {
     "doc_id": "trademark_license_agreement",
     "section_id": "6.2",              # or None if undetected
     "section_title": "Representations and Warranties; Limitations",
     "parent_section": "6. Representations and Warranties; Limitations",
     "page_number": 3,
     "text": "..."
   }
   ```
9. Validate generality, not just correctness on the sample docs: run chunking against a handful of contracts with *deliberately different* conventions (e.g. a Roman-numeral `ARTICLE I` doc, a plainly-typed doc with numbering but no bold/size distinction, and an unnumbered plain-prose NDA) in addition to the provided 5, and confirm no clause is cut mid-sentence and no section is silently merged into its neighbor. This is the highest-leverage check in the project — a chunking bug here undermines every downstream citation.

---

## Step 3 — Embedding + Indexing (`app/ingest.py`)

1. Start Qdrant locally for dev (`docker run -p 6333:6333 qdrant/qdrant`) before Compose is wired up.
2. Call the embedding model through the AI Gateway for each chunk's `text`, batching ~50 chunks per request.
3. Create the Qdrant collection once:
   ```python
   client.recreate_collection(
       collection_name="contracts",
       vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
   )
   ```
4. Upsert points with the chunk metadata as `payload` (so `doc_id`/`section_id`/`page_number`/`doc_date` come back with every search hit) — include the source document's filing/effective date where extractable, so conflicting clauses across an original and an amendment can be surfaced rather than silently merged.
5. In the same pass, build the BM25 index over the same chunk list (`rank_bm25.BM25Okapi` on tokenized chunk text) and pickle it to `data/bm25_index.pkl` alongside a `data/chunks.json` mapping chunk index → metadata (BM25 needs the raw corpus in memory; Qdrant only stores vectors + payload).
6. Wrap all of this in `POST /ingest` so it's re-runnable, not a one-off script — re-running should wipe and rebuild both indexes cleanly (idempotent).

---

## Step 4 — Hybrid Retrieval (`app/retrieval.py`)

1. **Dense leg**: embed the incoming question, `client.search(collection_name="contracts", query_vector=..., limit=20)`.
2. **Sparse leg**: tokenize the question, `bm25.get_scores(...)`, take top 20 chunk indices.
3. **Fuse with Reciprocal Rank Fusion**:
   ```python
   def rrf(dense_ranked_ids, sparse_ranked_ids, k=60):
       scores = defaultdict(float)
       for rank, cid in enumerate(dense_ranked_ids):
           scores[cid] += 1 / (k + rank + 1)
       for rank, cid in enumerate(sparse_ranked_ids):
           scores[cid] += 1 / (k + rank + 1)
       return sorted(scores.items(), key=lambda x: -x[1])
   ```
4. Take the fused **top 8–10** as final context (wider than a bare top-5, so a correct chunk ranked just outside the top few isn't silently dropped, which would otherwise push generation toward guessing rather than a truthful "not found").
5. **Resolve cross-references.** Scan the retrieved chunk text for patterns like `Section \d+(\.\d+)*` or `Article [IVXLC]+` that reference a section not already in the retrieved set, and fetch those chunks directly from the chunk store by ID — a document's own internal references are a stronger signal than search ranking.
6. **Inject defined-term definitions.** Check retrieved chunk text and the question itself against the defined-terms index (Step 1.5); if a known term appears, force-include its definition chunk even if it wasn't retrieved by search.
7. If the top fused score is below a floor threshold (and no cross-reference/defined-term chunks were found either), return an empty context list — this feeds the refusal path in Step 6.
8. Support an optional `doc_filter` param that adds a Qdrant payload filter (`doc_id == X`) before search, for questions scoped to one contract. For questions that aren't scoped to one document, tag the retrieved set with how many distinct `doc_id`s it spans — this flag feeds the multi-document synthesis guard in Step 6.
9. (Optional stretch: rerank the fused set with a cross-encoder call before truncating.)

---

## Step 5 — Prompt & Structured Generation (`app/generation.py`)

1. System prompt (paraphrased intent, not verbatim — write your own wording):
   - Answer only from the provided context chunks.
   - If the answer isn't in the context, say so explicitly rather than guessing.
   - Keep a professional, objective tone — no legal advice framing, describe what the contract says.
   - For every claim, include the exact section number and a short supporting quote.
2. Call the gateway with a JSON schema / tool-call forcing structured output:
   ```json
   {
     "answer": "string",
     "citations": [
       {"doc_id": "string", "section_id": "string", "quote": "string"}
     ],
     "insufficient_context": false
   }
   ```
3. `temperature=0`. Pass the fused chunks (from Step 4, ~8–10 plus any cross-reference/definition chunks) as labeled context blocks (`[doc_id | section_id | date] text`) so the model can copy exact identifiers instead of inventing them.
4. If retrieval returned no chunks at all, skip the LLM call entirely and return `insufficient_context: true` with a fixed "not found in the provided documents" message.
5. **Multi-document synthesis guard**: if the retrieved context spans more than one `doc_id` (flagged in Step 4.8) and the question isn't scoped to a single document, instruct the model to answer per-document rather than producing one merged claim — e.g. "Contract A: 30 days. Contract B: 15 days." — unless every contributing chunk states the same thing.

---

## Step 6 — Hallucination Guardrail: Citation Verification (`app/verification.py`)

1. **Quote grounding**: for each citation the model returns, look up the chunk it claims (`doc_id` + `section_id`) in the actual indexed chunk text, and fuzzy-match the `quote` against it (e.g. `rapidfuzz.fuzz.partial_ratio`, threshold ~85) — exact substring match is often too strict against minor whitespace/OCR differences, so fuzzy is more robust than `in`.
2. If a citation fails quote grounding: either drop it from the response and flag `unverified_citations`, or trigger one regeneration attempt with a stricter reminder — pick whichever you have time for; documenting the decision in the report is what matters.
3. **Claim entailment check** (separate from quote grounding — a quote can be accurate while the surrounding claim about it is not): for each answer sentence with a passing citation, make one additional lightweight LLM call asking only "does this sentence follow from this quoted text? yes/no", using a different/independent prompt so it isn't just re-asking the same question that produced the error. If "no", flag the sentence rather than returning it as-is.
4. **Exact numeric guard**: extract all numbers, dates, and percentages from both the generated answer and the specific chunk(s) it cites (a simple regex for `\d+` sequences, day/month/year patterns, and `%`/`$` amounts is enough). Any numeric value in the answer that doesn't literally appear in the cited chunk is a **hard fail** — no fuzzy tolerance here, since numeric drift (30 days vs. 15 days) is the highest-stakes hallucination category in a legal context and fuzzy matching is exactly what would let it slip through.
5. All of the above are deterministic Python checks, not further LLM calls (aside from the one independent entailment call in step 3) — the point is that verification doesn't inherit the same failure mode it's checking for.

---

## Step 7 — API Layer (`app/main.py`)

```python
from fastapi import FastAPI
app = FastAPI(title="Legal Contract RAG")

@app.post("/ingest")
def ingest(): ...

@app.post("/query")
def query(body: QueryRequest) -> QueryResponse: ...

@app.post("/evaluate")
def evaluate(): ...

@app.get("/health")
def health(): return {"status": "ok"}
```
Run locally: `uvicorn app.main:app --reload --port 8000`. Visit `/docs` for the free Swagger UI — useful both for your own testing and as a demo for reviewers.

---

## Step 8 — Faithfulness Evaluation (`app/evaluate.py`)

1. Hand-write 10–20 Q&A pairs into `data/eval_set/questions.json` by actually reading the 5 contracts — mix easy lookups ("What is the term of the Joint Venture Agreement?") with harder synthesis questions and at least 2–3 questions with **no answer in the documents** (to test the refusal path).
   ```json
   [{"question": "...", "expected_doc_id": "...", "expected_section": "..."}]
   ```
2. For each item, call the live `/query` pipeline, then score with RAGAS's `faithfulness` metric (decomposes the answer into claims, checks each against the retrieved context via an LLM judge) plus `context_precision`/`context_recall`.
3. Also compute your own simple **citation accuracy**: % of returned citations that passed Step 6's verification, and % of no-answer questions correctly refused.
4. `POST /evaluate` runs the full set and returns aggregate + per-question scores as JSON — keep it fast enough to run in under a minute so it's actually usable during development, not just at the end.

---

## Step 9 — Docker Compose

`Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:
```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]

  app:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on: [qdrant]
    volumes: ["./data:/app/data"]

volumes:
  qdrant_data:
```
Test with a clean run: `docker compose down -v && docker compose up --build`, then `curl -X POST localhost:8000/ingest` and a sample `/query` — this is exactly what a reviewer will do first.

---

## Step 10 — README

Must cover, in order: prerequisites, `.env` setup, `docker compose up --build`, how to trigger ingestion, an example `curl` for `/query` with a real question and expected shape of response, how to run `/evaluate`, and a one-line note on what each bonus task maps to in the code (helps grading).

---

## Step 11 — Report (max 5 pages)

Structure: Scenario recap (1 paragraph) → Architecture diagram + component choices with reasoning (reuse `architecture.md`, condensed) → Assumptions made (e.g. footer-stripping regex, chunk size, similarity threshold value) → Bonus task write-ups (one short section each, referencing where in the code they live) → Evaluation results table (faithfulness/precision/recall/citation-accuracy numbers from your actual eval run) → Limitations & what you'd do with more time (e.g. cross-encoder reranking, human-in-the-loop review for flagged/unverified citations, a larger hand-labeled eval set, multi-language contract support).

---

## Step 12 — Package Deliverables

```bash
zip -r code.zip app data docker-compose.yml Dockerfile requirements.txt README.md .env.example -x "venv/*" "*.pyc" "__pycache__/*"
```
Do **not** zip `venv/`, `qdrant_data/` volume contents, or `.env` with the real key — include `.env.example` instead. Send `report.pdf`, `code.zip`, and the README (or bundle README inside the zip) to `hr@dxdy.tech`.

---

## Suggested Order of Work (time-boxing)

1. Steps 0–3 (parsing → indexing) — get this rock solid first; everything downstream depends on chunk quality.
2. Step 4 (hybrid retrieval) — verify with a few manual queries before touching generation.
3. Steps 5–6 (generation + verification) — the core "answer the question" loop.
4. Step 7 (API) — wrap what already works; don't build the API before the pipeline works standalone.
5. Steps 9 (Docker) — do this early enough to catch environment issues, not last.
6. Step 8 (evaluation) — needs a working pipeline to evaluate, so naturally comes after, but budget real time for it — it's graded and easy to shortchange.
7. Steps 10–12 (README, report, packaging) — last, but don't compress this into 15 minutes; a sloppy README undercuts a good pipeline.
