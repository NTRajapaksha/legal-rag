# Legal Contract RAG — Recommended Architecture

**Scope:** Required task + all 4 bonus tasks (hybrid retrieval, hallucination prevention, faithfulness evaluation, Docker Compose), exposed as an API.

---

## 1. High-Level Architecture

```
                       ┌─────────────────────────────────────────┐
                       │              FastAPI Service              │
                       │                                           │
   PDFs ──ingest.py──▶│  /ingest   → parse, chunk, embed, index   │
                       │  /query    → hybrid retrieve → rerank →   │
                       │              generate → verify → cite     │
                       │  /evaluate → faithfulness scoring         │
                       └───────────────┬───────────────────────────┘
                                        │
                 ┌──────────────────────┼───────────────────────┐
                 ▼                      ▼                       ▼
         ┌───────────────┐     ┌───────────────┐       ┌────────────────┐
         │  Qdrant         │     │  BM25 index    │       │ Vercel AI      │
         │  (dense vectors)│     │  (sparse, on   │       │ Gateway        │
         │                 │     │  disk / SQLite)│       │ (LLM + embed)  │
         └───────────────┘     └───────────────┘       └────────────────┘
```

Everything runs as two containers: `app` (FastAPI) and `qdrant` (vector DB), wired together with Docker Compose. BM25 is kept in-process (rebuilt from a lightweight on-disk index) rather than as a separate service — it's cheap and avoids adding a third moving part for a prototype of this size.

---

## 2. Component Choices & Reasoning

### 2.1 Document Parsing & Chunking

| Choice | Reasoning |
|---|---|
| `pdfplumber` for text extraction (not `pypdf`) | The tool needs to work on **any** legal document the firm feeds it, not just the sample set — different contracts use completely different numbering conventions (`Section 1.1`, `ARTICLE I`, lettered `(a)`, unnumbered bolded topics, etc.). `pypdf` only returns a flat text stream and throws away font/weight/position, which is exactly the information needed to detect structure generically. `pdfplumber` preserves per-line font size, bold/weight, and position, which is what makes heading detection convention-independent rather than tied to one filing's wording. Full OCR is not the primary extraction path since most contracts of this kind are digitally text-based, but a `pytesseract` fallback is triggered automatically for any page that yields no usable text layer (see §2.6.5) — so scanned pages degrade gracefully rather than failing silently. |
| **Layered heading detection — layout signal, text-pattern signal, then semantic fallback** | Legal documents use inconsistent, unpredictable conventions (`Section 1.1`, `ARTICLE I`, lettered `(a)`, unnumbered bolded topics, or no numbering at all), so detection is built as three tiers, each catching what the previous one can't: **(1)** detect headings by visual/structural signal (short line, bold or above-average font size, all-caps, standing alone) — a property that holds across almost all legal drafting styles regardless of numbering scheme; **(2)** independently scan every line's *text* against a heading-pattern library regardless of formatting, to catch plainly-typed documents that number their clauses (`Section 1.1`, `ARTICLE I`) with no visual distinction at all; **(3)** only fall back to semantic chunking when *neither* signal finds anything, meaning the document truly has no structure rather than merely no formatting. Regex is used to *classify* a heading already found by either signal, never to search for one — so the pattern library can keep growing without ever changing what triggers a split. |
| **Semantic chunking as the fallback tier, not blind splitting** | For documents with no discoverable structure at all (plain-prose agreements, weak OCR, no numbering or formatting cues), blind fixed-size splitting can cut a clause exactly in half even where a real topic shift exists. Semantic chunking embeds consecutive sentences and splits where similarity between them drops sharply, treating that drop as a topic boundary — catching natural clause boundaries (e.g. the right to terminate vs. the effect of termination) that carry no visual or textual heading marker at all. Citation precision still degrades to page-level (`section_id = None`) since there's no heading to name, but chunk boundaries are meaningfully better than a fixed-size cut. The extra embedding cost only applies to this minority case, not to well-structured contracts. |
| **Never split inside an ALL-CAPS block** | This is a content property, not a document-specific pattern — warranty disclaimers, liability limitations, and jury-trial waivers are conventionally drafted in all-caps across essentially all US commercial contracts. Protecting these blocks from mid-clause splitting matters more than for ordinary prose, because a partial disclaimer citation in a legal-risk tool is actively misleading, not just incomplete. |
| Store `{doc_id, section_id, section_title, parent_section, page_number, text}` per chunk | This metadata is what makes "cite the specific section" (a required task item) possible and auditable, rather than citing "chunk 17". `parent_section` is included because sub-clauses (e.g. `1.2`) are close to meaningless in a citation without the heading they sit under — and because nesting depth itself varies by document, this is inferred from indentation/numbering depth at chunk time rather than assumed to follow one fixed hierarchy. |

### 2.2 Embeddings & Vector Store

| Choice | Reasoning |
|---|---|
| Embedding model via Vercel AI Gateway (e.g. an OpenAI/Voyage embedding model available through the gateway) | Reuses the provided key, no extra account needed. Voyage's `voyage-law-2` (if available through the gateway) is purpose-built for legal text and is worth trying first; fallback to a general-purpose embedding model otherwise. |
| **Qdrant** as the vector DB | Open source, has a first-class Docker image, supports metadata filtering (filter by `doc_id` before searching — useful when a user asks "in the NDA, what's the termination clause?"), and supports **hybrid search natively** (dense + sparse in one query), which directly serves bonus task #1. Alternatives considered: Chroma (simpler but weaker filtering/hybrid support), pgvector (more setup overhead for a prototype), FAISS (no persistence/service layer, harder to containerize cleanly). |
| **Defined-terms index built at ingest time** | Legal documents define a capitalized term once ("Brand", "Permitted Activity") and reuse it throughout — if the definition chunk isn't retrieved alongside a question that uses the term, the model risks substituting its own general-language assumption of what the term means. At ingest, scan each document for the pattern `"Capitalized Term"` near `means`/`shall mean`/`refers to`, and store a `{doc_id, term, definition_chunk_id}` lookup table. At query time, any capitalized term detected in the question or in retrieved chunks triggers an automatic fetch of its definition chunk, even if that chunk didn't score high enough to be retrieved normally. |
| **Document metadata includes version/date** | If a firm uploads both an original contract and a later amendment, chunks from each need to be distinguishable so conflicting clauses can be surfaced rather than silently merged or arbitrarily chosen between. Each chunk's payload includes the source document's filing/effective date where extractable, so generation can flag "Section 4 states X; the [date] amendment states Y" instead of picking one silently. |

### 2.3 Bonus Task 1 — Improving retrieval of legal terminology (Hybrid Search)

Pure dense vector search under-performs on exact legal terms — defined terms ("Indemnified Party"), statute references, monetary figures, and boilerplate phrasing are often near-duplicates in embedding space but must match *exactly* for a lawyer's purposes.

**Approach: Hybrid retrieval = dense vector search + BM25, combined with Reciprocal Rank Fusion (RRF).**

- Dense search captures semantic meaning ("what happens if the contract is broken" → "termination for breach").
- BM25 (sparse, term-frequency based) captures exact legal terminology and defined terms that embeddings can blur ("Section 9.3", "liquidated damages", "force majeure"), and helps guard against a specific embedding weakness: dense vectors often place "shall indemnify" and "shall not indemnify" close together in vector space, while exact-term matching preserves the negation.
- RRF merges the two ranked lists without needing to tune a weighting coefficient — a good default for a time-boxed prototype, and easy to justify in the report.
- **Cross-reference resolution**: contracts routinely reference other sections ("subject to Section 8", "as defined in Section 3"). After initial retrieval, scan the retrieved chunk text for these references and auto-fetch the referenced chunk by ID directly from the index, even if it didn't score high enough to be retrieved by search alone. Without this, the model either drops the reference (incomplete answer) or guesses at its content (hallucination).
- Retrieve a wider top-k (~8–10 chunks, not just top-5) before generation — context windows have room for it, and it reduces the chance that a correct answer ranked 6th–10th gets silently excluded, which would otherwise push the model toward guessing rather than truthfully reporting "not found."
- Optional (stretch, not required): a cross-encoder reranker (e.g. `bge-reranker` or a small model call) on the top ~20 fused results to reorder before they go to the LLM.

### 2.4 Bonus Task 2 — Hallucination Prevention

Layered defenses, cheap to implement, each independently useful:

1. **Strict, closed-book system prompt**: the LLM is instructed to answer *only* from the retrieved context, to say "I cannot find this in the provided documents" when the answer isn't present, and never to rely on outside legal knowledge.
2. **Structured output with inline citations**: force the LLM to respond in JSON (`{"answer": ..., "citations": [{"doc_id", "section_id", "quote"}]}`) via the gateway's structured-output / tool-calling mode. Free-text answers are much easier to hallucinate in; forcing a citation object per claim makes fabrication visible.
3. **Programmatic citation verification (grounding check)**: after generation, fuzzy-match each `quote` field against the chunk text it claims to cite. If a quote doesn't verify, the answer is flagged or regenerated — a deterministic, non-LLM safety net rather than trusting the model's self-report.
4. **Claim-level entailment check, separate from quote verification**: a correctly-quoted sentence can still be summarized incorrectly (e.g. quoting a termination-for-breach clause accurately but describing it as an unconditional right to terminate). After citation verification passes, run one additional lightweight check — a separate prompt asking only "does this specific answer sentence follow from this specific quoted text, yes/no" — since quote-existence and claim-accuracy are different failure modes and neither test catches the other.
5. **Exact-match numeric guard**: dates, day-counts, percentages, and monetary figures are the highest-stakes hallucination category, since small numeric drift (30 days vs. 15 days) is exactly the kind of error fuzzy text matching can miss. Extract all numbers/dates/percentages from the generated answer and from the cited chunk separately, and hard-fail (no fuzzy tolerance) if any numeric claim in the answer doesn't literally appear in the cited chunk.
6. **Temperature = 0** and a low `top_p` for generation — legal Q&A should be deterministic, not creative.
7. **Refusal on empty/irrelevant retrieval**: if the top retrieval score is below a similarity threshold, skip generation entirely and return "not found in the provided documents" rather than letting the LLM improvise.
8. **No blended cross-document claims by default**: if a question spans multiple contracts, each individual citation might verify correctly while the *synthesis* is still wrong (e.g. "all contracts require 30 days notice" when only some do). Answer per-document ("Contract A: X. Contract B: Y.") unless every contributing chunk actually supports the same generalized claim.

### 2.5 Bonus Task 3 — Faithfulness Evaluation

**Approach: RAGAS (or a lightweight custom equivalent) faithfulness metric, run as a batch evaluation script/endpoint, not a per-request cost.**

- **Faithfulness** = proportion of claims in the generated answer that are logically inferable from the retrieved context (decompose the answer into atomic claims → LLM-judge each claim against context → score).
- **Context precision / recall** as secondary metrics — do the retrieved chunks actually contain what's needed to answer, and is the top-ranked chunk the relevant one.
- Curate a small gold-standard test set (10–20 Q&A pairs across the 5 contracts, written by hand while reading them) — this is what makes the evaluation "methodical" rather than anecdotal, and is realistic to produce in the take-home's time budget.
- Expose as `POST /evaluate` which runs the test set against the live pipeline and returns aggregate scores + per-question breakdown — useful both for grading and for regression-checking future prompt changes.

### 2.6 Bonus Task 4 — Docker Compose

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]

  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - AI_GATEWAY_KEY=${AI_GATEWAY_KEY}
      - QDRANT_URL=http://qdrant:6333
    depends_on: [qdrant]
    volumes: ["./data:/app/data"]   # source PDFs + BM25 index persisted
volumes:
  qdrant_data:
```

`docker compose up --build` then `POST /ingest` (or an ingest-on-startup flag) is the entire setup story for the README.

### 2.6.5 PDF Extraction Edge Cases

| Failure | Mitigation |
|---|---|
| Multi-column layouts (fee schedules, side-by-side signature blocks) scrambled by naive left-to-right extraction | Use `pdfplumber`'s table detection (`extract_tables()`) and cluster text by x-coordinate before reading order, rather than assuming single-column flow. |
| Scanned pages with no real text layer return empty/near-empty extracted text, silently producing a gap in the index | Detect near-empty extracted text per page (below a character threshold) and either OCR it (`pytesseract`) or explicitly mark it `"unextracted"` in metadata, so missing content is visible rather than silently absent. |

### 2.7 API Surface (FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Parse PDFs in `/data`, chunk, embed, upsert to Qdrant, build/update BM25 index. Idempotent (re-running re-indexes cleanly). |
| `POST /query` | Body: `{"question": str, "doc_filter": Optional[str]}`. Runs hybrid retrieval → (optional rerank) → generation with citation verification → returns `{answer, citations, confidence}`. |
| `POST /evaluate` | Runs the gold-standard test set through `/query` internally and returns faithfulness/precision/recall scores. |
| `GET /health` | Trivial liveness check, useful for Compose/readiness probes. |

FastAPI is chosen over Flask/plain CLI because: it gives free request validation (Pydantic), auto-generated OpenAPI docs (nice deliverable polish for reviewers to click through), and async support for the LLM/embedding I/O calls.

---

## 3. Why This Is the "Best" Approach for This Assessment

- **Generalizes beyond the sample documents, by design.** Nothing in the pipeline assumes a specific contract's wording or numbering convention. Heading detection is layout-driven (font/weight/position) rather than regex-matched to the provided PDFs, with regex used only to label structure that's already found — so the tool doesn't quietly degrade the first time it sees a differently-formatted contract. This matters specifically because the deliverable is described as a firm-wide tool for a *collection* of contracts, not a one-off script for five known files.
- **Directly maps to the grading rubric.** Every required-task line item (parse → embed → store, retrieval + LLM generation, section-level citation, an interface) and all four bonus items have a named, justified component — nothing is bolted on generically.
- **Proportionate complexity.** Five PDFs don't need a distributed vector DB or a fine-tuned reranker; they do need clause-aware chunking and hybrid search, because that's precisely where naive RAG fails on legal text (exact terminology, section numbers). The design avoids both under-building (pure vector search, no citation verification) and over-building (multi-agent pipelines, fine-tuning) — matching the "prototype" framing in the brief.
- **Auditable over clever.** Programmatic citation verification and a hand-built gold test set are deliberately simple and inspectable — a reviewer can see *why* an answer is trusted, which matters more in a legal-risk context than a marginally higher benchmark score from a fancier reranker.
- **Deployable in one command.** Docker Compose with two services keeps the "how to run it" story trivial for the README, which is itself part of the deliverable quality.

---

## 4. Known Failure Modes & Mitigations (Summary Table)

| Failure Mode | Why It Happens | Mitigation |
|---|---|---|
| Unresolved cross-references ("subject to Section 8") | The referenced section isn't in the top-k retrieved chunks | Auto-detect section references in retrieved text and fetch them directly by ID |
| Defined term used far from its definition | Definition chunk not retrieved alongside usage | Defined-terms index built at ingest; auto-inject definitions when a term appears |
| Contradicting amendment vs. original clause | Two valid chunks disagree; no versioning signal | Store document date in metadata; surface conflicts explicitly instead of picking one |
| Negation blindness ("shall" vs. "shall not") | Dense embeddings place negated/non-negated clauses close together | BM25 exact-term matching + post-generation polarity check |
| Multi-document synthesis overgeneralizes | Each citation verifies individually but the merged claim doesn't hold | Answer per-document by default; only merge when all sources agree |
| Quote is accurate but claim is misstated | Citation verification checks quote existence, not interpretation | Separate entailment check: does the answer sentence follow from the quote |
| Numeric drift (30 days vs. 15 days) | Fuzzy text matching tolerates small numeric differences | Exact-match extraction and hard-fail on any unmatched number/date/amount |
| Correct answer ranked just outside top-k | Retrieval cutoff excludes a relevant chunk | Widen retrieved context (~8–10 chunks) before generation |
| Multi-column text scrambled | Naive extraction reads across columns | Table-aware / x-coordinate-clustered extraction |
| Scanned page silently indexed as empty | No OCR fallback, no detection of empty pages | Detect near-empty pages; OCR or flag as unextracted |

---

## 5. Suggested Repo Layout

```
.
├── app/
│   ├── main.py              # FastAPI app, endpoints
│   ├── ingest.py             # parsing + chunking + indexing
│   ├── defined_terms.py      # defined-terms extraction & lookup index
│   ├── retrieval.py          # hybrid search + RRF + cross-reference resolution
│   ├── generation.py         # prompt templates, structured LLM call
│   ├── verification.py       # citation grounding, entailment, and numeric checks
│   └── evaluate.py           # RAGAS-style faithfulness eval
├── data/
│   ├── contracts/            # source PDFs
│   └── eval_set.json         # gold Q&A pairs
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
└── report.pdf                # written report deliverable
```
