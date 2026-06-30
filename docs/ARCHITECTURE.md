# Axiom Architecture

## Mission

Build an air-gapped evidence assistant that can ingest local documents, images, and audio, then answer questions with citations that can be traced back to exact local source locations.

The design priority order is:

1. Evidence integrity.
2. Offline operation.
3. Citation traceability.
4. Operator speed.
5. Model quality.

Model quality matters, but it is not allowed to outrank evidence control.

## Improved Blueprint

The original prompt has the right ambition, but a production system should separate the trusted evidence plane from the probabilistic model plane.

```text
                 OFFLINE INTAKE DIRECTORY
                          |
                          v
             +--------------------------+
             | Extraction Adapters      |
             | PDF, DOCX, OCR, Audio    |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | Evidence Ledger          |
             | SQLite/Postgres          |
             | files, chunks, hashes,   |
             | timestamps, pages        |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | Retrieval Plane          |
             | BM25, embeddings, image  |
             | vectors, rerankers       |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | Context Gate             |
             | top-k chunks, allow-list |
             | citation IDs             |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | Local Answer Runtime     |
             | extractive fallback,     |
             | Ollama/vLLM adapter      |
             +------------+-------------+
                          |
                          v
             +--------------------------+
             | Citation Validator       |
             | reject unknown sources   |
             +--------------------------+

             +--------------------------+
             | Operator Plane           |
             | windows, tabs, folders,  |
             | open, run, audit         |
             +--------------------------+

             +--------------------------+
             | Vision Plane             |
             | screenshot, OCR, VLM,    |
             | image sidecar indexing   |
             +--------------------------+
```

## Key Hardening Changes

### 1. Treat the database as the evidence ledger

The vector index is not the source of truth. It can be rebuilt. The ledger stores:

- Source file path.
- SHA-256 hash.
- Extracted text chunks.
- Chunk kind: parent or child.
- Modality.
- Page or timestamp.
- Vector payload reference.
- Query context allow-list.

### 2. Use parent-child chunking by default

Child chunks improve recall. Parent chunks improve answer quality. The retriever ranks child chunks but passes the larger parent span into the answer context.

### 3. Never trust model citations

The model does not get to invent source IDs. A query creates an allow-list of chunk IDs. After generation, every citation token is checked against that allow-list. Unknown citations are marked as unverified.

### 4. Make heavy AI optional and swappable

In air-gapped environments, models and CUDA builds are deployment problems. The first working core should not collapse because a GPU package is missing. Axiom starts with deterministic local retrieval and then upgrades adapters.

### 5. Separate workstation action from model reasoning

The system can inspect and act on the local workstation, but those actions are not hidden behind free-form model output. They go through explicit tools and are written to `operator_audit`. This matters for air-gapped and sensitive environments because "the AI did it" is not an acceptable audit trail.

### 6. Treat screenshots as first-class evidence

Screenshots are captured, hashed, OCR-scanned, optionally described by a local VLM, and then indexed as image evidence. OCR text and VLM captions are stored as sidecars next to the source image so the same ingestion engine can process them without special trust rules.

## Storage Schema

The starter uses SQLite. For a multi-operator deployment, the same schema can move to local PostgreSQL.

```sql
document_registry(file_id, file_name, file_type, file_path, sha256, size_bytes, status, ingested_at)
content_chunks(chunk_id, parent_id, file_id, chunk_kind, text_content, modality, page_number, start_timestamp, end_timestamp, char_start, char_end, token_count)
chunk_vectors(chunk_id, vector_json, embedding_model)
cross_modal_links(link_id, source_chunk_id, target_chunk_id, confidence_score, link_type)
query_audit(query_id, query_text, created_at)
query_context(query_id, rank, chunk_id, score)
operator_audit(audit_id, action_type, target, parameters_json, executed, success, return_code, stdout_preview, stderr_preview, created_at)
```

## Ingestion Rules

- Every source file is hashed before extraction.
- Extraction failures are recorded instead of silently ignored.
- Sidecar files are supported for offline adapters:
  - `image.png.ocr.txt`
  - `image.png.caption.txt`
  - `audio.wav.transcript.json`
  - `audio.wav.transcript.txt`
- Production PDF, OCR, and audio extraction should be pinned to approved local binaries and model files.

## Retrieval Rules

1. Run dense local retrieval.
2. Run sparse lexical retrieval.
3. Merge with Reciprocal Rank Fusion.
4. Optionally rerank top candidates with a local cross-encoder.
5. Store the final query context as the citation allow-list.
6. Generate an answer using only the allowed context.
7. Validate citations after generation.

## Operator Rules

1. Read-only inspection tools can run directly.
2. Opening a file or folder requires an explicit execute flag.
3. Command execution defaults to dry-run.
4. Non-read-only or shell execution requires an unsafe override.
5. Every operator action is recorded in SQLite.
6. Browser tab URLs are available only when the browser exposes a local DevTools endpoint; otherwise the system reports visible window titles.

## Vision Rules

1. Screenshot capture never leaves the machine.
2. OCR prefers PaddleOCR when available and falls back to Tesseract.
3. Local VLM understanding uses Ollama or another offline runtime only.
4. OCR text and VLM summaries are stored as sidecars and indexed with the original image file.
5. If OCR or the VLM is unavailable, the system reports the missing adapter instead of inventing image content.

## Production Phases

### Phase 1: Evidence Core

- Finalize ledger schema.
- Implement ingestion and hashing.
- Add parent-child chunking.
- Add deterministic retrieval and citation validation.
- Build CLI and test harness.

### Phase 2: Local AI Adapters

- Add approved embedding model runtime.
- Add FAISS or Chroma for larger indexes.
- Add Whisper.cpp transcript ingestion.
- Add OCR and VLM caption workers.
- Add Ollama or vLLM answer runtime.

### Phase 3: Operator Interface

- FastAPI backend.
- React dashboard.
- PDF page viewer.
- Audio timestamp jump.
- Source inspection panel.
- Admin page for model and index health.
- Workstation panel for visible windows, browser tabs, local folders, and guarded task execution.
- Screenshot capture panel with OCR text, local VLM analysis, and one-click ingest.

### Phase 4: Evaluation and Accreditation

- Offline benchmark corpus.
- Retrieval recall checks.
- Faithfulness checks.
- Citation integrity tests.
- Tamper detection using file hashes.
- Resource profiling on target hardware.

## What Not To Do

- Do not make network calls during runtime.
- Do not let the LLM cite arbitrary strings.
- Do not use the vector database as the only metadata store.
- Do not process heavy media synchronously in the web request.
- Do not hide extraction failures.
- Do not auto-download models in production.
