# SIH25231 Axiom Plan

## Problem Statement

Organization: National Technical Research Organisation  
PS ID: SIH25231  
Category: Software  
Theme: Smart Automation  

Design and build a multimodal Retrieval-Augmented Generation system leveraging a Large Language Model for offline mode that can ingest, index, and query DOC, PDF, images, and voice recordings within a unified semantic retrieval framework.

## Stronger Product Interpretation

Most teams will build a basic offline RAG search box. Axiom should present as an offline intelligence operator:

1. Ingest evidence from local folders.
2. Extract text from documents, images, and audio.
3. Build local semantic and lexical indexes.
4. Answer with strict citations.
5. Let an analyst inspect cited files immediately.
6. Let the system see the local workstation state: open windows, exposed browser tabs, and project folders.
7. Let the system execute controlled local tasks with an audit trail.

That turns the project from "chat with files" into "offline mission workspace."

## Current Implemented Capabilities

- Local SQLite evidence ledger.
- Source hashing.
- Text ingestion.
- Sidecar support for OCR captions and audio transcripts.
- Parent-child chunking.
- Hashed local embeddings with no network dependency.
- SQLite FTS lexical search when available.
- Hybrid retrieval using dense plus sparse fusion.
- Strict citation validation.
- Citation inspection.
- Visible desktop window inventory.
- Browser tab inventory through local DevTools when available.
- Screenshot capture.
- OCR over screenshots and images.
- Offline VLM image understanding through a local Ollama model when available.
- Folder scan and file search.
- Local file/folder open action.
- Guarded command execution.
- Operator audit logging.
- Optional FastAPI endpoints for a future UI.

## Next Build Milestones

### Milestone 1: Multimodal Extractors

- Add pinned PyMuPDF PDF extraction.
- Add python-docx and LibreOffice fallback for DOC/DOCX conversion.
- Add PaddleOCR or Tesseract OCR adapter. Initial adapter layer is implemented.
- Add Whisper.cpp transcript worker.
- Add CLIP/SigLIP image-vector adapter.

### Milestone 2: Production Retrieval

- Replace hashed vectors with a local embedding model such as bge or E5.
- Add FAISS index persistence.
- Add BM25 ranking for larger corpora.
- Add local reranker.
- Add query-time citation allow-list cache.

### Milestone 3: Offline Local LLM

- Add Ollama/vLLM model health checks.
- Add context packing.
- Add answer JSON schema.
- Keep post-generation citation validation mandatory.

### Milestone 4: Operator Dashboard

- React dashboard with:
  - ingest queue,
  - chat/search pane,
  - evidence graph,
  - PDF/image/audio preview,
  - desktop windows panel,
  - folder/task panel,
  - audit log panel.

### Milestone 5: Evaluation

- Retrieval recall test set.
- Citation hallucination tests.
- Offline startup test.
- Large-folder ingestion benchmark.
- CPU-only fallback benchmark.
- GPU acceleration benchmark.

## Demo Story

1. Start fully offline.
2. Ingest a folder containing a PDF report, a transcript, an image OCR sidecar, and a voice transcript sidecar.
3. Ask: "Which evidence discusses international development in 2024 and references screenshots?"
4. Show cited answer.
5. Inspect the citation.
6. Open the source file locally.
7. Capture the current screen, run OCR, and ingest the screenshot as evidence.
8. Ask a follow-up question that retrieves screenshot-derived OCR/caption evidence.
9. Show visible workstation windows.
10. Run a dry-run command, then an approved command.
11. Show audit records in SQLite.

The judge takeaway should be: Axiom is not a chatbot bolted onto documents. It is an offline evidence and automation workstation with traceable controls.
