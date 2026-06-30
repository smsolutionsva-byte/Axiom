# Axiom

Axiom is an offline-first intelligence workspace for air-gapped environments. It ingests local evidence, builds a local retrieval index, and answers questions with citations that must map back to retrieved source chunks.

This starter is intentionally conservative:

- No cloud APIs.
- No network dependency at runtime.
- SQLite is the source of truth for files, chunks, links, vectors, query audit, and citation validation.
- A deterministic local retriever works immediately with the Python standard library.
- Heavy capabilities such as FAISS, Ollama, Whisper.cpp, PaddleOCR, Docling, or CLIP are adapter points, not fake requirements.

## Quick Start

Launch the local app:

```powershell
python -m axiom app --db .\data\axiom.sqlite
```

Then open:

```text
http://127.0.0.1:8765/
```

The app includes setup checks for OCR, screenshot capture, Tesseract, PaddleOCR, Ollama, and the local vision model.

```powershell
python -m axiom init --db .\data\axiom.sqlite
python -m axiom ingest .\samples --db .\data\axiom.sqlite
python -m axiom ask "international development targets for 2024 screenshots" --db .\data\axiom.sqlite
```

For the SIH25231 demo corpus:

```powershell
python -m axiom ingest .\samples\multimodal --db .\data\axiom.sqlite
python -m axiom mission --db .\data\axiom.sqlite
python -m axiom ask "Which 2024 international development evidence is supported by DOC, PDF, screenshot OCR, and voice transcript?" --db .\data\axiom.sqlite
```

Inspect cited evidence:

```powershell
python -m axiom sources list --db .\data\axiom.sqlite
python -m axiom sources inspect 2b72fb9f --db .\data\axiom.sqlite
```

Use local operator tools:

```powershell
python -m axiom operator windows --db .\data\axiom.sqlite
python -m axiom operator tabs --db .\data\axiom.sqlite
python -m axiom operator scan . --max-depth 2 --db .\data\axiom.sqlite
python -m axiom operator find . screenshot --content --db .\data\axiom.sqlite
python -m axiom operator open ".\samples\annexure_4b.txt" --execute --db .\data\axiom.sqlite
python -m axiom operator run --execute --db .\data\axiom.sqlite -- python --version
python -m axiom operator audit --db .\data\axiom.sqlite
```

Capture and understand screenshots:

```powershell
python -m axiom vision screenshot --analyze --ingest --db .\data\axiom.sqlite
python -m axiom vision screenshot --active-window --analyze --db .\data\axiom.sqlite
python -m axiom vision analyze ".\artifacts\screenshots\screenshot-YYYYMMDD-HHMMSS.png" --ingest --db .\data\axiom.sqlite
python -m axiom vision ocr ".\artifacts\screenshots\screenshot-YYYYMMDD-HHMMSS.png" --db .\data\axiom.sqlite
```

Run evidence analytics:

```powershell
python -m axiom analytics --query "international development 2024 screenshot" --db .\data\axiom.sqlite
```

Open the SIH25231 mission brief:

```powershell
python -m axiom mission --db .\data\axiom.sqlite
```

The local app also supports `/mission` or "show SIH readiness" in chat. This produces a judge-facing battlecard with modality coverage, offline adapter readiness, cross-modal links, audit counts, differentiators, gaps, and a demo script.

Generate images offline:

```powershell
python -m axiom imagegen status --db .\data\axiom.sqlite
python -m axiom imagegen generate "offline intelligence command center dashboard" --backend auto --db .\data\axiom.sqlite
```

Supported local image backends:

- AUTOMATIC1111/Forge running locally with API enabled at `http://127.0.0.1:7860`.
- Diffusers with a local model folder set through `AXIOM_DIFFUSION_MODEL` or `--model-path`.

Axiom does not download image models silently. Use the Setup page or install models through your approved offline model staging process.

Investigate a person, handle, email, or identifier:

```powershell
python -m axiom investigate "Jordan Vale" --root .\samples --db .\data\axiom.sqlite
python -m axiom report "Jordan Vale" --root .\samples --db .\data\axiom.sqlite
```

Investigation mode builds a local dossier with cited evidence, aliases, entities, timeline entries, file leads, risk flags, next actions, and a passive hallucination guard. It treats folder matches as leads until those files are ingested and citation-backed.

Report mode exports Markdown, HTML, and raw JSON case reports under `exports/reports`.

Run tests:

```powershell
python -m unittest discover -s tests
```

## What Works Now

- Ingests `.txt`, `.md`, `.csv`, `.json`, and `.log`.
- Uses optional local extractors for `.pdf`, `.doc`, `.docx`, image OCR, and audio transcripts when those tools are installed or sidecar transcripts are present.
- Includes dependency-free DOCX XML fallback, PyMuPDF/pypdf PDF fallback, legacy DOC hooks for local antiword/catdoc/LibreOffice adapters, and a Whisper.cpp hook for voice recordings.
- Creates parent and child chunks for precise retrieval with larger answer context.
- Stores source location metadata such as page number or audio timestamp when available.
- Combines local lexical search and hashed semantic retrieval with Reciprocal Rank Fusion.
- Produces extractive answers when no local LLM is configured.
- Validates every citation token against the exact query context.
- Lists visible desktop windows on Windows.
- Lists browser tabs when Chrome or Edge exposes local DevTools on `127.0.0.1`.
- Scans folders, searches files, opens local artifacts, and runs guarded local commands with audit logging.
- Captures screenshots, runs OCR, optionally asks a local offline vision model to understand the image, and ingests the OCR/caption output.
- Shows an Analytics tab with evidence graph, timeline, signal scoring, prediction brief, gaps, and next actions.
- Shows an SIH Mission Brief widget that maps the corpus to NTRO SIH25231 requirements and highlights win differentiators.
- Shows an Image Lab tab for offline image generation through local Stable Diffusion backends.
- Shows an Investigation tab for detective-style offline subject search and hallucination-resistant dossiers.
- Exports printable case reports with citations, evidence tables, timelines, leads, gaps, and guardrail rules.

## Optional Local Model Runtime

The starter does not require a model server. For production, add a local runtime:

- Text LLM: Ollama or vLLM with an approved open-weight instruct model.
- Embeddings: bge, E5, Nomic, or another approved local embedding model.
- Audio: Whisper.cpp with pinned local model files.
- Vision/OCR: PaddleOCR, Tesseract, CLIP/SigLIP, and an approved local VLM.

In an air-gapped deployment, stage wheels, model files, checksums, and container images through an approved offline artifact process. Do not let runtime code download anything.

## OCR And Screenshot Understanding

Axiom's best practical offline stack is:

- Screenshot capture: `mss` first, Pillow `ImageGrab` second, Windows PowerShell capture fallback.
- OCR: PaddleOCR first for production-quality OCR, Tesseract fallback for lightweight local installs.
- Vision understanding: Ollama local VLM through `AXIOM_VISION_MODEL`, defaulting to `llama3.2-vision`.
- Evidence ingestion: OCR text is saved as `.ocr.txt`, VLM summary as `.caption.txt`, and both are indexed against the image file.

For a serious offline demo, install or stage these locally:

```powershell
pip install -e ".[vision]"
# Optional production OCR if the environment supports it:
pip install paddleocr
# Then run a local vision model through Ollama:
set AXIOM_VISION_MODEL=llama3.2-vision
```

If no local VLM is running, Axiom still captures the screenshot, runs available OCR, saves sidecars, and produces an OCR-only visual summary instead of failing.

## Workstation Automation Layer

The SIH25231 problem is not only document search. The stronger product angle is an offline operator that can find evidence, inspect the machine state, and perform controlled local actions.

Axiom now has an operator layer:

- `operator windows` lists visible desktop windows.
- `operator tabs` lists browser tabs only when a browser is explicitly launched with local DevTools remote debugging. Without that, the operating system exposes window titles but not private tab internals.
- `operator scan` maps a local folder.
- `operator find` searches local files by name or content.
- `operator open` opens a source file or folder after `--execute`.
- `operator run` runs read-only allow-listed commands by default, with `--unsafe` required for arbitrary shell execution.
- Every operator action is logged into `operator_audit`.
- `operator audit` shows the recent local action trail.

This keeps the demo impressive while still defensible in a security environment: power exists, but it is visible and auditable.

## Project Layout

```text
axiom/
  local_app.py        Dependency-free local web app server
  app.py              Optional FastAPI wrapper
  chunking.py         Parent-child chunking
  citation.py         Citation parser and validator
  cli.py              Local command-line interface
  database.py         SQLite schema and persistence
  dependencies.py     Setup audit and guided installers
  analytics.py        Evidence graph, timeline, and prediction analysis
  image_generation.py Offline image generation adapters
  investigation.py    Detective search, dossiers, and hallucination guardrails
  reports.py          Markdown, HTML, and JSON case report export
  embeddings.py       Standard-library hashed embeddings
  extractors.py       Local file extraction adapters
  ingestion.py        Ingestion orchestration and cross-links
  retrieval.py        Hybrid retrieval and answer context
  answering.py        Extractive or local-model answer generation
  workstation.py      Local windows, folder, file, open, run tools
  vision.py           Screenshot capture, OCR, and local VLM image analysis
  web/                Local app UI
docs/
  ARCHITECTURE.md     Hardened architecture and implementation plan
  SIH25231_PLAN.md    Problem-specific build plan and demo story
tests/
  test_axiom_core.py  Offline unit tests
```

## Security Posture

Axiom treats citations as a control boundary. The answer layer is allowed to cite only chunks that retrieval returned for the active query. Unknown citation tokens are marked as unverified.

This does not replace classification review, chain-of-custody policy, or operational approval. It gives the software a strong default: evidence first, model second.
