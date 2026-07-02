from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .analytics import build_analytics
from .answering import answer_query
from .database import (
    connect,
    get_chunk_by_prefix,
    list_documents,
    list_operator_audit,
    record_operator_audit,
)
from .evaluation import DEFAULT_MODES, load_benchmark_cases, run_benchmark, write_benchmark_report
from .image_generation import ImageGenerationRequest, generate_image, image_generation_status
from .ingestion import ingest_path
from .investigation import investigate_subject
from .local_app import run_app
from .mission import build_mission_brief
from .reports import generate_case_report
from .retrieval import format_location
from .workstation import (
    browser_tab_note,
    find_files,
    list_browser_tabs,
    list_windows,
    open_path,
    payload_list,
    plan_operator_task,
    run_command,
    scan_folder,
)
from .vision import analyze_image, capture_screenshot, ingest_visual_analysis, run_ocr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axiom", description="Offline evidence retrieval with strict citations")
    parser.add_argument("--db", default=None, help="SQLite database path")
    subparsers = parser.add_subparsers(dest="root_command", required=True)

    init = subparsers.add_parser("init", help="Initialize the local evidence database")
    init.add_argument("--db", default=None, help="SQLite database path")

    ingest = subparsers.add_parser("ingest", help="Ingest a file or folder")
    ingest.add_argument("path", help="File or directory to ingest")
    ingest.add_argument("--db", default=None, help="SQLite database path")
    ingest.add_argument("--no-links", action="store_true", help="Skip semantic cross-link creation")

    ask = subparsers.add_parser("ask", help="Ask a question against local evidence")
    ask.add_argument("query", help="Question or search query")
    ask.add_argument("--db", default=None, help="SQLite database path")
    ask.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks to retrieve")
    ask.add_argument("--json", action="store_true", help="Print machine-readable output")

    analytics = subparsers.add_parser("analytics", help="Build local evidence graph, timeline, and prediction brief")
    analytics.add_argument("--db", default=None, help="SQLite database path")
    analytics.add_argument("--query", default=None, help="Optional focused query")
    analytics.add_argument("--limit", type=int, default=80)
    analytics.add_argument("--json", action="store_true")

    mission = subparsers.add_parser("mission", help="Show SIH25231 multimodal offline readiness brief")
    mission.add_argument("--db", default=None, help="SQLite database path")
    mission.add_argument("--json", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Run retrieval baselines for HiveRAG research tables")
    benchmark.add_argument("--db", default=None, help="SQLite database path")
    benchmark.add_argument("--dataset", default="benchmarks/local_axiom_eval.jsonl", help="JSONL benchmark cases")
    benchmark.add_argument("--corpus", default=None, help="Optional file/folder to ingest before benchmarking")
    benchmark.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated modes")
    benchmark.add_argument("--top-k", type=int, default=5)
    benchmark.add_argument("--case-offset", type=int, default=0, help="Skip this many benchmark cases before running")
    benchmark.add_argument("--case-limit", type=int, default=None, help="Run only this many benchmark cases")
    benchmark.add_argument("--out", default="exports/benchmarks")
    benchmark.add_argument("--label", default="biorag")
    benchmark.add_argument("--json", action="store_true")
    benchmark.add_argument("--evaluator", default=None, choices=["ollama", "openai"], help="Use official framework evaluator")
    benchmark.add_argument("--evaluator-model", default="qwen2.5:7b", help="Model to use for evaluator")
    benchmark.add_argument("--evaluator-url", default=None, help="Base URL for the evaluator endpoint")
    benchmark.add_argument("--embedding-model", default=None, help="Embedding model for RAGAS evaluator")

    imagegen = subparsers.add_parser("imagegen", help="Generate images with an offline local image model")
    imagegen.add_argument("--db", default=None, help="SQLite database path")
    imagegen_sub = imagegen.add_subparsers(dest="imagegen_command", required=True)
    image_status = imagegen_sub.add_parser("status", help="Check local image-generation backends")
    image_status.add_argument("--db", default=None, help="SQLite database path")
    image_status.add_argument("--json", action="store_true")
    image_generate = imagegen_sub.add_parser("generate", help="Generate an image")
    image_generate.add_argument("prompt")
    image_generate.add_argument("--db", default=None, help="SQLite database path")
    image_generate.add_argument("--negative", default="")
    image_generate.add_argument("--backend", default="auto", choices=["auto", "a1111", "automatic1111", "forge", "diffusers"])
    image_generate.add_argument("--width", type=int, default=768)
    image_generate.add_argument("--height", type=int, default=512)
    image_generate.add_argument("--steps", type=int, default=24)
    image_generate.add_argument("--cfg", type=float, default=7.0)
    image_generate.add_argument("--seed", type=int, default=-1)
    image_generate.add_argument("--model-path", default=None)
    image_generate.add_argument("--enhance", action="store_true")
    image_generate.add_argument("--json", action="store_true")

    investigate = subparsers.add_parser("investigate", help="Build an evidence-backed offline dossier for a person, handle, or identifier")
    investigate.add_argument("subject")
    investigate.add_argument("--db", default=None, help="SQLite database path")
    investigate.add_argument("--root", action="append", default=[], help="Optional local folder root for wider search")
    investigate.add_argument("--top-k", type=int, default=8)
    investigate.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report", help="Generate an offline case report with citations and guardrails")
    report.add_argument("subject")
    report.add_argument("--db", default=None, help="SQLite database path")
    report.add_argument("--root", action="append", default=[], help="Optional local folder root for wider search")
    report.add_argument("--top-k", type=int, default=8)
    report.add_argument("--out", default="exports/reports")
    report.add_argument("--json", action="store_true")

    sources = subparsers.add_parser("sources", help="List or inspect indexed evidence sources")
    sources.add_argument("--db", default=None, help="SQLite database path")
    sources_sub = sources.add_subparsers(dest="sources_command", required=True)
    sources_list = sources_sub.add_parser("list", help="List indexed source files")
    sources_list.add_argument("--db", default=None, help="SQLite database path")
    inspect = sources_sub.add_parser("inspect", help="Inspect a citation or chunk id")
    inspect.add_argument("citation", help="Full citation token or chunk id prefix")
    inspect.add_argument("--db", default=None, help="SQLite database path")

    operator = subparsers.add_parser("operator", help="Local workstation automation tools")
    operator.add_argument("--db", default=None, help="SQLite database path")
    operator_sub = operator.add_subparsers(dest="operator_command", required=True)

    windows = operator_sub.add_parser("windows", help="List visible desktop windows")
    windows.add_argument("--db", default=None, help="SQLite database path")
    windows.add_argument("--json", action="store_true")

    tabs = operator_sub.add_parser("tabs", help="List browser tabs exposed through local DevTools")
    tabs.add_argument("--db", default=None, help="SQLite database path")
    tabs.add_argument("--json", action="store_true")

    scan = operator_sub.add_parser("scan", help="Scan a local folder")
    scan.add_argument("path")
    scan.add_argument("--db", default=None, help="SQLite database path")
    scan.add_argument("--max-depth", type=int, default=2)
    scan.add_argument("--max-items", type=int, default=80)
    scan.add_argument("--json", action="store_true")

    find = operator_sub.add_parser("find", help="Find files by name or text content")
    find.add_argument("path")
    find.add_argument("query")
    find.add_argument("--db", default=None, help="SQLite database path")
    find.add_argument("--content", action="store_true", help="Search inside text-like files too")
    find.add_argument("--max-depth", type=int, default=5)
    find.add_argument("--max-results", type=int, default=50)
    find.add_argument("--ext", action="append", default=[], help="Extension filter such as .pdf or .txt")
    find.add_argument("--json", action="store_true")

    open_cmd = operator_sub.add_parser("open", help="Open a local file or folder")
    open_cmd.add_argument("path")
    open_cmd.add_argument("--db", default=None, help="SQLite database path")
    open_cmd.add_argument("--execute", action="store_true", help="Actually open the path")
    open_cmd.add_argument("--json", action="store_true")

    run = operator_sub.add_parser("run", help="Run a guarded local command")
    run.add_argument("--db", default=None, help="SQLite database path")
    run.add_argument("--cwd", default=None)
    run.add_argument("--execute", action="store_true", help="Actually run the command")
    run.add_argument("--unsafe", action="store_true", help="Allow commands outside the read-only allow-list")
    run.add_argument("--shell", action="store_true", help="Run through the OS shell; requires --unsafe")
    run.add_argument("--timeout", type=int, default=30)
    run.add_argument("--json", action="store_true")
    run.add_argument("run_args", nargs=argparse.REMAINDER)

    task = operator_sub.add_parser("task", help="Create a deterministic operator action plan")
    task.add_argument("request")
    task.add_argument("--db", default=None, help="SQLite database path")
    task.add_argument("--json", action="store_true")

    audit = operator_sub.add_parser("audit", help="Show recent operator audit records")
    audit.add_argument("--db", default=None, help="SQLite database path")
    audit.add_argument("--limit", type=int, default=20)
    audit.add_argument("--json", action="store_true")

    vision = subparsers.add_parser("vision", help="Screenshot, OCR, and offline vision understanding")
    vision.add_argument("--db", default=None, help="SQLite database path")
    vision_sub = vision.add_subparsers(dest="vision_command", required=True)

    screenshot = vision_sub.add_parser("screenshot", help="Capture a desktop or active-window screenshot")
    screenshot.add_argument("--db", default=None, help="SQLite database path")
    screenshot.add_argument("--out", default="artifacts/screenshots", help="Output directory")
    screenshot.add_argument("--active-window", action="store_true", help="Capture the foreground window on Windows")
    screenshot.add_argument("--analyze", action="store_true", help="Run OCR and local vision understanding")
    screenshot.add_argument("--ingest", action="store_true", help="Ingest OCR/caption output into the evidence index")
    screenshot.add_argument("--prompt", default=None, help="Extra instruction for the vision model")
    screenshot.add_argument("--ocr-engine", default="auto", choices=["auto", "paddle", "paddleocr", "tesseract", "none"])
    screenshot.add_argument("--lang", default="eng", help="OCR language, such as eng")
    screenshot.add_argument("--vision-model", default=None, help="Ollama vision model, default llama3.2-vision")
    screenshot.add_argument("--no-vlm", action="store_true", help="Skip local vision model and use OCR-only analysis")
    screenshot.add_argument("--json", action="store_true")

    analyze = vision_sub.add_parser("analyze", help="OCR and understand an existing image")
    analyze.add_argument("image")
    analyze.add_argument("--db", default=None, help="SQLite database path")
    analyze.add_argument("--prompt", default=None)
    analyze.add_argument("--ocr-engine", default="auto", choices=["auto", "paddle", "paddleocr", "tesseract", "none"])
    analyze.add_argument("--lang", default="eng")
    analyze.add_argument("--vision-model", default=None)
    analyze.add_argument("--no-vlm", action="store_true")
    analyze.add_argument("--ingest", action="store_true")
    analyze.add_argument("--json", action="store_true")

    ocr = vision_sub.add_parser("ocr", help="Run OCR over an image")
    ocr.add_argument("image")
    ocr.add_argument("--db", default=None, help="SQLite database path")
    ocr.add_argument("--engine", default="auto", choices=["auto", "paddle", "paddleocr", "tesseract", "none"])
    ocr.add_argument("--lang", default="eng")
    ocr.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="Run the optional FastAPI server")
    serve.add_argument("--db", default=None, help="SQLite database path")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    app = subparsers.add_parser("app", help="Run the local offline Axiom app")
    app.add_argument("--db", default=None, help="SQLite database path")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db or "data/axiom.sqlite"
    conn = connect(db_path)

    if args.root_command == "init":
        print(f"Initialized local evidence database: {Path(db_path).resolve()}")
        return 0

    if args.root_command == "ingest":
        report = ingest_path(conn, args.path, build_links=not args.no_links)
        print(f"Indexed files: {len(report.indexed_files)}")
        print(f"Skipped files: {len(report.skipped_files)}")
        print(f"Files needing adapters: {len(report.error_files)}")
        print(f"Chunks created: {report.chunks_created}")
        print(f"Cross-modal links created: {report.links_created}")
        if report.error_files:
            print("\nAdapter notes:")
            for file_path, message in report.error_files.items():
                print(f"- {file_path}: {message}")
        return 0

    if args.root_command == "ask":
        result = answer_query(conn, args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.__dict__, indent=2))
        else:
            print(result.answer)
            if result.sources:
                print("\nSources:")
                for source in result.sources:
                    print(
                        f"- {source['citation']} {source['file_name']} "
                        f"({source['location']}, sha256={str(source['sha256'])[:12]}...)"
                    )
        return 0

    if args.root_command == "sources":
        if args.sources_command == "list":
            rows = list_documents(conn)
            if not rows:
                print("No sources indexed yet.")
                return 0
            for row in rows:
                print(
                    f"{row['file_name']} | type={row['file_type']} | status={row['status']} | "
                    f"chunks={row['chunk_count']} | hash={row['sha256'][:12]}... | {row['file_path']}"
                )
            return 0
        if args.sources_command == "inspect":
            row = get_chunk_by_prefix(conn, args.citation)
            if row is None:
                print(f"No chunk matched: {args.citation}")
                return 1
            print(f"Citation: [Axiom:{row['chunk_id']}]")
            print(f"File: {row['file_name']}")
            print(f"Path: {row['file_path']}")
            print(f"Type: {row['file_type']}")
            print(f"Hash: {row['sha256']}")
            print(f"Modality: {row['modality']}")
            print(f"Location: {format_location(row)}")
            print("\nText:")
            print(row["text_content"])
            return 0

    if args.root_command == "analytics":
        result = build_analytics(conn, query=args.query, limit=args.limit)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            prediction = result["prediction"]
            metrics = result["metrics"]
            print(f"Confidence: {int(float(prediction['confidence']) * 100)}%")
            print(f"Forecast: {prediction['forecast']}")
            print(f"Trend: {prediction['trend']['summary']}")
            print(f"Graph: {metrics['graph_nodes']} nodes, {metrics['graph_edges']} edges")
            print(f"Timeline items: {metrics['timeline_items']}")
            print("\nNext actions:")
            for action in prediction["next_actions"]:
                print(f"- {action}")
        return 0

    if args.root_command == "mission":
        result = build_mission_brief(conn)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"{result['problem']['ps_id']} Mission Brief")
            print(f"Score: {result['score']}/100")
            print(f"Verdict: {result['verdict']}")
            print("\nCoverage:")
            for row in result["coverage"]:
                state = "ready" if row["ready"] else "gap"
                print(f"- {row['label']}: {state} ({row['indexed']} indexed, {row['needs_adapter']} adapter notes)")
            print("\nDifferentiators:")
            for item in result["differentiators"]:
                print(f"- {item['title']}: {item['body']}")
            if result["gaps"]:
                print("\nNext gaps:")
                for gap in result["gaps"]:
                    print(f"- {gap}")
        return 0

    if args.root_command == "benchmark":
        if args.corpus:
            ingest_path(conn, args.corpus)
        cases = load_benchmark_cases(args.dataset)
        if args.case_offset:
            cases = cases[max(args.case_offset, 0) :]
        if args.case_limit is not None:
            cases = cases[: max(args.case_limit, 0)]
        modes = [item.strip() for item in args.modes.split(",") if item.strip()]
        if args.evaluator_url:
            if args.evaluator == "openai":
                os.environ["AXIOM_OPENAI_BASE_URL"] = args.evaluator_url
            else:
                os.environ["AXIOM_OLLAMA_BASE_URL"] = args.evaluator_url
        if args.embedding_model:
            os.environ["AXIOM_RAGAS_EMBEDDING_MODEL"] = args.embedding_model
        result = run_benchmark(
            conn, cases, modes=modes, top_k=args.top_k,
            evaluator=args.evaluator, evaluator_model=args.evaluator_model
        )
        paths = write_benchmark_report(result, args.out, label=args.label)
        if args.json:
            print(json.dumps({"report": result, "paths": paths}, indent=2))
        else:
            print(f"Benchmark cases: {len(cases)}")
            print(f"Modes: {', '.join(modes)}")
            evaluator_status = result.get("evaluator_status", {})
            ragas_judge_used = isinstance(evaluator_status, dict) and bool(evaluator_status.get("ragas_llm_judge_used"))
            context_label = "ContextPrecision" if ragas_judge_used else "ContextPrecisionProxy"
            faithfulness_label = "Faithfulness" if ragas_judge_used else "FaithfulnessProxy"
            print("\nSummary:")
            for mode, metrics in result["summary"].items():
                print(
                    f"- {mode}: Hit@k={metrics['hit_at_k']} MRR={metrics['mrr']} "
                    f"SourceRecall={metrics['source_recall']} TermRecall={metrics['term_recall']} "
                    f"{context_label}={metrics['ragas_context_precision_proxy']} "
                    f"{faithfulness_label}={metrics['ragas_faithfulness_proxy']} "
                    f"Latency={metrics['avg_latency_ms']}ms"
                )
            print("\nEvaluator tracks:")
            for framework, info in result["evaluator_availability"].items():
                state = "official package ready" if info["official_available"] else "offline proxy only"
                print(f"- {framework}: {state}")
            if ragas_judge_used:
                print(
                    f"- ragas run: LLM judge used "
                    f"(model={evaluator_status.get('model') or args.evaluator_model or 'default'}, "
                    f"fallback metric cells={evaluator_status.get('fallback_metric_cells', 0)})"
                )
            elif isinstance(evaluator_status, dict) and evaluator_status.get("requested") and evaluator_status.get("error"):
                print(f"- ragas run skipped: {evaluator_status.get('error')}")
            framework_summary = result["framework_summary"]
            if "hiverag" in framework_summary["ragas"]:
                ragas = framework_summary["ragas"]["hiverag"]
                trulens = framework_summary["trulens"]["hiverag"]
                deepeval = framework_summary["deepeval"]["hiverag"]
                print(
                    "\nHiveRAG evaluator scores: "
                    f"RAGAS overall={ragas['overall_proxy']} "
                    f"TruLens triad={trulens['rag_triad_proxy']} "
                    f"DeepEval overall={deepeval['overall_proxy']}"
                )
            print(f"\nMarkdown: {paths['markdown']}")
            print(f"JSON: {paths['json']}")
        return 0

    if args.root_command == "imagegen":
        if args.imagegen_command == "status":
            status = image_generation_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"Ready: {status['ready']}")
                print(f"AUTOMATIC1111/Forge: {status['automatic1111']}")
                print(f"Diffusers: {status['diffusers']}")
                print(f"Model path: {status['model_path']}")
            return 0
        if args.imagegen_command == "generate":
            result = generate_image(
                ImageGenerationRequest(
                    prompt=args.prompt,
                    negative_prompt=args.negative,
                    backend=args.backend,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance_scale=args.cfg,
                    seed=args.seed,
                    model_path=args.model_path,
                    enhance_prompt=args.enhance,
                )
            )
            if args.json:
                print(json.dumps(result.__dict__, indent=2))
            else:
                if result.success:
                    print(f"Image saved: {result.image_path}")
                    print(f"Metadata: {result.metadata_path}")
                    print(f"Backend: {result.backend}")
                    print(f"Prompt: {result.prompt}")
                else:
                    print(f"Image generation failed: {result.error}")
            return 0 if result.success else 1

    if args.root_command == "investigate":
        result = investigate_subject(conn, args.subject, roots=args.root, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Subject: {result['subject']}")
            print(f"Confidence: {int(float(result['confidence']) * 100)}%")
            print(result["summary"])
            print("\nEvidence:")
            for item in result["evidence"][:8]:
                print(f"- {item['citation']} {item['file_name']} {item['location']}: {item['snippet']}")
            if result["file_leads"]:
                print("\nFile leads:")
                for item in result["file_leads"][:8]:
                    print(f"- {item['path']} ({item['reason']})")
            print("\nNext actions:")
            for action in result["next_actions"]:
                print(f"- {action}")
            print(f"\nGuard: {result['hallucination_guard']['status']}")
        return 0

    if args.root_command == "report":
        result = generate_case_report(conn, args.subject, roots=args.root, output_dir=args.out, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.__dict__, indent=2))
        else:
            print(f"Report generated for: {result.subject}")
            print(f"Confidence: {int(result.confidence * 100)}%")
            print(f"Evidence items: {result.evidence_count}")
            print(f"Timeline items: {result.timeline_count}")
            print(f"HTML: {result.html_path}")
            print(f"Markdown: {result.markdown_path}")
            print(f"JSON: {result.json_path}")
        return 0

    if args.root_command == "operator":
        return handle_operator(args, conn)

    if args.root_command == "vision":
        return handle_vision(args, conn)

    if args.root_command == "serve":
        os.environ["AXIOM_DB"] = db_path
        try:
            import uvicorn  # type: ignore
        except ImportError as exc:
            raise SystemExit("FastAPI server requires local install of optional dependency: uvicorn") from exc
        uvicorn.run("axiom.app:create_app", factory=True, host=args.host, port=args.port)
        return 0

    if args.root_command == "app":
        run_app(db_path=db_path, host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0

    return 2


def handle_operator(args: argparse.Namespace, conn) -> int:
    if args.operator_command == "windows":
        windows = list_windows()
        record_operator_audit(
            conn,
            action_type="windows",
            target=None,
            parameters_json="{}",
            executed=True,
            success=True,
            stdout_preview=json.dumps(payload_list(windows))[:2000],
        )
        if args.json:
            print(json.dumps(payload_list(windows), indent=2))
        else:
            print(f"Visible windows: {len(windows)}")
            for item in windows:
                print(f"- [{item.pid}] {item.process}: {item.title}")
        return 0

    if args.operator_command == "tabs":
        tabs = list_browser_tabs()
        record_operator_audit(
            conn,
            action_type="tabs",
            target=None,
            parameters_json="{}",
            executed=True,
            success=True,
            stdout_preview=json.dumps(payload_list(tabs))[:2000],
        )
        if args.json:
            print(json.dumps({"tabs": payload_list(tabs), "note": browser_tab_note()}, indent=2))
        else:
            print(f"DevTools browser tabs: {len(tabs)}")
            for item in tabs:
                print(f"- [{item.port}] {item.title} | {item.url}")
            if not tabs:
                print(browser_tab_note())
        return 0

    if args.operator_command == "scan":
        items = scan_folder(args.path, max_depth=args.max_depth, max_items=args.max_items)
        record_operator_audit(
            conn,
            action_type="scan",
            target=args.path,
            parameters_json=json.dumps({"max_depth": args.max_depth, "max_items": args.max_items}),
            executed=True,
            success=True,
            stdout_preview=json.dumps(payload_list(items))[:2000],
        )
        if args.json:
            print(json.dumps(payload_list(items), indent=2))
        else:
            print(f"Folder scan: {args.path}")
            for item in items:
                marker = "DIR " if item.is_dir else "FILE"
                print(f"- {marker} {item.path}")
        return 0

    if args.operator_command == "find":
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.ext} or None
        matches = find_files(
            args.path,
            args.query,
            max_depth=args.max_depth,
            max_results=args.max_results,
            content=args.content,
            extensions=extensions,
        )
        record_operator_audit(
            conn,
            action_type="find",
            target=args.path,
            parameters_json=json.dumps(
                {
                    "query": args.query,
                    "content": args.content,
                    "max_depth": args.max_depth,
                    "max_results": args.max_results,
                    "extensions": sorted(extensions or []),
                }
            ),
            executed=True,
            success=True,
            stdout_preview=json.dumps(payload_list(matches))[:2000],
        )
        if args.json:
            print(json.dumps(payload_list(matches), indent=2))
        else:
            print(f"Matches: {len(matches)}")
            for item in matches:
                print(f"- {item.path}")
        return 0

    if args.operator_command == "open":
        result = open_path(args.path, execute=args.execute)
        audit_command_result(conn, "open", args.path, {"execute": args.execute}, result)
        print_operator_result(result, as_json=args.json)
        return 0 if result.allowed and (result.return_code in (None, 0)) else 1

    if args.operator_command == "run":
        command_args = list(args.run_args)
        if command_args and command_args[0] == "--":
            command_args = command_args[1:]
        command = " ".join(command_args) if args.shell else command_args
        result = run_command(
            command,
            cwd=args.cwd,
            execute=args.execute,
            unsafe=args.unsafe,
            shell=args.shell,
            timeout=args.timeout,
        )
        audit_command_result(
            conn,
            "run",
            args.cwd,
            {
                "command": command,
                "execute": args.execute,
                "unsafe": args.unsafe,
                "shell": args.shell,
                "timeout": args.timeout,
            },
            result,
        )
        print_operator_result(result, as_json=args.json)
        return 0 if result.allowed and (result.return_code in (None, 0)) else 1

    if args.operator_command == "task":
        plan = plan_operator_task(args.request)
        record_operator_audit(
            conn,
            action_type="task_plan",
            target=None,
            parameters_json=json.dumps({"request": args.request}),
            executed=False,
            success=True,
            stdout_preview=json.dumps(plan),
        )
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(f"Planned action: {plan.get('action')}")
            for key, value in plan.items():
                if key != "action":
                    print(f"- {key}: {value}")
        return 0

    if args.operator_command == "audit":
        rows = list_operator_audit(conn, limit=args.limit)
        if args.json:
            print(json.dumps([dict(row) for row in rows], indent=2))
        else:
            if not rows:
                print("No operator audit records yet.")
                return 0
            for row in rows:
                status = "ok" if row["success"] else "failed"
                executed = "executed" if row["executed"] else "dry-run"
                print(
                    f"#{row['audit_id']} {row['created_at']} {row['action_type']} "
                    f"{executed}/{status} target={row['target'] or '-'}"
                )
        return 0

    return 2


def handle_vision(args: argparse.Namespace, conn) -> int:
    if args.vision_command == "screenshot":
        result = capture_screenshot(args.out, active_window=args.active_window)
        record_operator_audit(
            conn,
            action_type="screenshot",
            target=result.image_path,
            parameters_json=json.dumps(
                {
                    "active_window": args.active_window,
                    "analyze": args.analyze,
                    "ingest": args.ingest,
                    "ocr_engine": args.ocr_engine,
                }
            ),
            executed=True,
            success=True,
            stdout_preview=json.dumps(result.__dict__),
        )
        if args.analyze or args.ingest:
            analysis = analyze_image(
                result.image_path,
                prompt=args.prompt,
                ocr_engine=args.ocr_engine,
                lang=args.lang,
                vision_model=args.vision_model,
                use_vlm=not args.no_vlm,
            )
            if args.ingest:
                report = ingest_visual_analysis(conn, analysis)
            else:
                report = None
            print_vision_analysis(analysis, screenshot=result, ingest_report=report, as_json=args.json)
        else:
            if args.json:
                print(json.dumps(result.__dict__, indent=2))
            else:
                print(f"Screenshot saved: {result.image_path}")
                print(f"Method: {result.capture_method}")
                if result.width and result.height:
                    print(f"Size: {result.width}x{result.height}")
        return 0

    if args.vision_command == "analyze":
        analysis = analyze_image(
            args.image,
            prompt=args.prompt,
            ocr_engine=args.ocr_engine,
            lang=args.lang,
            vision_model=args.vision_model,
            use_vlm=not args.no_vlm,
        )
        report = ingest_visual_analysis(conn, analysis) if args.ingest else None
        record_operator_audit(
            conn,
            action_type="vision_analyze",
            target=analysis.image_path,
            parameters_json=json.dumps(
                {
                    "ingest": args.ingest,
                    "ocr_engine": args.ocr_engine,
                    "lang": args.lang,
                    "vision_model": analysis.model,
                }
            ),
            executed=True,
            success=True,
            stdout_preview=analysis.visual_summary,
        )
        print_vision_analysis(analysis, ingest_report=report, as_json=args.json)
        return 0

    if args.vision_command == "ocr":
        result = run_ocr(args.image, engine=args.engine, lang=args.lang)
        record_operator_audit(
            conn,
            action_type="ocr",
            target=args.image,
            parameters_json=json.dumps({"engine": args.engine, "lang": args.lang}),
            executed=True,
            success=result.error is None,
            stdout_preview=result.text,
            stderr_preview=result.error,
        )
        if args.json:
            print(json.dumps(result.__dict__, indent=2))
        else:
            print(f"Engine: {result.engine}")
            if result.confidence is not None:
                print(f"Confidence: {result.confidence:.3f}")
            if result.error:
                print(f"Error: {result.error}")
            print("\nText:")
            print(result.text or "(no text)")
        return 0 if result.error is None else 1

    return 2


def audit_command_result(conn, action_type: str, target: str | None, parameters: dict[str, object], result) -> None:
    record_operator_audit(
        conn,
        action_type=action_type,
        target=target,
        parameters_json=json.dumps(parameters),
        executed=result.executed,
        success=result.allowed and (result.return_code in (None, 0)),
        return_code=result.return_code,
        stdout_preview=result.stdout,
        stderr_preview=result.stderr,
    )


def print_operator_result(result, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result.__dict__, indent=2))
        return
    print(f"Allowed: {result.allowed}")
    print(f"Executed: {result.executed}")
    if result.return_code is not None:
        print(f"Return code: {result.return_code}")
    print(f"Reason: {result.reason}")
    if result.stdout:
        print("\nOutput:")
        print(result.stdout)
    if result.stderr:
        print("\nError:")
        print(result.stderr)


def print_vision_analysis(analysis, *, screenshot=None, ingest_report=None, as_json: bool) -> None:
    if as_json:
        payload = {"analysis": analysis.__dict__}
        payload["analysis"]["ocr"] = analysis.ocr.__dict__
        if screenshot is not None:
            payload["screenshot"] = screenshot.__dict__
        if ingest_report is not None:
            payload["ingest"] = ingest_report.__dict__
        print(json.dumps(payload, indent=2))
        return

    if screenshot is not None:
        print(f"Screenshot saved: {screenshot.image_path}")
        print(f"Method: {screenshot.capture_method}")
    print(f"Image: {analysis.image_path}")
    if analysis.width and analysis.height:
        print(f"Size: {analysis.width}x{analysis.height}")
    print(f"Hash: {analysis.sha256}")
    print(f"OCR engine: {analysis.ocr.engine}")
    if analysis.ocr.confidence is not None:
        print(f"OCR confidence: {analysis.ocr.confidence:.3f}")
    if analysis.ocr.error:
        print(f"OCR note: {analysis.ocr.error}")
    print(f"Vision model: {analysis.model or 'not available; OCR-only fallback'}")
    print("\nVisual understanding:")
    print(analysis.visual_summary)
    print("\nOCR text:")
    print(analysis.ocr.text or "(no OCR text)")
    print("\nSidecars:")
    for label, path in analysis.sidecars.items():
        print(f"- {label}: {path}")
    if ingest_report is not None:
        print("\nIngest:")
        print(f"- Indexed files: {len(ingest_report.indexed_files)}")
        print(f"- Chunks created: {ingest_report.chunks_created}")
        print(f"- Adapter notes: {len(ingest_report.error_files)}")
