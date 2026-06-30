from __future__ import annotations

import os
import json
from dataclasses import asdict

from .answering import answer_query, chat_with_ollama
from .biorag import biorag_status
from .database import connect, list_operator_audit, record_operator_audit
from .ingestion import ingest_path
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


def create_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("FastAPI API requires local install of fastapi and pydantic.") from exc

    db_path = os.environ.get("AXIOM_DB", "data/axiom.sqlite")
    app = FastAPI(title="Axiom Offline Evidence API", version="0.1.0")

    class IngestRequest(BaseModel):
        path: str
        build_links: bool = True

    class QueryRequest(BaseModel):
        query: str
        top_k: int = 5

    class ChatRequest(BaseModel):
        message: str
        history: list[dict[str, str]] = []
        model: str | None = None

    class ScanRequest(BaseModel):
        path: str
        max_depth: int = 2
        max_items: int = 80

    class FindRequest(BaseModel):
        path: str
        query: str
        content: bool = False
        max_depth: int = 5
        max_results: int = 50
        extensions: list[str] = []

    class OpenRequest(BaseModel):
        path: str
        execute: bool = False

    class RunRequest(BaseModel):
        command: list[str] | str
        cwd: str | None = None
        execute: bool = False
        unsafe: bool = False
        shell: bool = False
        timeout: int = 30

    class TaskRequest(BaseModel):
        request: str

    class ScreenshotRequest(BaseModel):
        output_dir: str = "artifacts/screenshots"
        active_window: bool = False
        analyze: bool = True
        ingest: bool = False
        prompt: str | None = None
        ocr_engine: str = "auto"
        lang: str = "eng"
        vision_model: str | None = None
        use_vlm: bool = True

    class VisionAnalyzeRequest(BaseModel):
        image: str
        prompt: str | None = None
        ocr_engine: str = "auto"
        lang: str = "eng"
        vision_model: str | None = None
        use_vlm: bool = True
        ingest: bool = False

    class OCRRequest(BaseModel):
        image: str
        engine: str = "auto"
        lang: str = "eng"

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "offline", "db": db_path}

    @app.post("/ingest")
    def ingest(request: IngestRequest):
        conn = connect(db_path)
        report = ingest_path(conn, request.path, build_links=request.build_links)
        return asdict(report)

    @app.post("/query")
    def query(request: QueryRequest):
        conn = connect(db_path)
        return asdict(answer_query(conn, request.query, top_k=request.top_k))

    @app.get("/biorag/status")
    def biorag():
        conn = connect(db_path)
        return biorag_status(conn)

    @app.post("/chat")
    def chat(request: ChatRequest):
        return asdict(chat_with_ollama(request.message, history=request.history, model=request.model))

    @app.get("/operator/windows")
    def operator_windows():
        conn = connect(db_path)
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
        return {"windows": payload_list(windows)}

    @app.get("/operator/tabs")
    def operator_tabs():
        conn = connect(db_path)
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
        return {"tabs": payload_list(tabs), "note": browser_tab_note()}

    @app.post("/operator/scan")
    def operator_scan(request: ScanRequest):
        conn = connect(db_path)
        items = scan_folder(request.path, max_depth=request.max_depth, max_items=request.max_items)
        record_operator_audit(
            conn,
            action_type="scan",
            target=request.path,
            parameters_json=model_json(request),
            executed=True,
            success=True,
            stdout_preview=json.dumps(payload_list(items))[:2000],
        )
        return {"items": payload_list(items)}

    @app.post("/operator/find")
    def operator_find(request: FindRequest):
        conn = connect(db_path)
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in request.extensions} or None
        matches = find_files(
            request.path,
            request.query,
            content=request.content,
            max_depth=request.max_depth,
            max_results=request.max_results,
            extensions=extensions,
        )
        record_operator_audit(
            conn,
            action_type="find",
            target=request.path,
            parameters_json=model_json(request),
            executed=True,
            success=True,
            stdout_preview=json.dumps(payload_list(matches))[:2000],
        )
        return {"matches": payload_list(matches)}

    @app.post("/operator/open")
    def operator_open(request: OpenRequest):
        conn = connect(db_path)
        result = open_path(request.path, execute=request.execute)
        record_operator_audit(
            conn,
            action_type="open",
            target=request.path,
            parameters_json=model_json(request),
            executed=result.executed,
            success=result.allowed and (result.return_code in (None, 0)),
            return_code=result.return_code,
            stdout_preview=result.stdout,
            stderr_preview=result.stderr,
        )
        return asdict(result)

    @app.post("/operator/run")
    def operator_run(request: RunRequest):
        conn = connect(db_path)
        result = run_command(
            request.command,
            cwd=request.cwd,
            execute=request.execute,
            unsafe=request.unsafe,
            shell=request.shell,
            timeout=request.timeout,
        )
        record_operator_audit(
            conn,
            action_type="run",
            target=request.cwd,
            parameters_json=model_json(request),
            executed=result.executed,
            success=result.allowed and (result.return_code in (None, 0)),
            return_code=result.return_code,
            stdout_preview=result.stdout,
            stderr_preview=result.stderr,
        )
        return asdict(result)

    @app.post("/operator/task")
    def operator_task(request: TaskRequest):
        conn = connect(db_path)
        plan = plan_operator_task(request.request)
        record_operator_audit(
            conn,
            action_type="task_plan",
            target=None,
            parameters_json=model_json(request),
            executed=False,
            success=True,
            stdout_preview=json.dumps(plan),
        )
        return plan

    @app.get("/operator/audit")
    def operator_audit(limit: int = 20):
        conn = connect(db_path)
        return {"records": [dict(row) for row in list_operator_audit(conn, limit=limit)]}

    @app.post("/vision/screenshot")
    def vision_screenshot(request: ScreenshotRequest):
        conn = connect(db_path)
        screenshot = capture_screenshot(request.output_dir, active_window=request.active_window)
        record_operator_audit(
            conn,
            action_type="screenshot",
            target=screenshot.image_path,
            parameters_json=model_json(request),
            executed=True,
            success=True,
            stdout_preview=json.dumps(asdict(screenshot)),
        )
        payload = {"screenshot": asdict(screenshot)}
        if request.analyze or request.ingest:
            analysis = analyze_image(
                screenshot.image_path,
                prompt=request.prompt,
                ocr_engine=request.ocr_engine,
                lang=request.lang,
                vision_model=request.vision_model,
                use_vlm=request.use_vlm,
            )
            payload["analysis"] = asdict(analysis)
            if request.ingest:
                payload["ingest"] = asdict(ingest_visual_analysis(conn, analysis))
        return payload

    @app.post("/vision/analyze")
    def vision_analyze(request: VisionAnalyzeRequest):
        conn = connect(db_path)
        analysis = analyze_image(
            request.image,
            prompt=request.prompt,
            ocr_engine=request.ocr_engine,
            lang=request.lang,
            vision_model=request.vision_model,
            use_vlm=request.use_vlm,
        )
        payload = {"analysis": asdict(analysis)}
        if request.ingest:
            payload["ingest"] = asdict(ingest_visual_analysis(conn, analysis))
        record_operator_audit(
            conn,
            action_type="vision_analyze",
            target=analysis.image_path,
            parameters_json=model_json(request),
            executed=True,
            success=True,
            stdout_preview=analysis.visual_summary,
        )
        return payload

    @app.post("/vision/ocr")
    def vision_ocr(request: OCRRequest):
        conn = connect(db_path)
        result = run_ocr(request.image, engine=request.engine, lang=request.lang)
        record_operator_audit(
            conn,
            action_type="ocr",
            target=request.image,
            parameters_json=model_json(request),
            executed=True,
            success=result.error is None,
            stdout_preview=result.text,
            stderr_preview=result.error,
        )
        return asdict(result)

    return app


def model_json(model: object) -> str:
    if hasattr(model, "model_dump_json"):
        return str(model.model_dump_json())  # type: ignore[attr-defined]
    if hasattr(model, "json"):
        return str(model.json())  # type: ignore[attr-defined]
    return json.dumps(model)
