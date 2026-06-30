from __future__ import annotations

import json
import mimetypes
import os
import socket
import threading
import urllib.parse
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .analytics import build_analytics
from .answering import chat_with_ollama, answer_query
from .biorag import biorag_status
from .database import connect, list_documents, list_operator_audit, record_operator_audit
from .dependencies import audit_dependencies, ensure_directories, install_dependency
from .image_generation import ImageGenerationRequest, generate_image, image_generation_status
from .ingestion import ingest_path
from .investigation import investigate_subject, validate_investigation_answer
from .mission import build_mission_brief
from .reports import generate_case_report
from .uploads import save_uploaded_files
from .workstation import (
    browser_tab_note,
    find_files,
    list_browser_tabs,
    list_windows,
    open_path,
    payload_list,
    run_command,
    scan_folder,
)
from .vision import analyze_image, capture_screenshot, ingest_visual_analysis, run_ocr, save_pasted_image


APP_DIR = Path(__file__).resolve().parent / "web"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AxiomAppState:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.install_lock = threading.Lock()
        self.install_jobs: dict[str, dict[str, Any]] = {}
        self.install_jobs_lock = threading.Lock()

    def conn(self):
        return connect(self.db_path)

    def start_install_job(self, key: str) -> dict[str, Any]:
        with self.install_jobs_lock:
            for job in self.install_jobs.values():
                if job["key"] == key and job["status"] == "running":
                    return dict(job)
            job_id = f"install-{len(self.install_jobs) + 1}-{key.replace(':', '-')}"
            job = {
                "job_id": job_id,
                "key": key,
                "status": "running",
                "started_at": utc_now(),
                "completed_at": None,
                "result": None,
            }
            self.install_jobs[job_id] = job
        thread = threading.Thread(target=self._run_install_job, args=(job_id, key), daemon=True)
        thread.start()
        return dict(job)

    def install_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self.install_jobs_lock:
            job = self.install_jobs.get(job_id)
            return dict(job) if job else None

    def _run_install_job(self, job_id: str, key: str) -> None:
        with self.install_lock:
            result = install_dependency(key, execute=True)
        with self.install_jobs_lock:
            job = self.install_jobs.get(job_id)
            if not job:
                return
            job["status"] = "done" if result.success else "failed"
            job["completed_at"] = utc_now()
            job["result"] = asdict(result)


class AxiomRequestHandler(BaseHTTPRequestHandler):
    server_version = "AxiomLocalApp/0.1"

    @property
    def state(self) -> AxiomAppState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api_get(parsed.path, urllib.parse.parse_qs(parsed.query))
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        payload = self.read_json()
        self.handle_api_post(parsed.path, payload)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        try:
            if path == "/api/health":
                self.send_json({"status": "ok", "db": self.state.db_path, "mode": "offline-app"})
            elif path == "/api/setup/status":
                self.send_json(audit_dependencies())
            elif path == "/api/setup/install-status":
                job_id = (query.get("job_id") or [""])[0]
                job = self.state.install_job_status(job_id)
                if job is None:
                    self.send_json({"error": "Unknown install job"}, HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(job)
            elif path == "/api/sources":
                conn = self.state.conn()
                self.send_json({"sources": [dict(row) for row in list_documents(conn)]})
            elif path == "/api/operator/windows":
                windows = list_windows()
                self.send_json({"windows": payload_list(windows)})
            elif path == "/api/operator/tabs":
                tabs = list_browser_tabs()
                self.send_json({"tabs": payload_list(tabs), "note": browser_tab_note()})
            elif path == "/api/operator/audit":
                limit = int((query.get("limit") or ["20"])[0])
                conn = self.state.conn()
                self.send_json({"records": [dict(row) for row in list_operator_audit(conn, limit=limit)]})
            elif path == "/api/analytics":
                conn = self.state.conn()
                query_text = (query.get("query") or [""])[0]
                limit = int((query.get("limit") or ["80"])[0])
                self.send_json(build_analytics(conn, query=query_text or None, limit=limit))
            elif path == "/api/mission/brief":
                conn = self.state.conn()
                self.send_json(build_mission_brief(conn))
            elif path == "/api/biorag/status":
                conn = self.state.conn()
                self.send_json(biorag_status(conn))
            elif path == "/api/imagegen/status":
                self.send_json(image_generation_status((query.get("model_path") or [""])[0] or None))
            elif path == "/api/artifact":
                self.serve_artifact((query.get("path") or [""])[0])
            else:
                self.send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - app API should return visible errors.
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def handle_api_post(self, path: str, payload: dict[str, Any]) -> None:
        try:
            if path == "/api/setup/install":
                key = str(payload.get("key", ""))
                execute = bool(payload.get("execute", False))
                with self.state.install_lock:
                    result = install_dependency(key, execute=execute)
                self.send_json(asdict(result))
            elif path == "/api/setup/install-start":
                key = str(payload.get("key", ""))
                job = self.state.start_install_job(key)
                self.send_json(job)
            elif path == "/api/query":
                conn = self.state.conn()
                result = answer_query(conn, str(payload.get("query", "")), top_k=int(payload.get("top_k", 5)))
                self.send_json(asdict(result))
            elif path == "/api/chat":
                result = chat_with_ollama(
                    str(payload.get("message", "")),
                    history=list(payload.get("history") or []),
                    model=payload.get("model") or None,
                )
                self.send_json(asdict(result))
            elif path == "/api/ingest":
                conn = self.state.conn()
                report = ingest_path(conn, str(payload.get("path", "")), build_links=bool(payload.get("build_links", True)))
                self.send_json(asdict(report))
            elif path == "/api/intake/upload":
                conn = self.state.conn()
                batch = save_uploaded_files(
                    list(payload.get("files") or []),
                    output_dir=payload.get("output_dir", "intake/uploads"),
                    batch_name=payload.get("batch_name") or None,
                )
                response: dict[str, Any] = {"batch": asdict(batch)}
                if bool(payload.get("ingest", True)):
                    report = ingest_path(conn, batch.root_path, build_links=bool(payload.get("build_links", True)))
                    response["ingest"] = asdict(report)
                self.send_json(response)
            elif path == "/api/operator/scan":
                items = scan_folder(
                    str(payload.get("path", ".")),
                    max_depth=int(payload.get("max_depth", 2)),
                    max_items=int(payload.get("max_items", 80)),
                )
                self.send_json({"items": payload_list(items)})
            elif path == "/api/operator/find":
                items = find_files(
                    str(payload.get("path", ".")),
                    str(payload.get("query", "")),
                    content=bool(payload.get("content", True)),
                    max_depth=int(payload.get("max_depth", 5)),
                    max_results=int(payload.get("max_results", 50)),
                )
                self.send_json({"matches": payload_list(items)})
            elif path == "/api/operator/open":
                conn = self.state.conn()
                result = open_path(str(payload.get("path", "")), execute=bool(payload.get("execute", False)))
                record_operator_audit(
                    conn,
                    action_type="open",
                    target=str(payload.get("path", "")),
                    parameters_json=json.dumps(payload),
                    executed=result.executed,
                    success=result.allowed and (result.return_code in (None, 0)),
                    return_code=result.return_code,
                    stdout_preview=result.stdout,
                    stderr_preview=result.stderr,
                )
                self.send_json(asdict(result))
            elif path == "/api/operator/run":
                conn = self.state.conn()
                command = payload.get("command", [])
                result = run_command(
                    command,
                    cwd=payload.get("cwd") or None,
                    execute=bool(payload.get("execute", False)),
                    unsafe=bool(payload.get("unsafe", False)),
                    shell=bool(payload.get("shell", False)),
                    timeout=int(payload.get("timeout", 30)),
                )
                record_operator_audit(
                    conn,
                    action_type="run",
                    target=payload.get("cwd") or None,
                    parameters_json=json.dumps(payload),
                    executed=result.executed,
                    success=result.allowed and (result.return_code in (None, 0)),
                    return_code=result.return_code,
                    stdout_preview=result.stdout,
                    stderr_preview=result.stderr,
                )
                self.send_json(asdict(result))
            elif path == "/api/vision/screenshot":
                conn = self.state.conn()
                screenshot = capture_screenshot(
                    payload.get("output_dir", "artifacts/screenshots"),
                    active_window=bool(payload.get("active_window", False)),
                )
                record_operator_audit(
                    conn,
                    action_type="screenshot",
                    target=screenshot.image_path,
                    parameters_json=json.dumps(payload),
                    executed=True,
                    success=True,
                    stdout_preview=json.dumps(asdict(screenshot)),
                )
                response: dict[str, Any] = {"screenshot": asdict(screenshot)}
                if bool(payload.get("analyze", True)) or bool(payload.get("ingest", False)):
                    analysis = analyze_image(
                        screenshot.image_path,
                        prompt=payload.get("prompt") or None,
                        ocr_engine=str(payload.get("ocr_engine", "auto")),
                        lang=str(payload.get("lang", "eng")),
                        vision_model=payload.get("vision_model") or None,
                        use_vlm=bool(payload.get("use_vlm", True)),
                    )
                    response["analysis"] = asdict(analysis)
                    if bool(payload.get("ingest", False)):
                        response["ingest"] = asdict(ingest_visual_analysis(conn, analysis))
                self.send_json(response)
            elif path == "/api/vision/upload":
                conn = self.state.conn()
                image = save_pasted_image(
                    str(payload.get("data_url", "")),
                    payload.get("output_dir", "artifacts/screenshots"),
                    file_name=payload.get("file_name") or None,
                )
                record_operator_audit(
                    conn,
                    action_type="clipboard_paste",
                    target=image.image_path,
                    parameters_json=json.dumps(
                        {
                            "file_name": payload.get("file_name"),
                            "analyze": bool(payload.get("analyze", True)),
                            "ingest": bool(payload.get("ingest", False)),
                        }
                    ),
                    executed=True,
                    success=True,
                    stdout_preview=json.dumps(asdict(image)),
                )
                response = {"image": asdict(image)}
                if bool(payload.get("analyze", True)) or bool(payload.get("ingest", False)):
                    analysis = analyze_image(
                        image.image_path,
                        prompt=payload.get("prompt") or None,
                        ocr_engine=str(payload.get("ocr_engine", "auto")),
                        lang=str(payload.get("lang", "eng")),
                        vision_model=payload.get("vision_model") or None,
                        use_vlm=bool(payload.get("use_vlm", True)),
                    )
                    response["analysis"] = asdict(analysis)
                    if bool(payload.get("ingest", False)):
                        response["ingest"] = asdict(ingest_visual_analysis(conn, analysis))
                self.send_json(response)
            elif path == "/api/vision/analyze":
                conn = self.state.conn()
                analysis = analyze_image(
                    str(payload.get("image", "")),
                    prompt=payload.get("prompt") or None,
                    ocr_engine=str(payload.get("ocr_engine", "auto")),
                    lang=str(payload.get("lang", "eng")),
                    vision_model=payload.get("vision_model") or None,
                    use_vlm=bool(payload.get("use_vlm", True)),
                )
                response = {"analysis": asdict(analysis)}
                if bool(payload.get("ingest", False)):
                    response["ingest"] = asdict(ingest_visual_analysis(conn, analysis))
                record_operator_audit(
                    conn,
                    action_type="vision_analyze",
                    target=analysis.image_path,
                    parameters_json=json.dumps(payload),
                    executed=True,
                    success=True,
                    stdout_preview=analysis.visual_summary,
                )
                self.send_json(response)
            elif path == "/api/vision/ocr":
                conn = self.state.conn()
                result = run_ocr(str(payload.get("image", "")), engine=str(payload.get("engine", "auto")))
                record_operator_audit(
                    conn,
                    action_type="ocr",
                    target=str(payload.get("image", "")),
                    parameters_json=json.dumps(payload),
                    executed=True,
                    success=result.error is None,
                    stdout_preview=result.text,
                    stderr_preview=result.error,
                )
                self.send_json(asdict(result))
            elif path == "/api/imagegen/generate":
                request = ImageGenerationRequest(
                    prompt=str(payload.get("prompt", "")),
                    negative_prompt=str(payload.get("negative_prompt", "")),
                    backend=str(payload.get("backend", "auto")),
                    width=int(payload.get("width", 768)),
                    height=int(payload.get("height", 512)),
                    steps=int(payload.get("steps", 24)),
                    guidance_scale=float(payload.get("guidance_scale", 7.0)),
                    seed=int(payload.get("seed", -1)),
                    model_path=payload.get("model_path") or None,
                    enhance_prompt=bool(payload.get("enhance_prompt", False)),
                )
                result = generate_image(request)
                self.send_json(asdict(result), HTTPStatus.OK if result.success else HTTPStatus.BAD_REQUEST)
            elif path == "/api/investigate":
                conn = self.state.conn()
                roots = payload.get("roots") or []
                if isinstance(roots, str):
                    roots = [item.strip() for item in roots.split(";") if item.strip()]
                result = investigate_subject(
                    conn,
                    str(payload.get("subject", "")),
                    roots=list(roots),
                    top_k=int(payload.get("top_k", 8)),
                    max_file_results=int(payload.get("max_file_results", 40)),
                )
                self.send_json(result)
            elif path == "/api/investigate/validate":
                result = validate_investigation_answer(
                    str(payload.get("answer", "")),
                    list(payload.get("allowed_citations", [])),
                )
                self.send_json(result)
            elif path == "/api/report/generate":
                conn = self.state.conn()
                roots = payload.get("roots") or []
                if isinstance(roots, str):
                    roots = [item.strip() for item in roots.split(";") if item.strip()]
                result = generate_case_report(
                    conn,
                    str(payload.get("subject", "")),
                    roots=list(roots),
                    top_k=int(payload.get("top_k", 8)),
                )
                self.send_json(asdict(result))
            else:
                self.send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - local app should surface tool errors.
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_static(self, path: str) -> None:
        target = APP_DIR / ("index.html" if path in {"", "/"} else path.lstrip("/"))
        resolved = target.resolve()
        if not str(resolved).startswith(str(APP_DIR.resolve())) or not resolved.exists() or resolved.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(str(resolved))
        data = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def serve_artifact(self, raw_path: str) -> None:
        target = Path(raw_path).expanduser().resolve()
        allowed_roots = [(Path.cwd() / "artifacts").resolve(), (Path.cwd() / "exports").resolve()]
        if not any(str(target).startswith(str(root)) for root in allowed_roots) or not target.exists() or target.is_dir():
            self.send_json({"error": "Artifact not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(str(target))
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def run_app(db_path: str | Path = "data/axiom.sqlite", host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    ensure_directories()
    state = AxiomAppState(db_path)
    connect(state.db_path).close()
    selected_port = available_port(host, port)
    server = ThreadingHTTPServer((host, selected_port), AxiomRequestHandler)
    server.state = state  # type: ignore[attr-defined]
    url = f"http://{host}:{selected_port}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"Axiom local app running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def available_port(host: str, preferred: int) -> int:
    for candidate in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    return preferred
