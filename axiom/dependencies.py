from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DependencyCheck:
    key: str
    label: str
    category: str
    required: bool
    installed: bool
    version: str | None
    install_plan: list[list[str]]
    note: str


@dataclass(frozen=True)
class InstallResult:
    key: str
    executed: bool
    success: bool
    command: list[str]
    return_code: int | None
    stdout: str
    stderr: str
    note: str


PYTHON_DEPS = {
    "pillow": ("PIL", "Pillow", "Python imaging and screenshot fallback."),
    "mss": ("mss", "mss", "Fast multi-monitor screenshot capture."),
    "pytesseract": ("pytesseract", "pytesseract", "Python bridge for the Tesseract OCR binary."),
    "paddleocr": ("paddleocr", "paddleocr", "Production OCR engine for screenshots and scanned images."),
    "pymupdf": ("fitz", "PyMuPDF", "Primary local PDF text extraction adapter."),
    "pypdf": ("pypdf", "pypdf", "Lightweight fallback PDF text extraction adapter."),
    "python-docx": ("docx", "python-docx", "DOCX paragraph and table extraction adapter."),
    "fastapi": ("fastapi", "fastapi", "Optional API server for integrations."),
    "uvicorn": ("uvicorn", "uvicorn", "Optional API server runner."),
    "diffusers": ("diffusers", "diffusers", "Embedded Stable Diffusion pipeline support."),
    "torch": ("torch", "torch", "Local tensor runtime for embedded image generation."),
    "transformers": ("transformers", "transformers", "Model loader support for Diffusers pipelines."),
    "accelerate": ("accelerate", "accelerate", "Efficient local model loading and execution."),
    "safetensors": ("safetensors", "safetensors", "Safe local checkpoint loading."),
}

SYSTEM_DEPS = {
    "tesseract": ("tesseract", "Tesseract OCR binary", "Required fallback OCR executable."),
    "ollama": ("ollama", "Ollama runtime", "Required for local image understanding through a VLM."),
}

VISION_MODEL_KEY = "llama3.2-vision"


def audit_dependencies() -> dict[str, object]:
    checks: list[DependencyCheck] = []
    for key, (module_name, label, note) in PYTHON_DEPS.items():
        checks.append(check_python_dep(key, module_name, label, note, required=key in {"pillow", "mss", "pytesseract", "paddleocr"}))
    for key, (binary, label, note) in SYSTEM_DEPS.items():
        checks.append(check_binary_dep(key, binary, label, note, required=True))
    checks.append(check_ollama_model(VISION_MODEL_KEY, required=True))

    required = [item for item in checks if item.required]
    installed_required = [item for item in required if item.installed]
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "ready": len(installed_required) == len(required),
        "required_total": len(required),
        "required_installed": len(installed_required),
        "checks": [asdict(item) for item in checks],
    }


def check_python_dep(key: str, module_name: str, label: str, note: str, *, required: bool) -> DependencyCheck:
    spec = importlib.util.find_spec(module_name)
    version = None
    if spec is not None:
        try:
            version = importlib.metadata.version(package_for_key(key))
        except importlib.metadata.PackageNotFoundError:
            version = None
    return DependencyCheck(
        key=key,
        label=label,
        category="python",
        required=required,
        installed=spec is not None,
        version=version,
        install_plan=[[sys.executable, "-m", "pip", "install", package_for_key(key)]],
        note=note,
    )


def check_binary_dep(key: str, binary: str, label: str, note: str, *, required: bool) -> DependencyCheck:
    path = resolve_binary(binary)
    install_plan = binary_install_plan(binary)
    return DependencyCheck(
        key=key,
        label=label,
        category="system",
        required=required,
        installed=path is not None,
        version=path,
        install_plan=install_plan,
        note=note,
    )


def resolve_binary(binary: str) -> str | None:
    path = shutil.which(binary)
    if path:
        return path
    for candidate in common_binary_paths(binary):
        if candidate.exists():
            return str(candidate)
    return None


def common_binary_paths(binary: str) -> list[Path]:
    if platform.system().lower() != "windows":
        return []
    program_files = [Path(os.environ.get("ProgramFiles", r"C:\Program Files"))]
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        program_files.append(Path(program_files_x86))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    if binary == "tesseract":
        return [root / "Tesseract-OCR" / "tesseract.exe" for root in program_files]
    if binary == "ollama":
        return [
            local_app_data / "Programs" / "Ollama" / "ollama.exe",
            Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        ]
    return []


def check_ollama_model(model: str, *, required: bool) -> DependencyCheck:
    installed = False
    note = "Required local vision model for screenshot/image understanding."
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.6) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        models = data.get("models", []) if isinstance(data, dict) else []
        installed = any(str(item.get("name", "")).split(":")[0] == model for item in models if isinstance(item, dict))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        note = "Ollama is not running or no local model list is exposed."
    return DependencyCheck(
        key=f"model:{model}",
        label=model,
        category="model",
        required=required,
        installed=installed,
        version=model if installed else None,
        install_plan=[[resolve_binary("ollama") or "ollama", "pull", model]],
        note=note,
    )


def package_for_key(key: str) -> str:
    if key == "pillow":
        return "Pillow"
    if key == "pymupdf":
        return "PyMuPDF"
    return key


def binary_install_plan(binary: str) -> list[list[str]]:
    if platform.system().lower() == "windows":
        if binary == "tesseract":
            return [["winget", "install", "--id", "UB-Mannheim.TesseractOCR", "-e"]]
        if binary == "ollama":
            return [["winget", "install", "--id", "Ollama.Ollama", "-e"]]
    return []


def install_dependency(key: str, *, execute: bool = False, timeout: int = 900) -> InstallResult:
    check = dependency_by_key(key)
    if check is None:
        return InstallResult(key, False, False, [], None, "", f"Unknown dependency: {key}", "Unknown dependency.")
    if not check.install_plan:
        return InstallResult(
            key,
            False,
            False,
            [],
            None,
            "",
            "No automatic installer is available for this dependency.",
            "Install manually, then refresh setup.",
        )
    command = check.install_plan[0]
    if not execute:
        return InstallResult(key, False, True, command, None, "Dry run only.", "", "Set execute=true to install.")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return InstallResult(key, True, False, command, None, "", str(exc), "Installer failed before completion.")
    return InstallResult(
        key=key,
        executed=True,
        success=completed.returncode == 0,
        command=command,
        return_code=completed.returncode,
        stdout=completed.stdout.strip()[-4000:],
        stderr=completed.stderr.strip()[-4000:],
        note="Installer completed." if completed.returncode == 0 else "Installer returned an error.",
    )


def dependency_by_key(key: str) -> DependencyCheck | None:
    if key in PYTHON_DEPS:
        module_name, label, note = PYTHON_DEPS[key]
        return check_python_dep(key, module_name, label, note, required=key in {"pillow", "mss", "pytesseract", "paddleocr"})
    if key in SYSTEM_DEPS:
        binary, label, note = SYSTEM_DEPS[key]
        return check_binary_dep(key, binary, label, note, required=True)
    if key == f"model:{VISION_MODEL_KEY}":
        return check_ollama_model(VISION_MODEL_KEY, required=True)
    return None


def ensure_directories() -> None:
    for directory in ("data", "artifacts", "artifacts/screenshots", "intake", "exports"):
        Path(directory).mkdir(parents=True, exist_ok=True)
