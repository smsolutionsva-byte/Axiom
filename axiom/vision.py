from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .dependencies import resolve_binary
from .ingestion import ingest_path, sha256_file


@dataclass(frozen=True)
class OCRResult:
    engine: str
    text: str
    confidence: float | None
    blocks: list[dict[str, object]]
    error: str | None = None


@dataclass(frozen=True)
class ScreenshotResult:
    image_path: str
    capture_method: str
    active_window: bool
    width: int | None
    height: int | None


@dataclass(frozen=True)
class VisionAnalysis:
    image_path: str
    sha256: str
    width: int | None
    height: int | None
    ocr: OCRResult
    visual_summary: str
    model: str | None
    prompt: str
    sidecars: dict[str, str]


DEFAULT_VISION_MODEL = "llama3.2-vision"
DEFAULT_VISION_PROMPT = (
    "Analyze this screenshot or image for an offline intelligence workstation. "
    "Use the visible image and the OCR text together. Identify UI state, document type, "
    "important entities, visible warnings/errors, and useful next operator actions. "
    "Do not invent text that is not visible or supplied by OCR."
)

PASTED_IMAGE_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "image/gif": "gif",
}


def capture_screenshot(
    output_dir: str | Path = "artifacts/screenshots",
    *,
    active_window: bool = False,
    file_name: str | None = None,
) -> ScreenshotResult:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = file_name or f"screenshot-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.png"
    output_path = directory / name

    methods = []
    if active_window and platform.system().lower() == "windows":
        methods.append(_capture_windows_powershell)
    methods.extend([_capture_mss, _capture_pillow])
    if platform.system().lower() == "windows":
        methods.append(_capture_windows_powershell)

    errors: list[str] = []
    for method in methods:
        try:
            capture_method = method(output_path, active_window=active_window)
        except Exception as exc:  # noqa: BLE001 - capture fallbacks should keep trying.
            errors.append(f"{method.__name__}: {exc}")
            continue
        width, height = image_dimensions(output_path)
        return ScreenshotResult(
            image_path=str(output_path),
            capture_method=capture_method,
            active_window=active_window,
            width=width,
            height=height,
        )

    raise RuntimeError("Screenshot capture failed. " + " | ".join(errors))


def save_pasted_image(
    data_url: str,
    output_dir: str | Path = "artifacts/screenshots",
    *,
    file_name: str | None = None,
    max_bytes: int = 25 * 1024 * 1024,
) -> ScreenshotResult:
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.startswith("data:") or ";base64" not in header:
        raise ValueError("Expected a base64 image data URL from the clipboard.")

    mime_type = header[5:].split(";", 1)[0].lower()
    extension = PASTED_IMAGE_TYPES.get(mime_type)
    if extension is None:
        supported = ", ".join(sorted(PASTED_IMAGE_TYPES))
        raise ValueError(f"Unsupported clipboard image type: {mime_type}. Supported types: {supported}.")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Clipboard image data was not valid base64.") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"Clipboard image is too large. Limit is {max_bytes // (1024 * 1024)} MB.")

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = _safe_pasted_file_name(file_name, extension)
    output_path = directory / name
    output_path.write_bytes(raw)
    width, height = image_dimensions(output_path)
    return ScreenshotResult(
        image_path=str(output_path),
        capture_method="clipboard-paste",
        active_window=False,
        width=width,
        height=height,
    )


def analyze_image(
    image_path: str | Path,
    *,
    prompt: str | None = None,
    ocr_engine: str = "auto",
    lang: str = "eng",
    vision_model: str | None = None,
    use_vlm: bool = True,
    timeout: float = 45.0,
) -> VisionAnalysis:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    ocr = run_ocr(path, engine=ocr_engine, lang=lang)
    selected_prompt = prompt or DEFAULT_VISION_PROMPT
    model = vision_model
    if model is None:
        model = os.environ.get("AXIOM_VISION_MODEL", DEFAULT_VISION_MODEL)

    visual_summary = ""
    if use_vlm and model:
        visual_summary = call_ollama_vision(
            path,
            model=model,
            prompt=build_vision_prompt(selected_prompt, ocr.text),
            timeout=timeout,
        )
    if not visual_summary:
        visual_summary = fallback_visual_summary(ocr)
        model = None

    width, height = image_dimensions(path)
    sidecars = write_analysis_sidecars(path, ocr, visual_summary)
    return VisionAnalysis(
        image_path=str(path),
        sha256=sha256_file(path),
        width=width,
        height=height,
        ocr=ocr,
        visual_summary=visual_summary,
        model=model,
        prompt=selected_prompt,
        sidecars=sidecars,
    )


def ingest_visual_analysis(conn, analysis: VisionAnalysis, *, build_links: bool = True):
    return ingest_path(conn, analysis.image_path, build_links=build_links)


def run_ocr(image_path: str | Path, *, engine: str = "auto", lang: str = "eng") -> OCRResult:
    path = Path(image_path).expanduser().resolve()
    sidecar_text = read_ocr_sidecar(path)
    if sidecar_text:
        return OCRResult(engine="sidecar", text=sidecar_text, confidence=None, blocks=[])

    normalized = engine.lower()
    if normalized in {"auto", "paddle", "paddleocr"}:
        result = try_paddleocr(path, lang=lang)
        if result and result.text.strip():
            return result
        if normalized in {"paddle", "paddleocr"}:
            return result or OCRResult("paddleocr", "", None, [], "PaddleOCR is not available or returned no text.")

    if normalized in {"auto", "tesseract"}:
        result = try_tesseract_cli(path, lang=lang) or try_pytesseract(path, lang=lang)
        if result and result.text.strip():
            return result
        if normalized == "tesseract":
            return result or OCRResult("tesseract", "", None, [], "Tesseract is not available or returned no text.")

    if normalized == "none":
        return OCRResult("none", "", None, [])

    return OCRResult(
        engine="none",
        text="",
        confidence=None,
        blocks=[],
        error="No OCR engine was available. Install PaddleOCR for best results or Tesseract for a lightweight fallback.",
    )


def try_paddleocr(path: Path, *, lang: str) -> OCRResult | None:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None

    try:
        paddle_lang = "en" if lang in {"eng", "en"} else lang
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
        except TypeError:
            ocr = PaddleOCR(lang=paddle_lang)
        raw = ocr.ocr(str(path), cls=True)
    except Exception as exc:  # noqa: BLE001 - optional adapter should fail softly.
        return OCRResult("paddleocr", "", None, [], str(exc))

    blocks: list[dict[str, object]] = []
    texts: list[str] = []
    confidences: list[float] = []
    for page in raw or []:
        if not page:
            continue
        for item in page:
            if not item or len(item) < 2:
                continue
            box = item[0]
            rec = item[1]
            text = str(rec[0]) if isinstance(rec, (list, tuple)) and rec else ""
            confidence = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else None
            if not text.strip():
                continue
            texts.append(text.strip())
            if confidence is not None:
                confidences.append(confidence)
            blocks.append({"text": text.strip(), "confidence": confidence, "box": box})
    average = sum(confidences) / len(confidences) if confidences else None
    return OCRResult("paddleocr", "\n".join(texts), average, blocks)


def try_tesseract_cli(path: Path, *, lang: str) -> OCRResult | None:
    executable = resolve_binary("tesseract")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, str(path), "stdout", "-l", lang, "--oem", "1", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return OCRResult("tesseract", "", None, [], str(exc))
    if completed.returncode != 0:
        return OCRResult("tesseract", "", None, [], completed.stderr.strip())
    return OCRResult("tesseract", completed.stdout.strip(), None, [])


def try_pytesseract(path: Path, *, lang: str) -> OCRResult | None:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return None
    try:
        executable = resolve_binary("tesseract")
        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
        text = pytesseract.image_to_string(Image.open(path), lang=lang).strip()
    except Exception as exc:  # noqa: BLE001 - optional adapter should fail softly.
        return OCRResult("pytesseract", "", None, [], str(exc))
    return OCRResult("pytesseract", text, None, [])


def call_ollama_vision(path: Path, *, model: str, prompt: str, timeout: float) -> str:
    url = os.environ.get("AXIOM_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return ""
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(data.get("response", "")).strip() if isinstance(data, dict) else ""


def build_vision_prompt(prompt: str, ocr_text: str) -> str:
    ocr_section = ocr_text.strip() or "No OCR text was extracted."
    return f"""{prompt}

OCR TEXT:
{ocr_section}

Return concise sections:
1. Visual Summary
2. OCR Interpretation
3. Entities and UI Elements
4. Suggested Operator Actions
5. Uncertainty
"""


def fallback_visual_summary(ocr: OCRResult) -> str:
    if ocr.text.strip():
        compact = " ".join(ocr.text.split())
        return (
            "OCR-only visual analysis: a local vision model was not available, but OCR extracted "
            f"this visible text: {compact[:1200]}"
        )
    return (
        "No local vision model or OCR text was available. Capture succeeded, but image understanding "
        "requires PaddleOCR/Tesseract and an offline VLM such as llama3.2-vision through Ollama."
    )


def write_analysis_sidecars(path: Path, ocr: OCRResult, visual_summary: str) -> dict[str, str]:
    sidecars = {
        "ocr": str(path) + ".ocr.txt",
        "caption": str(path) + ".caption.txt",
        "analysis": str(path) + ".analysis.json",
    }
    Path(sidecars["ocr"]).write_text(ocr.text, encoding="utf-8")
    Path(sidecars["caption"]).write_text(visual_summary, encoding="utf-8")
    Path(sidecars["analysis"]).write_text(
        json.dumps(
            {
                "image_path": str(path),
                "ocr": asdict(ocr),
                "visual_summary": visual_summary,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sidecars


def read_ocr_sidecar(path: Path) -> str | None:
    candidates = [Path(str(path) + ".ocr.txt"), path.with_suffix(".ocr.txt")]
    for candidate in candidates:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text
    return None


def _safe_pasted_file_name(file_name: str | None, extension: str) -> str:
    fallback = f"paste-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.{extension}"
    raw = Path(file_name or fallback).name.strip() or fallback
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw)
    suffix = Path(safe).suffix.lower().lstrip(".")
    if suffix not in set(PASTED_IMAGE_TYPES.values()):
        safe = f"{Path(safe).stem or 'paste'}.{extension}"
    return safe


def image_dimensions(path: str | Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None, None
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None, None


def _capture_mss(output_path: Path, *, active_window: bool) -> str:
    if active_window:
        raise RuntimeError("MSS captures monitors, not foreground windows.")
    from mss import mss  # type: ignore

    with mss() as sct:
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        from PIL import Image  # type: ignore

        image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        image.save(output_path)
    return "mss"


def _capture_pillow(output_path: Path, *, active_window: bool) -> str:
    if active_window:
        raise RuntimeError("Pillow fallback captures screens, not foreground windows.")
    from PIL import ImageGrab  # type: ignore

    image = ImageGrab.grab(all_screens=True)
    image.save(output_path)
    return "pillow-imagegrab"


def _capture_windows_powershell(output_path: Path, *, active_window: bool) -> str:
    if platform.system().lower() != "windows":
        raise RuntimeError("Windows PowerShell capture is only available on Windows.")
    mode = "active" if active_window else "desktop"
    script = _windows_capture_script(str(output_path), active_window=active_window)
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if not output_path.exists():
        raise RuntimeError("PowerShell capture completed but did not create the image.")
    return f"windows-powershell-{mode}"


def _windows_capture_script(output_path: str, *, active_window: bool) -> str:
    escaped = output_path.replace("'", "''")
    if active_window:
        target_block = r"""
$code = @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
}
public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
"@
Add-Type $code
$handle = [Win32]::GetForegroundWindow()
$rect = New-Object RECT
[void][Win32]::GetWindowRect($handle, [ref]$rect)
$left = $rect.Left
$top = $rect.Top
$width = [Math]::Max(1, $rect.Right - $rect.Left)
$height = [Math]::Max(1, $rect.Bottom - $rect.Top)
"""
    else:
        target_block = r"""
$bounds = [System.Windows.Forms.SystemInformation]::VirtualScreen
$left = $bounds.Left
$top = $bounds.Top
$width = $bounds.Width
$height = $bounds.Height
"""
    return f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
{target_block}
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($left, $top, 0, 0, $bitmap.Size)
$bitmap.Save('{escaped}', [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
"""
