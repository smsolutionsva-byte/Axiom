from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class UploadedFile:
    original_name: str
    relative_path: str
    saved_path: str
    mime_type: str
    size: int


@dataclass(frozen=True)
class UploadBatch:
    root_path: str
    files: list[UploadedFile]


def save_uploaded_files(
    files: list[dict[str, object]],
    output_dir: str | Path = "intake/uploads",
    *,
    batch_name: str | None = None,
    max_file_bytes: int = 25 * 1024 * 1024,
    max_total_bytes: int = 100 * 1024 * 1024,
) -> UploadBatch:
    if not files:
        raise ValueError("No files were supplied.")

    root = Path(output_dir).expanduser().resolve() / safe_segment(
        batch_name or f"upload-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}"
    )
    root.mkdir(parents=True, exist_ok=True)

    saved: list[UploadedFile] = []
    total = 0
    used_paths: set[Path] = set()
    for index, item in enumerate(files, start=1):
        data_url = str(item.get("data_url", ""))
        mime_type, raw = decode_data_url(data_url)
        if len(raw) > max_file_bytes:
            raise ValueError(f"{item.get('name') or 'File'} is larger than {max_file_bytes // (1024 * 1024)} MB.")
        total += len(raw)
        if total > max_total_bytes:
            raise ValueError(f"Upload batch is larger than {max_total_bytes // (1024 * 1024)} MB.")

        original_name = str(item.get("name") or f"upload-{index}")
        relative_path = safe_relative_path(str(item.get("relative_path") or original_name), mime_type, fallback=f"upload-{index}")
        target = unique_target(root, relative_path, used_paths)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        used_paths.add(target.relative_to(root))
        saved.append(
            UploadedFile(
                original_name=original_name,
                relative_path=str(target.relative_to(root)),
                saved_path=str(target),
                mime_type=mime_type,
                size=len(raw),
            )
        )

    return UploadBatch(root_path=str(root), files=saved)


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.startswith("data:") or ";base64" not in header:
        raise ValueError("Expected a base64 data URL.")
    mime_type = header[5:].split(";", 1)[0].lower() or "application/octet-stream"
    try:
        return mime_type, base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Uploaded file data was not valid base64.") from exc


def safe_relative_path(value: str, mime_type: str, *, fallback: str) -> Path:
    parts = []
    for part in value.replace("\\", "/").split("/"):
        if part.strip() in {"", ".", ".."}:
            continue
        segment = safe_segment(part)
        if segment:
            parts.append(segment)
    if not parts:
        parts = [safe_segment(fallback)]
    leaf = Path(parts[-1])
    if not leaf.suffix:
        extension = mimetypes.guess_extension(mime_type) or ""
        if extension:
            parts[-1] = parts[-1] + extension
    return Path(*parts)


def safe_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "-" for ch in str(value)).strip(" .")
    return cleaned or "upload"


def unique_target(root: Path, relative_path: Path, used_paths: set[Path]) -> Path:
    candidate = root / relative_path
    candidate = candidate.resolve()
    candidate.relative_to(root)
    rel = candidate.relative_to(root)
    if rel not in used_paths and not candidate.exists():
        return candidate

    stem = candidate.stem or "upload"
    suffix = candidate.suffix
    parent = candidate.parent
    for index in range(2, 10_000):
        next_candidate = (parent / f"{stem}-{index}{suffix}").resolve()
        next_candidate.relative_to(root)
        next_rel = next_candidate.relative_to(root)
        if next_rel not in used_paths and not next_candidate.exists():
            return next_candidate
    raise ValueError("Could not choose a unique upload file name.")
