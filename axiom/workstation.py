from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WindowInfo:
    pid: int
    process: str
    title: str
    path: str | None


@dataclass(frozen=True)
class BrowserTabInfo:
    port: int
    browser_context: str
    title: str
    url: str
    tab_type: str


@dataclass(frozen=True)
class FileInfo:
    path: str
    name: str
    extension: str
    size_bytes: int
    modified_at: str
    is_dir: bool


@dataclass(frozen=True)
class CommandResult:
    command: list[str] | str
    cwd: str
    executed: bool
    allowed: bool
    return_code: int | None
    stdout: str
    stderr: str
    reason: str


SAFE_READ_ONLY_COMMANDS = {
    "python": {"--version", "-V"},
    "py": {"--version", "-V"},
    "node": {"--version", "-v"},
    "npm": {"--version", "-v"},
    "git": {"status", "log", "show", "diff", "branch", "remote", "rev-parse"},
    "where": None,
}


def list_windows() -> list[WindowInfo]:
    if platform.system().lower() != "windows":
        return []

    script = r"""
$items = Get-Process | Where-Object { $_.MainWindowTitle -and $_.MainWindowTitle.Trim().Length -gt 0 } | ForEach-Object {
    $p = $null
    try { $p = $_.Path } catch { $p = $null }
    [PSCustomObject]@{
        pid = $_.Id
        process = $_.ProcessName
        title = $_.MainWindowTitle
        path = $p
    }
}
$items | Sort-Object process, pid | ConvertTo-Json -Depth 4
"""
    items = _powershell_json(script, timeout=8)
    return [
        WindowInfo(
            pid=int(item.get("pid", 0)),
            process=str(item.get("process", "")),
            title=str(item.get("title", "")),
            path=item.get("path"),
        )
        for item in items
        if item.get("title")
    ]


def list_browser_tabs(ports: Iterable[int] = (9222, 9223, 9224, 9225)) -> list[BrowserTabInfo]:
    tabs: list[BrowserTabInfo] = []
    for port in ports:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=0.35) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            tabs.append(
                BrowserTabInfo(
                    port=port,
                    browser_context=str(item.get("browserContextId", "")),
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    tab_type=str(item.get("type", "")),
                )
            )
    return tabs


def browser_tab_note() -> str:
    return (
        "Full browser tab URLs require Chrome or Edge to be launched with local DevTools "
        "remote debugging, for example --remote-debugging-port=9222. Without that, Axiom "
        "can still list visible browser window titles."
    )


def scan_folder(root: str | Path, *, max_depth: int = 2, max_items: int = 200) -> list[FileInfo]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(str(base))
    results: list[FileInfo] = []
    for item in _walk_limited(base, max_depth=max_depth):
        try:
            stat = item.stat()
        except OSError:
            continue
        results.append(
            FileInfo(
                path=str(item),
                name=item.name,
                extension=item.suffix.lower(),
                size_bytes=0 if item.is_dir() else stat.st_size,
                modified_at=_format_mtime(stat.st_mtime),
                is_dir=item.is_dir(),
            )
        )
        if len(results) >= max_items:
            break
    return results


def find_files(
    root: str | Path,
    query: str,
    *,
    max_depth: int = 5,
    max_results: int = 100,
    content: bool = False,
    extensions: set[str] | None = None,
) -> list[FileInfo]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(str(base))

    needle = query.lower()
    results: list[FileInfo] = []
    for item in _walk_limited(base, max_depth=max_depth):
        if item.is_dir():
            continue
        suffix = item.suffix.lower()
        if extensions and suffix not in extensions:
            continue
        name_match = needle in item.name.lower()
        content_match = False
        if content and suffix in {".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts", ".tsx", ".html", ".css"}:
            try:
                content_match = needle in item.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                content_match = False
        if not (name_match or content_match):
            continue
        try:
            stat = item.stat()
        except OSError:
            continue
        results.append(
            FileInfo(
                path=str(item),
                name=item.name,
                extension=suffix,
                size_bytes=stat.st_size,
                modified_at=_format_mtime(stat.st_mtime),
                is_dir=False,
            )
        )
        if len(results) >= max_results:
            break
    return results


def open_path(path: str | Path, *, execute: bool = False) -> CommandResult:
    target = str(Path(path).expanduser().resolve())
    if not execute:
        return CommandResult(
            command=["open", target],
            cwd=str(Path.cwd()),
            executed=False,
            allowed=True,
            return_code=None,
            stdout=f"Dry run: would open {target}",
            stderr="",
            reason="Pass --execute to open the path.",
        )
    if not Path(target).exists():
        return CommandResult(
            command=["open", target],
            cwd=str(Path.cwd()),
            executed=True,
            allowed=False,
            return_code=None,
            stdout="",
            stderr=f"Path does not exist: {target}",
            reason="Missing path.",
        )
    try:
        if platform.system().lower() == "windows":
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except OSError as exc:
        return CommandResult(
            command=["open", target],
            cwd=str(Path.cwd()),
            executed=True,
            allowed=True,
            return_code=None,
            stdout="",
            stderr=str(exc),
            reason="Open request failed.",
        )
    return CommandResult(
        command=["open", target],
        cwd=str(Path.cwd()),
        executed=True,
        allowed=True,
        return_code=0,
        stdout=f"Opened {target}",
        stderr="",
        reason="Path opened by the operating system.",
    )


def run_command(
    command: list[str] | str,
    *,
    cwd: str | Path | None = None,
    execute: bool = False,
    unsafe: bool = False,
    shell: bool = False,
    timeout: int = 30,
) -> CommandResult:
    workdir = str(Path(cwd or Path.cwd()).expanduser().resolve())
    allowed, reason = command_allowed(command, unsafe=unsafe, shell=shell)
    if not execute:
        return CommandResult(
            command=command,
            cwd=workdir,
            executed=False,
            allowed=allowed,
            return_code=None,
            stdout=f"Dry run: would run {command}",
            stderr="",
            reason=reason if allowed else f"Blocked unless --unsafe is supplied: {reason}",
        )
    if not allowed:
        return CommandResult(
            command=command,
            cwd=workdir,
            executed=False,
            allowed=False,
            return_code=None,
            stdout="",
            stderr=reason,
            reason="Command blocked by operator policy.",
        )

    try:
        if shell:
            completed = subprocess.run(
                str(command),
                cwd=workdir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            completed = subprocess.run(
                [str(part) for part in command],
                cwd=workdir,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(
            command=command,
            cwd=workdir,
            executed=True,
            allowed=True,
            return_code=None,
            stdout="",
            stderr=str(exc),
            reason="Command failed before completion.",
        )

    return CommandResult(
        command=command,
        cwd=workdir,
        executed=True,
        allowed=True,
        return_code=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        reason="Command completed.",
    )


def command_allowed(command: list[str] | str, *, unsafe: bool, shell: bool) -> tuple[bool, str]:
    if unsafe:
        return True, "Explicit unsafe override supplied."
    if shell:
        return False, "Shell commands require --unsafe."
    if not isinstance(command, list) or not command:
        return False, "Command must be a non-empty argument list."
    executable = Path(str(command[0])).name.lower()
    if executable.endswith(".exe"):
        executable = executable[:-4]
    allowed_subcommands = SAFE_READ_ONLY_COMMANDS.get(executable)
    if executable not in SAFE_READ_ONLY_COMMANDS:
        return False, f"Executable is not in the read-only allow-list: {executable}"
    if allowed_subcommands is None:
        return True, "Executable is allow-listed."
    if len(command) == 1:
        return False, f"{executable} requires an allowed read-only argument."
    first_arg = str(command[1])
    if first_arg in allowed_subcommands:
        return True, "Read-only command is allow-listed."
    return False, f"Argument is not allow-listed for {executable}: {first_arg}"


def plan_operator_task(text: str) -> dict[str, object]:
    normalized = text.strip()
    lower = normalized.lower()
    if "window" in lower or "open app" in lower:
        return {"action": "windows", "confidence": 0.82, "reason": "The task asks for visible windows or apps."}
    if "tab" in lower or "browser" in lower:
        return {"action": "tabs", "confidence": 0.78, "reason": "The task asks for browser tab visibility."}
    if lower.startswith("open "):
        return {"action": "open", "target": normalized[5:].strip(), "confidence": 0.74}
    match = re.search(r"find\s+(.+?)\s+in\s+(.+)$", normalized, flags=re.IGNORECASE)
    if match:
        return {
            "action": "find",
            "query": match.group(1).strip(),
            "root": match.group(2).strip(),
            "confidence": 0.7,
        }
    return {
        "action": "manual_review",
        "confidence": 0.25,
        "reason": "No deterministic local operator pattern matched this task.",
    }


def payload_list(items: Iterable[object]) -> list[dict[str, object]]:
    return [asdict(item) for item in items]


def _powershell_json(script: str, *, timeout: int) -> list[dict[str, object]]:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    data = json.loads(completed.stdout)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _walk_limited(root: Path, *, max_depth: int) -> Iterable[Path]:
    root_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [name for name in dirs if not name.startswith(".") and name not in {"__pycache__", ".git"}]
        for dirname in dirs:
            yield current_path / dirname
        for filename in files:
            yield current_path / filename


def _format_mtime(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")
