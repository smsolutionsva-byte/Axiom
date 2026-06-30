from __future__ import annotations

import base64
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: str = ""
    backend: str = "auto"
    width: int = 768
    height: int = 512
    steps: int = 24
    guidance_scale: float = 7.0
    seed: int = -1
    model_path: str | None = None
    output_dir: str = "artifacts/generated"
    enhance_prompt: bool = False


@dataclass(frozen=True)
class ImageGenerationResult:
    success: bool
    backend: str
    image_path: str | None
    metadata_path: str | None
    prompt: str
    negative_prompt: str
    seed: int
    elapsed_seconds: float
    error: str | None = None


A1111_URL = os.environ.get("AXIOM_A1111_URL", "http://127.0.0.1:7860")


def image_generation_status(model_path: str | None = None) -> dict[str, object]:
    a1111 = check_a1111()
    diffusers = check_diffusers()
    model = configured_model_path(model_path)
    ready = bool(a1111["ready"] or (diffusers["ready"] and model["exists"]))
    return {
        "automatic1111": a1111,
        "diffusers": diffusers,
        "model_path": model,
        "ready": ready,
        "note": (
            "Image generation needs a Stable Diffusion backend. "
            "llama3.2-vision is for understanding images/screenshots, not creating images."
        ),
        "next_steps": image_backend_next_steps(a1111, diffusers, model),
    }


def generate_image(request: ImageGenerationRequest) -> ImageGenerationResult:
    started = time.time()
    prompt = request.prompt.strip()
    if request.enhance_prompt:
        prompt = enhance_prompt(prompt) or prompt
    if not prompt:
        return fail_result(request, "No prompt supplied.", started)

    backend = request.backend.lower()
    if backend == "auto":
        if check_a1111()["ready"]:
            backend = "a1111"
        elif check_diffusers()["ready"] and configured_model_path(request.model_path)["exists"]:
            backend = "diffusers"
        else:
            return fail_result(
                request,
                "No offline image backend is ready. Start AUTOMATIC1111 with --api or set AXIOM_DIFFUSION_MODEL to a local Diffusers model folder.",
                started,
            )

    if backend in {"a1111", "automatic1111", "forge"}:
        return generate_a1111(request, prompt=prompt, started=started)
    if backend == "diffusers":
        return generate_diffusers(request, prompt=prompt, started=started)
    return fail_result(request, f"Unknown backend: {request.backend}", started)


def generate_a1111(request: ImageGenerationRequest, *, prompt: str, started: float) -> ImageGenerationResult:
    payload = {
        "prompt": prompt,
        "negative_prompt": request.negative_prompt,
        "steps": request.steps,
        "width": request.width,
        "height": request.height,
        "cfg_scale": request.guidance_scale,
        "seed": request.seed,
    }
    try:
        data = post_json(f"{A1111_URL}/sdapi/v1/txt2img", payload, timeout=900)
    except Exception as exc:  # noqa: BLE001 - local backend errors should surface.
        return fail_result(request, f"AUTOMATIC1111 generation failed: {exc}", started, backend="a1111", prompt=prompt)

    images = data.get("images", []) if isinstance(data, dict) else []
    if not images:
        return fail_result(request, "AUTOMATIC1111 returned no image.", started, backend="a1111", prompt=prompt)

    output = output_paths(request.output_dir)
    raw = images[0].split(",", 1)[-1]
    output["image"].write_bytes(base64.b64decode(raw))
    metadata = {
        "backend": "a1111",
        "request": asdict(request),
        "prompt_used": prompt,
        "response_parameters": data.get("parameters") if isinstance(data, dict) else None,
        "info": safe_json(data.get("info")) if isinstance(data, dict) else None,
        "created_at": utc_now(),
    }
    output["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    seed = extract_seed(metadata.get("info"), request.seed)
    return ImageGenerationResult(True, "a1111", str(output["image"]), str(output["metadata"]), prompt, request.negative_prompt, seed, round(time.time() - started, 2))


def generate_diffusers(request: ImageGenerationRequest, *, prompt: str, started: float) -> ImageGenerationResult:
    model = configured_model_path(request.model_path)
    if not model["exists"]:
        return fail_result(request, "Diffusers requires a local model folder path. Set AXIOM_DIFFUSION_MODEL or pass model_path.", started, backend="diffusers", prompt=prompt)
    try:
        import torch  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as exc:
        return fail_result(request, f"Diffusers dependencies are missing: {exc}", started, backend="diffusers", prompt=prompt)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = DiffusionPipeline.from_pretrained(str(model["path"]), torch_dtype=dtype, local_files_only=True)
        pipe = pipe.to(device)
        generator = None
        if request.seed >= 0:
            generator = torch.Generator(device=device).manual_seed(request.seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        ).images[0]
    except Exception as exc:  # noqa: BLE001 - local model errors should surface.
        return fail_result(request, f"Diffusers generation failed: {exc}", started, backend="diffusers", prompt=prompt)

    output = output_paths(request.output_dir)
    image.save(output["image"])
    metadata = {
        "backend": "diffusers",
        "request": asdict(request),
        "prompt_used": prompt,
        "model_path": model["path"],
        "created_at": utc_now(),
    }
    output["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ImageGenerationResult(True, "diffusers", str(output["image"]), str(output["metadata"]), prompt, request.negative_prompt, request.seed, round(time.time() - started, 2))


def check_a1111() -> dict[str, object]:
    try:
        data = get_json(f"{A1111_URL}/sdapi/v1/options", timeout=0.8)
        model = data.get("sd_model_checkpoint") if isinstance(data, dict) else None
        return {"ready": True, "url": A1111_URL, "model": model}
    except Exception:
        return {"ready": False, "url": A1111_URL, "model": None, "note": "Start Stable Diffusion WebUI/Forge with --api."}


def check_diffusers() -> dict[str, object]:
    missing: list[str] = []
    for module in ("torch", "diffusers", "transformers", "accelerate"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return {"ready": not missing, "missing": missing}


def configured_model_path(value: str | None = None) -> dict[str, object]:
    raw = value or os.environ.get("AXIOM_DIFFUSION_MODEL", "")
    if not raw:
        return {"path": None, "exists": False}
    path = Path(raw).expanduser().resolve()
    return {"path": str(path), "exists": path.exists()}


def image_backend_next_steps(a1111: dict[str, object], diffusers: dict[str, object], model: dict[str, object]) -> list[str]:
    steps: list[str] = []
    if not a1111["ready"]:
        steps.append("Start AUTOMATIC1111/Forge locally with --api at http://127.0.0.1:7860.")
    if diffusers["ready"] and not model["exists"]:
        steps.append("Set AXIOM_DIFFUSION_MODEL or enter a local Stable Diffusion model folder in Image Lab.")
    if not diffusers["ready"]:
        missing = ", ".join(str(item) for item in diffusers.get("missing", []))
        steps.append(f"Install missing Diffusers packages: {missing}.")
    return steps


def enhance_prompt(prompt: str) -> str | None:
    model = os.environ.get("AXIOM_PROMPT_MODEL") or os.environ.get("AXIOM_OLLAMA_MODEL")
    if not model:
        return None
    payload = {
        "model": model,
        "prompt": (
            "Rewrite this into a concise, high-quality Stable Diffusion prompt. "
            "Do not add unsafe content. Return only the prompt.\n\n"
            f"{prompt}"
        ),
        "stream": False,
    }
    try:
        data = post_json(os.environ.get("AXIOM_OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate"), payload, timeout=45)
    except Exception:
        return None
    value = str(data.get("response", "")).strip() if isinstance(data, dict) else ""
    return value[:1200] or None


def output_paths(output_dir: str) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return {
        "image": root / f"image-{stamp}.png",
        "metadata": root / f"image-{stamp}.json",
    }


def fail_result(request: ImageGenerationRequest, error: str, started: float, *, backend: str | None = None, prompt: str | None = None) -> ImageGenerationResult:
    return ImageGenerationResult(False, backend or request.backend, None, None, prompt or request.prompt, request.negative_prompt, request.seed, round(time.time() - started, 2), error)


def get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def safe_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def extract_seed(info: object, fallback: int) -> int:
    if isinstance(info, dict):
        for key in ("seed", "all_seeds"):
            value = info.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, list) and value and isinstance(value[0], int):
                return value[0]
    return fallback


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
