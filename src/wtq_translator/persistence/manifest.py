from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..prompts import PROMPT_SET_VERSION, prompt_snapshot


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sanitized_config(config: AppConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    if "model" in data and "api_key" in data["model"]:
        data["model"]["api_key"] = "***REDACTED***"
    if "embeddings" in data and data["embeddings"].get("api_key") is not None:
        data["embeddings"]["api_key"] = "***REDACTED***"
    return data


def write_manifest(path: str | Path, *, run_id: str, config: AppConfig, extra: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.project.seed,
        "provider": config.model.provider,
        "model": config.model.model_name,
        "base_url": config.model.base_url,
        "prompt_version": PROMPT_SET_VERSION,
        "prompts": prompt_snapshot(),
        "generation": {
            "temperature": config.model.temperature,
            "max_output_tokens": config.model.max_output_tokens,
            "retries": config.model.retries,
        },
        "config": _sanitized_config(config),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "libraries": {
            "pydantic": _version("pydantic"),
            "PyYAML": _version("PyYAML"),
            "httpx": _version("httpx"),
        },
    }
    if extra:
        data.update(extra)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
