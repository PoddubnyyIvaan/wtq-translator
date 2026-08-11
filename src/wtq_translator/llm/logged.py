from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import LLMProvider
from .capabilities import ProviderCapabilities
from .tracing import current_llm_trace_context


class LoggedLLMProvider:
    """Append every logical LLM request and response to JSONL.

    The wrapper should sit OUTSIDE the cache wrapper. That way cached responses are
    still visible in the per-run interaction log, which is useful for reproducing
    and debugging a question even when no HTTP call was made during that run.
    """

    def __init__(
        self,
        wrapped: LLMProvider,
        path: str | Path,
        *,
        identity: dict[str, Any] | None = None,
    ) -> None:
        self.wrapped = wrapped
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = identity or {}

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self.wrapped.capabilities

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_output_tokens: int,
        seed: int | None = None,
    ) -> str:
        started = datetime.now(timezone.utc).isoformat()
        context = current_llm_trace_context()
        base_event: dict[str, Any] = {
            "timestamp": started,
            "kind": "llm_chat",
            "identity": self.identity,
            "context": context,
            "request": {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "seed": seed,
            },
        }
        try:
            response = await self.wrapped.generate(
                system=system,
                user=user,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                seed=seed,
            )
        except Exception as exc:
            event = dict(base_event)
            event["status"] = "error"
            event["error"] = f"{type(exc).__name__}: {exc}"
            self._append(event)
            raise

        event = dict(base_event)
        event["status"] = "ok"
        event["response"] = response
        self._append(event)
        return response
