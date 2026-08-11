from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_TRACE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("wtq_llm_trace_context", default={})


def current_llm_trace_context() -> dict[str, Any]:
    """Return a copy of metadata associated with the current logical LLM call."""
    return dict(_TRACE_CONTEXT.get())


@contextmanager
def llm_trace_context(**fields: Any) -> Iterator[None]:
    """Temporarily enrich LLM trace metadata.

    Context variables are used instead of passing question/stage identifiers through
    every translation function. This also works correctly with asyncio tasks.
    """
    merged = dict(_TRACE_CONTEXT.get())
    merged.update({k: v for k, v in fields.items() if v is not None})
    token = _TRACE_CONTEXT.set(merged)
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)
