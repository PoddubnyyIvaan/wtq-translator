"""Определение предметной области (домена) таблицы по TAPEX-фрагменту."""
from __future__ import annotations

import functools
import json
from typing import Awaitable, Callable

from .config import AppConfig
from .llm.base import LLMProvider
from .llm.tracing import llm_trace_context
from .models import TableData
from .prompts import DOMAIN_DETECTION_SYSTEM
from .translation.table import _parse_json

# Колбэк подсчёта токенов: принимает текст, возвращает число токенов.
_Count = Callable[[str], Awaitable[int]]


async def count_text(provider: LLMProvider, text: str) -> int:
    """Тонкая обёртка над провайдером: считает токены одного фрагмента текста."""
    return await provider.count_tokens(text)


def _render_header(headers: list[str], k: int) -> str:
    """Строка заголовка TAPEX-префикса: ``col : [H1] | [H2] | ...`` (k колонок)."""
    cols = " | ".join(f"[{h}]" for h in headers[:k])
    return f"col : {cols}"


def _render_row(row: list[str], k: int) -> str:
    """Строка таблицы TAPEX: ``[a] | [b] | ...`` (k колонок).

    Префикс ``row N: `` добавляется в ``linearize_table``, т.к. имя строки
    известно только по позиции в таблице, а не из значений ячеек.
    """
    return " | ".join(f"[{x}]" for x in row[:k])


async def linearize_table(
    table: TableData,
    max_tokens: int,
    count: _Count,
) -> str:
    """Линеаризует таблицу в формат TAPEX жадным бюджетом токенов.

    Сначала в жадном режиме отбираются колонки: пока заголовочная строка умещается
    в ``max_tokens`` — колонка сохраняется. Если не влезает даже первая колонка,
    возвращается пустая строка. Затем в жадном режиме добавляются строки, пока
    накопленное число токенов строк + токенов очередной строки не превышает
    ``max_tokens``; на первой не влезающей строке добавление прекращается.
    """
    # Жадный отбор колонок: acc строится как итоговая строка заголовка.
    acc = "col : "
    k = 0
    for h in table.headers:
        if not k:
            candidate = f"{acc}[{h}]"
        else:
            candidate = f"{acc} | [{h}]"
        if await count(candidate) <= max_tokens:
            acc = candidate
            k += 1
        else:
            break

    if not k:
        return ""

    header = _render_header(table.headers, k)
    lines = [header]
    used = 0
    for r, row in enumerate(table.rows):
        line = f"row {r}: {_render_row(row, k)}"
        n = await count(line)
        if used + n <= max_tokens:
            lines.append(line)
            used += n
        else:
            break

    return "\n".join(lines)


async def detect_domain(
    provider: LLMProvider,
    config: AppConfig,
    table: TableData,
) -> tuple[str, str]:
    """Определяет предметную область таблицы.

    Возвращает ``(domain, warning)``. При выключенной конфигурации — ``("", "")``.
    При исчерпании ретраев или некорректном JSON — ``("", warning)``, не бросает
    исключений; warning предназначен для добавления в отчёт.
    """
    if not config.domain_detection.enabled:
        return ("", "")

    lang = config.domain_detection.language or config.project.source_language
    count = functools.partial(count_text, provider)
    snippet = await linearize_table(table, config.domain_detection.max_tokens, count)

    payload = {
        "task": "detect_domain",
        "language": lang,
        "snippet": snippet,
    }

    last_error: Exception | None = None
    for attempt in range(config.model.retries + 1):
        request_payload = dict(payload)
        if last_error is not None:
            request_payload["retry_instruction"] = (
                "The previous response was invalid. Detect the table domain again. "
                "Return exactly one JSON DATA object containing only the ``domain`` key."
            )
            request_payload["previous_error"] = str(last_error)[:1400]

        try:
            with llm_trace_context(stage="detect_domain", attempt=attempt):
                raw = await provider.generate(
                    system=DOMAIN_DETECTION_SYSTEM,
                    user=json.dumps(request_payload, ensure_ascii=False),
                    temperature=config.model.temperature,
                    max_output_tokens=config.model.max_output_tokens,
                    seed=config.project.seed + 900_000 + attempt,
                )

            parsed = _parse_json(raw)
            domain = str(parsed.get("domain", "")).strip()
            return (domain, "")
        except Exception as exc:
            last_error = exc

    warning = f"Domain detection failed after retries: {last_error}"
    return ("", warning)
