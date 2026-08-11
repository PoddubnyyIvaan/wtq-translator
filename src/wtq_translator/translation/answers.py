from __future__ import annotations

import json
from pydantic import BaseModel

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.tracing import llm_trace_context
from ..models import AnswerMatch, TableCoordinate, TableData
from ..prompts import ANSWER_TRANSLATION_SYSTEM
from ..mentions.normalize import normalize_for_match
from ..validation.invariants import is_immutable_value


def split_target_value(value: str) -> list[str]:
    return value.split("|")


def find_answer_matches(target_value: str, table: TableData, normalization_cfg=None) -> list[AnswerMatch]:
    answers = split_target_value(target_value)
    result: list[AnswerMatch] = []
    for i, answer in enumerate(answers):
        norm = normalize_for_match(answer, normalization_cfg)
        coords: list[TableCoordinate] = []
        for c, value in enumerate(table.headers):
            if normalize_for_match(value, normalization_cfg) == norm:
                coords.append(TableCoordinate(None, c))
        for r, row in enumerate(table.rows):
            for c, value in enumerate(row):
                if normalize_for_match(value, normalization_cfg) == norm:
                    coords.append(TableCoordinate(r, c))
        result.append(AnswerMatch(i, answer, tuple(coords)))
    return result


class AnswerResponse(BaseModel):
    translated_text: str


def _extract_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            return json.loads(s[a:b + 1])
        raise


async def _translate_direct(provider: LLMProvider, text: str, config: AppConfig, source_language: str, target_language: str, seed_offset: int) -> str:
    if is_immutable_value(text):
        return text
    system = ANSWER_TRANSLATION_SYSTEM
    base = {"source_language": source_language, "target_language": target_language, "text": text}
    last_error: Exception | None = None
    for attempt in range(config.model.retries + 1):
        payload = dict(base)
        if last_error is not None:
            payload["retry_instruction"] = "Previous output was invalid. Return the translated data object, not a schema."
            payload["previous_error"] = str(last_error)[:1000]
        try:
            with llm_trace_context(stage="translate_answer", attempt=attempt, answer_index=seed_offset):
                raw = await provider.generate(
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    temperature=config.model.temperature,
                    max_output_tokens=min(config.model.max_output_tokens, 512),
                    seed=config.project.seed + seed_offset + attempt,
                )
            return AnswerResponse.model_validate(_extract_json(raw)).translated_text
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Direct answer translation failed after retries: {last_error}")


async def build_target_value(
    provider: LLMProvider,
    *,
    source_target_value: str,
    source_table: TableData,
    output_table: TableData,
    config: AppConfig,
    desired_language: str,
    source_language: str,
    table_language_is_desired: bool,
) -> tuple[str, list[str]]:
    matches = find_answer_matches(source_target_value, source_table, config.mentions.normalization)
    answers = split_target_value(source_target_value)
    out: list[str] = []
    warnings: list[str] = []

    for i, (answer, match) in enumerate(zip(answers, matches)):
        if match.coordinates and table_language_is_desired:
            values = {output_table.value_at(c) for c in match.coordinates}
            if len(values) == 1:
                out.append(next(iter(values)))
                continue
            warnings.append(f"Answer {i} has multiple translated values at matching coordinates")
        if desired_language == source_language or is_immutable_value(answer):
            out.append(answer)
        else:
            out.append(await _translate_direct(provider, answer, config, source_language, desired_language, 9000 + i * 100))
            warnings.append(f"Answer {i} translated directly because coordinate recovery was not usable")
    return "|".join(out), warnings
