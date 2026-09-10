from __future__ import annotations

import json
from pydantic import BaseModel

from ..config import ModelConfig
from ..llm.base import LLMProvider
from ..llm.tracing import llm_trace_context
from ..prompts import SHELL_TRANSLATION_SYSTEM, SHELL_TRANSLATION_TAGGED_SYSTEM
from ..validation.masks import validate_masks, validate_tags


class ShellResponse(BaseModel):
    translated_text: str


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as second_error:
                preview = cleaned[:1200].replace("\n", "\\n")
                raise ValueError(f"Invalid shell JSON: {second_error}. Raw preview: {preview}") from second_error
        preview = cleaned[:1200].replace("\n", "\\n")
        raise ValueError(f"Invalid shell JSON: {first_error}. Raw preview: {preview}") from first_error


def _tagged_worked_example(source_language: str, target_language: str) -> dict[str, str] | None:
    """Returns a direction-matched worked example for the tagged shell prompt.

    The example demonstrates the two rules that models most often violate:
    1) a tag pair is NOT dropped or unwrapped - it stays glued around its content;
    2) the pair may travel together with its content to follow the target-language
    word order, and the content is the tag_glossary word inflected to agree with
    the sentence (en: "<the team>'s captain" -> ru: "капитан <команды>").
    """
    if source_language == "en" and target_language == "ru":
        return {
            "input": "who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?",
            "glossary": {"<TABLE_REF_0000>": "команда"},
            "output": "кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?",
        }
    if source_language == "ru" and target_language == "en":
        return {
            "input": "кем был капитан <TABLE_REF_0000>команды</TABLE_REF_0000> в 2001 году?",
            "glossary": {"<TABLE_REF_0000>": "the team"},
            "output": "who was <TABLE_REF_0000>the team</TABLE_REF_0000>'s captain in 2001?",
        }
    return None


async def translate_shell(
    provider: LLMProvider,
    text: str,
    *,
    source_language: str,
    target_language: str,
    model: ModelConfig,
    seed: int,
    trace_stage: str = "translate_shell",
    tag_glossary: dict[str, str] | None = None,
) -> str:
    # tag_glossary=None -> старое поведение (mask-валидация); иначе теги + глоссарий.
    tagged = tag_glossary is not None
    system = SHELL_TRANSLATION_TAGGED_SYSTEM if tagged else SHELL_TRANSLATION_SYSTEM
    base_payload = {
        "source_language": source_language,
        "target_language": target_language,
        "text": text,
    }
    if tagged:
        base_payload["tag_glossary"] = tag_glossary
        example = _tagged_worked_example(source_language, target_language)
        if example is not None:
            base_payload["worked_example"] = example

    last_error: Exception | None = None
    for attempt in range(model.retries + 1):
        payload = dict(base_payload)
        if last_error is not None:
            if tagged:
                payload["retry_instruction"] = (
                    "The previous response was invalid. Return the translated data object itself, not a schema, "
                    "and preserve every opening and closing TABLE_REF tag exactly. "
                    "Do not drop or unwrap tags: a tag pair moves together with its content to fit the "
                    "target-language word order, and the content inside the pair must be the matching "
                    "tag_glossary word inflected to agree with the sentence."
                )
            else:
                payload["retry_instruction"] = (
                    "The previous response was invalid. Return the translated data object itself, not a schema, "
                    "and preserve all TABLE_REF placeholders exactly."
                )
            payload["previous_error"] = str(last_error)[:1000]
        try:
            with llm_trace_context(stage=trace_stage, attempt=attempt):
                raw = await provider.generate(
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    temperature=model.temperature,
                    max_output_tokens=model.max_output_tokens,
                    seed=seed + attempt,
                )
            parsed = ShellResponse.model_validate(_extract_json(raw))
            mv = validate_tags(text, parsed.translated_text) if tagged else validate_masks(text, parsed.translated_text)
            if not mv.ok:
                raise ValueError(f"LLM changed placeholders: missing={mv.missing}, added={mv.added}")
            return parsed.translated_text
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Shell translation failed after retries: {last_error}")
