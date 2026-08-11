from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from ..config import AppConfig
from ..llm.base import LLMProvider
from ..llm.tracing import llm_trace_context
from ..models import TableData
from ..prompts import TABLE_TRANSLATION_SYSTEM
from ..validation.invariants import (
    is_immutable_value,
    validate_immutable_values,
    validate_nonempty_preservation,
    validate_untranslated_columns,
)
from ..validation.structure import validate_table_shape


class UnitTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    translation: str


class TableTranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translations: list[UnitTranslation]


@dataclass(frozen=True)
class _Destination:
    row: int | None
    column: int


@dataclass
class _TranslationUnit:
    id: str
    source: str
    destinations: list[_Destination] = field(default_factory=list)


def _parse_json(text: str) -> dict:
    """Parse a model response as one JSON object, tolerating a surrounding code fence."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as first_error:
        # Some models occasionally add a short prefix/suffix despite the prompt.
        # We still require exactly one object-shaped payload after extraction.
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(s[a : b + 1])
            except json.JSONDecodeError as second_error:
                preview = s[:1600].replace("\n", "\\n")
                raise ValueError(
                    f"Invalid JSON returned by model: {second_error}. Raw preview: {preview}"
                ) from second_error
        preview = s[:1600].replace("\n", "\\n")
        raise ValueError(
            f"Invalid JSON returned by model: {first_error}. Raw preview: {preview}"
        ) from first_error


def _build_translation_units(table: TableData, columns: set[int]) -> list[_TranslationUnit]:
    """Build exact unique textual values and keep their coordinates only in Python.

    The LLM never receives row/column coordinates, the complete table, protected
    values, or values from unselected columns. Exact duplicate source strings are
    represented by one translation unit and are later propagated to every stored
    destination by Python.
    """
    units: list[_TranslationUnit] = []
    by_source: dict[str, _TranslationUnit] = {}

    def add(source: str, destination: _Destination) -> None:
        if is_immutable_value(source):
            return
        unit = by_source.get(source)
        if unit is None:
            unit = _TranslationUnit(id=f"u{len(units):04d}", source=source)
            units.append(unit)
            by_source[source] = unit
        unit.destinations.append(destination)

    # Headers are translation units too, but only for selected columns.
    for c in sorted(columns):
        add(table.headers[c], _Destination(row=None, column=c))

    # Cells are traversed in stable row/column order. Duplicates are exact-string
    # duplicates: no case-folding or punctuation normalization is used here.
    for r, row in enumerate(table.rows):
        for c in sorted(columns):
            add(row[c], _Destination(row=r, column=c))

    return units


def _validate_response(units: list[_TranslationUnit], response: TableTranslationResponse) -> dict[str, str]:
    expected_ids = [u.id for u in units]
    returned_ids = [item.id for item in response.translations]

    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for item_id in returned_ids:
        if item_id in seen and item_id not in duplicate_ids:
            duplicate_ids.append(item_id)
        seen.add(item_id)
    if duplicate_ids:
        raise ValueError(f"Model returned duplicate translation IDs: {duplicate_ids[:10]}")

    missing = [item_id for item_id in expected_ids if item_id not in returned_ids]
    unexpected = [item_id for item_id in returned_ids if item_id not in expected_ids]
    if missing or unexpected or len(returned_ids) != len(expected_ids):
        raise ValueError(
            "Translation ID set mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"expected_count={len(expected_ids)}, returned_count={len(returned_ids)}"
        )

    mapping = {item.id: item.translation for item in response.translations}
    empty = [u.id for u in units if u.source.strip() and not mapping[u.id].strip()]
    if empty:
        raise ValueError(f"Model returned empty translations for non-empty values: {empty[:10]}")
    return mapping


def _apply_translations(table: TableData, units: list[_TranslationUnit], mapping: dict[str, str]) -> TableData:
    candidate = table.copy()
    for unit in units:
        translated = mapping[unit.id]
        for destination in unit.destinations:
            if destination.row is None:
                candidate.headers[destination.column] = translated
            else:
                candidate.rows[destination.row][destination.column] = translated
    return candidate


async def translate_table(
    provider: LLMProvider,
    table: TableData,
    *,
    config: AppConfig,
    source_language: str,
    target_language: str,
    columns: set[int] | None = None,
    trace_stage_prefix: str = "translate_table",
) -> TableData:
    """Translate selected table text as one dictionary-style LLM request.

    Python owns table structure and coordinates. It extracts only exact unique,
    translatable values from the selected columns (including selected headers),
    sends those values to the model once, validates the returned ID->translation
    mapping, and writes each translation back to every stored coordinate.

    Consequences:
    * exact duplicates are translated once and therefore cannot diverge;
    * immutable values and unselected columns are never sent to the model;
    * the model cannot reorder/delete rows or columns because it never receives
      responsibility for reconstructing the table;
    * SHELL_ONLY makes no table call at all; MENTIONS_ONLY passes values only from
      mention-linked columns because the pipeline supplies ``columns``.

    A failed/invalid response may retry the whole unit-list request according to
    ``model.retries``. A successful first attempt is one LLM request.
    """
    columns = set(range(len(table.headers))) if columns is None else set(columns)
    for c in columns:
        if c < 0 or c >= len(table.headers):
            raise ValueError(f"Column index out of range: {c}")

    units = _build_translation_units(table, columns)
    if not units:
        return table.copy()

    payload = {
        "task": "translate_table_values",
        "source_language": source_language,
        "target_language": target_language,
        "values": [{"id": unit.id, "text": unit.source} for unit in units],
        "requirements": [
            "Translate every supplied value into the target language.",
            "Return exactly one translation for every supplied id.",
            "Preserve every id exactly; do not add, remove, rename, or duplicate ids.",
            "Translate the text only; do not answer questions, calculate, summarize, or add facts.",
            "Do not return any table coordinates, rows, columns, source text, explanations, or extra keys.",
        ],
    }

    last_error: Exception | None = None
    for attempt in range(config.model.retries + 1):
        request_payload = dict(payload)
        if last_error is not None:
            request_payload["retry_instruction"] = (
                "The previous response was invalid. Translate the same values again. Return exactly one JSON DATA "
                "object containing translations, with exactly one entry for every supplied id and no extra ids."
            )
            request_payload["previous_error"] = str(last_error)[:1400]

        try:
            with llm_trace_context(
                stage=f"{trace_stage_prefix}.values",
                attempt=attempt,
                unique_values=len(units),
                selected_columns=sorted(columns),
            ):
                raw = await provider.generate(
                    system=TABLE_TRANSLATION_SYSTEM,
                    user=json.dumps(request_payload, ensure_ascii=False),
                    temperature=config.model.temperature,
                    max_output_tokens=config.model.max_output_tokens,
                    seed=config.project.seed + 900_000 + attempt,
                )

            parsed = TableTranslationResponse.model_validate(_parse_json(raw))
            mapping = _validate_response(units, parsed)
            candidate = _apply_translations(table, units, mapping)

            # These checks are intentionally retained even though Python constructs
            # the output: they guard future refactors and catch accidental mutations.
            validate_table_shape(table, candidate)

            immutable_errors = validate_immutable_values(table, candidate)
            if config.validation.strict_immutable_values and immutable_errors:
                raise ValueError(f"Immutable values changed: {immutable_errors[:5]}")

            changed_unselected = validate_untranslated_columns(table, candidate, columns)
            if changed_unselected:
                raise ValueError(f"Unselected columns changed: {changed_unselected[:5]}")

            disappeared = validate_nonempty_preservation(table, candidate, columns)
            if disappeared:
                raise ValueError(
                    f"Non-empty values disappeared during translation: {disappeared[:5]}"
                )

            return candidate
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Table value-list translation failed after retries: {last_error}")
