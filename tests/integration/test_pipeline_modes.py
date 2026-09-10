import json

import pytest

from wtq_translator.config import AppConfig
from wtq_translator.enums import ProcessingStatus, TranslationMode
from wtq_translator.models import QualityReport, QuestionRecord, TableData
from wtq_translator.pipeline import TranslationPipeline


class RuleBasedProvider:
    """Tiny deterministic provider used to exercise the real pipeline without a model."""

    def __init__(self):
        self.payloads = []

    @property
    def capabilities(self):
        class Caps:
            seed = True
        return Caps()

    async def count_tokens(self, text: str) -> int:
        # Детерминированный подсчёт токенов (по словам).
        return len(text.split())

    async def generate(self, *, system, user, temperature, max_output_tokens, seed=None):
        payload = json.loads(user)
        self.payloads.append(payload)
        src = payload.get("source_language")
        dst = payload.get("target_language")
        task = payload.get("task")

        en_ru = {"Name": "Имя", "City": "Город", "Alice": "Алиса", "London": "Лондон"}
        ru_en = {v: k for k, v in en_ru.items()}
        mapping = en_ru if (src, dst) == ("en", "ru") else ru_en

        if task == "detect_domain":
            return json.dumps({"domain": "geography"}, ensure_ascii=False)

        if task == "translate_table_values":
            return json.dumps(
                {
                    "translations": [
                        {"id": item["id"], "translation": mapping.get(item["text"], item["text"])}
                        for item in payload["values"]
                    ]
                },
                ensure_ascii=False,
            )

        if "text" in payload:
            text = payload["text"]
            shell_mapping = {
                ("en", "ru", "Where is <TABLE_REF_0000>?"): "Где находится <TABLE_REF_0000>?",
                ("ru", "en", "Где находится <TABLE_REF_0000>?"): "Where is <TABLE_REF_0000>?",
            }
            if (src, dst, text) in shell_mapping:
                return json.dumps({"translated_text": shell_mapping[(src, dst, text)]}, ensure_ascii=False)
            return json.dumps({"translated_text": mapping.get(text, text)}, ensure_ascii=False)

        raise AssertionError(payload)


class IdenticalEmbeddingProvider:
    async def embed(self, texts):
        # The scripted back translation is exact, so this is enough to verify wiring.
        return [[1.0, 0.0] for _ in texts]


def _config(tmp_path, mode, answer_language_mode="table"):
    return AppConfig.model_validate({
        "project": {
            "input_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "source_language": "en",
            "target_language": "ru",
            "seed": 42,
        },
        "translation": {"mode": mode, "answer_language_mode": answer_language_mode},
        "model": {"retries": 0},
        "embeddings": {"enabled": False},
        "back_translation": {"enabled": True},
        "table": {"rows_per_chunk": 20},
        "cache": {"enabled": False},
    })


def _input():
    q = QuestionRecord("q1", "Where is Alice?", "table.tsv", "London")
    table = TableData(["Name", "City"], [["Alice", "London"]])
    return q, table


@pytest.mark.asyncio
async def test_shell_only_pipeline_end_to_end(tmp_path):
    cfg = _config(tmp_path, "shell_only")
    q, table = _input()
    provider = RuleBasedProvider()
    result = await TranslationPipeline(cfg, provider, IdenticalEmbeddingProvider()).process(q, table)

    assert result.translated_question == "Где находится Alice?"
    assert result.translated_target_value == "London"
    assert result.output_table.headers == ["Name", "City"]
    assert result.output_table.rows == [["Alice", "London"]]
    assert result.quality.back_translated_question == q.utterance
    assert result.quality.back_table_groups_compared == 0
    assert result.quality.status == ProcessingStatus.ACCEPTED
    assert not any(p.get("task") == "translate_table_values" for p in provider.payloads)


@pytest.mark.asyncio
async def test_mentions_only_pipeline_end_to_end(tmp_path):
    cfg = _config(tmp_path, "mentions_only")
    q, table = _input()
    provider = RuleBasedProvider()
    result = await TranslationPipeline(cfg, provider, IdenticalEmbeddingProvider()).process(q, table)

    assert result.translated_question == "Where is Алиса?"
    assert result.translated_target_value == "London"  # answer is in untouched City column
    assert result.output_table.headers == ["Имя", "City"]
    assert result.output_table.rows == [["Алиса", "London"]]
    assert result.quality.back_translated_question == q.utterance
    assert result.quality.back_table_min_similarity == 1.0
    assert result.quality.status == ProcessingStatus.ACCEPTED
    table_payloads = [p for p in provider.payloads if p.get("task") == "translate_table_values"]
    assert len(table_payloads) == 2  # selected-column forward + selected-column back-translation
    assert [v["text"] for v in table_payloads[0]["values"]] == ["Name", "Alice"]
    for payload in table_payloads:
        texts = [v["text"] for v in payload["values"]]
        assert "City" not in texts
        assert "London" not in texts


@pytest.mark.asyncio
async def test_back_table_threshold_needs_review(tmp_path):
    """Порог back-table схожести: min < threshold -> NEEDS_REVIEW + warning; без порога статус не меняется."""
    class LowSimilarityEmbeddingProvider:
        async def embed(self, texts):
            return [[1.0, 0.0] if "lice" in t else [0.0, 1.0] for t in texts]

    cfg = AppConfig.model_validate({
        "project": {"input_root": str(tmp_path), "output_root": str(tmp_path / "out"),
                    "source_language": "en", "target_language": "ru", "seed": 42},
        "translation": {"mode": "mentions_only", "answer_language_mode": "table"},
        "model": {"retries": 0},
        "embeddings": {"enabled": False},
        "back_translation": {"enabled": True},
        "table": {"rows_per_chunk": 20},
        "cache": {"enabled": False},
        "validation": {"back_table_similarity_threshold": 0.9},
    })

    pipeline = TranslationPipeline(cfg, RuleBasedProvider(), LowSimilarityEmbeddingProvider())
    src = TableData(["Name"], [["Alice"]])
    back = TableData(["Name"], [["Bob"]])

    report = QualityReport(
        question_id="q1",
        mode=TranslationMode.MENTIONS_ONLY,
        table_mentions_found=True,
        source_question="",
        status=ProcessingStatus.PROCESSING,
    )
    await pipeline._apply_back_table_quality(src, back, report)

    assert report.back_table_min_similarity == 0.0
    assert report.status == ProcessingStatus.NEEDS_REVIEW
    assert any("below threshold" in w for w in report.warnings)

    # Без порога статус не меняется.
    cfg0 = _config(tmp_path, "mentions_only")
    pipeline0 = TranslationPipeline(cfg0, RuleBasedProvider(), LowSimilarityEmbeddingProvider())
    report2 = QualityReport(
        question_id="q1",
        mode=TranslationMode.MENTIONS_ONLY,
        table_mentions_found=True,
        source_question="",
        status=ProcessingStatus.PROCESSING,
    )
    await pipeline0._apply_back_table_quality(src, back, report2)

    assert report2.back_table_min_similarity == 0.0
    assert report2.status == ProcessingStatus.PROCESSING
