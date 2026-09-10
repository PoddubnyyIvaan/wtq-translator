from __future__ import annotations

from dataclasses import asdict

from .backtranslation import back_translate_shell, back_translate_table, back_translate_table_candidates
from .config import AppConfig
from .enums import AnswerLanguageMode, ProcessingStatus, TranslationMode
from .llm.base import EmbeddingProvider, LLMProvider
from .llm.tracing import llm_trace_context
from .mentions import build_mention_map, find_table_mentions, mask_question, restore_from_table, restore_original
from .mentions.candidates import load_spacy_model
from .mentions.finder_full import find_table_mentions_full
from .mentions.masking import mask_question_tagged, strip_tags
from .models import MentionMap, QualityReport, QuestionRecord, TableData, TranslationResult
from .persistence.cache import SQLiteCache
from .translation import build_target_value, translate_shell, translate_table
from .translation.table_candidates import reconstruct_table, translate_table_candidates, validate_untranslated
from .validation import (
    compare_back_translated_table,
    cosine_similarity,
    extract_masks,
    validate_immutable_values,
    validate_masks,
    validate_table_shape,
    validate_untranslated_columns,
)
from .validation.masks import extract_open_tags, validate_tags


class TranslationPipeline:
    def __init__(self, config: AppConfig, llm: LLMProvider, embeddings: EmbeddingProvider | None = None, cache: SQLiteCache | None = None) -> None:
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.cache = cache

    async def process(self, question: QuestionRecord, table: TableData) -> TranslationResult:
        # All nested model calls inherit the WTQ question id in their LLM I/O log.
        with llm_trace_context(question_id=question.id, source_question=question.utterance):
            return await self._process(question, table)

    async def _process(self, question: QuestionRecord, table: TableData) -> TranslationResult:
        mode = self.config.translation.mode
        if mode == TranslationMode.FULL:
            return await self._process_full(question, table)

        search = find_table_mentions(question.utterance, table, self.config.mentions)
        mention_map = build_mention_map(search)
        report = QualityReport(
            question_id=question.id,
            mode=mode,
            table_mentions_found=bool(mention_map.mentions),
            source_question=question.utterance,
            original_masks=mention_map.masks(),
            ambiguous_mentions=[asdict(a) for a in search.ambiguous],
            status=ProcessingStatus.PROCESSING,
        )
        if not mention_map.mentions:
            report.warnings.append("No exact normalized table mentions found in question.")
        if search.ambiguous:
            report.warnings.append("Ambiguous table references found; no automatic coordinate chosen for those spans.")
            if self.config.mentions.on_ambiguity == "failed":
                report.status = ProcessingStatus.FAILED
                raise ValueError("Ambiguous table reference")

        masked_source = mask_question(question.utterance, mention_map.mentions)

        if mode == TranslationMode.SHELL_ONLY:
            result = await self._shell_only(question, table, mention_map, masked_source, report)
        elif mode == TranslationMode.MENTIONS_ONLY:
            result = await self._mentions_only(question, table, mention_map, masked_source, report)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if search.ambiguous and result.quality.status in {ProcessingStatus.ACCEPTED, ProcessingStatus.ACCEPTED_WITH_WARNINGS}:
            result.quality.status = ProcessingStatus.NEEDS_REVIEW
        return result

    async def _process_full(self, question: QuestionRecord, table: TableData) -> TranslationResult:
        cfg = self.config
        nlp = load_spacy_model(cfg.mentions.spacy_model)
        search, candidate_set = find_table_mentions_full(
            question.utterance, table, cfg.mentions, nlp, cfg.project.target_language
        )
        mention_map = build_mention_map(search)
        report = QualityReport(
            question_id=question.id,
            mode=TranslationMode.FULL,
            table_mentions_found=bool(mention_map.mentions),
            source_question=question.utterance,
            original_masks=mention_map.masks(),          # открывающие теги
            ambiguous_mentions=[],                        # всегда пусто для full
            status=ProcessingStatus.PROCESSING,
        )
        if not mention_map.mentions:
            report.warnings.append("No table mentions found in question.")

        # 1) Перевод таблицы (кандидаты) — ДО shell
        await translate_table_candidates(
            self.llm, candidate_set.candidates,
            config=cfg, source_language=cfg.project.source_language,
            target_language=cfg.project.target_language, cache=self.cache,
        )
        lemma_translations = {c.lemma_key: c.translation for c in candidate_set.candidates}
        untranslated = validate_untranslated(candidate_set, lemma_translations)
        if untranslated:
            report.errors.append(f"Untranslated table runs: {untranslated[:5]}")
            report.status = ProcessingStatus.FAILED
            raise ValueError(report.errors[-1])
        translated_table = reconstruct_table(table, candidate_set, lemma_translations)

        # 2) Теговая маска
        masked_source = mask_question_tagged(question.utterance, mention_map.mentions)

        # 3) Shell с глоссарием
        id_to_translation = {c.id: c.translation for c in candidate_set.candidates}
        tag_glossary = {m.mask: id_to_translation[m.candidate_key] for m in mention_map.mentions}
        masked_ru = await translate_shell(
            self.llm, masked_source,
            source_language=cfg.project.source_language,
            target_language=cfg.project.target_language,
            model=cfg.model, seed=cfg.project.seed,
            tag_glossary=tag_glossary,
        )
        self._apply_tag_report(masked_source, masked_ru, report, forward=True)
        final_question = strip_tags(masked_ru)

        # 4) Ответ
        answer, warnings = await build_target_value(
            self.llm, source_target_value=question.target_value,
            source_table=table, output_table=translated_table, config=cfg,
            desired_language=cfg.project.target_language,
            source_language=cfg.project.source_language,
            table_language_is_desired=True,
        )
        report.warnings.extend(warnings)
        if any("translated directly" in w or "multiple translated values" in w for w in warnings):
            report.status = ProcessingStatus.NEEDS_REVIEW

        # 5) Back-shell (RU→EN) с обратным глоссарием
        back_glossary = {m.mask: m.question_text for m in mention_map.mentions}
        back_masked = await back_translate_shell(
            self.llm, masked_ru, config=cfg,
            source_language=cfg.project.source_language,
            translated_language=cfg.project.target_language,
            tag_glossary=back_glossary,
        )
        self._apply_tag_report(masked_source, back_masked, report, forward=False)
        back_question = strip_tags(back_masked)

        # 6) Back-таблица (кандидаты, RU→EN)
        back_table = await back_translate_table_candidates(
            self.llm, translated_table, config=cfg,
            source_language=cfg.project.source_language,
            translated_language=cfg.project.target_language,
            nlp=nlp, cache=self.cache,
        )
        validate_table_shape(table, back_table)
        self._apply_back_table_quality(table, back_table, report)
        immutable_back_errors = validate_immutable_values(table, back_table)
        if cfg.validation.strict_immutable_values and immutable_back_errors:
            report.errors.append(f"Back-translated table changed immutable values: {immutable_back_errors[:5]}")
            report.status = ProcessingStatus.FAILED
            raise ValueError(report.errors[-1])

        report.translated_question = final_question
        report.back_translated_question = back_question
        await self._score_question(question.utterance, back_question, report)
        if report.status == ProcessingStatus.PROCESSING:
            report.status = ProcessingStatus.ACCEPTED_WITH_WARNINGS if report.warnings else ProcessingStatus.ACCEPTED
        return TranslationResult(question, table, translated_table, final_question, answer, report, mention_map)

    async def _score_question(self, source: str, back: str, report: QualityReport) -> None:
        if self.embeddings is None:
            report.warnings.append("Embedding provider disabled; cosine similarity was not computed.")
            return
        vectors = await self.embeddings.embed([source, back])
        score = cosine_similarity(vectors[0], vectors[1])
        report.question_cosine_similarity = score
        threshold = self.config.validation.question_cosine_threshold
        if threshold is not None and score < threshold:
            report.warnings.append(f"Question cosine similarity {score:.4f} is below threshold {threshold:.4f}.")
            report.status = ProcessingStatus.NEEDS_REVIEW

    def _apply_back_table_quality(
        self,
        source: TableData,
        back_translated: TableData,
        report: QualityReport,
        *,
        columns: set[int] | None = None,
    ) -> None:
        metrics = compare_back_translated_table(
            source,
            back_translated,
            normalization=self.config.mentions.normalization,
            columns=columns,
        )
        report.back_table_values_compared = metrics["compared"]
        report.back_table_normalized_exact_matches = metrics["normalized_exact_matches"]
        report.back_table_normalized_exact_ratio = metrics["normalized_exact_ratio"]
        report.back_table_mismatch_examples = metrics["mismatch_examples"]

    def _apply_mask_report(self, expected: str, actual: str, report: QualityReport, forward: bool) -> None:
        mv = validate_masks(expected, actual)
        actual_masks = extract_masks(actual)
        missing = [k for k, n in mv.missing.items() for _ in range(n)]
        added = [k for k, n in mv.added.items() for _ in range(n)]
        if forward:
            report.forward_masks = actual_masks
            report.missing_forward_masks = missing
            report.added_forward_masks = added
        else:
            report.backward_masks = actual_masks
            report.missing_backward_masks = missing
            report.added_backward_masks = added
        if not mv.ok:
            report.errors.append(f"Mask validation failed: missing={mv.missing}, added={mv.added}")
            report.status = ProcessingStatus.FAILED
            raise ValueError(report.errors[-1])

    def _apply_tag_report(self, expected: str, actual: str, report: QualityReport, forward: bool) -> None:
        mv = validate_tags(expected, actual)
        actual_tags = extract_open_tags(actual)
        missing = [k for k, n in mv.missing.items() for _ in range(n)]
        added = [k for k, n in mv.added.items() for _ in range(n)]
        if forward:
            report.forward_masks = actual_tags
            report.missing_forward_masks = missing
            report.added_forward_masks = added
        else:
            report.backward_masks = actual_tags
            report.missing_backward_masks = missing
            report.added_backward_masks = added
        if not mv.ok:
            report.errors.append(f"Tag validation failed: missing={mv.missing}, added={mv.added}")
            report.status = ProcessingStatus.FAILED
            raise ValueError(report.errors[-1])

    async def _shell_only(self, question: QuestionRecord, table: TableData, mentions: MentionMap, masked_source: str, report: QualityReport) -> TranslationResult:
        cfg = self.config
        masked_ru = await translate_shell(self.llm, masked_source, source_language=cfg.project.source_language, target_language=cfg.project.target_language, model=cfg.model, seed=cfg.project.seed)
        self._apply_mask_report(masked_source, masked_ru, report, True)
        output_table = table.copy()
        final_question = restore_original(masked_ru, mentions)

        answer_target_lang = cfg.project.source_language if cfg.translation.answer_language_mode == AnswerLanguageMode.TABLE else cfg.project.target_language
        answer, warnings = await build_target_value(
            self.llm,
            source_target_value=question.target_value,
            source_table=table,
            output_table=output_table,
            config=cfg,
            desired_language=answer_target_lang,
            source_language=cfg.project.source_language,
            table_language_is_desired=(answer_target_lang == cfg.project.source_language),
        )
        report.warnings.extend(warnings)
        if any("translated directly" in w or "multiple translated values" in w for w in warnings):
            report.status = ProcessingStatus.NEEDS_REVIEW

        back_masked = await back_translate_shell(self.llm, masked_ru, config=cfg, source_language=cfg.project.source_language, translated_language=cfg.project.target_language)
        self._apply_mask_report(masked_source, back_masked, report, False)
        back_question = restore_original(back_masked, mentions)
        validate_table_shape(table, output_table)
        if table.headers != output_table.headers or table.rows != output_table.rows:
            raise ValueError("SHELL_ONLY changed the table")

        report.translated_question = final_question
        report.back_translated_question = back_question
        await self._score_question(question.utterance, back_question, report)
        if report.status == ProcessingStatus.PROCESSING:
            report.status = ProcessingStatus.ACCEPTED_WITH_WARNINGS if report.warnings else ProcessingStatus.ACCEPTED
        return TranslationResult(question, table, output_table, final_question, answer, report, mentions)

    async def _mentions_only(self, question: QuestionRecord, table: TableData, mentions: MentionMap, masked_source: str, report: QualityReport) -> TranslationResult:
        cfg = self.config
        translated_columns = {m.coordinate.column for m in mentions.mentions}
        if translated_columns:
            output_table = await translate_table(self.llm, table, config=cfg, source_language=cfg.project.source_language, target_language=cfg.project.target_language, columns=translated_columns)
        else:
            output_table = table.copy()

        changed = validate_untranslated_columns(table, output_table, translated_columns)
        if changed:
            raise ValueError(f"MENTIONS_ONLY changed unselected columns: {changed[:5]}")

        final_question = restore_from_table(masked_source, mentions, output_table)
        report.forward_masks = extract_masks(masked_source)

        # Answer language: table means language of answer cell; if it is not known to be
        # in a translated column, direct translation is not forced. Question means source
        # language because the shell remains in the source language.
        desired = cfg.project.source_language
        table_lang_desired = cfg.translation.answer_language_mode == AnswerLanguageMode.TABLE
        if cfg.translation.answer_language_mode == AnswerLanguageMode.QUESTION:
            desired = cfg.project.source_language
        answer, warnings = await build_target_value(
            self.llm,
            source_target_value=question.target_value,
            source_table=table,
            output_table=output_table,
            config=cfg,
            desired_language=desired,
            source_language=cfg.project.source_language,
            table_language_is_desired=table_lang_desired,
        )
        report.warnings.extend(warnings)
        if any("translated directly" in w or "multiple translated values" in w for w in warnings):
            report.status = ProcessingStatus.NEEDS_REVIEW

        if translated_columns:
            back_table = await back_translate_table(self.llm, output_table, config=cfg, source_language=cfg.project.source_language, translated_language=cfg.project.target_language, columns=translated_columns)
            validate_table_shape(table, back_table)
            self._apply_back_table_quality(table, back_table, report, columns=translated_columns)
            changed_back_unselected = validate_untranslated_columns(output_table, back_table, translated_columns)
            if changed_back_unselected:
                report.errors.append(f"Back translation changed unselected columns: {changed_back_unselected[:5]}")
                report.status = ProcessingStatus.FAILED
                raise ValueError(report.errors[-1])
            back_question_semantic = restore_from_table(masked_source, mentions, back_table)
        else:
            back_question_semantic = restore_original(masked_source, mentions)

        # Deterministic reconstruction uses original mention values.
        deterministic_back = restore_original(masked_source, mentions)
        report.backward_masks = extract_masks(masked_source)
        report.translated_question = final_question
        report.back_translated_question = back_question_semantic
        await self._score_question(question.utterance, back_question_semantic, report)
        if deterministic_back != question.utterance:
            report.errors.append("Deterministic mask reconstruction does not reproduce the original question")
            report.status = ProcessingStatus.FAILED
            raise ValueError(report.errors[-1])
        if report.status == ProcessingStatus.PROCESSING:
            report.status = ProcessingStatus.ACCEPTED_WITH_WARNINGS if report.warnings else ProcessingStatus.ACCEPTED
        return TranslationResult(question, table, output_table, final_question, answer, report, mentions)
