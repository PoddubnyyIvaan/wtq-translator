# Changelog

## Unreleased — candidate-based `full` translation pipeline

Режим `full` переведён на candidate-based пайплайн (план `.kilo/plans/1788603643092-full-translate-candidate-pipeline.md`). Режимы `shell_only`/`mentions_only` и их тесты не изменились.

- **Audit** (`audit.py`): команда `audit` выбирает матчер по `translation.mode` — `full` → лемма-матчер `find_table_mentions_full` (через spaCy), `shell_only`/`mentions_only` → точный нормализованный матчер `find_table_mentions`; в отчёт добавлены поля `mode` и `matcher`.

- **spaCy** (`pyproject.toml`, `README.md`): зависимость `spacy>=3.7`, загрузка модели `python -m spacy download en_core_web_trf`; имя модели — `mentions.spacy_model` (по умолчанию `en_core_web_trf`), ленивая загрузка с кэшем на процесс.
- **Конфиг** (`config.py`): в `MentionsConfig` добавлены `spacy_model`, `stop_pos` (`DET`/`ADP`/`PART`), `exclude_numbers`, `exclude_dates` (дефолты сохраняют старые режимы).
- **Модели** (`models.py`): `TableMention.candidate_key`; новые датаклассы `QuestionCandidate`, `TableCandidate`, `ContentRun`, `CandidateSet`.
- **Кандидаты** (`mentions/candidates.py`): `load_spacy_model`, классификация токенов (word/number/date/punct), лемма-ключи (word — нормализованная лемма, number/date — поверхностная форма, Unicode-корректно), `extract_question_candidates` / `extract_table_candidates` (дедуп по лемма-ключу с объединением координат, in-memory кэш таблиц).
- **Матчер** (`mentions/finder_full.py`): `find_table_mentions_full` — match по лемма-ключу, longest non-overlap; `ambiguous` всегда пуст для `full`.
- **Теги** (`mentions/masking.py`, `validation/masks.py`): `mask_question_tagged` (`<TABLE_REF_XXXX>…</TABLE_REF_XXXX>` без внутренних пробелов), `strip_tags`, `closing_mask`; `extract_open_tags`/`extract_close_tags`/`validate_tags`.
- **Промпты** (`prompts.py`): `SHELL_TRANSLATION_TAGGED_SYSTEM`, `CANDIDATE_TRANSLATION_SYSTEM`, `PROMPT_SET_VERSION` → `2026-08-13-v9`.
- **Перевод таблицы** (`translation/table_candidates.py`): `translate_table_candidates` (ярусы длины с глоссарием более коротких подпоследовательностей, кэш in-memory + `SQLiteCache.candidate_translations`), `reconstruct_table` (замена content-run'ов, ведущая/хвостовая пунктуация и числа/даты verbatim), `validate_untranslated`.
- **Shell** (`translation/shell.py`, `backtranslation/shell.py`): `translate_shell(..., tag_glossary=None)` — при переданном глоссарии используется `SHELL_TRANSLATION_TAGGED_SYSTEM` и валидация через `validate_tags`.
- **Back-перевод** (`backtranslation/table.py`): `back_translate_table_candidates` — кандидаты из переведённой таблицы, перевод src/dst наоборот, реконструкция.
- **Кэш** (`persistence/cache.py`): таблица `candidate_translations` + `get_candidate_translation`/`put_candidate_translation`.
- **Пайплайн** (`pipeline.py`, `api.py`): `_process_full` (таблица ДО shell) с ранней веткой `if mode == FULL` в `_process`; кэш прокидывается в `TranslationPipeline`.
- **Тесты**: новые unit-тесты кандидатов, full-матчера, тегов, кандидатного перевода и обратного перевода; новый интеграционный тест `full`-режима (`tests/integration/test_pipeline_full_new.py`); старый `test_full_pipeline_end_to_end` заменён.

## 0.8.0 — unique-value dictionary translation

- Модель больше не получает таблицу, строки, координаты, `protected_cells` или невыбранные столбцы. Python полностью владеет структурой таблицы.
- Для выбранной части таблицы Python извлекает только точные уникальные текстовые значения, которые действительно нужно переводить; числа, пустые значения, URL/идентификаторы и другие immutable-значения не отправляются в LLM.
- Заголовки выбранных столбцов входят в тот же список переводимых значений.
- Один прямой перевод таблицы по-прежнему выполняется одним LLM-запросом: `values=[{id,text}, ...]` → `translations=[{id,translation}, ...]`.
- Координаты и все вхождения каждого уникального значения хранятся только в Python. Один ответ модели программно вставляется во все исходные координаты, поэтому точные дубликаты не могут получить разные переводы.
- `full` передаёт уникальные переводимые значения из всех столбцов; `mentions_only` — только из столбцов, связанных с найденными табличными упоминаниями; `shell_only` вообще не делает LLM-запрос для таблицы.
- Обратный перевод таблицы использует тот же механизм и ту же область выбранных столбцов.
- Ответ модели строго проверяется по множеству ID: пропущенные, лишние и дублированные ID, пустые переводы и лишние JSON-поля отклоняются.
- Стадии LLM-лога для таблиц: `translate_table.values` и `back_translate_table.values`.

## 0.7.0 — whole-table single-request translation

- Перевод одной таблицы теперь выполняется одним логическим LLM-запросом: модель получает заголовки и все строки целиком и возвращает всю таблицу одним JSON-объектом.
- Удалён поклеточный перевод и программное распространение переводов дубликатов. Консистентность повторяющихся значений теперь является обязанностью модели и явно указана в prompt.
- `mentions_only` также отправляет таблицу одним запросом, но модель обязана изменить только выбранные столбцы; остальные проверяются на точное совпадение.
- Back-translation таблицы выполняется тем же whole-table механизмом одним отдельным запросом.
- Повторный LLM-запрос выполняется только если первый ответ невалиден или не проходит детерминированные проверки, в пределах `model.retries`.
- В `llm_io.jsonl` стадия перевода таблицы теперь `translate_table.table`, а обратного перевода — `back_translate_table.table`.

## v0.6

- Заголовки таблицы переводятся ровно один раз отдельным LLM-запросом.
- Перевод строк чанками заменён на translation memory для ячеек: каждое уникальное точное текстовое значение переводится один раз, после чего тот же перевод подставляется во все точные дубликаты в выбранных столбцах.
- Для каждой новой ячейки модели передаются заголовки, перевод заголовков и исходная строка как контекст.
- Обратный перевод таблицы использует тот же механизм: один перевод на уникальное значение.
- Добавлен полный JSONL-лог каждого LLM request/response, включая retry и ответы из кэша.
- В LLM-логе сохраняются `question_id`, стадия pipeline, номер попытки и координата ячейки, если применимо.
- Для dataset run лог находится в `runs/<run_id>/reports/llm_io.jsonl`; для `run-one` — в том же run-каталоге.
- Локальный пример конфигурации переведён на `qwen2.5:3b`.
- Prompt set обновлён до `2026-08-10-v4`.

## v0.5

- Заголовки таблицы стали переводиться один раз вместо сравнения перевода между чанками.
- Улучшены ошибки при невалидном JSON.

## v0.4

- Добавлен audit табличных упоминаний.
- Улучшена обработка повторов значения в одном столбце.
- Добавлена проверка исчезновения непустых значений.
