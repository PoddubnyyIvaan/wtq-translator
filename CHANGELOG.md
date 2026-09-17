## Unreleased — config toggle for transliteration forcing

- новый флаг `translation.candidates.transliteration_enabled` (дефолт `false`) в `CandidateTranslationConfig`;
- при выключенном флаге `transliterations` не попадают в payload перевода кандидатов, а требование о форсинге исключается из `requirements` (поведение возвращается к качеству baseline-спектра);
- при включённом — прежнее поведение форсинга NER-токенов (для A/B-сравнения);
- пример в `configs/qwen_full.yaml` (`transliteration_enabled: false`);
- тесты: новые `test_transliteration_disabled_by_default`, остальные транслит-тесты явно включают флаг.

## Unreleased — structured question translation (MENTION-CONSTRAINED SHELL)

- **Промпт** (`prompts.py`): новый шаблон `QUESTION_TRANSLATION_SYSTEM` (англ. версия контракта «перевод вопроса с упоминаниями») с наполняемыми полями `{source_language}`/`{target_language}`/`{domain}` — функция `question_translation_system_prompt()`; в ответе модель отдаёт plain-перевод, tagged-перевод, `mentions_used` (id, surface, lemma) и `missing_mentions`. Имена полей параметризованы суффиксом целевого языка (`question_ru`/`question_en`), что делает шаблон пригодным и для обратного перевода. `PROMPT_SET_VERSION` → `2026-09-13-v13`, шаблон добавлен в `prompt_snapshot`.
- **Протокол** (`translation/shell.py`): новый режим `translate_shell(mentions=..., id_to_canonical=..., domain=..., on_missing=...)`. В user-JSON кладутся `tagged_question` и `mentions` (id, tag, current_text, canonical translation, lemma, `inflect_allowed`); `inflect_allowed=false` для имён колонок и их частей (любая координата упоминания — заголовок), `true` для остальных. Legacy-режимы (`tag_glossary`, маски) не изменены.
- **Парсер**: `_parse_question_translation` валидирует покрытие id (каждое упоминание использовано или помечено `missing_mentions`), сохранение пар тегов, совпадение `surface` с содержимым пары тегов и согласованность plain- и tagged-вариантов (после `strip_tags`); при непустом `missing_mentions` вызывается `on_missing(ids)`, а не выбрасывается ошибка.
- **Пацплайн** (`pipeline.py`): `_process_full` переводит оболочку вопроса новым протоколом в обе стороны (forward: канонический перевод — перевод кандидата; back: канонический перевод — исходный фрагмент вопроса), домен передаётся в системный промпт; пропавшие упоминания пишутся в `report.warnings`.
- **Тесты**: `tests/unit/test_question_translation.py` (наполнение промпта, inflect_allowed, success/missing/dropped/surface-mismatch/plain-tagged-mismatch); интеграционный мок `test_pipeline_full_new.py` переведён на новый формат ответа.

## Unreleased — BERT-score для вопроса и таблицы

- параллельная оценка BERTScore (F1 / precision / recall) рядом с косинусным сходством для **вопроса** и **таблицы**;
- мастер-включатель `bert_score.enabled` (дефолт `false`) + per-site тогглы `validation.question_cosine_enabled`, `validation.question_bert_enabled`, `validation.back_table_bert_enabled`;
- новые поля отчёта `question_bert_f1/precision/recall` и `back_table_bert_groups_compared`, `back_table_bert_min_f1`, `back_table_bert_mean_f1`, `back_table_bert_min/mean_precision`, `back_table_bert_min/mean_recall`;
- опциональные пороги `bert_score.question_threshold` / `bert_score.table_threshold` — warning + `NEEDS_REVIEW` при `f1` (вопрос) / `min_f1` (таблица) ниже порога, null = только диагностика;
- новая зависимость `bert-score>=0.3.13` (тянет torch); ленивый импорт, тяжёлый вызов через `asyncio.to_thread`, скорер кэшируется (модель грузится один раз);
- деградация: при ошибке загрузки/инференса BERT поля остаются `None`, пишется warning, статус не меняется.

## Unreleased — back-table semantic similarity

- замена exact-match reverse-перевода таблицы на семантическую близость той же embedding-моделью (без новых зависимостей);
- новая настройка `validation.back_table_similarity_mode` — гранулярность `context` (дефолт) / `row` / `cell`;
- новые поля отчёта `back_table_groups_compared`, `back_table_min_similarity`, `back_table_mean_similarity`, `back_table_similarity_examples` (старые normalized-exact поля удалены);
- опциональный порог `back_table_similarity_threshold` — warning + `NEEDS_REVIEW` при `min` ниже порога, null = только диагностика;
- `back_table_similarity_max_chunk_tokens` — бюджет токенов для чанков в `context`-режиме.

## Unreleased — descending candidate tiers: per-item glossary + column context

- **Порядок ярусов** (`translation/table_candidates.py`): кандидаты переводятся от большего яруса к меньшему (`sorted(by_length, reverse=True)`); глоссарий меньшего яруса — переводы самых длинных уже переведённых последовательностей, в которые переводимая подпоследовательность вложена как непрерывное подокно токенов `lemma_key`, с дедупом по `(source_text, translation)`.
- **Структурированный ввод** (`translation/table_candidates.py`, `prompts.py`): user-JSON теперь `items: [{"id","text","column","glossary"}]` — каждый кандидат со своим глоссарием и полным оригинальным заголовком столбца (`column`, только контекст, первая координата кандидата). Формат ответа `{"translations":[{"id","translation"}]}` и downstream (валидация по id, кэш, реконструкция, трассировка) не изменились.
- **Домен** (`prompts.py`): домен указывается один раз в системном промпте (`candidate_system_prompt(domain)`, `CANDIDATE_TRANSLATION_DOMAIN_SUFFIX`), из user-JSON ключ `domain` удалён. `PROMPT_SET_VERSION` → `2026-09-10-v12`.
- **Call sites** (`pipeline.py`, `backtranslation/table.py`): передают `headers=table.headers`; обратный перевод таблицы наследует новую логику (языки меняются местами).

## Unreleased — tagged shell: movable tags + glossary-anchored content

- **Промпт** (`prompts.py`): `SHELL_TRANSLATION_TAGGED_SYSTEM` переформулирован — теги описаны как перемещаемые скобки, а не разметка: пара тегов едет вместе со своим контентом по правилам порядка слов целевого языка (позиция и взаимный порядок пар не фиксированы), но не может быть удалена/развёрнута/продублирована/слита. Контент внутри пары обязан быть словом из `tag_glossary`, согласованным (инфлексия рода/числа/падежа/лица/времени, при необходимости порядок слов внутри фразы); «hint for consistency» и запрет самостоятельного перевода содержимого тега ужесточены. `PROMPT_SET_VERSION` → `2026-09-09-v10`.
- **Shell** (`translation/shell.py`): для tagged-ветки в user-payload добавляется направленный `worked_example` (en→ru и ru→en) — пример переноса пары тегов вместе со словом и согласования словарной формы; для неизвестных языковых пар пример не добавляется. `retry_instruction` при потере тегов повторяет правила перемещения/словаря вместо абстрактного запрета.

## Unreleased — non-overlapping glossary windows

- **Сжатие глоссария кандидатов** (`translation/table_candidates.py`): в пределах одной фразы-значения в глоссарий chunk'а попадают только непересекающиеся по токен-позициям окна — из пересекающихся остаётся самое длинное (при равенстве — первое/левое), вложенные удаляются; результаты по значениям запроса объединяются. Порядок записей сохранён по порядку более коротких кандидатов.

## Unreleased — relevant candidate glossary

- **Глоссарий кандидатов** (`translation/table_candidates.py`): в промпт chunk'а яруса передаётся только релевантный глоссарий — переводы более коротких кандидатов, чей lemma_key является непрерывным подокном lemma_key хотя бы одного кандидата этого же chunk-запроса, а не весь глоссарий всех более коротких кандидатов яруса.

## Unreleased — concurrency + progress + native Ollama

Конкурентная обработка датасета с прогресс-баром и нативный Ollama-провайдер (план `.kilo/plans/1788851084786-optimize-run-time-concurrency.md`).

- **Конфиг** (`config.py`): новые поля `runner.concurrency`, `model.max_concurrent_requests`, `translation.candidates.{max_candidates_per_request, request_concurrency}`; дефолты сохраняют прежнее последовательное поведение.
- **Runner** (`runner.py`): ограниченный пул воркеров (top-up), checkpoint/вывод сохраняют порядок исходного TSV.
- **Прогресс** (`cli.py`, `pyproject.toml`): tqdm-бар в `run`/`resume` (stderr, `--no-progress`), stdout остаётся чистым для финального JSON.
- **Надёжность при параллелизме** (`persistence/state.py`, `persistence/cache.py`, `llm/`): SQLite WAL + `busy_timeout`, keep-alive `httpx.AsyncClient`, семафор в LLM-провайдере, потокобезопасная запись `llm_io.jsonl`.
- **Известное следствие**: порядок строк в append-only jsonl (`events`, `quality`, `failures`, `llm_io`) — порядок завершения, а не исходного файла; `output_questions.tsv` сохраняет порядок исходного файла.
- **Нативный Ollama** (`llm/ollama.py`, `llm/factory.py`): `model.provider: ollama` → `POST {base_url}/api/chat` (base_url без `/v1`); `model.ollama.{think, stream, options}` (stream только `false`), options по умолчанию `{temperature, num_predict=max_output_tokens, seed}` перекрываются конфигом.
- **Кэш/логи** (`api.py`): при `provider: ollama` `think`/`options` входят в identity кэша и логов.
- **Конфиг**: изменение значений меняет `_config_hash` → новый `run_id` (старые прогоны не резюмируются).

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
