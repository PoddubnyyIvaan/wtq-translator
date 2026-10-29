## Unreleased — перевод и обратный перевод упоминаний в трейсе

- `models.py`: в `TableMention` добавлены поля `translated_text` (surface упоминания
  в переведённом вопросе) и `back_translated_text` (surface в обратном переводе вопроса).
  Frozen + default — существующие конструкторы и тесты не ломаются; `runner.py:_write_trace`
  сериализует поля в `mentions` трейса автоматически через `asdict`.
- `mentions/masking.py`: новый helper `extract_tagged_spans(text)` — извлекает пары
  `<TABLE_REF_XXXX>...</TABLE_REF_XXXX>` в `{mask: content}` (regex с `re.DOTALL`).
- `pipeline.py`: в `_process_full` поля заполняются surface-формами из тегированного
  перевода (`masked_ru`) и тегированного обратного перевода (`back_masked`); в
  `_mentions_only` — значениями переведённой и обратно переведённой таблицы через
  `mention_map.table_value` (только при непустых `translated_columns`). SHELL_ONLY не
  трогается: упоминания не переводятся, поля остаются `None`.
- **Тесты**: новый `tests/unit/test_masking.py` (извлечение пар: обычный текст, пустой
  ввод, порядок пар, содержимое с пробелами/пунктуацией/внутренними тегами, многострочное);
  в `test_pipeline_full_new` и `test_pipeline_modes` добавлены проверки
  `translated_text`/`back_translated_text` в mentions результата. Валидация: 21 целевой
  тест зелёные; полный прогон — 254 passed, 5 известных pre-existing failures
  (`test_alphabet_detection` — 2, `test_question_translation` — 3, воспроизводятся без
  этих изменений).

## Unreleased — дедупликация reports jsonl перед статистиками

- Новый модуль `reporting/jsonl_dedupe.py`: `load_jsonl_rows(path)` — устойчивое чтение
  JSONL (пустые строки и битый JSON пропускаются), `dedupe_last_wins(rows, key_fn)` —
  last-wins по ключу с сохранением порядка первого вхождения (записи без ключа
  сохраняются все), `dedupe_events(rows)` — по сессии вопроса: для `question_id`
  остаются последний `START` и события после него; без `START` — last-wins по
  `(question_id, code)`; события без `question_id` не трогаются. Функции экспортируются
  из `reporting/__init__.py`.
- `reporting/stats.py`: `load_quality_rows` переведён на общие хелперы — дедуп по
  `question_id` (last wins) теперь выполняется до агрегации `run_stats`, поэтому
  `reports/stats.json` (runner.py) и CLI `wtq-translate stats` считают `quality_rows`,
  `avg`/`min`/`metric_counts` и warnings по уникальным вопросам при resume
  (`retry_failed_on_resume`) и `rerun_needs_review`. Сигнатуры и формат отчёта не менялись.
- `quality_stats.py`: ручной dict-дедуп в `_load_quality` заменён на общий
  `dedupe_last_wins` (то же last-wins поведение, убрана дублирующаяся реализация).
- Файлы не переписываются: дедупликация на стороне чтения, раннер и SQLite не менялись.
- **Тесты**: новый `test_jsonl_dedupe.py` (пустые/битые строки, записи без ключа,
  last-wins и порядок, две попытки в events, события без `START`/без `question_id`);
  `test_run_stats.py` — кейс с двумя строками одного `question_id` (метрики и warnings
  из последней строки). Валидация: 13 unit-тестов (run_stats + jsonl_dedupe),
  `test_filter_run`, `test_table_quality`, `test_pipeline_modes` зелёные; на реальном
  ране `test-a02b5003577e` дедуп — no-op (events 8688→8688, quality 4129 уникальных,
  failures 206→206, `stats.json` побайтово идентичен).

## Unreleased — candidate_id зависит от содержимого таблицы

- `candidates.py`: `candidate_id` теперь включает хеш содержимого таблицы (`_table_content_hash`
  от `[table.headers, table.rows]`, то же содержимое, что в `_table_cache_key`, но без языка).
  Перевод кандидата кэшируется per-table: одинаковая таблица → тот же id (переиспользование
  сохранено), разные таблицы → разные id (нет ложных кросс-табличных hits из `candidate_translations`
  и `llm_cache`). Хеш вычисляется один раз в `extract_table_candidates` после проверки `_TABLE_CACHE`
  (на hit не считается); форма id (16 hex) и стабильность в рамках таблицы не изменились —
  shell-протокол «вернуть id 1-в-1» и валидации множеств не затронуты.
- `cache.py`: одноразовая миграция в `SQLiteCache._init` под гвардом `PRAGMA user_version`
  (0 → 1): `candidate_translations` со старыми записями (без table-hash) удаляется и
  пересоздаётся; идемпотентно на свежих БД и при повторной инициализации. `llm_cache`,
  `embedding_cache` и state-таблицы (`StateStore`) не трогаются.
- **Кэш перевода зависит от контекста перевода** (`table_candidates.py`): ключ записи
  `candidate_translations` и `_MEMORY_CACHE` = `candidate.id + \x00 + контекст(модель,
  сид, системный промпт)`. Контекст — `sha256(model.model_name + project.seed +
  candidate_system_prompt(domain))`; берётся конфигурационный сид (не seed попытки/чанка),
  чтобы ключ был детерминирован между ранами и resume работал. Переводы не перетекают
  между моделями/сидами/доменами; `id`-якорь в LLM-промптах и shell-протокол не меняются
  (`llm_cache` кандидатских запросов и раньше изолировал модель/сид/промпт через
  identity+system+seed в `request_hash`). Миграция поднята до `user_version = 2`
  (старые записи по candidate.id очищаются одноразово).
- **Тесты**: `test_candidates.py` — один и тот же источник кандидата в двух разных таблицах
  даёт разные id, в одной таблице + языке id стабилен; `test_cache.py` — миграция очищает
  старые записи и ставит `user_version = 2`, повторная инициализация новые записи не удаляет,
  свежая БД сразу получает `user_version = 2`; `test_table_candidates.py` —
  `test_translation_cache_separated_by_model_seed_domain`: смена модели/сида/домена даёт
  свой ключ кэша (повторный перевод, старые строки целы), повтор с той же конфигурацией —
  cache-hit без вызовов LLM.
- Валидация: 63 unit-теста (5 целевых файлов плана) + `test_pipeline_full_new.py` проходят.
  Примечание: 5 предшествующих failure в `tests/unit/` (транслитерация NER — 2 теста
  `KeyError: 'transliterations'`; `test_question_translation.py` — 3 structured-shell теста;
  `test_alphabet_detection.py` — 1 regex-кейс) воспроизводятся и на HEAD — к данной задаче
  не относятся.

## Unreleased — RUNBOOK (руководство по запуску скриптов)

- Новый `RUNBOOK.md`: руководство по запуску всех исполняемых команд (`wtq-translate` — 11 подкоманд, оба скрипта `scripts/`, `pytest`) с оглавлением-навигацией, подготовкой окружения, аргументами, выводом и типовым рабочим процессом. Документация, код не менялся.

## Unreleased — актуализация TRANSLATION_FULL_REVIEW.md

- Документ приведён в соответствие с текущим кодом: добавлены разделы про алфавитную детекцию (шаг 0.5, `alphabet.py`, статус `excluded`, TSV исключений), стратегии транслитерации (`off/letterwise/quoted`), пер-кандидатную транслитерацию и глоссарий, изоляцию ошибок chunk'ов, режимы `back_table_similarity_mode`, `reporting/stats.py` (`stats.json`), а также новые CLI-инструменты и скрипты (`alphabet-screen`, `entities`, `stats`, `filter` + `scripts/extract_original_examples.py`, `scripts/tsv_to_eval_jsonl.py`).
- Раздел «Результаты и метрики» обновлён по последнему прогону `pristine-unseen-tables-cd2268448ba3` (статусы с `excluded`, метрики по уникальным вопросам, причины отказов по свежему `failures.jsonl`).

## Unreleased — сбор сущностей из таблиц WTQ через spaCy NER (wtq-translate entities)

- **Конфиг** (`config.py`): новая секция `entities` (`EntitiesConfig`): `input_tsv` (опционально, переопределяется `--input-tsv`; относительно `project.input_root`), `output_tsv` / `processed_path` (резолвятся в `resolve_and_validate` под `project.output_root`), `template` (по умолчанию `{cell} is {col}`), `allowed_types` (порядок значим — первое совпадение), `all_types_sep`, `normalize_name` (strip + collapse whitespace), `max_words` (в отсев идут только ячейки с числом слов ≤ max_words, дефолт 5), `table_suffix`, `missing_table: skip|error`, `device: gpu|cpu|auto`, `batch_size`, `checkpoint_every_tables`; валидация положительных batch_size/checkpoint/max_words и непустых allowed_types. Пример — `configs/entities_example.yaml`.
- **Модуль** (`entities/ner.py`, новый): `EntityExtractor` — батч-классификация ячеек запросом `<cell> is <col>` на `en_core_web_trf`; `device=gpu` → `spacy.require_gpu()` до `spacy.load` (без CuPy — `ValueError` с подсказкой `pip install cupy-cudaXXx`); `device=auto` → пробует GPU и при недоступности CuPy/устройства молча падает на CPU (как `wtq-translate run`); отключаются `tagger/parser/attribute_ruler/lemmatizer` (остаются transformer+ner); `nlp.pipe` без `n_process` (spaCy запрещает для trf) с length bucketing; выравнивание спанов **по char-offset**: спаны с `end_char <= len(cell)` — ячейки, спаны колонки отбрасываются (устойчиво к `1,000`/`1950s`/дефисам); `classify` — одиночный запрос (для тестов), `classify_many` — боевой батч; ошибки `OSError/RuntimeError` → понятное сообщение с `python -m spacy download en_core_web_trf`. spaCy импортируется лениво.
- **Реестр** (`entities/registry.py`, новый): `EntitiesRegistry` — TSV `name\ttype\tcount\tall_types` (`|`-сепаратор настраивается), `add` накапливает count и объединяет `all_types` (type первичной встречи), `load` совместим со старым 3-колоночным форматом (`all_types = {type}`) и пропускает заголовок; `ProcessedTables` — список обработанных контекстов по строке на файл; запись файлов атомарная (`tempfile` + `os.replace`).
- **Оркестратор** (`entities/runner.py`, новый): `run_entities(config, input_tsv, progress=...)` — уникальные context-пути входного WTQ-файла, пропуск уже отмеченных таблиц (`skipped_processed`), отсев ячеек (пустые + с числом слов > `max_words` в NER не уходят и в реестр не попадают), dedup-кэш `(cell, col)` уровня запуска, `missing_table=skip` помечает отсутствующую таблицу обработанной, чекпоинты каждые `checkpoint_every_tables` таблиц и в конце (реестр + список обработанных); `remaining_tables` — тотал для прогресс-бара; `EntitiesSummary` (total_tables/skipped_processed/missing_tables/new_entities/updated_entities/output_tsv/processed_path).
- **CLI** (`cli.py`): подкоманда `wtq-translate entities <config> [--input-tsv PATH] [--no-progress]`; tqdm-бар (единица — таблица, stderr) через расширенный `_make_progress_bar(total, ..., desc=, unit=)`; stdout — финальный JSON-отчёт.
- **Тесты**: `tests/unit/test_ner_entities.py` (без GPU/модели): char-offset выравнивание (спан колонки отброшен), `type` = первый по `allowed_types` + полнота `all_types`, неизвестный тип → `None`, порядок `classify_many` при length bucketing, режимы `device` (auto фолбэк на CPU, gpu без CuPy → понятная ошибка); реестр (add/merge, roundtrip, legacy 3 колонки, заголовок, кастомный сепаратор); `ProcessedTables` (load/add/flush, дубликаты); runner (skip обработанных, `missing_table=skip/error`, чекпоинты пишут и реестр, и processed, отсев ячеек по `max_words`) — NER мокается фейковым `nlp.pipe`.

## Unreleased — transliteration method (letterwise / quoted)

- **Конфиг** (`config.py`): `translation.candidates.transliteration_enabled: bool` заменён на enum `transliteration: Literal["off", "letterwise", "quoted"]` (дефолт `"off"`). Миграция старого флага в `model_validator(mode="before")`: `true → "letterwise"`, `false → "off"`; конфликт обоих ключей с противоречивыми значениями → `ValueError`. В `configs/qwen_full.yaml` — `transliteration: "off"` (в кавычках: YAML-1.1 парсер читает голое `off` как bool) с комментарием по трём значениям.
- **Новый метод `quoted`** (`translation/table_candidates.py`): NER-упоминания (PERSON/ORG/GPE/LOC/PRODUCT, идущие подряд — один span) оборачиваются в маркеры «…» в `text` кандидата, в `requirements` добавляется `_QUOTED_TRANSLITERATION_REQUIREMENT`; модель транслитерирует их сама, после валидации ответа Python вырезает «» из переводов (до кэширования/глоссария). Guard по языку — только кириллические цели (`_CYRILLIC_TARGETS`), back-перевод (target=`en`) — no-op.
- **Letterwise/off не меняются**: `transliteration == "letterwise"` — прежний форсинг через `transliterations` + `_TRANSLITERATION_REQUIREMENT`; `"off"` — baseline. Стрип «» применяется только в `quoted`.
- **Тесты**: `_cfg(..., transliteration=...)` вместо `transliteration_enabled`; форсинг-тесты переведены на `"letterwise"`; новые quoted-тесты (один NER-токен, многословный span, не-NER вне кавычек, no-op для `en`, требования промпта) и миграция/конфликт конфига.

## Unreleased — статистика по рану (avg/min метрик, статусы, warnings)

- **Модуль** (`reporting/stats.py`, переписан): `run_stats(run_dir, *, status_counts, total, extra, save=True)` — единый источник правды для `reports/stats.json`; `aggregate_metrics` (avg/min/metric_counts по `KNOWN_METRICS` из `quality.py`, + warnings records/total + quality_rows) и `load_quality_rows` (построчное чтение с пропуском пустых строк).
- **Схема** `reports/stats.json`: к существующим ключам (`run_id`, `total`, `statuses`, `excluded_share`, `*_this_invocation`, tsv-пути) добавлены `status_shares` (count/total по статусам), `quality_rows`, `warnings.{records,total,records_share}` и `avg`/`min`/`metric_counts` по всем 12 метрикам (метрика без значений — `null`/0). Статусы — только из `StateStore`, метрики — только из `quality.jsonl`.
- **Runner** (`runner.py`): inline-словарь в конце `run_dataset_from_tsv` заменён на `run_stats` — новые раны получают полную статистику.
- **CLI** (`cli.py`): подкоманда `wtq-translate stats <config> <questions_tsv> [--run-id]` (как `report`) — для любого готового рана читает статусы из state, вызывает `run_stats` (сохраняет и печатает); при пустом state — ошибка `no state for run`.
- **Экспорт** (`reporting/__init__.py`): `summarize` → `run_stats` (мёртвый код заменён), `append_event`/`append_quality` сохранены.
- **Тесты**: `tests/unit/test_run_stats.py` (avg/min/counts c игнором None, warnings, backward-compat ключи, пустой quality.jsonl, `total=0` без деления на ноль).

## Unreleased — regex-метод алфавитной детекции

- **Конфиг** (`config.py`): новая опция `alphabet_detection.method: Literal["llm", "regex"]` (дефолт `llm` — полная обратная совместимость); `regex` — детерминированная классификация без вызова LLM. В `configs/qwen_full.yaml` включён `method: regex`.
- **Детекция** (`alphabet.py`): `detect_alphabet_intent_rules(question_text)` — компилированные regex с `\b`-границами и `re.IGNORECASE`: `alphabet*|lexicograph*` → intent 1, `letter(s)|character(s)` → intent 4 (приоритет у intent 1); `why = "regex: keyword(s) '…'"` с полными найденными словоформами; ошибок не бывает (`error=None`). В `detect_alphabet_intent` после проверки `enabled` добавлена ветка `method == "regex"` — провайдер в этом режиме не вызывается.
- **Screen** (`alphabet_screen.py`): LLM-провайдер строится только для `method == "llm"`; в regex-режиме `provider=None`, никаких LLM/кэш/лог-вызовов.
- **Тесты**: `test_alphabet_detection.py` (словоформы → intent, ложные друзья → `(None, None)`, приоритет + `why` с обоими словами, диспетчер без вызова провайдера, disabled) и `test_alphabet_screen.py` (regex-отсев без вызова LLM).

## Unreleased — алфавитный отсев датасета без перевода (alphabet-screen)

- **Модуль** (`alphabet_screen.py`, новый): `run_alphabet_screen(config, questions_tsv, ...)` прогоняет `detect_alphabet_intent` по каждому вопросу датасета БЕЗ полного перевода (concurrency — `runner.concurrency`), разделяет на переводимые (kept) и исключаемые (excluded; намерение 1..4 или fail-closed-ошибка детекции с `class=""`).
- **Вывод**: `<output_root>/alphabet-screen/<run_id>/dataset/{stem}.tsv` (kept, исходный формат с extra-полями), `dataset/{stem}-alphabet-excluded.tsv` (class + why), `reports/alphabet-screen-stats.json` (total/kept/excluded/excluded_share/intent_classes/detection_errors).
- **LLM-цепочка**: тот же провайдер+SQLite-кэш+llm_io-лог, что у `api.Translator`, но без embeddings/BERTScore (детекту они не нужны).
- **CLI** (`cli.py`): новая подкоманда `wtq-translate alphabet-screen <config> <questions_tsv> [--no-progress]`, stdout — JSON-статистика; config — конфиг run_full (configs/qwen_full.yaml).
- **Тесты**: `tests/unit/test_alphabet_screen.py` (split kept/excluded, fail-closed, persisted stats).

## Unreleased — alphabet-dependent question detection (FULL mode)

- **Цель**: в режиме `full` после определения домена стадия детектирует вопросы, чья семантика зависит от алфавита языка; такие вопросы **не переводятся** и сохраняются в отдельный TSV `{stem}-alphabet-excluded.tsv` с колонками `(id, utterance, context, targetValue, class, why)` (`class` — намерение из таблицы намерений, `why` — обоснование).
- **Промпт** (`prompts.py`): таблица намерений `ALPHABET_INTENT_TABLE` (алфавитная сортировка / положение и группирование по алфавиту / сравнение по алфавиту / конкретная буква) и системный промпт `ALPHABET_DETECTION_SYSTEM` (русский; `intent_id: null` при отсутствии намерения; при нескольких — наиболее специфичное; `why` на русском); добавлен в `prompt_snapshot`.
- **Детекция** (`alphabet.py`, новый): `detect_alphabet_intent` — один LLM-вызов per question, retry-цикл `model.retries + 1` (паттерн `domain.detect_domain`), валидация `intent_id` (int, не bool, 1..4) и непустого `why`; провал детекции — fail-closed `AlphabetDetection(error=...)`. `AlphabetIntentError` — исключение с `question_id`, `intent_label`, `why`.
- **Пацплайн** (`pipeline.py`): `_process_full` вызывает детекцию сразу после `detect_domain` (LLM #2); при намерении или ошибке детекции бросает `AlphabetIntentError` — вопрос исключается.
- **Конфиг** (`config.py`): новая секция `alphabet_detection.enabled` (дефолт `true`); пример в `configs/qwen_full.yaml`.
- **Раннер** (`runner.py`): новый статус `ProcessingStatus.EXCLUDED`; обработчик `AlphabetIntentError` в главном цикле (без fallback-таблицы и failures.jsonl), событие `EXCLUDED`, `excluded_by_id` + перезапись TSV в `checkpoint()`, resume/rehydrate исключённых из state, `excluded_this_invocation`, `alphabet_excluded_tsv` и `excluded_share = excluded/total` в `stats.json`; `RunSummary.excluded`/`excluded_share`; `run_one_from_tsv` пишет однорядный TSV исключений.
- **CLI** (`cli.py`): вывод `excluded`/`excluded_share` в run-summary; `run-one` при `AlphabetIntentError` печатает `{"question_id", "status": "excluded", "class", "why"}`.
- **Тесты**: `tests/unit/test_alphabet_detection.py` (детекция: успех, null, невалидные ответы/ретраи, не-JSON, disabled, pipeline-level до переводов таблицы) и `tests/unit/test_alphabet_runner.py` (TSV исключений, state, share/stats.json, resume, fail-closed). Интеграционный `test_pipeline_full_new.py` и `test_filter_run.py` обновлены под новые статусы.

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
