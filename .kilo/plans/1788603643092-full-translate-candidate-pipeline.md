# План: модификация полного перевода (режим `full`) на candidate-based пайплайн

## Цель

Переделать режим `full` (`pipeline.TranslationPipeline._full`) на новую механику:
- поиск сущностей по лемматизированным подпоследовательностям (spaCy + POS);
- маска, оборачивающая исходный фрагмент в открывающий/закрывающий тег;
- перевод таблицы через набор кандидатов-подпоследовательностей (по ярусам длины);
- снятие тегов вместо фазы восстановления.

Режимы `shell_only` и `mentions_only`, а также старые `find_table_mentions`, `mask_question`, `translate_table` и `validate_masks` **остаются без изменений** и продолжают работать как сейчас.

---

## Принятые решения (согласованы с заказчиком)

| # | Вопрос | Решение |
|---|--------|---------|
| 1 | Движок перевода таблицы | Заменить: output_table реконструируется из переводов кандидатов (для `full`). Старый `translate_table` остаётся для `mentions_only`/back-перевода старых режимов. |
| 2 | Область применения | Только `full`. `shell_only`/`mentions_only` не трогаем. |
| 3 | spaCy | Добавить зависимость; имя модели конфигурируется (`mentions.spacy_model`, default `en_core_web_trf`), ленивая загрузка, кэш на процесс; в тестах подменяется лёгкой заглушкой. |
| 4 | Неоднозначность | Резолвимо по построению: кандидат таблицы = лемма-ключ, объединяющий все координаты, переводится один раз. `AmbiguousMention`/`on_ambiguity` для `full` не используются. |
| 5 | Порядок перевода кандидатов | Несколько LLM-запросов по ярусам длины: сначала длина 1, потом длина 2 с глоссарием переводов длины 1, и т.д. |
| 6 | Формат тега | `<TABLE_REF_0000>Красная</TABLE_REF_0000>` — БЕЗ внутренних пробелов. Снятие = удаление `</?TABLE_REF_\d{4}>`. |

Дополнительные проектные решения (зафиксированы в этом плане, легко корректируются):

- **Стоп-слова** (только для однозначных подпоследовательностей вопроса): POS `DET` (артикли/определители), `ADP` (предлоги), `PART` (частицы `to`/`not`), а также притяжательные определители (`PRON` с `Poss=Yes`). Точный набор тегов валидируется на реальном выводе `en_core_web_trf` в ходе реализации.
- **Числа и даты**: токены с `token.like_num == True` или `token.pos_ == "NUM"`, либо `token.ent_type_ == "DATE"`, исключаются только из **одиночных** кандидатов, но могут присутствовать внутри многословной подпоследовательности. В лемма-ключе number/date-токены представлены своей поверхностной формой (raw), word-токены — леммой. **Чисто числовые / чисто датные / immutable (URL, ID) / пустые ячейки не дают кандидатов вообще** — не сопоставляются и не переводятся.
- **Лемма-ключ**: `" ".join(...)` — для word-токенов `lemma.casefold()` с удалением не-алфавитно-цифровых символов из леммы; для number/date-токенов — поверхностная форма. Одинаков для вопроса и таблицы.
- **Репрезентативный исходный текст** кандидата (для перевода): первый встреченный исходный фрагмент (детерминированный порядок обхода таблицы).
- **Идентификатор кандидата**: `sha256(лемма-ключ + source_text + target_lang)[:16]` — стабильный id для кэша переводов.
- **Финальный вопрос**: результат shell-перевода с удалёнными тегами (грамматическая форма — как дала модель). Обратный перевод выполняется над версией С тегами; косинус — по версии без тегов.

---

## Целевой поток данных (режим `full`)

```
question, table
  │
  ├─ nlp = load_spacy_model(cfg.mentions.spacy_model)          # лениво, кэш на процесс
  │
  ├─ find_table_mentions_full(question, table, cfg, nlp)
  │      └─ extract_question_candidates  → [QuestionCandidate]
  │      └─ extract_table_candidates     → TableCandidateSet (дедуп по лемма-ключу,
  │                                         координаты объединены, кэш по sha256(таблицы))
  │      └─ match (по лемма-ключу)       → links
  │      └─ select (longest non-overlap) → MentionSearchResult + candidate links
  │
  ├─ translated_table, cand_translations = translate_table_candidates(...)
  │      └─ ярусы длины 1..max: 1 LLM-запрос на ярус + глоссарий короче
  │      └─ reconstruct_table: замена span'а максимального content-кандидата (raw
  │         source_text с внутренними числами/датами/пунктуацией) на его перевод;
  │         ведущая/хвостовая пунктуация сохраняются; чисто числовые/датные/пустые
  │         ячейки не переводятся; проверка непереведённых ячеек
  │
  ├─ masked_source = mask_question_tagged(question, mentions)
  │      └─ `<TABLE_REF_0000>фрагмент</TABLE_REF_0000>` (без пробелов)
  │
  ├─ masked_ru = translate_shell(..., tag_glossary=per_mask_translations)
  │      └─ модель сохраняет теги, переводит содержимое по глоссарию в нужной форме
  │      └─ validate_tags(masked_source, masked_ru)
  ├─ final_question = strip_tags(masked_ru)                    # БЕЗ тегов
  │
  ├─ answer = build_target_value(..., output_table=translated_table)   # без изменений
  │
  ├─ back_masked = back_translate_shell(masked_ru, tag_glossary=back_glossary)
  │      └─ validate_tags; back_question = strip_tags(back_masked)
  ├─ back_table = back_translate_table_candidates(translated_table, ...)  # кэшируется
  │      └─ validate_table_shape + качество обратного перевода
  │
  └─ _score_question(question.utterance, back_question)        # без тегов
```

---

## Задачи (порядок, каждая проверяема)

### Шаг 0. Зависимость spaCy
Файлы: `pyproject.toml`.
- Добавить `spacy>=3.7` в `dependencies`.
- Документировать в `README` загрузку модели: `python -m spacy download en_core_web_trf`.
- Проверка: `pip install -e .` и `python -m spacy download en_core_web_trf` в venv.

### Шаг 1. Конфиг
Файлы: `config.py`.
- В `MentionsConfig` добавить `spacy_model: str = "en_core_web_trf"` (и, опционально, под-конфиг кандидатов: `stop_pos: list[str]`, `exclude_numbers: bool = True`, `exclude_dates: bool = True`). Поля аддитивные, дефолты сохраняют старые режимы.
- Проверка: существующие тесты конфига/пайплайна проходят; `AppConfig.model_validate` принимает `mentions.spacy_model`.

### Шаг 2. Модели данных
Файлы: `models.py`.
- В `TableMention` добавить `candidate_key: str = ""` (опционально, без нарушения существующих конструкторов).
- Добавить датаклассы: `QuestionCandidate` (lemma_key, source_text, start, end, length), `TableCandidate` (id, lemma_key, source_text, coordinates, length, translation), `CandidateSet` (candidates + per-coordinate максимальные content-run'ы + token layout).
- Проверка: `python -c "import wtq_translator.models"`, старые тесты упоминаний/реставрации проходят.

### Шаг 3. Кандидаты (spaCy)
Файлы: `mentions/candidates.py` (новый).
- `load_spacy_model(model_name)` — ленивый синглтон, `nlp = spacy.load(...)`; при ошибке загрузки модели — понятное исключение.
- Классификация токенов: word / number / date / punct / other.
- `content_runs(doc)` — непрерывные run'ы контентных токенов (всё, кроме пунктуации/пробелов; числа и даты ВХОДЯТ в run), с offset-картой к исходной строке.
- `extract_question_candidates(question, nlp, cfg)` — все подпоследовательности каждого content-run'а (n-граммы 1..len); из одиночных кандидатов отбрасываются стоп-слова и одиночные числа/даты; `QuestionCandidate` со span'ом (start/end по первому/последнему контентному токену) и `source_text` (raw-подстрока, включая внутреннюю пунктуацию).
- `extract_table_candidates(table, nlp, cfg)` — для каждого header (row=-1) и ячейки: content-run'ы → подпоследовательности; чисто числовые/датные/immutable/пустые ячейки пропускаются целиком; дедуп по лемма-ключу с объединением координат и сохранением репрезентативного `source_text`; возвращает `CandidateSet` + layout для реконструкции. Кэш in-memory по `sha256(headers+rows)`.
- Проверка: unit-тесты на заглушке `nlp` (объект с `__call__` → `Doc`-like с `token.lemma_/pos_/like_num/ent_type_/is_punct/idx/id`).

### Шаг 4. Новый матчер
Файлы: `mentions/finder_full.py` (новый).
- `find_table_mentions_full(question, table, cfg, nlp) -> (MentionSearchResult, list[MatchedCandidate])`.
- match: `dict[lemma_key → TableCandidate]`; для каждой question-кандидатуры (включая дубликаты с разными span) ищем лемма-ключ.
- select: сортировка по `(-length, start, end)`, жадный отбор непересекающихся span'ов (длинный «New York City» блокирует вложенные «New York», «York»). Каждый выбранный span → `TableMention` (mask, question_text=фрагмент, normalized_text=лемма-ключ, start/end, coordinates=все координаты кандидата, table_source_text=репрезентативный, match_type="lemma", candidate_key=id кандидата).
- `ambiguous` всегда пуст для `full`.
- Проверка: unit-тесты матчинга/селекции на заглушке.

### Шаг 5. Маскирование тегами + снятие
Файлы: `mentions/masking.py`.
- `mask_question_tagged(question, mentions)` → `<TABLE_REF_XXXX>фрагмент</TABLE_REF_XXXX>` (обратный порядок вставки справа налево, как в старом `mask_question`).
- `strip_tags(text)` → удаление `</?TABLE_REF_\d{4}>`.
- `closing_mask(mask)` → `</TABLE_REF_XXXX>`.
- Старые `mask_question`/`restore_*` не трогаем.
- Проверка: unit-тесты roundtrip `strip_tags(mask_question_tagged(q))` — без лишних пробелов.

### Шаг 6. Валидация тегов
Файлы: `validation/masks.py`.
- `extract_open_tags` / `extract_close_tags` / `validate_tags(expected, actual)` → счётчики открытых и закрытых тегов, проверка сбалансированности (порядок, парность). Старые `extract_masks`/`validate_masks` не трогаем.
- Проверка: unit-тесты (missing/added/несбалансированный тег).

### Шаг 7. Промпты
Файлы: `prompts.py`.
- `SHELL_TRANSLATION_TAGGED_SYSTEM`: перевести весь текст; теги `<TABLE_REF_XXXX>…</TABLE_REF_XXXX>` сохранить без изменений; содержимое тегов переводить словами из предоставленного глоссария (per-tag), в корректной грамматической форме; не добавлять/удалять теги.
- `CANDIDATE_TRANSLATION_SYSTEM`: перевод словаря `id+text`, с глоссарием коротких подпоследовательностей для согласованности.
- Добавить новые промпты в `prompt_snapshot()`.
- Проверка: snapshot содержит новые хэши.

### Шаг 8. Перевод кандидатов таблицы + реконструкция
Файлы: `translation/table_candidates.py` (новый).
- `translate_table_candidates(provider, candidates, config, src, dst)`:
  - группировка кандидатов по длине; для каждого яруса (по возрастанию) один LLM-запрос: `values=[{id,text}]` + `glossary=[{key,translation}]` всех короче; retry/валидация id как в `translate_table`; перевод сохраняется в кандидата.
- `reconstruct_table(table, candidate_set, translations) -> TableData`: для каждого header/ячейки заменить span максимального content-кандидата (raw `source_text`, содержащий внутренние числа/даты/пунктуацию) на его перевод по лемма-ключу; ведущая/хвостовая пунктуация сохраняются verbatim; чисто числовые/датные/immutable/пустые ячейки остаются без изменений (кандидатов нет).
- `validate_untranslated(candidate_set, translations)`: проверить, что у каждого максимального content-run'а есть перевод (иначе ошибка/фолбэк).
- Кэш переводов: `SQLiteCache` — новая таблица `candidate_translations(candidate_id, source_text, target_lang, translation)`; заполняется по мере перевода ярусов; lookup перед LLM-вызовом. Плюс in-memory `dict` на процесс.
- Проверка: unit-тесты на mock-провайдере (несколько запросов по ярусам, наличие глоссария, реконструкция чисел/дат, ошибка непереведённого run'а).

### Шаг 9. Shell-перевод с глоссарием
Файлы: `translation/shell.py`.
- Расширить `translate_shell` опциональным параметром `tag_glossary: dict[str, str] | None = None` (default None → старое поведение, `validate_masks`). При наличии глоссария: system = `SHELL_TRANSLATION_TAGGED_SYSTEM`, в payload добавить per-tag переводы, валидация через `validate_tags`.
- Проверка: unit-тест с mock-провайдером (передаёт глоссарий, валидирует теги).

### Шаг 10. Back-перевод (симметричный + кэш)
Файлы: `backtranslation/table.py`, `backtranslation/shell.py`.
- `back_translate_shell(..., tag_glossary=...)` — прокидывает глоссарий (переводы обратно на исходный язык).
- `back_translate_table_candidates(translated_table, ...)`: строит кандидаты из переведённой таблицы, переводит по ярусам обратно (через тот же `translate_table_candidates` с `src/dst` наоборот), реконструирует back-таблицу; переиспользует кэш `candidate_translations`.
- Проверка: unit-тест симметрии.

### Шаг 11. Пайплайн `_full`
Файлы: `pipeline.py`.
- В `_process` сделать раннюю ветку: `if mode == FULL: return await self._process_full(question, table)` (с сохранением `llm_trace_context` в `process`). Существующая общая часть (старый `find_table_mentions`, маскирование, `_shell_only`, `_mentions_only`) остаётся для остальных режимов.
- `_process_full` реализует поток из раздела выше: поиск → перевод таблицы (ПЕРЕД shell) → теговая маска → shell с глоссарием → strip → answer → back shell → back table → качество → cosine.
- Порядок обязателен: **таблица раньше оболочки**.
- Поля `QualityReport` (original_masks/forward_masks/backward_masks) заполняются открывающими тегами; `ambiguous_mentions` пуст.
- Проверка: integration-тест `full` с mock-провайдером и stub-nlp.

### Шаг 12. Кэш кандидатов в persistence
Файлы: `persistence/cache.py`.
- Методы `get_candidate_translation`/`put_candidate_translation` + инициализация таблицы `candidate_translations` в `_init()`.
- Проверка: unit-тест кэша.

### Шаг 13. Тесты и регресс
Файлы: `tests/unit/test_candidates.py`, `tests/unit/test_finder_full.py`, `tests/unit/test_masking_tagged.py`, `tests/unit/test_tag_validation.py`, `tests/unit/test_table_candidates.py`, `tests/integration/test_pipeline_full_new.py`.
- Убедиться, что существующие тесты `shell_only`/`mentions_only`/`test_unique_cell_translation`/`test_mentions` проходят (старые пути не тронуты).
- Проверка: `pytest -q`.

---

## Риски и граничные случаи

1. **Контракции/клитики в ячейках** (`City's`, `don't`): spaCy токенизирует как `City`+`'s`. Правило: `'s`/клитика причисляется к предыдущему word-токену (не разрывает run) либо считается пунктуацией — зафиксировать и покрыть тестом. В таблицах WTQ редки.
2. **Дефисы/составные токены** (`USL A-League`, `U.S.A.`): полагаемся на токенизацию spaCy; лемма-ключ одинаков с обеих сторон, поэтому матчинг консистентен даже при разных разбиениях.
3. **Числа/даты внутри подпоследовательности** (`4th, Western`, `New York City (2010)`, `12 October 1998`): одиночный number/date-токен не является кандидатом, но входит в многословные кандидаты и в `source_text`. В смешанной ячейке перевод кандидата сохраняет числа/даты (промпт + `source_text` raw); целиком числовая/датная ячейка не переводится вовсе. Ординалы (`4th`) не переводятся.
4. **Датная ячейка как «граница» матчинга**: если ячейка целиком — дата (`12 October 1998`), она не даёт кандидатов, поэтому отдельный word внутри неё (например `October`) тоже не станет табличным кандидатом и не будет заматчен — документируем как ожидаемое поведение.
5. **Грамматическая форма содержимого тега**: модель может изменить форму/род; финальный вопрос берёт форму модели. Косинус по обратному переводу — единственная метрика (детерминированной реконструкции больше нет).
6. **Производительность**: O(N²) подпоследовательностей на вопрос/ячейку — допустимо (WTQ короткие). Кэш таблиц (in-memory) снимает повторную лемматизацию общих таблиц.
7. **`audit` команда**: продолжает использовать старый `find_table_mentions` (не новый full-матчер). Отметить как опциональный follow-up (`audit` для нового матчера).
8. **Пустой вопрос/таблица без слов**: кандидатов нет → перевод таблицы не делает LLM-запросов (как в текущем `translate_table`), вопрос без тегов.

---

## Открытые вопросы (не блокируют реализацию)

- Точный набор stop-POS: валидировать на реальном `en_core_web_trf` (сейчас предложен `DET`, `ADP`, `PART`, притяжательные `PRON`). Возможна правка по результатам audit.
- Нужно ли зеркалить новый матчер в `audit` (отдельная задача) — решить после основной реализации.

---

## Критерий готовности

- `pytest -q` зелёный (старые + новые тесты).
- `run-one`/`run` в режиме `full` на `configs/qwen_full.yaml` отрабатывает end-to-end: финальный вопрос без тегов, таблица реконструирована из кандидатов, числа/даты/пустые не переведены, обратный перевод и косинус работают, кэш кандидатов переиспользуется.
