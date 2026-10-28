# RUNBOOK — запуск всех скриптов WTQ Translator

Руководство по запуску каждой исполняемой команды проекта: пайплайн перевода, пост-обработка, утилиты, скрипты `scripts/` и тесты. Команды даны для Linux/macOS; на Windows замените `.venv/bin/` на `.venv\Scripts\`.

> Рабочая директория всех команд — корень репозитория. Пакет запускается консольной командой `wtq-translate` (установленной из `pyproject.toml`) либо как модуль: `python -m wtq_translator.cli <...>`.

---

## Содержание

- [0. Подготовка окружения](#0-подготовка-окружения)
- **1. Пайплайн перевода**
  - [wtq-translate validate-config](#wtq-translate-validate-config)
  - [wtq-translate inspect](#wtq-translate-inspect)
  - [wtq-translate audit](#wtq-translate-audit)
  - [wtq-translate run](#wtq-translate-run)
  - [wtq-translate resume](#wtq-translate-resume)
  - [wtq-translate run-one](#wtq-translate-run-one)
- **2. Пост-обработка результатов**
  - [wtq-translate report](#wtq-translate-report)
  - [wtq-translate stats](#wtq-translate-stats)
  - [wtq-translate filter](#wtq-translate-filter)
- **3. Вспомогательные инструменты**
  - [wtq-translate alphabet-screen](#wtq-translate-alphabet-screen)
  - [wtq-translate entities](#wtq-translate-entities)
- **4. Скрипты `scripts/`**
  - [extract_original_examples.py](#extract_original_examplespy)
  - [tsv_to_eval_jsonl.py](#tsv_to_eval_jsonlpy)
- **5. Тесты**
  - [pytest](#pytest)
- [Типовой рабочий процесс](#типовой-рабочий-процесс)

---

## 0. Подготовка окружения

Установка пакета в виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # команда wtq-translate станет доступной
pip install -e ".[dev]"          # если нужны тесты (pytest, pytest-asyncio)
```

Дополнительно:

- **spaCy-модель** (обязательна для режимов `full` и для команд `audit`/`entities` в полном режиме):
  ```bash
  python -m spacy download en_core_web_trf
  ```
  Имя модели задаётся в конфиге: `mentions.spacy_model`.
- **LLM-сервер**: команды перевода требуют доступного OpenAI-совместимого сервера (по умолчанию Ollama на `http://localhost:11434/v1`) или нативного Ollama (`model.provider: ollama`). Адрес и модель — в секции `model` конфига.
- **Embeddings** (если `embeddings.enabled: true`): embedding-модель, например `nomic-embed-text`, на указанном `embeddings.base_url`.
- **BERT-score** (если `bert_score.enabled: true`): тяжёлая зависимость torch; модель `bert-base-multilingual-cased` качается с Hugging Face при первом запуске. Для CI/CPU рекомендуется выключить.
- **Скрипты `scripts/`**: `tsv_to_eval_jsonl.py` требует `pandas` и `tabulate` (в этом репозитории уже стоят в venv).

Входные датасеты лежат в `data/wtq/` (`training.tsv`, `test.tsv`, `pristine-unseen-tables.tsv` — колонки `id | utterance | context | targetValue`). Готовые конфиги — в `configs/`:

| Конфиг | Режим | Назначение |
|---|---|---|
| `configs/qwen_full.yaml` | `full` | Полный перевод (все столбцы), кандидатная таблица, embeddings + BERT-score + обратный перевод. Требует Qwen3-30B и embedding-сервер. |
| `configs/local_qwen.yaml` | `full` | Лёгкий локальный прогон на `qwen2.5:3b`, embeddings и BERT выключены. |
| `configs/qwen_mentions_only.yaml` | `mentions_only` | Перевод только столбцов, связанных с табличными упоминаниями. |
| `configs/qwen_shell_only.yaml` | `shell_only` | Перевод только оболочки вопроса (таблица в LLM не отправляется). |

Результаты каждого рана — в `output/<output_root>/runs/<run_id>/`: `manifest.json`, `dataset/`, `traces/`, `reports/` (`llm_io.jsonl`, `quality.jsonl`, `failures.jsonl`, `events.jsonl`, `stats.json`).

---

## 1. Пайплайн перевода

### wtq-translate validate-config

Проверка конфига и дамп его нормализованного JSON-представления (без обращений к LLM).

```bash
.venv/bin/wtq-translate validate-config configs/local_qwen.yaml
# или configs/qwen_full.yaml, configs/qwen_mentions_only.yaml, configs/qwen_shell_only.yaml
```

**Аргументы:** `config` — путь до YAML-конфига.

**Вывод:** JSON-схема распарсенного `AppConfig` в stdout; при ошибке валидации — понятное сообщение и ненулевой код возврата. Полезно как первый шаг перед прогоном.

---

### wtq-translate inspect

Разбор табличных упоминаний **одного вопроса** без перевода: маски, нормализованный текст, координаты, `source_type`, `match_type`, неоднозначности.

```bash
.venv/bin/wtq-translate inspect configs/local_qwen.yaml data/wtq/training.tsv nt-0
```

**Аргументы:** `config`, `questions_tsv`, `question_id`.

**Вывод:** JSON с полями `question`, `table`, `mentions[]`, `ambiguous[]`. Не делает ни одного LLM-вызова — удобно отлаживать детекцию упоминаний.

---

### wtq-translate audit

Аудит покрытия табличных упоминаний по **всему датасету** без LLM-вызовов. Матчер выбирается по `translation.mode` (`full` → лемма-матчер через spaCy, иначе → точный нормализованный).

```bash
.venv/bin/wtq-translate audit configs/qwen_full.yaml data/wtq/training.tsv \
    --output output/mention-audit.json --max-examples 20
```

**Аргументы:**
- `--output PATH` — сохранить отчёт в JSON-файл (иначе только stdout);
- `--max-examples N` — лимит разобранных вопросов (по умолчанию 20; `0` — все).

**Вывод:** JSON-отчёт со статистикой покрытия. Используется для оценки качества детекции до затрат на перевод.

---

### wtq-translate run

Перевод датасета целиком (все вопросы из TSV). По умолчанию **продолжает** существующий детерминированный раунд (то же поведение, что `resume`); новый независимый раунд — флагом `--fresh`.

```bash
# Полный прогон (embeddings + BERT + обратный перевод)
.venv/bin/wtq-translate run configs/qwen_full.yaml data/wtq/training.tsv

# Лёгкий прогон на локальной модели без embeddings/BERT
.venv/bin/wtq-translate run configs/local_qwen.yaml data/wtq/test.tsv

# Новый независимый раунд с тем же конфигом (флаг --fresh)
.venv/bin/wtq-translate run configs/local_qwen.yaml data/wtq/training.tsv --fresh

# Без прогресс-бара (например, в неинтерактивном окружении/логах)
.venv/bin/wtq-translate run configs/local_qwen.yaml data/wtq/training.tsv --no-progress
```

**Аргументы:**
- `--run-id ID` — зафиксировать идентификатор раунда (по умолчанию вычисляется из конфига+файла);
- `--fresh` — не резюмировать существующее состояние, создать новый `run_id` (если не задан `--run-id`);
- `--no-progress` — отключить tqdm-бар.

**Вывод:** JSON-сводка `run_id`, `total`, `processed`, `skipped`, `failed`, `excluded`, `excluded_share`, `statuses`, пути `output_questions_tsv` и `run_dir`. Каждый LLM-запрос пишется в `<run_dir>/reports/llm_io.jsonl` — основной файл для диагностики модели.

**Требования:** работающий LLM-сервер (и embedding-сервер для `qwen_full.yaml`); для `full` — spaCy-модель; для `qwen_full.yaml` ещё и BERT-score.

---

### wtq-translate resume

Возобновление детерминированного раунда после сбоя/остановки: пропускает уже обработанные вопросы, валидирует сохранённые TSV-артефакты перед пропуском.

```bash
.venv/bin/wtq-translate resume configs/qwen_full.yaml data/wtq/training.tsv

# Продолжить конкретный раунд
.venv/bin/wtq-translate resume configs/local_qwen.yaml data/wtq/training.tsv --run-id test-a02b5003577e
```

**Аргументы:** `--run-id`, `--no-progress` (аналогично `run`).

**Вывод:** та же сводка, что у `run`. Физически `run` без `--fresh` работает как `resume` — команда существует для явности и обратной совместимости.

---

### wtq-translate run-one

Перевод **одного вопроса** (для отладки и быстрых проверок).

```bash
.venv/bin/wtq-translate run-one configs/local_qwen.yaml data/wtq/training.tsv nt-0
```

**Аргументы:** `config`, `questions_tsv`, `question_id`.

**Вывод:** JSON `question_id`, `status`, `translated_question`, `translated_target_value`, `cosine_similarity`, `warnings`. Если вопрос исключён алфавитной детекцией — `{"status": "excluded", "class", "why"}`.

---

## 2. Пост-обработка результатов

### wtq-translate report

Сводка готового раунда из его артефактов.

```bash
.venv/bin/wtq-translate report configs/qwen_full.yaml data/wtq/training.tsv
.venv/bin/wtq-translate report configs/local_qwen.yaml data/wtq/training.tsv --run-id test-a02b5003577e
```

**Аргументы:** `config`, `questions_tsv`, опционально `--run-id` (по умолчанию — детерминированный `run_id` из конфига).

**Вывод:** JSON-отчёт по раунду (статусы, пути артефактов).

---

### wtq-translate stats

Агрегация статистики раунда: статусы + `status_shares`, `excluded_share`, `quality_rows`, warnings и avg/min по всем метрикам качества. Пересчитывает `reports/stats.json` для любого раунда (в т.ч. после ручной пост-обработки).

```bash
.venv/bin/wtq-translate stats configs/qwen_full.yaml data/wtq/training.tsv
.venv/bin/wtq-translate stats configs/qwen_full.yaml data/wtq/pristine-unseen-tables.tsv \
    --run-id pristine-unseen-tables-cd2268448ba3
```

**Аргументы:** `config`, `questions_tsv`, опционально `--run-id`.

**Вывод:** JSON со сводной статистикой; ошибка `no state for run: <id>`, если раунд не найден в `state.sqlite`.

---

### wtq-translate filter

Постфильтрация переведённого датасета по статусам и порогам метрик. Использует **отдельную схему конфига** `FilterConfig`.

```bash
.venv/bin/wtq-translate filter configs/filter_example.yaml \
    output/wtq-ru-full/runs/pristine-unseen-tables-a22315711e3c
```

**Аргументы:**
- `config` — путь до фильтр-конфига (пример: `configs/filter_example.yaml`), где:
  - `status_include` — список принимаемых статусов (пустое/отсутствующее = все);
  - `metrics` — словарь `имя_метрики: нижняя_граница` (неизвестные имена отклоняются как опечатки; `null` — без границы, метрика только отображается);
- `run_dir` — полный путь до каталога раунда.

**Вывод:** JSON-отчёт; результат пишется в `<run_dir>/filtered/<run>-sorted-<params>.tsv` (сортировка по конкурентному рангу, лучшие первыми) + `-stats.json` с причиной каждой отбраковки. Это вход для скриптов из раздела 4.

**Пример фильтр-конфига** (`configs/filter_example.yaml`):

```yaml
status_include:
  - accepted
metrics:
  question_bert_f1: 0.60
  back_table_bert_mean_f1: 0.75
  question_cosine_similarity: null
```

---

## 3. Вспомогательные инструменты

### wtq-translate alphabet-screen

Предфильтрация датасета по алфавитно-зависимому намерению **без полного перевода**: разделяет вопросы на переводимые и исключаемые (сортировка/положение/сравнение по алфавиту, «содержит букву X»). В режиме `alphabet_detection.method: regex` LLM не вызывается вообще.

```bash
.venv/bin/wtq-translate alphabet-screen configs/qwen_full.yaml data/wtq/training.tsv
```

**Аргументы:** `config`, `questions_tsv`, `--no-progress`.

**Результат (вывод JSON в stdout):** `total`, `kept`, `excluded`, `excluded_share`, `intent_classes`, `detection_errors`.

**Файлы на диске:** `<output_root>/alphabet-screen/<run_id>/dataset/<stem>.tsv` (переводимые) и `<stem>-alphabet-excluded.tsv` (класс + причина), плюс `reports/alphabet-screen-stats.json`. Экономит ~7 LLM-вызовов на каждый отсеянный вопрос.

---

### wtq-translate entities

Сбор именованных сущностей из ячеек таблиц через spaCy NER (запрос «`<cell> is <col>`»). Накладывает реестр `name <tab> type <tab> count <tab> all_types` в `<output_root>/entities.tsv`.

```bash
.venv/bin/wtq-translate entities configs/entities_example.yaml \
    --input-tsv data/wtq/pristine-unseen-tables.tsv
```

**Аргументы:**
- `--input-tsv PATH` — переопределяет `entities.input_tsv` из конфига (путь относительно `input_root`);
- `--no-progress` — отключить tqdm-бар.

**Вывод JSON:** `total_tables`, `skipped_processed`, `missing_tables`, `new_entities`, `updated_entities`, `output_tsv`, `processed_path`.

**Ключевые настройки конфига:** `device: gpu|cpu|auto` (gpu без CuPy — понятная ошибка), `batch_size`, `allowed_types`, `max_words`, `checkpoint_every_tables`. Требует spaCy-модель `en_core_web_trf`; прогон идёт с чекпоинтами и возобновляется (список обработанных таблиц).

---

## 4. Скрипты `scripts/`

Оба скрипта работают поверх отфильтрованных датасетов из `<run>/filtered/` и запускаются как обычные Python-модули. Внутри у них заданы пути по умолчанию к конкретному прогону, поэтому **без аргументов они тоже запускаются** — но явные аргументы надёжнее.

### extract_original_examples.py

Собирает строки **исходного** (английского) датасета по `id` из отфильтрованного TSV — для сравнения переведённых вопросов с эталонными примерами. Выходной файл — рядом с входным, суффикс `-original.tsv`.

```bash
# По умолчанию (пути из шапки скрипта)
.venv/bin/python scripts/extract_original_examples.py

# Явно
.venv/bin/python scripts/extract_original_examples.py \
    --tsv "output/wtq-ru-full/runs/pristine-unseen-tables-a22315711e3c/filtered/pristine-unseen-tables-a22315711e3c-sorted-question_bert_f1-0.6|back_table_bert_mean_f1-0.75.tsv" \
    --source data/wtq/pristine-unseen-tables.tsv \
    --output /tmp/examples-original.tsv
```

**Аргументы:** `--tsv` (отфильтрованный вход, откуда берутся `id`), `--source` (TSV оригинального датасета), `--output` (по умолчанию `<вход>-original.tsv`).

**Вывод:** краткий отчёт в stdout (`ID во входе`, `Записей в выходе`). **Код возврата 1**, если часть `id` отсутствует в источнике (файл всё равно создаётся с найденными).

**Детали:** разбор идёт вручную по `\t` с `maxsplit=3` (одна запись = одна физическая строка, многострочные вопросы экранированы как `\n`); экранирование не раскодируется — выход байт-в-байт повторяет источник.

---

### tsv_to_eval_jsonl.py

Конвертирует отфильтрованный TSV в JSONL для eval-скрипта LoRA (формат `test.jsonl`): `{id, query, context, targetValue, table}`, где `table` — markdown-рендер переведённой таблицы (`pandas.DataFrame.to_markdown`), `query` — из `utterance` с раскодированием `\n`.

```bash
# Быстрая проверка на 5 записях
.venv/bin/python scripts/tsv_to_eval_jsonl.py --limit 5

# Явно
.venv/bin/python scripts/tsv_to_eval_jsonl.py \
    --tsv "output/wtq-ru-full/runs/pristine-unseen-tables-a22315711e3c/filtered/pristine-unseen-tables-a22315711e3c-sorted-question_bert_f1-0.6|back_table_bert_mean_f1-0.75.tsv" \
    --output /tmp/test_ru.jsonl
```

**Аргументы:**
- `--tsv` — входной отфильтрованный TSV (по умолчанию — конкретный файл из шапки скрипта);
- `--tables-root PATH` — каталог, относительно которого разрешаются пути из колонки `context` (по умолчанию `<run>/dataset`, вычисляется автоматически);
- `--output PATH` — по умолчанию тот же путь, что у TSV, с расширением `.jsonl`;
- `--limit N` — первые N записей (проверка);
- `--missing-value nan|empty` — как рендерить пустые ячейки (`nan` — pandas по умолчанию, `empty` — пустая строка).

**Вывод:** отчёт в stdout (прочитано/записано, пропущенные таблицы). **Код возврата 1**, если не все записи попали в выход (нет файлов таблиц или битые строки).

**Требования:** `pandas`, `tabulate` (установлены в venv; при отсутствии — подсказка `pip install ...`).

---

## 5. Тесты

### pytest

Все тесты (unit + integration). Интеграционные — с моками LLM, GPU не требуется.

```bash
.venv/bin/pytest -q
# Быстрый прогон с коротким трейсбеком и остановкой на первой ошибке (стиль AGENTS.md):
.venv/bin/pytest -q --tb=short --maxfail=1
# Только unit-тесты:
.venv/bin/pytest -q tests/unit
```

**Требования:** установка `pip install -e ".[dev]"`. `pythonpath` и `testpaths` уже настроены в `pyproject.toml`.

---

## Типовой рабочий процесс

1. **Проверка конфига** → `validate-config configs/qwen_full.yaml`
2. **Оценка детекции упоминаний без затрат** → `audit configs/qwen_full.yaml data/wtq/training.tsv`
3. **Отсев алфавитных вопросов** → `alphabet-screen configs/qwen_full.yaml data/wtq/training.tsv`
4. **Проба на одном вопросе** → `run-one configs/qwen_full.yaml data/wtq/training.tsv nt-0`
5. **Полный прогон** → `run configs/qwen_full.yaml data/wtq/training.tsv`
6. **Возобновление после сбоя** → `resume configs/qwen_full.yaml data/wtq/training.tsv`
7. **Статистика и диагноз** → `stats ...`, `report ...`, `reports/llm_io.jsonl` (каждый запрос/ответ LLM), `reports/failures.jsonl`
8. **Постфильтрация по метрикам** → `filter configs/filter_example.yaml output/wtq-ru-full/runs/<run_id>`
9. **Подготовка eval-данных для LoRA** → `tsv_to_eval_jsonl.py --tsv <filtered/*.tsv>`; восстановление оригиналов — `extract_original_examples.py --tsv <filtered/*.tsv>`