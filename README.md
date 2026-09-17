# WTQ Translator v0.8

Controlled translation pipeline for WikiTableQuestions with deterministic table mentions, three translation modes and mandatory back translation.

**Поведение v0.8:** Python извлекает из выбранной части таблицы только точные уникальные текстовые значения, которые действительно требуют перевода, и отправляет их модели одним списком `id + text`. Координаты, строки, столбцы, immutable-значения и невыбранные значения модели не передаются. Модель возвращает только `id + translation`, после чего Python сам вставляет один перевод во все сохранённые координаты. Поэтому точные дубликаты физически не могут разойтись. `full` использует все столбцы, `mentions_only` — только столбцы найденных упоминаний, а `shell_only` не отправляет таблицу в LLM вообще. Обратный перевод использует тот же механизм в той же области столбцов.

## Реализовано

- TSV reading/writing with preservation of unknown question columns (`targetCanon` и др.);
- deterministic table-mention matching after basic normalization;
- maximal-fragment / longest-match-first selection;
- ambiguity detection without LLM guessing;
- repeated identical mentions in one column are retained as one logical mention with all coordinates;
- stable `<TABLE_REF_XXXX>` masks and strict multiplicity validation;
- exact reconstruction of the original question fragment after masking, including original case/punctuation;
- `full`, `shell_only`, `mentions_only`;
- mention reinsertion only from linked translated table coordinates;
- exact duplicate source values are translated once and propagated by Python to all stored coordinates;
- one-request unique-value translation with strict response-ID / immutable-value / non-empty-value checks;
- mandatory back translation of every component changed in the forward pass;
- embedding-based (semantic) similarity for the back-translated table (context/row/cell modes);
- optional question cosine similarity through a separate embedding provider;
- answer matching and coordinate-based recovery;
- OpenAI-compatible chat and embedding providers;
- SQLite cache for chat and embedding requests;
- dataset-wide `run` / `resume`;
- per-question output table;
- run-specific dataset artifacts: independent runs no longer overwrite each other;
- state protection against reusing a `run_id` with another config or source split;
- resume validates persisted TSV artifacts before skipping them;
- atomic TSV/table writes and full-source fallback on failure;
- manifest with seed, config, exact prompt snapshots/hashes, model parameters and library versions;
- traces, events, failures, quality reports and stats;
- local RNG seeding where Python/NumPy/Torch are available;
- `audit` command for measuring exact table-mention coverage before any LLM calls.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -e .
```

For the `full` (candidate-based) translation mode, a spaCy model is used for lemmatization:

```bash
python -m spacy download en_core_web_trf
```

The model name is configured via `mentions.spacy_model` (default `en_core_web_trf`).

For tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## CLI

Validate config:

```bash
wtq-translate validate-config configs/local_qwen.yaml
```

Inspect mention detection for one question without translating:

```bash
wtq-translate inspect configs/local_qwen.yaml data/wtq/train.tsv nt-0
```

Audit the whole split without LLM calls:

```bash
wtq-translate audit configs/local_qwen.yaml data/wtq/train.tsv --output mention-audit.json
```

Translate one question:

```bash
wtq-translate run-one configs/local_qwen.yaml data/wtq/train.tsv nt-0
```

Translate a split:

```bash
wtq-translate run configs/local_qwen.yaml data/wtq/train.tsv
```

Resume the deterministic run:

```bash
wtq-translate resume configs/local_qwen.yaml data/wtq/train.tsv
```

Create a new independent run with the same config:

```bash
wtq-translate run configs/local_qwen.yaml data/wtq/train.tsv --fresh
```

Report:

```bash
wtq-translate report configs/local_qwen.yaml data/wtq/train.tsv
```

## Output

Every run owns its dataset artifacts, so a later experiment cannot overwrite an older run:

```text
output/wtq-ru/
├── state.sqlite
└── runs/
    └── <run_id>/
        ├── manifest.json
        ├── dataset/
        │   ├── train.tsv
        │   └── tables/
        │       └── train/
        │           ├── nt-0.tsv
        │           └── ...
        ├── traces/
        │   └── train/*.json
        └── reports/
            ├── events.jsonl
            ├── failures.jsonl
            ├── quality.jsonl
            └── stats.json
```

The `context` field in the output question TSV remains relative to that run's `dataset/` directory, for example `tables/train/nt-0.tsv`.

## Current quality policy

The system performs hard deterministic checks for masks and table structure. Back-translated question and table checks are optional and independently toggled:

- **Question cosine similarity** — gated by `validation.question_cosine_enabled` (default `true`). Records `question_cosine_similarity`; an optional `validation.question_cosine_threshold` (not fixed intentionally) marks `NEEDS_REVIEW` when below it. The numeric threshold is intentionally not fixed yet; it must be calibrated on a manually reviewed WTQ sample. If embeddings are disabled, the run records a warning instead of pretending the cosine check happened.
- **Back-table cosine similarity** — gated by `validation.back_table_similarity_enabled`. Granularity is set by `validation.back_table_similarity_mode`: `context` (header plus greedy chunked rows), `row` (header and rows compared separately) or `cell` (per-cell). The report records `min`/`mean` similarity across the comparison units. An optional `back_table_similarity_threshold` is intentionally not fixed and must be calibrated for the chosen mode; when set and `min` falls below it, a warning plus `NEEDS_REVIEW` is recorded. If embeddings are disabled, a warning is recorded instead.
- **BERT-score** — master switch `bert_score.enabled` (default `false`) plus per-site gates `validation.question_bert_enabled` and `validation.back_table_bert_enabled` (both default `true`, effective only while the master switch is on). When enabled, the question is scored against its back translation and records `question_bert_f1`/`question_bert_precision`/`question_bert_recall`; the back-translated table is scored per comparison unit (same granularity/modes as cosine) and records `back_table_bert_groups_compared` plus `back_table_bert_min_*`/`back_table_bert_mean_*` for `f1`, `precision` and `recall`. Optional `bert_score.question_threshold` / `bert_score.table_threshold` mark `NEEDS_REVIEW` when the question `f1` (resp. table `min_f1`) falls below them; `null` = diagnostics only. `bert_score.model_type`, `lang`, `device`, `batch_size` and `rescale_with_baseline` tune the underlying `bert-score` scorer. If BERT loading/inference fails, the fields stay `None` and a warning is recorded; the status is untouched. Because `bert-score` pulls in torch, the heavy packages are imported lazily, the model is loaded once per process and scoring runs via `asyncio.to_thread`; keep `runner.concurrency` moderate when BERT is enabled.

## Изменения v0.8: словарь уникальных значений

Для прямого перевода таблицы LLM получает только такой тип данных:

```json
{
  "task": "translate_table_values",
  "source_language": "en",
  "target_language": "ru",
  "values": [
    {"id": "u0000", "text": "Year"},
    {"id": "u0001", "text": "USL A-League"}
  ]
}
```

Модель возвращает:

```json
{
  "translations": [
    {"id": "u0000", "translation": "Год"},
    {"id": "u0001", "translation": "USL А-лига"}
  ]
}
```

Связь `id → координаты` существует только внутри Python. Числа, пустые значения, URL/идентификаторы, невыбранные столбцы и прочие immutable-значения в запрос не попадают. В `mentions_only` список строится только из столбцов, связанных с табличными упоминаниями; в `shell_only` такого запроса нет.

## Изменения v0.6: таблицы и LLM-логи

Перевод таблицы теперь работает по принципу translation memory. Заголовки переводятся один раз. Затем первое вхождение каждого уникального текстового значения ячейки переводится один раз с контекстом строки и столбца. Все точные дубликаты этого исходного значения получают тот же перевод без нового LLM-запроса.

При `run-one` и `run` каждый LLM-запрос и каждый сырой ответ модели сохраняются в:

```text
output/wtq-ru/runs/<run_id>/reports/llm_io.jsonl
```

В каждой JSONL-записи есть `question_id`, `stage`, `attempt`, полный system/user prompt и сырой `response` либо `error`. Это основной файл для диагностики Qwen.

Локальный конфиг по умолчанию использует:

```yaml
model_name: qwen2.5:3b
```

Поле `table.rows_per_chunk` сохранено только ради совместимости конфигов. В v0.8 таблица не отправляется модели целиком: Python формирует список уникальных переводимых значений только из выбранных столбцов, а после одного LLM-ответа сам собирает таблицу по сохранённым координатам.
