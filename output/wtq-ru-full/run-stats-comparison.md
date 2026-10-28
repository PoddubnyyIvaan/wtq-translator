# Статистика прогонов перевода WTQ (en → ru)

Источники данных: `output/wtq-ru-full/runs/{run}/reports/` (`stats.json`, `quality.jsonl`, `events.jsonl`, `failures.jsonl`, `llm_io.jsonl`).

| | **Test** (`test-a02b5003577e`) | **Training** (`training-4d5b0e0c616c`) |
|---|---|---|
| Датасет | Test | Training |
| Модель | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 |
| Всего вопросов | 4344 | 14152 |
| Обработано в этом запуске | 4344 | 13250 |
| Строк с качественной оценкой (quality rows) | 4129 | 13250 |
| Успешно (accepted + accepted_with_warnings) | 3736 | 12028 |
| Длительность прогона | 2.81 ч | 9.89 ч |
| LLM-вызовов / токенизаций | 20896 / 101534 | 70468 / 333222 |

---

## 1. Итоговые статусы обработки вопросов

| Статус | Test, шт | Test, доля | Training, шт | Training, доля |
|---|---:|---:|---:|---:|
| accepted | 3390 | 78.04% | 10980 | 77.59% |
| accepted_with_warnings | 346 | 7.97% | 1048 | 7.41% |
| needs_review | 393 | 9.05% | 1222 | 8.63% |
| failed | 206 | 4.74% | 864 | 6.11% |
| excluded | 9 | 0.21% | 38 | 0.27% |

**Активность перевода** (по `events.jsonl`): Test — `START 4344 / DONE 4129 / FAILED 206 / EXCLUDED 9`; Training — `START 14152 / DONE 13250 / FAILED 864 / EXCLUDED 38`.

---

## 2. Качество перевода — метрики по всем вопросам

| Метрика | Test avg | Training avg | Test min | Training min |
|---|---:|---:|---:|---:|
| Cosine similarity (вопрос) | 0.9619 | 0.9630 | 0.6175 | 0.5151 |
| BERT precision (вопрос) | 0.7571 | 0.7590 | 0.0687 | 0.0610 |
| BERT recall (вопрос) | 0.7733 | 0.7729 | 0.1417 | 0.0879 |
| BERT F1 (вопрос) | 0.7650 | 0.7658 | 0.1318 | 0.0920 |
| Cosine min (таблица) | 0.5599 | 0.9004 | 0.2824 | 0.6409 |
| Cosine mean (таблица) | 0.9610 | 0.9732 | 0.7814 | 0.8565 |
| BERT min F1 (таблица) | 0.2353 | 0.7746 | -0.2756 | 0.0859 |
| BERT mean F1 (таблица) | 0.9079 | 0.9182 | 0.5144 | 0.3841 |
| BERT min precision (таблица) | 0.2451 | 0.7862 | -0.2621 | 0.0137 |
| BERT mean precision (таблица) | 0.9096 | 0.9223 | 0.5469 | 0.3310 |
| BERT min recall (таблица) | 0.2088 | 0.7557 | -0.3276 | 0.0968 |
| BERT mean recall (таблица) | 0.9064 | 0.9141 | 0.4819 | 0.4362 |

Средняя длина вопроса: исходный Test/Training — 53.9/54.5 симв., перевод — 55.7/56.7 симв.

---

## 3. Качество по итоговому статусу (усреднённые метрики)

### Test (`test-a02b5003577e`)

| Статус | BERT F1 (вопрос) | BERT mean F1 (таблица) | Cosine mean (таблица) |
|---|---:|---:|---:|
| accepted | 0.7676 | 0.9084 | 0.9610 |
| accepted_with_warnings | 0.7674 | 0.9139 | 0.9624 |
| needs_review | 0.7409 | 0.8983 | 0.9601 |

### Training (`training-4d5b0e0c616c`)

| Статус | BERT F1 (вопрос) | BERT mean F1 (таблица) | Cosine mean (таблица) |
|---|---:|---:|---:|
| accepted | 0.7688 | 0.9193 | 0.9736 |
| accepted_with_warnings | 0.7707 | 0.9264 | 0.9712 |
| needs_review | 0.7349 | 0.9008 | 0.9717 |

---

## 4. Предупреждения (warnings)

| Warning (сгруппировано) | Test, предупреждений | Training, предупреждений |
|---|---:|---:|
| Всего записей с warnings / суммарно предупреждений | 739 / 789 | 2270 / 2453 |
| Нет упоминаний таблицы в вопросе | 393 | 1178 |
| Ответ переведён напрямую (координаты непригодны) | 388 | 1240 |
| Множественные переведённые значения по координатам | 8 | 35 |

Вопросы без найденных упоминаний таблиц (`table_mentions_found=False`): Test — **393**, Training — **1178**. Статус `needs_review`: Test — **393**, Training — **1222** (в Test все попадают в `needs_review`, в Training ещё 44 строки дополнительно).

---

## 5. Сбои (failures)

| Категория ошибки | Test | Training |
|---|---:|---:|
| `RuntimeError`: Invalid JSON returned by model | 64 | 438 |
| `RuntimeError`: Validation errors for response | 0 | 8 |
| `RuntimeError`: Translation ID set mismatch | 14 | 26 |
| `RuntimeError`: Server disconnected | 0 | 4 |
| `RuntimeError`: Empty translations | 0 | 8 |
| `RuntimeError`: Tags changed | 103 | 326 |
| `RuntimeError`: Other runtime | 0 | 0 |
| `RuntimeError`: Model returned duplicate translation IDs | 8 | 54 |
| `ValueError`: Rows have wrong width | 17 | 0 |

---

## 6. LLM-взаимодействие (модель: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)

### 6.1 Вызовы по стадиям (llm_chat)

| Стадия | Test, вызовов | Training, вызовов |
|---|---:|---:|
| detect_domain | 4327 | 14152 |
| translate_shell | 4581 | 14689 |
| back_translate_shell | 4396 | 14067 |
| translate_table_candidates | 4099 | 15202 |
| back_translate_table_candidates | 3096 | 11092 |
| translate_answer | 397 | 1266 |
| **Итого chat** | 20896 | 70468 |
| Токенизаций (llm_tokenize) | 101534 | 333222 |
| Ошибок LLM (chat/tokenize) | 18 / 3 | 82 / 6 |

### 6.2 Ретраи (попытки вызова)

| Попытка | Test, вызовов | Training, вызовов |
|---|---:|---:|
| attempt 0 | 19986 | 66731 |
| attempt 1 | 491 | 1868 |
| attempt 2 | 218 | 969 |
| attempt 3 | 201 | 900 |

---

## 7. Топ доменов таблиц (по freq)

### Test — топ-15

| Домен | Кол-во |
|---|---:|
| sports | 889 |
| sports athletics | 193 |
| auto racing | 134 |
| Olympic Games medal standings | 105 |
| tennis | 101 |
| motor racing | 84 |
| music | 83 |
| geography | 75 |
| television series | 58 |
| entertainment | 45 |
| public transportation | 40 |
| sports competitions | 38 |
| education | 35 |
| politics | 33 |
| Formula 1 racing results | 33 |

### Training — топ-15

| Домен | Кол-во |
|---|---:|
| sports | 3212 |
| music | 410 |
| sports athletics | 377 |
| sports competitions | 329 |
| geography | 299 |
| auto racing | 270 |
| motor racing | 236 |
| Olympic Games medal standings | 188 |
| election results | 156 |
| tennis | 152 |
| college football | 134 |
| video games | 115 |
| demographics | 115 |
| politics | 110 |
| music charts | 88 |
