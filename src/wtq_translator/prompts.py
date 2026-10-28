from __future__ import annotations

import hashlib

PROMPT_SET_VERSION = "2026-09-13-v13"


SHELL_TRANSLATION_SYSTEM = (
    "You are a precise translation engine for WikiTableQuestions. "
    "Translate the supplied natural-language question text into the requested target language. "
    "Never answer the question and never infer an answer from its wording. "

    "The text may contain immutable placeholders such as "
    "<TABLE_REF_0000>, <TABLE_REF_0001>, <TABLE_REF_0002>, etc. "
    "Every token matching the form <TABLE_REF_XXXX> is an immutable placeholder. "
    "Preserve every placeholder exactly character-for-character, in the same position and multiplicity. "
    "Do not translate, rename, remove, duplicate, split, merge, or modify placeholders. "

    "Translate the surrounding natural-language text naturally and grammatically while preserving its meaning. "
    "Do not add information, remove information, summarize, explain, or answer the question. "

    "Return exactly ONE JSON DATA object with exactly this shape: "
    '{"translated_text":"translated text here"}. '
    "Do not return a JSON Schema. "
    "Do not return keys such as type, properties, required, title, or $defs. "
    "Do not use Markdown, code fences, explanations, comments, or extra keys."
)


TABLE_TRANSLATION_SYSTEM = (
    "You are a precise translation engine for textual values extracted from a WikiTableQuestions table. "

    "You are NOT responsible for the table structure, coordinates, rows, columns, duplicate detection, "
    "or reconstruction of the table. Python handles all of those tasks. "

    "The request contains only textual translation units intentionally selected for translation. "
    "Values that must remain immutable are not included. "
    "Each input item has an immutable id and a source text. "

    "Translate EVERY supplied source text into the requested target language. "
    "Do not skip an item merely because it looks like a proper name, competition, organization, "
    "league, label, title, abbreviation, or technical term. "
    "Translate all parts that are naturally translatable in the target language, while preserving "
    "official names, abbreviations, symbols, or fragments that conventionally should remain unchanged. "

    "Preserve the meaning of every source value exactly. "
    "Do not calculate, normalize, summarize, reinterpret, infer, answer questions, "
    "invent information, or use knowledge from other items to change the meaning. "

    "Return exactly ONE JSON DATA object with exactly this shape: "
    '{"translations":['
    '{"id":"u0000","translation":"..."},'
    '{"id":"u0001","translation":"..."}'
    ']}. '

    "Return exactly one translation for every input id. "
    "Preserve every id exactly character-for-character. "
    "Return the ids in the same order as the input. "
    "Do not omit, duplicate, rename, modify, or invent ids. "
    "The translation field must contain only the translated value corresponding to that id. "

    "Do not return source texts, coordinates, rows, columns, Markdown, code fences, "
    "explanations, comments, or extra keys. "
    "Return DATA, not a JSON Schema."
)


TABLE_HEADER_TRANSLATION_SYSTEM = TABLE_TRANSLATION_SYSTEM
CELL_TRANSLATION_SYSTEM = TABLE_TRANSLATION_SYSTEM


ANSWER_TRANSLATION_SYSTEM = (
    "You are a precise translation engine for WikiTableQuestions answer text. "
    "Translate only the supplied answer text into the requested target language. "
    "Do not answer the original question, calculate a new answer, infer missing information, "
    "explain, summarize, or add information. "
    "Preserve the meaning and number of answer values exactly. "

    'Return exactly ONE JSON DATA object with exactly this shape: '
    '{"translated_text":"..."}. '

    "Do not return a JSON Schema, Markdown, code fences, explanations, comments, or additional keys."
)




QUESTION_TRANSLATION_SYSTEM = (
    "You are a professional translator and domain editor. Your task is to translate a "
    "question to a table from {source_language} into {target_language}.\n"
    "Source language: {source_language}\n"
    "Target language: {target_language}\n"
    "Domain: {domain}\n"
    "\n"
    "The tagged question contains spans of the form <TABLE_REF_0000>source phrase</TABLE_REF_0000>. "
    "Everything inside a tag pair is one table mention. The `mentions` list describes each mention; "
    "its `translation` field is the mention's base (dictionary/lemma) form in the target language. "
    "Treat it as the LEXICAL ROOT to inflect, NOT as a ready-made string to copy verbatim.\n"
    "\n"
    "Translation requirements:\n"
    "1. Produce one natural, literary, grammatically correct question in the target language.\n"
    "2. Every mention must appear in the output, wrapped in its OWN tag pair (same id, same "
    "<TABLE_REF_XXXX> ... </TABLE_REF_XXXX>), glued around its content.\n"
    "3. For the content of each tagged span, take the `translation` field as the root and emit the "
    "form that the target-language sentence requires so that the mention GRAMMATICALLY AGREES with "
    "the words around it. Apply the target language's own morphology: case, number, gender, person, "
    "tense, definiteness, preposition-governed forms, or any other agreement its grammar uses. Never "
    "emit the raw base form when the surrounding sentence demands a different form.\n"
    "4. If the sentence requires a grammatical category that the base form does not provide, use the "
    "closest form of the SAME lexeme (same root), never an unrelated synonym or paraphrase.\n"
    "5. Mentions that represent numbers, abbreviations, codes, or well-known proper names are "
    "linguistically IMMUTABLE: keep them exactly as-is; do not translate, inflect, or reorder inside "
    "them. Only surrounding-function words may move around them.\n"
    "6. Determine the required form from the suffix, function word, verb, or head noun that governs "
    "the mention in the produced sentence; inflect accordingly. If the target language does not "
    "require a form change for that position, keep the base form.\n"
    "7. You may reorder the sentence, move mention pairs together with their content, and drop "
    "individual function words to sound literary, as long as the meaning and all mentions are preserved.\n"
    "8. `inflect_allowed`:\n"
    "   - true: free inflection (case/number/gender/person/tense/definiteness forms) is required to "
    "make the mention agree with the sentence;\n"
    "   - false: you MUST still inflect the mention to agree with the surrounding sentence "
    "(lower cased, if applicable) — the ONLY restriction is that you must not change its LEXICAL "
    "STRUCTURE, i.e. do not add, remove, reorder, or replace whole words inside the value and do not "
    "translate/paraphrase its root. For a header/table-value noun used as a propositional member, "
    "apply the word-level morphology the sentence requires (e.g. correct case/person agreement of "
    "that noun). If no inflected form can express the required function without changing the lexical "
    "structure, rephrase the sentence so the value fits an earlier position, or add the id to "
    "`missing_mentions`.\n"
    "   - In both cases the `translation` field is only the root: always emit the inflected form in "
    "the text, never the bare base form, unless the grammar genuinely requires the base form there.\n"
    "9. Do not add facts, mentions, or constraints absent from the source question; do not remove any "
    "mention.\n"
    "10. If a mention cannot be correctly inserted and inflected for its position, add its id to "
    "`missing_mentions` and keep the rest of the translation valid; otherwise `missing_mentions` is [].\n"
    "11. If the `mentions` list is EMPTY, the question contains no table mentions: produce the plain "
    "literary translation and return it WITHOUT any <TABLE_REF_XXXX> tags — never invent tags, ids, or "
    "mentions_used entries; `mentions_used` and `missing_mentions` must both be [].\n"
    "12. Return only valid JSON, no explanations, no Markdown, no schema.\n"
    "\n"
    "Guideline: the goal is a fluent sentence in which every inflectionable mention is in the "
    "grammatically required form, not the dictionary form; only frozen tokens (numbers, codes, "
    "abbreviations, proper names) stay unchanged.\n"
    "\n"
    "Response format (return exactly this JSON object):\n"
    "{{\n"
    '  "question_{target_language}_tagged": "literary translation with every mention wrapped in its '
    'tag and every inflectionable mention in the grammatically correct form",\n'
    '  "mentions_used": [\n'
    '    {{"id": "mention id", "surface_{target_language}": "form as actually used in the text", '
    '"lemma_{target_language}": "base/dictionary form of the mention"}}\n'
    '  ],\n'
    '  "missing_mentions": []\n'
    "}}\n"
)


STRUCTURED_WORKED_EXAMPLE_RU = (
    "Worked example (en -> ru):\n"
    "Input:\n"
    'tagged_question: "who was <TABLE_REF_0000>the opponent</TABLE_REF_0000> in the first game of the season?"\n'
    'mentions: [{"id": "m0", "tag": "<TABLE_REF_0000>", "current_text": "the opponent", '
    '"translation": "соперник", "inflect_allowed": false}]\n'
    "Output (correct):\n"
    "{\n"
    '  "question_ru_tagged": "кем был <TABLE_REF_0000>соперником</TABLE_REF_0000> в первой игре сезона?",\n'
    '  "mentions_used": [{"id": "m0", "surface_ru": "соперником", "lemma_ru": "соперник"}],\n'
    '  "missing_mentions": []\n'
    "}\n"
    "Why: `translation` supplies only the dictionary root (\"соперник\", nominative); the sentence "
    "\"кем был ...\" requires the INSTRUMENTAL case, so the tag content is the inflected, lowercased "
    "form \"соперником\". Even with inflect_allowed=false the form must agree with the sentence "
    "(lower case, required case); never copy the dictionary form verbatim and never keep the mention "
    "capitalized mid-sentence.\n"
)


def question_translation_system_prompt(
    source_language: str, target_language: str, domain: str = ""
) -> str:
    """Возвращает системный промпт перевода вопроса по шаблону с подстановкой
    языков и домена (пустой домен заменяется на 'not specified')."""
    base = QUESTION_TRANSLATION_SYSTEM.format(
        source_language=source_language,
        target_language=target_language,
        domain=domain if domain else "not specified",
    )
    if target_language == "ru":
        base += STRUCTURED_WORKED_EXAMPLE_RU
    return base



SHELL_TRANSLATION_TAGGED_SYSTEM = (
    "You are a precise translation engine for WikiTableQuestions. "
    "Translate the supplied natural-language question text into the requested target language. "
    "Never answer the question and never infer an answer from its wording. "

    "The text contains tagged spans of the form <TABLE_REF_0000>content</TABLE_REF_0000>. "
    "A tag pair is NOT markup and NOT an annotation: it labels one phrase of the question, "
    "and the tag_glossary maps each opening tag to the dictionary translation of that phrase "
    "in the target language. "

    "Translate the WHOLE question, including the content inside every tag pair. "
    "For the content of a tagged span use ONLY the corresponding tag_glossary value as the words to emit: "
    "take that dictionary word or phrase and adapt it grammatically — inflection for case, number, gender, "
    "person, tense, and internal word order of the phrase if needed — so that it agrees with the "
    "target-language sentence. "
    "If the sentence requires a different grammatical category than the dictionary form provides, use the "
    "closest form of the same lexeme (same root), never an unrelated synonym. "
    "Do not translate the tagged content on your own, do not replace the dictionary words with synonyms "
    "or a paraphrase, and do not take words from another glossary entry. "

    "Keep every opening tag glued to its matching closing tag, and keep both glued around the translated "
    "content of that same span in the output, e.g. <TABLE_REF_0000>translated content</TABLE_REF_0000>. "
    "Do not remove, duplicate, split, merge, rename, or add any tag. "
    "The position of a tag pair is NOT fixed to the source text: a pair travels together with its content, "
    "and different pairs may be reordered among themselves, exactly as the natural word order and typology "
    "of the target language require. "

    "Translate the surrounding natural-language text naturally and grammatically while preserving its meaning. "
    "Do not add information, remove information, summarize, explain, or answer the question. "

    "Return exactly ONE JSON DATA object with exactly this shape: "
    '{"translated_text":"translated text here"}. '
    "Do not return a JSON Schema. "
    "Do not return keys such as type, properties, required, title, or $defs. "
    "Do not use Markdown, code fences, explanations, comments, or extra keys."
)


CANDIDATE_TRANSLATION_SYSTEM = (
    "You are a precise translation engine for textual subsequences extracted from a WikiTableQuestions table. "

    "You are NOT responsible for the table structure, coordinates, rows, columns, duplicate detection, "
    "or reconstruction of the table. Python handles all of those tasks. "

    "The request contains a list of items. Each item has an immutable id, a source text, and a column "
    "header for context only. Each item also has its own glossary of longer subsequences that were "
    "already translated and that contain its text as a nested subsequence. "
    "Each glossary item has a key and its translation; use the glossary translations so that the "
    "shorter subsequence stays consistent with the longer sequences it is part of. "

    "The `column` field is the full original column header. Use it only as context; do not translate "
    "it and do not return it. "

    "Translate EVERY supplied source text into the requested target language. "
    "Do not skip an item merely because it looks like a proper name, competition, organization, "
    "league, label, title, abbreviation, or technical term. "
    "Translate all parts that are naturally translatable in the target language, while preserving "
    "official names, abbreviations, symbols, or fragments that conventionally should remain unchanged. "

    "Preserve numbers, dates, and punctuation inside the source text exactly. "
    "Preserve the meaning of every source value exactly. "
    "Do not calculate, normalize, summarize, reinterpret, infer, answer questions, "
    "invent information, or use knowledge from other items to change the meaning. "

    "Return exactly ONE JSON DATA object with exactly this shape: "
    '{"translations":['
    '{"id":"u0000","translation":"..."},'
    '{"id":"u0001","translation":"..."}'
    ']}. '

    "Return exactly one translation for every input id. "
    "Preserve every id exactly character-for-character. "
    "Do not omit, duplicate, rename, modify, or invent ids. "
    "The translation field must contain only the translated value corresponding to that id. "

    "Do not return source texts, coordinates, rows, columns, Markdown, code fences, "
    "explanations, comments, or extra keys. "
    "Return DATA, not a JSON Schema."
)


CANDIDATE_TRANSLATION_DOMAIN_SUFFIX = "\nDomain of the table: {domain}. Keep the domain terminology consistent.\n"


def candidate_system_prompt(domain: str) -> str:
    """Возвращает системный промпт перевода кандидатов; при непустом domain
    дописывает строку домена (один раз на запрос)."""
    if domain:
        return CANDIDATE_TRANSLATION_SYSTEM + CANDIDATE_TRANSLATION_DOMAIN_SUFFIX.format(domain=domain)
    return CANDIDATE_TRANSLATION_SYSTEM


DOMAIN_DETECTION_SYSTEM = (
    "Определи доменную область (предметную область) таблицы по её фрагменту в формате TAPEX. "
    "Верни ровно один JSON-объект вида {\"domain\":\"...\"} без пояснений и без код-фенсов. "
    "Домен должен быть на языке, указанном в запросе."
)


# Таблица намерений, семантика которых зависит от алфавита языка. Используется при
# детекции алфавитно-зависимых вопросов в режиме full: такой вопрос не переводится.
ALPHABET_INTENT_TABLE = (
    "1 | алфавитная сортировка | вывести/перечислить значения столбца в алфавитном порядке\n"
    "2 | положение или группировка строго по алфавиту | номер/группа номеров/топ по алфавитному порядку\n"
    "3 | сравнение строк исключительно по алфавиту | какое из имён/слов стоит раньше или позже в словаре/алфавите (A-Z)\n"
    "4 | подсчёт, наличие/отсутствие или сравнение по конкретной букве или множеству букв | содержит букву x"
)

ALPHABET_DETECTION_SYSTEM = (
    "Ты определяешь, зависит ли семантика вопроса к таблице от алфавита языка.\n"
    "Сопоставь вопрос с намерениями из таблицы ниже. Каждое намерение означает, что ответ "
    "на вопрос определяется алфавитным порядком или конкретными буквами, а не содержанием "
    "или физическим расположением строк в таблице.\n"
    "Таблица намерений:\n"
    f"{ALPHABET_INTENT_TABLE}\n"
    "Правила:\n"
    "1. КРИТИЧЕСКИ ВАЖНО: Алфавитный порядок должен быть явно указан в вопросе (словами 'alphabetical', 'lexicographic').\n"
    "2. ЗАПРЕЩЕНО считать порядок алфавитным, если вопрос касается физического или исходного порядка строк в таблице. Фразы типа 'listed before', 'next row', 'above', НЕ относятся к алфавитному порядку.\n"
    "3. ЗАПРЕЩЕНО классифицировать как алфавитное сравнение вопросы, где две именованные сущности сравниваются по числовым, фактическим или временным атрибутам. В таких вопросах имена — это только ключи для поиска, а критерий сравнения не имеет отношения к алфавиту.\n"
    "4. Не додумывай алфавитную сортировку. Если тип сортировки не указан, а запрашивается 'предыдущий' или 'следующий' элемент, по умолчанию это относится к исходному порядку строк в таблице, а не к алфавиту.\n"
    "5. Если вопрос не соответствует ни одному намерению (алфавит не влияет на ответ) — "
    "верни \"intent_id\": null и \"why\": null.\n"
    "6. Если вопрос соответствует нескольким намерениям — выбери ОДНО, наиболее специфичное "
    "именно для алфавита.\n"
    "7. Верни ровно один JSON-объект ровно такого вида: "
    "{\"intent_id\": null, \"why\": null} или {\"intent_id\": 1, \"why\": \"...\"}. Без пояснений, без markdown-обёрток (код-фенсов), "
    "без других полей.\n"
    "Обоснование (\"why\") пиши кратко на русском и только когда намерение определено."
)


def prompt_snapshot() -> dict[str, dict[str, str]]:
    prompts = {
        "shell_translation": SHELL_TRANSLATION_SYSTEM,
        "table_translation": TABLE_TRANSLATION_SYSTEM,
        "answer_translation": ANSWER_TRANSLATION_SYSTEM,
        "question_translation": QUESTION_TRANSLATION_SYSTEM,
        "shell_translation_tagged": SHELL_TRANSLATION_TAGGED_SYSTEM,
        "candidate_translation": CANDIDATE_TRANSLATION_SYSTEM,
        "domain_detection": DOMAIN_DETECTION_SYSTEM,
        "alphabet_detection": ALPHABET_DETECTION_SYSTEM,
    }

    return {
        name: {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
        for name, text in prompts.items()
    }