"""Unit-тесты candidates: лемма-ключи, классификация токенов, дедуп, id кандидата."""
from wtq_translator.config import MentionsConfig
from wtq_translator.models import TableCoordinate, TableData
from wtq_translator.mentions.candidates import (
    extract_question_candidates,
    extract_table_candidates,
)


class _Morph:
    def __init__(self, poss=None):
        self._poss = poss

    def get(self, key):
        if key == "Poss" and self._poss is not None:
            return [self._poss]
        return []


class _Tok:
    def __init__(self, text, idx, *, lemma=None, pos="", num=False, date=False, punct=False, poss=None, ent=""):
        self.text = text
        self.idx = idx
        self.lemma_ = text.lower() if lemma is None else lemma
        self.pos_ = pos
        self.like_num = num
        self.ent_type_ = "DATE" if date else ent
        self.is_punct = punct
        self.is_space = False
        self.morph = _Morph(poss)


def make_nlp(mapping):
    """Stub-nlp: `__call__(text)` возвращает итерируемый список токенов для заданного текста."""
    def nlp(text):
        return list(mapping[text])
    return nlp


def test_lemma_key_casefold_and_normalization():
    nlp = make_nlp({
        "New York": [_Tok("New", 0, pos="PROPN"), _Tok("York", 4, pos="PROPN")],
    })
    cands = extract_question_candidates("New York", nlp, MentionsConfig())
    keys = {c.lemma_key for c in cands}
    assert "new york" in keys
    ny = next(c for c in cands if c.lemma_key == "new york")
    assert ny.source_text == "New York"
    assert (ny.start, ny.end) == (0, 8)


def test_number_excluded_single_kept_in_multiword():
    nlp = make_nlp({
        "4 teams": [_Tok("4", 0, num=True), _Tok("teams", 2, pos="NOUN", lemma="team")],
    })
    keys = {c.lemma_key for c in extract_question_candidates("4 teams", nlp, MentionsConfig())}
    assert "4" not in keys
    assert "team" in keys
    assert "4 team" in keys


def test_date_excluded_single():
    nlp = make_nlp({"October": [_Tok("October", 0, date=True)]})
    assert extract_question_candidates("October", nlp, MentionsConfig()) == []


def test_content_run_not_broken_by_number():
    nlp = make_nlp({
        "World War 2 facts": [
            _Tok("World", 0, pos="PROPN"),
            _Tok("War", 6, pos="PROPN"),
            _Tok("2", 10, num=True),
            _Tok("facts", 12, pos="NOUN"),
        ],
    })
    keys = {c.lemma_key for c in extract_question_candidates("World War 2 facts", nlp, MentionsConfig())}
    assert "world war 2 facts" in keys


def test_question_stop_words_discarded():
    nlp = make_nlp({
        "the team": [_Tok("the", 0, pos="DET"), _Tok("team", 4, pos="NOUN")],
        "his team": [_Tok("his", 0, pos="PRON", poss="Yes"), _Tok("team", 4, pos="NOUN")],
    })
    cfg = MentionsConfig()
    det_keys = {c.lemma_key for c in extract_question_candidates("the team", nlp, cfg)}
    assert "the" not in det_keys
    assert "the team" in det_keys
    poss_keys = {c.lemma_key for c in extract_question_candidates("his team", nlp, cfg)}
    assert "his" not in poss_keys
    assert "his team" in poss_keys


def test_table_stop_words_not_discarded():
    table = TableData(["the"], [["a"]])
    nlp = make_nlp({
        "the": [_Tok("the", 0, pos="DET")],
        "a": [_Tok("a", 0, pos="DET")],
    })
    keys = {c.lemma_key for c in extract_table_candidates(table, nlp, MentionsConfig(), "ru").candidates}
    assert "the" in keys
    assert "a" in keys


def test_table_pure_numeric_and_date_cells_skipped():
    table = TableData(["", ""], [["2010", "12 October 1998"]])
    nlp = make_nlp({
        "12 October 1998": [
            _Tok("12", 0, num=True),
            _Tok("October", 3, date=True),
            _Tok("1998", 11, num=True),
        ],
    })
    result = extract_table_candidates(table, nlp, MentionsConfig(), "ru")
    assert result.candidates == []
    assert result.runs == {}


def test_table_date_cell_used_when_dates_not_excluded():
    table = TableData(["", ""], [["12 October 1998", "2010"]])
    nlp = make_nlp({
        "12 October 1998": [
            _Tok("12", 0, num=True),
            _Tok("October", 3, date=True),
            _Tok("1998", 11, num=True),
        ],
        "2010": [_Tok("2010", 0, num=True)],
    })
    cfg = MentionsConfig(exclude_dates=False)
    result = extract_table_candidates(table, nlp, cfg, "ru")
    keys = {c.lemma_key for c in result.candidates}
    # дата целиком используется как кандидат (лемма-ключ по поверхностной форме токенов
    # в нижнем регистре: не-числа приводятся к lower case)
    assert "12 october 1998" in keys
    assert "12 October 1998" not in keys
    # чистое число по-прежнему не порождает кандидатов (immutable)
    assert "2010" not in keys
    # у дата-ячейки появился content-run для реконструкции
    assert TableCoordinate(0, 0) in result.runs


def test_dedup_merges_coordinates():
    table = TableData(["City"], [["city"]])
    nlp = make_nlp({
        "City": [_Tok("City", 0, pos="PROPN", lemma="city")],
        "city": [_Tok("city", 0, pos="NOUN", lemma="city")],
    })
    result = extract_table_candidates(table, nlp, MentionsConfig(), "ru")
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.lemma_key == "city"
    assert c.source_text == "City"
    assert c.coordinates == [TableCoordinate(None, 0), TableCoordinate(0, 0)]


def test_id_stable_and_depends_on_target_lang():
    table = TableData(["City"], [["London"]])
    nlp = make_nlp({
        "City": [_Tok("City", 0, pos="PROPN", lemma="city")],
        "London": [_Tok("London", 0, pos="PROPN")],
    })
    ru1 = extract_table_candidates(table, nlp, MentionsConfig(), "ru")
    ru2 = extract_table_candidates(table, nlp, MentionsConfig(), "ru")
    de = extract_table_candidates(table, nlp, MentionsConfig(), "de")
    by1 = {c.lemma_key: c for c in ru1.candidates}
    by2 = {c.lemma_key: c for c in ru2.candidates}
    by_de = {c.lemma_key: c for c in de.candidates}
    assert by1["london"].id == by2["london"].id
    assert by1["london"].id != by_de["london"].id
    assert len(by1["london"].id) == 16


def test_lemma_key_keeps_cyrillic():
    # Кириллические леммы не должны схлопываться в пустую строку (важно для back-перевода на русский).
    nlp = make_nlp({
        "Нью-Йорк": [_Tok("Нью-Йорк", 0, pos="PROPN", lemma="нью-йорк")],
    })
    cands = extract_question_candidates("Нью-Йорк", nlp, MentionsConfig())
    assert "ньюйорк" in {c.lemma_key for c in cands}


def test_table_candidates_carry_tokens_pos_ner_annotations():
    table = TableData(["New York"], [])
    nlp = make_nlp({
        "New York": [
            _Tok("New", 0, pos="PROPN", ent="GPE"),
            _Tok("York", 4, pos="PROPN", ent="GPE"),
        ],
    })
    result = extract_table_candidates(table, nlp, MentionsConfig(), "ru")
    ny = next(c for c in result.candidates if c.lemma_key == "new york")
    assert ny.tokens == ["New", "York"]
    assert ny.pos == ["PROPN", "PROPN"]
    assert ny.ner == ["GPE", "GPE"]
    assert len(ny.tokens) == len(ny.pos) == len(ny.ner) == 2


def test_question_candidates_have_no_token_annotations():
    nlp = make_nlp({
        "New York": [
            _Tok("New", 0, pos="PROPN", ent="GPE"),
            _Tok("York", 4, pos="PROPN", ent="GPE"),
        ],
    })
    cands = extract_question_candidates("New York", nlp, MentionsConfig())
    ny = next(c for c in cands if c.lemma_key == "new york")
    assert not hasattr(ny, "tokens")
    assert not hasattr(ny, "pos")
    assert not hasattr(ny, "ner")
