"""Unit-тесты full-матчера: longest non-overlap, матчинг по лемме, метаданные mention."""
from wtq_translator.config import MentionsConfig
from wtq_translator.enums import MentionSourceType
from wtq_translator.models import TableCoordinate, TableData
from wtq_translator.mentions.finder_full import find_table_mentions_full


class _Morph:
    def __init__(self, poss=None):
        self._poss = poss

    def get(self, key):
        if key == "Poss" and self._poss is not None:
            return [self._poss]
        return []


class _Tok:
    def __init__(self, text, idx, *, lemma=None, pos="", num=False, date=False, punct=False, poss=None):
        self.text = text
        self.idx = idx
        self.lemma_ = text.lower() if lemma is None else lemma
        self.pos_ = pos
        self.like_num = num
        self.ent_type_ = "DATE" if date else ""
        self.is_punct = punct
        self.is_space = False
        self.morph = _Morph(poss)


def make_nlp(mapping):
    def nlp(text):
        return list(mapping[text])
    return nlp


def test_longest_non_overlap():
    table = TableData(["Place"], [["New York City"], ["New York"], ["York"]])
    nlp = make_nlp({
        "Place": [_Tok("Place", 0, pos="PROPN")],
        "New York City": [
            _Tok("New", 0, pos="PROPN"),
            _Tok("York", 4, pos="PROPN"),
            _Tok("City", 9, pos="PROPN"),
        ],
        "New York": [_Tok("New", 0, pos="PROPN"), _Tok("York", 4, pos="PROPN")],
        "York": [_Tok("York", 0, pos="PROPN")],
        "Was New York City first?": [
            _Tok("Was", 0, pos="VERB"),
            _Tok("New", 4, pos="PROPN"),
            _Tok("York", 8, pos="PROPN"),
            _Tok("City", 13, pos="PROPN"),
            _Tok("first", 18, pos="ADJ"),
            _Tok("?", 23, punct=True),
        ],
    })
    result, _ = find_table_mentions_full("Was New York City first?", table, MentionsConfig(), nlp, "ru")
    assert [m.question_text for m in result.mentions] == ["New York City"]


def test_lemma_match_and_metadata():
    table = TableData(["City"], [["London"]])
    nlp = make_nlp({
        "City": [_Tok("City", 0, pos="PROPN", lemma="city")],
        "London": [_Tok("London", 0, pos="PROPN")],
        "where is london?": [
            _Tok("where", 0, pos="ADV"),
            _Tok("is", 6, pos="AUX"),
            _Tok("london", 9, pos="PROPN"),
            _Tok("?", 15, punct=True),
        ],
    })
    result, candidate_set = find_table_mentions_full("where is london?", table, MentionsConfig(), nlp, "ru")
    assert result.ambiguous == []
    assert [m.question_text for m in result.mentions] == ["london"]
    m = result.mentions[0]
    assert m.normalized_text == "london"
    assert m.source_type == MentionSourceType.CELL
    assert m.coordinates == (TableCoordinate(0, 0),)
    assert m.match_type == "lemma"
    assert m.candidate_key
    assert len(m.candidate_key) == 16
    london = {c.lemma_key: c for c in candidate_set.candidates}["london"]
    assert m.candidate_key == london.id


def test_merged_coordinates_flow_to_mention():
    table = TableData(["London"], [["London"]])
    nlp = make_nlp({
        "London": [_Tok("London", 0, pos="PROPN")],
        "in London": [_Tok("in", 0, pos="ADP"), _Tok("London", 3, pos="PROPN")],
    })
    result, _ = find_table_mentions_full("in London", table, MentionsConfig(), nlp, "ru")
    m = result.mentions[0]
    assert m.question_text == "London"
    assert m.coordinates == (TableCoordinate(None, 0), TableCoordinate(0, 0))
    assert m.source_type == MentionSourceType.HEADER


def test_normalized_text_lowercased():
    # Лемма-ключ хранится по поверхностной форме токенов, но в нижнем регистре,
    # если содержимое — не число (месяц даты не должен сохранять заглавную букву).
    table = TableData([""], [["12 October 1998"]])
    nlp = make_nlp({
        "12 October 1998": [
            _Tok("12", 0, num=True),
            _Tok("October", 3, date=True),
            _Tok("1998", 11, num=True),
        ],
    })
    result, candidate_set = find_table_mentions_full(
        "12 October 1998", table, MentionsConfig(exclude_dates=False), nlp, "ru"
    )
    assert result.mentions
    assert result.mentions[0].normalized_text == "12 october 1998"
    keys = {c.lemma_key for c in candidate_set.candidates}
    assert "12 october 1998" in keys
    assert "12 October 1998" not in keys
