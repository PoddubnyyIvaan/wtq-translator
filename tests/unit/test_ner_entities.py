"""Unit-тесты сбора сущностей: NER-классификатор, реестр, список обработанных, runner.

NER тестируется на фейковом nlp.pipe без GPU и реальной spaCy-модели.
"""
from wtq_translator.config import AppConfig, EntitiesConfig, ProjectConfig
from wtq_translator.entities import EntityExtractor
from wtq_translator.entities.registry import EntitiesRegistry, ProcessedTables
from wtq_translator.entities.runner import run_entities
import wtq_translator.entities.runner as runner_module

# --------------------------------------------------------------------------- fakes


class _FakeSpan:
    def __init__(self, text, label_, start_char, end_char):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


class _FakeDoc:
    def __init__(self, text, ents):
        self.text = text
        self.ents = ents


class _FakeNlp:
    """Фейк spaCy: mapping text -> [(text, label, start_char, end_char), ...] ents."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __call__(self, text):
        return _FakeDoc(text, [_FakeSpan(*s) for s in self._mapping[text]])

    def pipe(self, texts, batch_size=1):
        return [self(t) for t in texts]


def _make_extractor(nlp, allowed_types=None):
    """EntityExtractor без загрузки модели: инстанс через __new__ + ручные атрибуты."""
    ext = object.__new__(EntityExtractor)
    ext._nlp = nlp
    ext._template = "{cell} is {col}"
    ext._allowed_types = tuple(allowed_types or ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"])
    ext._batch_size = 64
    return ext


# ---------------------------------------------------------------------- extractor


def test_cell_span_kept_column_span_dropped():
    # "<mit> is Country": <mit> = chars 0..5, Country = chars 11..18.
    nlp = _FakeNlp({
        "<mit> is Country": [
            ("<mit>", "PERSON", 0, 5),
            ("Country", "GPE", 11, 18),
        ],
    })
    ext = _make_extractor(nlp)
    ent_type, all_types, spans = ext.classify("<mit>", "Country")
    assert ent_type == "PERSON"
    assert all_types == frozenset({"PERSON"})
    assert spans == [("<mit>", "PERSON", "0:5")]


def test_first_allowed_type_and_full_all_types():
    # "Bill Gates, MSFT is Company": Bill Gates = 0..10 (PERSON), MSFT = 12..16 (ORG),
    # Company = 20..27 (PRODUCT, спаны колонки отбрасываются).
    nlp = _FakeNlp({
        "Bill Gates, MSFT is Company": [
            ("Bill Gates", "PERSON", 0, 10),
            ("MSFT", "ORG", 12, 16),
            ("Company", "PRODUCT", 20, 27),
        ],
    })
    ext = _make_extractor(nlp)
    ent_type, all_types, _ = ext.classify("Bill Gates, MSFT", "Company")
    assert ent_type == "PERSON"  # PERSON идёт раньше ORG в allowed_types
    assert all_types == frozenset({"PERSON", "ORG"})


def test_unrecognized_cell_type_is_none():
    # "42 is Number": ячейка имеет тип CARDINAL, не входящий в allowed → None.
    nlp = _FakeNlp({
        "42 is Number": [("42", "CARDINAL", 0, 2), ("Number", "CARDINAL", 5, 11)],
    })
    ext = _make_extractor(nlp)
    ent_type, all_types, _ = ext.classify("42", "Number")
    assert ent_type is None
    assert all_types == frozenset({"CARDINAL"})


def test_classify_many_preserves_order_with_length_bucketing():
    nlp = _FakeNlp({
        "A is Letter": [("A", "PERSON", 0, 1)],
        "B is Letter": [("B", "GPE", 0, 1)],
        "Longer Corporation is Company": [("Longer Corporation", "ORG", 0, 18)],
    })
    ext = _make_extractor(nlp)
    items = [("Longer Corporation", "Company"), ("A", "Letter"), ("B", "Letter")]
    results = ext.classify_many(items)
    assert results == [
        ("ORG", frozenset({"ORG"})),
        ("PERSON", frozenset({"PERSON"})),
        ("GPE", frozenset({"GPE"})),
    ]


# ------------------------------------------------------------------- device modes


class _FakeLoadedModel:
    def select_pipes(self, disable):
        pass


class _FakeSpacyNoCupy:
    """spacy без CuPy: require_gpu бросает ValueError, как thinc на Linux."""

    @staticmethod
    def require_gpu():
        raise ValueError("Cannot use GPU, CuPy is not installed")

    @staticmethod
    def load(model):
        return _FakeLoadedModel()


class _FakeSpacyGpu:
    @staticmethod
    def require_gpu():
        return True

    @staticmethod
    def load(model):
        return _FakeLoadedModel()


def test_gpu_without_cupy_raises_actionable_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "spacy", _FakeSpacyNoCupy)
    import pytest
    with pytest.raises(ValueError, match="cupy-cuda"):
        EntityExtractor(device="gpu")


def test_auto_falls_back_to_cpu_without_cupy(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "spacy", _FakeSpacyNoCupy)
    ext = EntityExtractor(device="auto")
    assert ext._nlp is not None


def test_auto_uses_gpu_when_cupy_available(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "spacy", _FakeSpacyGpu)
    ext = EntityExtractor(device="auto")
    assert ext._nlp is not None


# ------------------------------------------------------------------------ registry


def test_registry_add_merge_and_roundtrip(tmp_path):
    registry = EntitiesRegistry()
    assert registry.add("VMware", "ORG", frozenset({"ORG"})) is True
    assert registry.add("VMware", "ORG", frozenset({"ORG", "PRODUCT"})) is False
    assert registry.add("Apple", "ORG", frozenset({"ORG"})) is True
    assert registry.store["VMware"] == {"type": "ORG", "count": 2, "all_types": {"ORG", "PRODUCT"}}

    path = tmp_path / "entities.tsv"
    registry.save(path)
    loaded = EntitiesRegistry.load(path)
    assert loaded.store == registry.store
    assert loaded._order == ["VMware", "Apple"]


def test_registry_load_legacy_three_columns(tmp_path):
    path = tmp_path / "entities.tsv"
    path.write_text("VMware\tORG\t3\nApple\tPRODUCT\t1\n", encoding="utf-8")
    registry = EntitiesRegistry.load(path)
    assert registry.store["VMware"] == {"type": "ORG", "count": 3, "all_types": {"ORG"}}
    assert registry.store["Apple"] == {"type": "PRODUCT", "count": 1, "all_types": {"PRODUCT"}}


def test_registry_load_skips_header(tmp_path):
    path = tmp_path / "entities.tsv"
    path.write_text(
        "name\ttype\tcount\tall_types\n"
        "VMware\tORG\t2\tORG|PRODUCT\n",
        encoding="utf-8",
    )
    registry = EntitiesRegistry.load(path)
    assert registry.store == {
        "VMware": {"type": "ORG", "count": 2, "all_types": {"ORG", "PRODUCT"}}
    }


def test_registry_load_uses_configured_sep(tmp_path):
    path = tmp_path / "entities.tsv"
    path.write_text("VMware\tORG\t1\tORG,PRODUCT\n", encoding="utf-8")
    registry = EntitiesRegistry.load(path, sep=",")
    assert registry.store["VMware"]["all_types"] == {"ORG", "PRODUCT"}


# ------------------------------------------------------------------ processed tables


def test_processed_tables_load_add_flush(tmp_path):
    path = tmp_path / "entities-processed.txt"
    tables = ProcessedTables.load(path)  # файла нет — пустой список
    assert tables.seen == set()

    tables.add("csv/1.csv")
    tables.add("csv/1.csv")  # дубликат не добавляется
    tables.add("csv/2.csv")
    tables.flush(path)
    assert path.read_text(encoding="utf-8") == "csv/1.csv\ncsv/2.csv\n"

    reloaded = ProcessedTables.load(path)
    assert reloaded.seen == {"csv/1.csv", "csv/2.csv"}
    assert reloaded._rows == ["csv/1.csv", "csv/2.csv"]


# ---------------------------------------------------------------------------- runner


def _make_config(tmp_path, **entities_kwargs) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            input_root=str(tmp_path / "in"),
            output_root=str(tmp_path / "out"),
        ),
        entities=EntitiesConfig(**entities_kwargs),
    )


def _write_input_tsv(path, rows):
    lines = ["id\tutterance\tcontext\ttargetValue"]
    lines += [f"q{i}\tq{i}\t{ctx}\tv{i}" for i, ctx in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_runner_skips_processed_tables(tmp_path):
    config = _make_config(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    _write_input_tsv(input_tsv, ["csv/1.csv", "csv/2.csv"])

    processed_path = config.entities.processed_path
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text("csv/1.csv\ncsv/2.csv\n", encoding="utf-8")

    summary = run_entities(config, input_tsv)
    assert summary.total_tables == 2
    assert summary.skipped_processed == 2
    assert summary.missing_tables == 0
    assert (summary.new_entities, summary.updated_entities) == (0, 0)


def test_runner_missing_table_skip(tmp_path):
    config = _make_config(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    _write_input_tsv(input_tsv, ["csv/ghost.csv"])

    summary = run_entities(config, input_tsv)
    assert summary.missing_tables == 1
    assert summary.skipped_processed == 0
    assert summary.new_entities == 0
    # Пропущенная таблица помечается обработанной — повторный запуск её не трогает.
    assert config.entities.processed_path.read_text(encoding="utf-8") == "csv/ghost.csv\n"


def test_runner_missing_table_error_raises(tmp_path):
    config = _make_config(tmp_path, missing_table="error")
    input_tsv = tmp_path / "in.tsv"
    _write_input_tsv(input_tsv, ["csv/ghost.csv"])

    import pytest
    with pytest.raises(FileNotFoundError):
        run_entities(config, input_tsv)


class _FakeExtractor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def classify_many(self, pairs):
        return [
            ("ORG", frozenset({"ORG", "PRODUCT"})) if cell == "VMware" else (None, frozenset())
            for cell, _ in pairs
        ]


def test_runner_classifies_checkpoints_and_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "EntityExtractor", _FakeExtractor)
    config = _make_config(tmp_path, checkpoint_every_tables=1)

    (tmp_path / "in" / "csv").mkdir(parents=True)
    (tmp_path / "in" / "csv" / "1.tsv").write_text(
        "Company\tType\nVMware\tSoftware\nEmpire\tFilm\n", encoding="utf-8",
    )
    (tmp_path / "in" / "csv" / "2.tsv").write_text(
        "Company\tCountry\nVMware\tUSA\nApple\tUSA\n", encoding="utf-8",
    )
    input_tsv = tmp_path / "in.tsv"
    _write_input_tsv(input_tsv, ["csv/1.csv", "csv/2.csv"])

    summary = run_entities(config, input_tsv)
    assert summary.total_tables == 2
    assert summary.skipped_processed == 0
    assert summary.missing_tables == 0
    assert summary.new_entities == 1       # VMware
    assert summary.updated_entities == 1   # повторная встреча VMware

    registry_text = config.entities.output_tsv.read_text(encoding="utf-8")
    header, *rows = registry_text.strip().splitlines()
    assert header == "name\ttype\tcount\tall_types"
    assert rows == ["VMware\tORG\t2\tORG|PRODUCT"]

    processed = config.entities.processed_path.read_text(encoding="utf-8").splitlines()
    assert processed == ["csv/1.csv", "csv/2.csv"]


def test_runner_filters_cells_by_max_words(tmp_path, monkeypatch):
    class _RecordingExtractor:
        def __init__(self, **kwargs):
            self.seen: list[tuple[str, str]] = []

        def classify_many(self, pairs):
            self.seen.extend(pairs)
            return [("ORG", frozenset({"ORG"})) for _ in pairs]

    extractor = _RecordingExtractor()
    monkeypatch.setattr(runner_module, "EntityExtractor", lambda **kw: extractor)

    config = _make_config(tmp_path)  # max_words по умолчанию = 5
    (tmp_path / "in" / "csv").mkdir(parents=True)
    (tmp_path / "in" / "csv" / "1.tsv").write_text(
        "Company\tType\nVMware\tSoftware\n"
        "A Very Long Company Name With Many Words\tFilm\n",
        encoding="utf-8",
    )
    input_tsv = tmp_path / "in.tsv"
    _write_input_tsv(input_tsv, ["csv/1.csv"])

    summary = run_entities(config, input_tsv)
    # "A Very Long Company Name With Many Words" (8 слов) в NER не уходит и в реестре его нет.
    assert extractor.seen == [
        ("VMware", "Company"), ("Software", "Type"), ("Film", "Type"),
    ]
    assert summary.new_entities == 3
    registry_text = config.entities.output_tsv.read_text(encoding="utf-8")
    assert "A Very Long Company Name" not in registry_text