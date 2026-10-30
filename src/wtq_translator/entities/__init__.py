"""Сбор сущностей из ячеек таблиц через spaCy NER (команда `wtq-translate entities`)."""
from .ner import EntityExtractor
from .registry import EntitiesRegistry, ProcessedTables
from .runner import EntitiesSummary, remaining_tables, run_entities

__all__ = [
    "EntityExtractor",
    "EntitiesRegistry",
    "ProcessedTables",
    "EntitiesSummary",
    "remaining_tables",
    "run_entities",
]