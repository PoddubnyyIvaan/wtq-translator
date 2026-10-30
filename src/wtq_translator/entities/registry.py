"""Реестр накопленных сущностей (TSV) и список обработанных таблиц."""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


def _atomic_write(path: Path, content: str) -> None:
    """Атомарная запись текста: tempfile в том же каталоге + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class EntitiesRegistry:
    """Накопительный реестр сущностей: ``name -> {type, count, all_types}``.

    ``type`` — тип первой встречи (первичная встреча); ``count`` — число встреч;
    ``all_types`` — объединение всех NER-типов по встречам. Мелкие партии не пишутся
    в файл — ``save`` вызывается на чекпоинтах и в конце.
    """

    _HEADER = ("name", "type", "count", "all_types")

    def __init__(self, *, sep: str = "|") -> None:
        self._sep = sep
        self.store: dict[str, dict] = {}
        self._order: list[str] = []

    @classmethod
    def load(cls, path: str | Path, *, sep: str = "|") -> "EntitiesRegistry":
        registry = cls(sep=sep)
        p = Path(path)
        if not p.exists():
            return registry
        with p.open("r", encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.reader(f, delimiter="\t")):
                if not row or (i == 0 and row[0] == cls._HEADER[0]):
                    continue  # заголовок и пустые/битые строки
                name, ent_type, count = row[0], row[1], row[2]
                # Совместимость: старый 3-колоночный файл → all_types = {type}.
                if len(row) >= 4 and row[3]:
                    all_types = set(row[3].split(sep))
                else:
                    all_types = {ent_type}
                registry.store[name] = {"type": ent_type, "count": int(count), "all_types": all_types}
                registry._order.append(name)
        return registry

    def add(self, name: str, ent_type: str, cell_types: frozenset[str]) -> bool:
        """Накопить одну встречу сущности; True — сущность новая.

        Новая: ``type=ent_type``, ``count=1``, ``all_types=set(cell_types)``.
        Существующая: ``count += 1``, ``all_types |= cell_types`` (type не меняется).
        """
        record = self.store.get(name)
        if record is None:
            self.store[name] = {"type": ent_type, "count": 1, "all_types": set(cell_types)}
            self._order.append(name)
            return True
        record["count"] += 1
        record["all_types"] |= set(cell_types)
        return False

    def save(self, path: str | Path) -> None:
        lines = ["\t".join(self._HEADER)]
        for name in self._order:
            record = self.store[name]
            all_types = self._sep.join(sorted(record["all_types"]))
            lines.append("\t".join([name, record["type"], str(record["count"]), all_types]))
        _atomic_write(Path(path), "\n".join(lines) + "\n" if lines else "")


class ProcessedTables:
    """Список обработанных таблиц: относительные context-пути, по строке на файл."""

    def __init__(self) -> None:
        self._rows: list[str] = []
        self.seen: set[str] = set()

    @classmethod
    def load(cls, path: str | Path) -> "ProcessedTables":
        tables = cls()
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    ctx = line.rstrip("\r\n")
                    if ctx:
                        tables._rows.append(ctx)
                        tables.seen.add(ctx)
        return tables

    def add(self, ctx: str) -> None:
        if ctx not in self.seen:
            self.seen.add(ctx)
            self._rows.append(ctx)

    def flush(self, path: str | Path) -> None:
        _atomic_write(Path(path), "".join(ctx + "\n" for ctx in self._rows))