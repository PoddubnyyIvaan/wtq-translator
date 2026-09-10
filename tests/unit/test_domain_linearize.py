"""Тесты жадной линеаризации таблицы в TAPEX-формат (linearize_table).

Стиль: pytest + pytest-asyncio, как в остальном tests/unit. ``count`` —
асинхронный колбэк, поэтому в тестах используется async-функция подсчёта
(в роли прокси-подсчёта токенов выступает длина строки).
"""
import pytest

from wtq_translator.domain import _render_header, _render_row, linearize_table
from wtq_translator.models import TableData


async def _count_by_length(text: str) -> int:
    """Асинхронный колбэк подсчёта «токенов»: длина строки (детерминированно)."""
    return len(text)


def _table(headers, rows):
    return TableData(headers=headers, rows=rows)


@pytest.mark.asyncio
async def test_header_greedy_trimming_adds_only_full_columns():
    """Заголовок: жадный отбор колонок останавливается на первой не влезающей.

    Бюджет 24 умещает 3 колонки (длина `col : [H1] | [H2] | [H3]` == 24), но
    4-я колонка уже не влезает (31 > 24) → K = 3, все колонки в строке целые.
    """
    table = _table(["H1", "H2", "H3", "H4"], [])
    result = await linearize_table(table, max_tokens=24, count=_count_by_length)

    assert result == "col : [H1] | [H2] | [H3]"
    # 1 строка вывода (только заголовок), 3 колонки.
    assert result.count("\n") == 0
    assert result.count("[") == 3
    assert "H4" not in result


@pytest.mark.asyncio
async def test_whole_rows_added_with_k_column_width_of_header():
    """Строки добавляются целиком, ширина K берётся из заголовка.

    Бюджета хватает на все строки (16 токенов каждая) и заголовок.
    """
    table = _table(["a", "b"], [["1", "2"], ["3", "4"]])
    header = _render_header(table.headers, 2)
    assert header == "col : [a] | [b]"

    result = await linearize_table(table, max_tokens=100, count=_count_by_length)

    assert result == (
        "col : [a] | [b]\n"
        "row 0: [1] | [2]\n"
        "row 1: [3] | [4]"
    )
    # 3 строки вывода: заголовок + 2 строки таблицы.
    assert result.count("\n") == 2
    # Каждая строка использует ровно K=2 колонок.
    for line in result.split("\n"):
        assert line.count("[") == 2


@pytest.mark.asyncio
async def test_stops_at_first_row_that_does_not_fit():
    """Добавление строк прекращается на первой не влезающей строке.

    Бюджет 47 умещает header (15) + row0 (16) + row1 (16) = 47; row2 (16)
    не влезает → добавляются только row0 и row1, row2 отбрасывается.
    """
    table = _table(["a", "b"], [["1", "2"], ["3", "4"], ["5", "6"]])
    result = await linearize_table(table, max_tokens=47, count=_count_by_length)

    assert result == (
        "col : [a] | [b]\n"
        "row 0: [1] | [2]\n"
        "row 1: [3] | [4]"
    )
    assert "row 2" not in result
    # 1 заголовок + 2 добавленные строки.
    assert len(result.split("\n")) == 3
    assert "5" not in result


@pytest.mark.asyncio
async def test_k_and_rows_count_reflected_in_output():
    """Число колонок K и добавленных строк читаются из вывода."""
    table = _table(["a", "b", "c", "d"], [["1", "2", "3", "4"], ["5", "6", "7", "8"]])
    # K=4 (заголовок из 4 колонок влезает), все строки влезают.
    result = await linearize_table(table, max_tokens=500, count=_count_by_length)

    lines = result.split("\n")
    assert lines[0] == "col : [a] | [b] | [c] | [d]"
    assert len(lines) == 3  # header + 2 строки.
    assert [l.count("[") for l in lines] == [4, 4, 4]  # все с K=4 колонками.


@pytest.mark.asyncio
async def test_empty_case_when_nothing_fits():
    """Если не влезает даже первая колонка, возвращается пустая строка."""
    table = _table(["some-long-header"], [["x"]])
    result = await linearize_table(table, max_tokens=5, count=_count_by_length)

    assert result == ""
    # Ни одной строки вывода (заголовок не добавлен).
    assert result.count("\n") == 0


@pytest.mark.asyncio
async def test_render_row_only_uses_first_k_cells():
    """_render_row ограничивает вывод первыми k колонками даже для широкой строки."""
    assert _render_row(["1", "2", "3"], k=2) == "[1] | [2]"
    assert _render_row(["1", "2", "3"], k=0) == ""
