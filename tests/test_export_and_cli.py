"""
Tests for the exporter, the terminal report, the cache and the CLI's error
handling (the fixes from a code review on 2026-11-08).

They build tiny PDFs and results by hand, so they run in about a second.
"""

import json
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from rich.console import Console
from typer.testing import CliRunner

from scoresplit import cache, cli
from scoresplit.exporter import (
    InvalidResultError,
    check_export,
    export_parts,
    extra_filename,
    part_filenames,
    safe_filename,
)
from scoresplit.metadata import _vote
from scoresplit.models import MAX_PAGE_NUMBER, Extra, Part, Result, parse_page_ranges
from scoresplit.page_analyzer import AnalysisSettings
from scoresplit.pdf_reader import MAX_RENDER_PIXELS, fit_dpi
from scoresplit.pipeline import inspect_pdf
from scoresplit.report import print_result

runner = CliRunner()


def make_pdf(path: Path, pages: int) -> Path:
    """Helper: write an empty PDF with `pages` blank A4 pages."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(595, 842)  # A4 in PDF points
    pdf.save(str(path))
    pdf.close()
    return path


def make_map(parts: list[Part], extra: list[Extra] | None = None, page_count: int = 10) -> Result:
    """Helper: a Result with just the fields the tests care about."""
    return Result(
        source="test.pdf", sha256="0" * 64, page_count=page_count, parts=parts, extra=extra or []
    )


def page_count_of(path: Path) -> int:
    pdf = pdfium.PdfDocument(str(path))
    count = len(pdf)
    pdf.close()
    return count


# --- Finding 1: empty page lists ---------------------------------------------


def test_exported_part_without_pages_is_rejected(tmp_path: Path) -> None:
    page_map = make_map([Part(name="Flute", pages="1-2"), Part(name="Oboe", pages="")])
    problems = check_export(page_map, page_count=10).errors
    assert any("'Oboe': has no pages" in problem for problem in problems)

    # export_parts checks too, and writes nothing (it used to write the WHOLE PDF).
    pdf = make_pdf(tmp_path / "score.pdf", 10)
    with pytest.raises(InvalidResultError):
        export_parts(page_map, pdf, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_switched_off_part_and_empty_extra_are_fine(tmp_path: Path) -> None:
    page_map = make_map(
        [Part(name="Flute", pages="1-2"), Part(name="Oboe", pages="", export=False)],
        extra=[Extra(reason="blank", pages="")],
    )
    assert check_export(page_map, page_count=10).errors == []

    pdf = make_pdf(tmp_path / "score.pdf", 10)
    _, written = export_parts(page_map, pdf, tmp_path / "out")
    # The flute, pages 3-10 (in no part) as unassigned extra pages, the
    # three export files and the copy of the original PDF; the switched-off
    # oboe and the empty "blank" group write nothing.
    assert len(written) == 6
    assert written[0].part == 0  # the Flute: index 0 in page_map.parts
    assert page_count_of(written[0].path) == 2
    assert written[1].part is None
    assert written[1].path.name == "unassigned_pages_3-10.pdf"
    assert page_count_of(written[1].path) == 8


# --- Finding 2: square brackets in names -------------------------------------


def test_report_shows_square_brackets_literally() -> None:
    page_map = make_map([Part(name="Perc [a2]", pages="1"), Part(name="Odd [/] name", pages="2")])
    page_map.title = "Title [/b]"
    console = Console(record=True, width=200)
    print_result(console, page_map)  # used to raise rich.errors.MarkupError
    text = console.export_text()
    assert "Perc [a2]" in text
    assert "Odd [/] name" in text
    assert "Title [/b]" in text


def test_the_report_box_title_shows_the_version() -> None:
    # The box around the title block says "scoresplit 0.2.0", so a run in
    # the terminal shows which version made it.
    from scoresplit import __version__

    page_map = make_map([Part(name="Flute", pages="1")])
    console = Console(record=True, width=200)
    print_result(console, page_map)
    # The first line of the report is the top edge of the box, with the
    # title in it: "╭─── scoresplit 0.2.0 ───╮".
    first_line = console.export_text().splitlines()[0]
    assert f"scoresplit {__version__}" in first_line


# --- Finding 3: long file names ----------------------------------------------


def test_long_extra_page_list_gets_a_short_file_name() -> None:
    scattered = list(range(6, 240, 6))  # 6, 12, 18, ... 234
    name = extra_filename(Extra(reason="blank", pages=scattered))
    assert name == f"blank_{len(scattered)}_pages_6-234.pdf"
    assert extra_filename(Extra(reason="cover", pages="1")) == "cover_pages_1.pdf"


def test_safe_filename_limits_the_length_in_bytes() -> None:
    name = safe_filename("Š" * 300)  # Š takes 2 bytes in UTF-8
    assert len(name.encode("utf-8")) <= 200
    assert set(name) == {"Š"}  # no half-cut letter at the end


# --- Finding 5: parts that end up in the same file ----------------------------


def test_parts_with_the_same_name_are_kept_apart_by_their_pages() -> None:
    # Two parts may share a printed name (a Euphonium in treble clef and
    # one in bass clef): the pages in the file name keep them apart.
    page_map = make_map([Part(name="Trombone 1/2", pages="1"), Part(name="trombone 1-2", pages="2")])
    assert check_export(page_map, page_count=10).errors == []


def test_parts_with_the_same_name_and_pages_get_their_own_file() -> None:
    # Same name AND same pages: the row number in front keeps the two
    # files apart, so this is no longer an error.
    page_map = make_map([Part(name="Trombone 1/2", pages="1"), Part(name="trombone 1-2", pages="1")])
    assert check_export(page_map, page_count=10).errors == []
    assert part_filenames(page_map) == {
        0: "01_Trombone_1-2_pages_1.pdf",
        1: "02_trombone_1-2_pages_1.pdf",
    }


# --- Finding 7: cache file with the wrong shape -------------------------------


@pytest.mark.parametrize("content", ['{"not": "a list"}', '[{"page_number": 1}]', "[1, 2]", "not json"])
def test_broken_cache_file_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = cache._cache_file("a" * 64, "test")
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    assert cache.load_facts("a" * 64, "test") == {}


# --- Finding 9: vote key keeps non-ASCII letters -------------------------------


def test_vote_keeps_croatian_and_cyrillic_letters_apart() -> None:
    # "Šuma" is NOT "uma": uma/UMA together have 3 votes and win.
    assert _vote([("Šuma", 2), ("uma", 2), ("UMA", 1)]) == "uma"
    # Cyrillic titles used to all become "" (one big group, so Шума won).
    assert _vote([("Шума", 2), ("река", 2), ("Река", 1)]) == "река"


# --- Findings 4 and 8: CLI exit codes and JSON errors --------------------------


def test_missing_pdf_gives_json_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["inspect", str(tmp_path / "nope.pdf"), "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    error = json.loads(result.stdout)
    assert error["status"] == "error"
    assert "does not exist" in error["message"]


def test_missing_yes_fails_before_any_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("the PDF was analysed although --yes was missing")

    monkeypatch.setattr(cli, "_run_inspection", must_not_run)
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    result = runner.invoke(cli.app, ["split", str(pdf), "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    assert "--yes" in json.loads(result.stdout)["message"]


def test_unexpected_error_gives_exit_code_5_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("Tesseract crashed")

    monkeypatch.delenv("SCORESPLIT_DEBUG", raising=False)
    monkeypatch.setattr(cli, "_run_inspection", broken)
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    result = runner.invoke(cli.app, ["inspect", str(pdf), "--json"])
    assert result.exit_code == cli.EXIT_ERROR
    error = json.loads(result.stdout)
    assert error["code"] == cli.EXIT_ERROR
    assert "Tesseract crashed" in error["message"]


# --- Security hardening: inputs that could exhaust the memory ----------------


def test_huge_page_range_in_a_map_is_rejected_quickly(tmp_path: Path) -> None:
    # Without a limit, "1-999999999999" would try to build a trillion page
    # numbers in memory and crash the machine. Now it's a normal bad-input error.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    page_map = make_map([Part(name="Flute", pages="1")]).model_dump()
    page_map["parts"][0]["pages"] = "1-999999999999"
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps(page_map), encoding="utf-8")

    result = runner.invoke(
        cli.app, ["split", str(pdf), "--edits", str(map_file), "--yes", "--force", "--json"]
    )
    assert result.exit_code == cli.EXIT_BAD_INPUT
    error = json.loads(result.stdout)
    assert any("too high" in problem for problem in error["problems"])


def test_page_numbers_above_the_limit_are_rejected_in_every_form() -> None:
    too_high = MAX_PAGE_NUMBER + 1
    for value in (f"1-{too_high}", str(too_high), [1, too_high]):
        with pytest.raises(ValueError, match="too high"):
            parse_page_ranges(value)
    # The limit itself is still allowed.
    assert parse_page_ranges(str(MAX_PAGE_NUMBER)) == [MAX_PAGE_NUMBER]


def test_fit_dpi_keeps_normal_pages_and_shrinks_giant_ones() -> None:
    assert fit_dpi(595, 842, 200) == 200  # A4: untouched
    assert fit_dpi(1191, 1684, 200) == 200  # A2: untouched
    giant_dpi = fit_dpi(14400, 14400, 200)  # 200 x 200 inches, the PDF maximum
    assert giant_dpi < 200
    pixels = (14400 / 72 * giant_dpi) ** 2
    assert pixels <= MAX_RENDER_PIXELS


def test_giant_page_is_analysed_at_a_lower_resolution(tmp_path: Path) -> None:
    # A 200 x 200 inch page would be a 1.6 GB picture at 200 dpi. With the
    # limit it's rendered small and the analysis just works.
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(14400, 14400)
    path = tmp_path / "giant.pdf"
    pdf.save(str(path))
    pdf.close()

    page_map = inspect_pdf(path, AnalysisSettings(ocr_mode="never"), use_cache=False, workers=1)
    assert page_map.page_count == 1


# --- the parts table -----------------------------------------------------------


def test_report_fits_a_narrow_terminal() -> None:
    # In a narrow terminal the name and the instrument wrap onto a second
    # line; the number, key, clef, pages and confidence are never cut.
    page_map = make_map(
        [
            Part(name="Baritone Saxophone 1", pages="1", instrument="baritone_saxophone", key="Eb", clef="treble"),
            Part(name="Shaker, Susp. Cymbal, Wind Chimes", pages="2-3,5", clef="percussion"),
        ]
    )
    console = Console(record=True, width=70)
    print_result(console, page_map)
    lines = console.export_text().splitlines()
    assert all(len(line) <= 70 for line in lines)
    rows = [line.split() for line in lines if line.strip().startswith(("1 ", "2 "))]
    assert rows[0][0] == "1" and rows[0][-4:] == ["Eb", "treble", "1", "100%"]
    assert rows[1][0] == "2" and rows[1][-4:] == ["-", "percussion", "2-3,5", "100%"]


def test_report_shows_notes_under_the_table_and_marks_assumed_keys() -> None:
    page_map = make_map(
        [
            Part(name="Flute", pages="1", instrument="flute", key="C", key_assumed=True),
            Part(name="Horn", pages="2", instrument="horn", key="F", notes=["please check it"]),
        ]
    )
    console = Console(record=True, width=100)
    print_result(console, page_map)
    text = console.export_text()
    assert "  2: please check it" in text
    assert " C* " in text  # assumed key
    assert " F " in text and " F* " not in text
    assert "* = key not printed" in text
    assert "Extra pages: unassigned 3-10" in text
