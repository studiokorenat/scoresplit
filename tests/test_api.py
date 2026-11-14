"""
Tests for api.py, the Python interface under the CLI.

They call the api functions directly (no terminal, no Typer) and check two
things: the results, and that every planned failure is a ScoresplitError
carrying the right exit code. They build tiny PDFs by hand, so they run in
well under a second.
"""

import json
import time
from pathlib import Path

import pytest

from scoresplit import api
from scoresplit.cache import file_sha256
from scoresplit.models import (
    CONTROL_CHARACTERS,
    MAX_LISTED_PAGES,
    MAX_PAGE_NUMBER,
    MAX_PARTS,
    Extra,
    Part,
    Result,
)
from scoresplit.ocr import OcrUnavailableError

# The tiny-PDF and page-map helpers of the exporter tests. pytest puts the
# tests/ folder on the import path, so another test file can be imported
# like a normal module.
from test_export_and_cli import make_map, make_pdf


def test_every_error_carries_its_exit_code() -> None:
    # The exit codes are part of the contract with calling programs.
    assert api.BadInputError("x").exit_code == 2
    assert api.InvalidEditsError(["x"]).exit_code == 2
    assert api.OcrMissingError("x").exit_code == 3
    assert api.OutputExistsError("x").exit_code == 4
    assert api.OutputWriteError("x").exit_code == 5
    # issubclass(A, B): "is every A also a B?", like PHP's instanceof for classes.
    assert issubclass(api.InvalidEditsError, api.BadInputError)
    assert all(
        issubclass(error, api.ScoresplitError)
        for error in (api.BadInputError, api.OcrMissingError, api.OutputExistsError, api.OutputWriteError)
    )


# --- inspect -----------------------------------------------------------------


def test_inspect_missing_file_is_bad_input(tmp_path: Path) -> None:
    # "with pytest.raises(...) as caught" passes only if the block raises
    # that error; caught.value is the error object.
    with pytest.raises(api.BadInputError) as caught:
        api.inspect(tmp_path / "nope.pdf")
    assert caught.value.exit_code == 2
    assert "does not exist" in caught.value.message


def test_inspect_file_that_is_not_a_pdf_is_bad_input(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(api.BadInputError, match="cannot read fake.pdf: it is not a PDF"):
        api.inspect(fake, api.InspectOptions(use_cache=False, workers=1))


def test_inspect_refuses_an_unknown_ocr_mode(tmp_path: Path) -> None:
    # The CLI's --ocr option can't send this, but a Python caller can.
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    with pytest.raises(api.BadInputError, match="unknown OCR mode 'sometimes'"):
        api.inspect(pdf, api.InspectOptions(ocr_mode="sometimes"))
    with pytest.raises(api.BadInputError, match="workers must be 1 or more"):
        api.inspect(pdf, api.InspectOptions(workers=0))


def test_a_language_name_brings_no_control_character_into_the_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A language that isn't installed is named in a warning, which ends up in
    # the table and the export files. The name comes from the caller (--lang).
    def ocr_was_used(pdf: Path, *args, **kwargs) -> Result:
        return Result(source=pdf.name, sha256="0" * 64, page_count=1, ocr_pages=1)

    monkeypatch.setattr(api, "inspect_pdf", ocr_was_used)
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    result = api.inspect(pdf, api.InspectOptions(lang="eng+x\x1b[2J"))
    assert any("x [2J" in warning for warning in result.warnings)
    assert not any(CONTROL_CHARACTERS.search(warning) for warning in result.warnings)


def test_inspect_without_tesseract_is_ocr_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_tesseract(*args, **kwargs):
        raise OcrUnavailableError("Tesseract is not installed")

    # Replace the slow analysis with a fake that fails like a missing Tesseract.
    monkeypatch.setattr(api, "inspect_pdf", no_tesseract)
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    with pytest.raises(api.OcrMissingError) as caught:
        api.inspect(pdf)
    assert caught.value.exit_code == 3
    assert "sudo pacman -S tesseract" in caught.value.message


def test_inspect_reports_progress_and_returns_a_page_map(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    calls: list[tuple[int, int]] = []

    # ocr_mode "never": the blank test pages need no Tesseract.
    page_map = api.inspect(
        pdf,
        api.InspectOptions(ocr_mode="never", use_cache=False, workers=1),
        # A lambda is a tiny inline function: fn($done, $total) => ... in PHP.
        progress=lambda done, total: calls.append((done, total)),
    )
    assert page_map.page_count == 3
    assert page_map.source == "score.pdf"
    assert calls and calls[-1] == (3, 3)


# --- save_edits / load_edits -------------------------------------------------


def test_saved_map_loads_again(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    page_map = make_map([Part(name="Flute", pages="1-2")])
    page_map.sha256 = file_sha256(pdf)  # make it the map of this PDF
    map_file = tmp_path / "map.json"

    api.save_edits(page_map, map_file)
    loaded = api.load_edits(map_file, pdf)
    assert loaded.parts[0].name == "Flute"
    assert loaded.parts[0].pages == "1-2"


def test_save_edits_into_a_missing_folder_is_bad_input(tmp_path: Path) -> None:
    page_map = make_map([Part(name="Flute", pages="1")])
    with pytest.raises(api.BadInputError, match="cannot write edits file"):
        api.save_edits(page_map, tmp_path / "no_such_folder" / "map.json")


def test_load_edits_for_another_pdf_is_bad_input(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    map_file = tmp_path / "map.json"
    # make_map's sha256 is "000...", which is not this PDF's fingerprint.
    api.save_edits(make_map([Part(name="Flute", pages="1")]), map_file)

    with pytest.raises(api.BadInputError, match="made for a different PDF"):
        api.load_edits(map_file, pdf)
    # force=True accepts it anyway.
    assert api.load_edits(map_file, pdf, force=True).parts[0].name == "Flute"


def test_an_export_made_with_force_loads_back_without_force(tmp_path: Path) -> None:
    # Review step 21: split a.pdf with edits made for another PDF (--force).
    # The export JSON then describes a.pdf, so loading it back as edits on
    # a.pdf needs no --force; the warning says where the edits came from.
    pdf = make_pdf(tmp_path / "a.pdf", 2)
    map_file = tmp_path / "map.json"
    page_map = make_map([Part(name="Flute", pages="1")], page_count=2)  # for test.pdf
    page_map.title = "Song"
    api.save_edits(page_map, map_file)

    loaded = api.load_edits(map_file, pdf, force=True)
    result = api.split(pdf, loaded, tmp_path / "out")
    export_json = result.output_dir / "export" / "Song.json"

    again = api.load_edits(export_json, pdf)  # no force
    assert (again.source, again.sha256) == ("a.pdf", file_sha256(pdf))
    assert any("edits made for another PDF (test.pdf)" in warning for warning in again.warnings)
    assert again.parts[0].name == "Flute"


def test_load_edits_with_broken_json_is_bad_input(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps({"schema_version": 4, "parts": "not a list"}), encoding="utf-8")
    with pytest.raises(api.BadInputError, match="is not a valid scoresplit result") as caught:
        api.load_edits(map_file, pdf)
    # Every mistake at once, each saying where it is: "parts" has the wrong
    # type and "source", "sha256", ... are missing.
    problems = caught.value.problems
    assert "parts: Input should be a valid list" in problems
    assert "source: Field required" in problems
    assert len(problems) > 2
    with pytest.raises(api.BadInputError, match="edits file .* does not exist"):
        api.load_edits(tmp_path / "nope.json", pdf)


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_load_edits_refuses_a_file_of_an_older_scoresplit(tmp_path: Path, version: int | None) -> None:
    # Versions 1 and 2 were the old "page map"; version 1 had no schema_version.
    # Version 3 had the same JSON as 4, but the CSV layout before row_type.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    old = make_map([Part(name="Flute", pages="1")]).model_dump()
    if version is None:
        del old["schema_version"]
    else:
        old["schema_version"] = version
    map_file = tmp_path / "map.json"
    map_file.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(api.BadInputError, match=f"has schema_version {version or 1}, but this scoresplit reads only version 4"):
        api.load_edits(map_file, pdf)


def test_load_edits_refuses_a_small_file_that_lists_millions_of_pages(tmp_path: Path) -> None:
    # The review's measurement: 300 parts that all say "1-100000" are an
    # 11 KB file, but took 25 seconds and 1.2 GB. The web tool gets edits
    # files from anyone, so this must be a quick, ordinary bad-input error.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    data = make_map([Part(name="Flute", pages="1")]).model_dump()
    data["parts"] = [{"name": f"Part {i}", "pages": f"1-{MAX_PAGE_NUMBER}"} for i in range(300)]
    edits_file = tmp_path / "edits.json"
    edits_file.write_text(json.dumps(data), encoding="utf-8")
    assert edits_file.stat().st_size < 15_000  # about 13 KB

    start = time.perf_counter()
    with pytest.raises(api.BadInputError, match="is not a valid scoresplit result") as caught:
        api.load_edits(edits_file, pdf, force=True)
    assert time.perf_counter() - start < 1
    assert caught.value.problems == [
        f"parts: Value error, the parts list 30000000 pages together, more than {MAX_LISTED_PAGES} "
        "(a page in two parts counts twice)"
    ]


def test_load_edits_refuses_too_many_parts(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    data = make_map([]).model_dump()
    data["parts"] = [{"name": f"Part {i}", "pages": "1"} for i in range(MAX_PARTS + 1)]
    edits_file = tmp_path / "edits.json"
    edits_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(api.BadInputError, match="is not a valid scoresplit result") as caught:
        api.load_edits(edits_file, pdf, force=True)
    assert caught.value.problems == [
        f"parts: List should have at most {MAX_PARTS} items after validation, not {MAX_PARTS + 1}"
    ]


def test_load_edits_reads_back_edited_pages_and_works_out_the_extra_pages(tmp_path: Path) -> None:
    # A person deleted the Oboe from the saved file: its page becomes an extra page.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    map_file = tmp_path / "map.json"
    api.save_edits(make_map([Part(name="Flute", pages="1"), Part(name="Oboe", pages="2")], page_count=2), map_file)
    data = json.loads(map_file.read_text(encoding="utf-8"))
    del data["parts"][1]
    map_file.write_text(json.dumps(data), encoding="utf-8")

    loaded = api.load_edits(map_file, pdf, force=True)
    assert [part.name for part in loaded.parts] == ["Flute"]
    assert loaded.extra == [Extra(reason="unassigned", pages="2")]


# --- check_edits -------------------------------------------------------------


def test_check_edits_lists_the_problems(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    good = make_map([Part(name="Flute", pages="1-2")])
    bad = make_map([Part(name="Flute", pages="1-5"), Part(name="Oboe", pages="")])

    assert api.check_edits(good, pdf) == api.CheckReport()
    problems = api.check_edits(bad, pdf).errors
    assert len(problems) == 2
    assert "don't exist (PDF has 2)" in problems[0]
    assert "'Oboe': has no pages" in problems[1]


# --- split -------------------------------------------------------------------


def test_split_writes_the_files_and_returns_them(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 4)
    page_map = make_map(
        [Part(name="Flute", pages="1-2"), Part(name="Oboe", pages="3")],
        extra=[Extra(reason="blank", pages="4")],
    )
    page_map.title = "Song"

    result = api.split(pdf, page_map, tmp_path / "out")

    assert result.output_dir == tmp_path / "out" / "Song"
    # Parts first, then the extra pages, then the export files and the
    # copy of the original PDF.
    assert [file.relative_path for file in result.files] == [
        "01_Song_Flute_pages_1-2.pdf",
        "02_Song_Oboe_pages_3.pdf",
        "extra/blank_pages_4.pdf",
        "export/Song.json",
        "export/Song.csv",
        "export/Song.xml",
        "export/score.pdf",
    ]
    assert all(file.file.is_file() for file in result.files)


def test_split_result_links_each_file_to_its_part(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    page_map = make_map(
        [
            Part(name="Horn 2", pages="1", instrument="horn", number="2"),
            # Switched off: not written, so no file points at index 1.
            Part(name="Horn 2 (copy)", pages="1", export=False),
            Part(name="Kazoo", pages="2"),
        ],
        extra=[Extra(reason="cover", pages="3")],
    )
    page_map.title = "Song"

    result = api.split(pdf, page_map, tmp_path / "out")

    # "part" is the index in page_map.parts; None for extra pages and exports.
    assert [file.part for file in result.files] == [0, 2, None, None, None, None, None]
    horn = result.files[0]
    assert horn.file == (tmp_path / "out" / "Song" / "01_Song_Horn_2_pages_1.pdf").resolve()
    assert horn.file.is_absolute()
    assert result.files[2].relative_path == "extra/cover_pages_3.pdf"


def test_split_without_out_dir_writes_next_to_the_pdf(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    page_map = make_map([Part(name="Flute", pages="1")])
    page_map.title = "Song"

    result = api.split(pdf, page_map, include_extra=False)
    assert result.output_dir == tmp_path / "Song"


def test_split_twice_into_the_same_folder_is_output_exists(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    page_map = make_map([Part(name="Flute", pages="1-2")])
    api.split(pdf, page_map, tmp_path / "out")

    with pytest.raises(api.OutputExistsError) as caught:
        api.split(pdf, page_map, tmp_path / "out")
    assert caught.value.exit_code == 4
    assert "(use --force)" in caught.value.message

    # force=True writes into the existing folder (the part, 3 export files
    # and the copy of the original PDF).
    assert len(api.split(pdf, page_map, tmp_path / "out", force=True).files) == 5


def test_split_with_an_invalid_map_writes_nothing(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    page_map = make_map([Part(name="Flute", pages="1-9")])

    with pytest.raises(api.InvalidEditsError) as caught:
        api.split(pdf, page_map, tmp_path / "out")
    assert len(caught.value.problems) == 1
    assert caught.value.message == "the parts can't be split as they are"
    assert not (tmp_path / "out").exists()


def test_split_that_cannot_write_is_output_write_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(api, "export_parts", disk_full)
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    with pytest.raises(api.OutputWriteError) as caught:
        api.split(pdf, make_map([Part(name="Flute", pages="1")]), tmp_path / "out")
    assert caught.value.exit_code == 5
    assert caught.value.message == "cannot write the output files: [Errno 28] No space left on device"
