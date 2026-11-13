"""
Tests for the output folder that exporter.py writes: the part file names,
the folder layout, and the three export files (JSON, CSV, XML), read back
and compared with the result.

They build tiny PDFs and results by hand, so they run in about a second.
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scoresplit.exporter import (
    CREDIT_FIELDS,
    CSV_COLUMNS,
    MAX_FILENAME_BYTES,
    MAX_NAME_BYTES,
    PART_FIELDS,
    _write_xml,
    export_parts,
    extra_filename,
    part_filename,
    part_filenames,
    safe_filename,
)
from scoresplit.models import Extra, Part, Result

# The tiny-PDF and result helpers of the other exporter tests.
from test_export_and_cli import make_map, make_pdf, page_count_of


# ---------------------------------------------------------------------------
# Part file names
# ---------------------------------------------------------------------------

# One row per rule: (the part, the file name it must get). pytest runs the
# test below once per row, and shows the row's id when one fails.
FILE_NAMES = [
    pytest.param(
        Part(name="Euphonium", instrument="euphonium", key="C", clef="bass", pages="171-174"),
        "Euphonium_C_bc_pages_171-174.pdf",
        id="name, key, bass clef, pages",
    ),
    pytest.param(
        Part(name="Clarinet 2", instrument="clarinet", number="2", key="Bb", clef="treble", pages="19-21"),
        "Clarinet_2_Bb_tc_pages_19-21.pdf",
        id="spaces become underscores",
    ),
    pytest.param(
        Part(name="Flute 1", instrument="flute", number="1", key="C", key_assumed=True,
             clef="treble", pages="25-27"),
        "Flute_1_C_tc_pages_25-27.pdf",
        id="an assumed key is in the name too",
    ),
    pytest.param(
        Part(name="2. klarinet in Es", instrument="clarinet", number="2", key="Eb", clef="treble",
             pages="40-42"),
        "2_klarinet_in_Es_tc_pages_40-42.pdf",
        id="key already in the name (another spelling)",
    ),
    pytest.param(
        Part(name="Bb Clarinet 1", instrument="clarinet", number="1", key="Bb", clef="treble", pages="5"),
        "Bb_Clarinet_1_tc_pages_5.pdf",
        id="key already in front of the name",
    ),
    pytest.param(
        Part(name="2. klarinet in Es", instrument="clarinet", number="2", key="Bb", clef="treble",
             pages="40-42"),
        "2_klarinet_in_Es_Bb_tc_pages_40-42.pdf",
        id="the name says another key than the part: the part's key is added",
    ),
    pytest.param(
        Part(name="Conductor Score", kind="score", key="C", clef="treble", pages="1-24"),
        "Conductor_Score_pages_1-24.pdf",
        id="the conductor score gets no key and no clef",
    ),
    pytest.param(
        Part(name="Timpani", instrument="timpani", pages="200-201"),
        "Timpani_pages_200-201.pdf",
        id="no key, no clef",
    ),
    pytest.param(
        Part(name="Timpani", instrument="timpani", clef="bass", pages="237-238"),
        "Timpani_bc_pages_237-238.pdf",
        id="no key, bass clef",
    ),
    pytest.param(
        Part(name="Drums", instrument="drum_set", clef="percussion", pages="230-233"),
        "Drums_pages_230-233.pdf",
        id="a percussion clef gets no sign",
    ),
    pytest.param(
        Part(name="Viola", instrument="viola", key="C", clef="alto", pages="11-12"),
        "Viola_C_pages_11-12.pdf",
        id="an alto clef gets no sign",
    ),
    pytest.param(
        Part(name="Piano", instrument="piano", clef=None, pages="7-9"),
        "Piano_pages_7-9.pdf",
        id="mixed clefs (piano): no clef",
    ),
    pytest.param(
        Part(name="Trombone 1/2", instrument="trombone", number="1&2", key="C", clef="bass",
             pages="5-7,9"),
        "Trombone_1-2_C_bc_pages_5-7_9.pdf",
        id="slash becomes a dash, several page ranges",
    ),
    pytest.param(
        Part(name='Perc.: "Shaker"?  <a2>', pages="3"),
        "Perc_Shaker_a2_pages_3.pdf",
        id="characters not allowed in file names are removed",
    ),
]


@pytest.mark.parametrize(("part", "expected"), FILE_NAMES)
def test_part_file_name(part: Part, expected: str) -> None:
    assert part_filename(part) == expected


def test_a_long_scattered_page_list_is_shortened() -> None:
    scattered = list(range(6, 240, 6))  # 6, 12, 18, ... 234
    part = Part(name="Flute", pages=scattered)
    assert part_filename(part) == f"Flute_{len(scattered)}_pages_6-234.pdf"


def test_a_very_long_name_stays_within_the_file_name_limit() -> None:
    part = Part(name="Š" * 300, key="Bb", clef="treble", pages="1-2")
    name = part_filename(part)
    assert len(name.encode("utf-8")) < 255  # the limit of most file systems
    assert name.endswith("_Bb_tc_pages_1-2.pdf")


def _byte_length(name: str) -> int:
    """How many bytes a file name takes on disk (UTF-8)."""
    return len(name.encode("utf-8"))


# The part of review step 2 that crashed `split` (OSError: File name too
# long): the name alone is under the 200-byte cap, but with the row number,
# key, clef and a long page list the whole file name went over 255 bytes.
LONG_NAME = "Shaker Tambourine Wind Chimes " * 8
LONG_PAGES = "100-101,103-104,106-107,109-110,112-113"


def _long_parts_result(name: str) -> Result:
    result = make_map(
        [Part(name=name, key="Bb", clef="treble", pages=LONG_PAGES)], page_count=120
    )
    result.title = "A Long Title"
    result.arranger = "Some Arranger"
    return result


@pytest.mark.parametrize("name", [LONG_NAME, "š" * 150], ids=["ascii", "two-byte letters"])
def test_a_long_part_name_is_shortened_to_fit_the_file_name_limit(name: str) -> None:
    file_name = part_filenames(_long_parts_result(name))[0]
    assert _byte_length(file_name) <= MAX_FILENAME_BYTES
    # The key, clef and pages are kept: the name is shortened instead.
    assert file_name.endswith("_Bb_tc_pages_100-101_103-104_106-107_109-110_112-113.pdf")
    # The credits went first (the part's own name matters more), and what
    # is left of the name is still its start.
    assert file_name.startswith("01_" + name.replace(" ", "_")[:20])
    # decode() would fail if a two-byte letter had been cut in half.
    file_name.encode("utf-8").decode("utf-8")


def test_a_garbage_key_from_an_edits_file_cannot_make_the_name_too_long() -> None:
    # The key is free text in an edits file, so it can be just as long.
    part = Part(name=LONG_NAME, key="X" * 300, clef="treble", pages=LONG_PAGES)
    file_name = part_filenames(make_map([part], page_count=120))[0]
    assert _byte_length(file_name) <= MAX_FILENAME_BYTES
    assert file_name.endswith("_tc_pages_100-101_103-104_106-107_109-110_112-113.pdf")


def test_split_writes_a_part_with_a_very_long_name(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 120)
    target, written = export_parts(_long_parts_result(LONG_NAME), pdf, tmp_path / "out")
    part_file = written[0].path
    assert part_file.exists()
    assert page_count_of(part_file) == 10
    assert _byte_length(part_file.name) <= MAX_FILENAME_BYTES


def test_the_longest_extra_file_name_is_short() -> None:
    # The longest reason, and the longest page list that isn't shortened.
    scattered = list(range(1, 200, 2))  # 1, 3, 5 ... 199
    name = extra_filename(Extra(reason="unassigned", pages=scattered))
    assert _byte_length(name) <= MAX_FILENAME_BYTES
    assert _byte_length(name) < 100


def test_the_main_folder_name_stays_below_the_limit() -> None:
    result = make_map([Part(name="Flute", pages="1")])
    result.title = "Š" * 300
    result.arranger = "Ž" * 300
    assert _byte_length(safe_filename(result.folder_name())) <= MAX_NAME_BYTES < 255


def test_extra_file_names() -> None:
    assert extra_filename(Extra(reason="cover", pages="1")) == "cover_pages_1.pdf"
    assert extra_filename(Extra(reason="blank", pages="237-238")) == "blank_pages_237-238.pdf"
    assert extra_filename(Extra(reason="unassigned", pages="2,9")) == "unassigned_pages_2_9.pdf"


# ---------------------------------------------------------------------------
# The folder layout and the export files
# ---------------------------------------------------------------------------


def _example(tmp_path: Path) -> tuple[Path, Result]:
    """A 10-page PDF with a score, three parts (one a duplicate) and extra pages."""
    pdf = make_pdf(tmp_path / "score.pdf", 10)
    result = make_map(
        [
            Part(name="Conductor Score", kind="score", pages="2-3"),
            Part(name="Flute 1", instrument="flute", number="1", key="C", key_assumed=True,
                 clef="treble", pages="4-5"),
            Part(name="2. klarinet in Es", instrument="clarinet", number="2", key="Eb",
                 clef="treble", pages="6-7"),
            Part(name="2. klarinet in Es", instrument="clarinet", number="2", key="Eb",
                 clef="treble", pages="8", export=False, duplicate_of="2. klarinet in Es",
                 notes=["duplicate of '2. klarinet in Es' (pages 6-7)"]),
        ],
        extra=[Extra(reason="cover", pages="1"), Extra(reason="blank", pages="10")],
    )
    # Page 9 is in no part and detection gave it no reason: "unassigned".
    result.title = "Song & Dance"
    result.arranger = "A. Person"
    return pdf, result


def test_split_writes_the_planned_layout(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, written = export_parts(result, pdf, tmp_path / "out")

    assert target == tmp_path / "out" / "Song & Dance - A. Person"
    # rglob("*") lists everything below the folder; keep only the files.
    files = sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
    assert files == sorted(
        [
            "01_Song_&_Dance_A_Person_Conductor_Score_pages_2-3.pdf",
            "02_Song_&_Dance_A_Person_Flute_1_C_tc_pages_4-5.pdf",
            "03_Song_&_Dance_A_Person_2_klarinet_in_Es_tc_pages_6-7.pdf",
            "extra/cover_pages_1.pdf",
            "extra/unassigned_pages_9.pdf",
            "extra/blank_pages_10.pdf",
            "export/Song & Dance - A. Person.json",
            "export/Song & Dance - A. Person.csv",
            "export/Song & Dance - A. Person.xml",
            "export/score.pdf",
        ]
    )
    # The written list: parts, then extras, then export files.
    assert [file.path for file in written] == [
        target / "01_Song_&_Dance_A_Person_Conductor_Score_pages_2-3.pdf",
        target / "02_Song_&_Dance_A_Person_Flute_1_C_tc_pages_4-5.pdf",
        target / "03_Song_&_Dance_A_Person_2_klarinet_in_Es_tc_pages_6-7.pdf",
        target / "extra" / "cover_pages_1.pdf",
        target / "extra" / "unassigned_pages_9.pdf",
        target / "extra" / "blank_pages_10.pdf",
        target / "export" / "Song & Dance - A. Person.json",
        target / "export" / "Song & Dance - A. Person.csv",
        target / "export" / "Song & Dance - A. Person.xml",
        target / "export" / "score.pdf",
    ]
    assert page_count_of(target / "01_Song_&_Dance_A_Person_Conductor_Score_pages_2-3.pdf") == 2
    assert page_count_of(target / "extra" / "blank_pages_10.pdf") == 1
    # The original PDF is copied as it is: the very same bytes.
    assert (target / "export" / "score.pdf").read_bytes() == pdf.read_bytes()


def test_without_extra_pages_no_extra_folder_is_made(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out", include_extra=False)
    assert not (target / "extra").exists()
    assert (target / "export").is_dir()


def test_the_json_export_reads_back_as_the_same_result(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out")
    data = json.loads((target / "export" / f"{target.name}.json").read_text(encoding="utf-8"))
    # model_validate builds a Result from the JSON; the duplicate is in it too.
    # The added "file" keys are ignored, so the JSON works as --edits.
    assert Result.model_validate(data) == result
    assert data["parts"][1]["file"] == "02_Song_&_Dance_A_Person_Flute_1_C_tc_pages_4-5.pdf"
    assert data["parts"][3]["file"] == ""  # the duplicate wasn't written
    assert data["extra"][0] == {"reason": "cover", "pages": "1", "file": "extra/cover_pages_1.pdf"}


def test_the_csv_export_has_one_row_per_part_and_extra_group(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out")
    # "utf-8-sig" reads past the marker the CSV starts with (for Excel).
    with (target / "export" / f"{target.name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "name", "instrument", "number", "key", "key_assumed", "clef", "clef_assumed",
        "kind", "pages", "export", "duplicate_of", "confidence", "notes", "file",
        "schema_version", "source", "sha256", "page_count", "ocr_pages",
        "title", "composer", "arranger", "warnings",
    ]
    assert rows[0]["name"] == "Conductor Score"
    assert rows[0]["file"] == "01_Song_&_Dance_A_Person_Conductor_Score_pages_2-3.pdf"
    assert rows[0]["confidence"] == "1.0"
    # The credits are on every row.
    assert {row["title"] for row in rows} == {"Song & Dance"}
    assert rows[1]["key"] == "C" and rows[1]["key_assumed"] == "true"
    # The duplicate is listed, with its note, but wasn't written: no file.
    assert (rows[3]["export"], rows[3]["file"]) == ("false", "")
    assert rows[3]["duplicate_of"] == "2. klarinet in Es"
    assert rows[3]["notes"] == "duplicate of '2. klarinet in Es' (pages 6-7)"
    # After the parts, one row per group of extra pages.
    assert [(row["kind"], row["name"], row["pages"], row["file"]) for row in rows[4:]] == [
        ("cover", "Cover", "1", "extra/cover_pages_1.pdf"),
        ("unassigned", "Unassigned", "9", "extra/unassigned_pages_9.pdf"),
        ("blank", "Blank", "10", "extra/blank_pages_10.pdf"),
    ]
    # Every listed file exists, relative to the main folder.
    assert all((target / row["file"]).is_file() for row in rows if row["file"])


def test_the_csv_export_never_starts_a_cell_with_a_formula(tmp_path: Path) -> None:
    # A spreadsheet runs a cell starting with "=", "+", "-", "@" (or a tab,
    # or a carriage return) as a formula. Titles and part names come from an
    # uploaded PDF, so a crafted PDF could run one in the reader's Excel.
    # The CSV puts a "'" in front; the JSON and XML keep the text as it is.
    pdf, result = _example(tmp_path)
    result.title = "=1+1"
    result.parts[1].name = "@SUM(A1)"
    result.parts[2].notes = ["-2+3"]
    target, _ = export_parts(result, pdf, tmp_path / "out")
    export = target / "export"

    with (export / f"{target.name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["title"] for row in rows} == {"'=1+1"}
    assert rows[1]["name"] == "'@SUM(A1)"
    assert rows[2]["notes"] == "'-2+3"
    # No cell anywhere starts with a formula character.
    formula_starts = ("=", "+", "-", "@", "\t", "\r")
    assert not [cell for row in rows for cell in row.values() if cell.startswith(formula_starts)]

    data = json.loads((export / f"{target.name}.json").read_text(encoding="utf-8"))
    assert data["title"] == "=1+1"
    assert data["parts"][1]["name"] == "@SUM(A1)"
    root = ET.parse(export / f"{target.name}.xml").getroot()
    assert root.findtext("title") == "=1+1"
    assert root.findall("parts/part")[1].findtext("name") == "@SUM(A1)"


def test_the_xml_export_holds_the_credits_parts_and_extra_pages(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out")
    root = ET.parse(target / "export" / f"{target.name}.xml").getroot()

    assert root.tag == "scoresplit"
    assert root.get("schema_version") == "3"
    # findtext gives the text of the first matching element.
    assert root.findtext("title") == "Song & Dance"  # "&" written as "&amp;" and read back
    assert root.findtext("arranger") == "A. Person"
    assert root.findtext("composer") == ""
    assert root.findtext("ocr_pages") == "0"

    parts = root.findall("parts/part")
    assert len(parts) == 4
    flute = {child.tag: child.text or "" for child in parts[1]}
    assert flute == {
        "name": "Flute 1", "instrument": "flute", "number": "1", "key": "C",
        "key_assumed": "true", "clef": "treble", "clef_assumed": "false", "kind": "part", "pages": "4-5",
        "export": "true", "duplicate_of": "", "confidence": "1.0", "notes": "",
        "file": "02_Song_&_Dance_A_Person_Flute_1_C_tc_pages_4-5.pdf",
    }
    assert parts[3].findtext("file") == ""  # the duplicate wasn't written
    assert [note.text for note in parts[3].findall("notes/note")] == [
        "duplicate of '2. klarinet in Es' (pages 6-7)"
    ]

    extras = [(pages.get("reason"), pages.text, pages.get("file")) for pages in root.findall("extra/pages")]
    assert extras == [
        ("cover", "1", "extra/cover_pages_1.pdf"),
        ("unassigned", "9", "extra/unassigned_pages_9.pdf"),
        ("blank", "10", "extra/blank_pages_10.pdf"),
    ]


def test_the_xml_export_parses_whatever_control_characters_the_text_holds(tmp_path: Path) -> None:
    # XML 1.0 forbids most control characters (\x0c = form feed, \x03 ...).
    # They can come from a PDF text layer, OCR or an edits file; before,
    # the XML was written but couldn't be read ("not well-formed").
    pdf, result = _example(tmp_path)
    result.title = "Song\x0c & Dance"
    result.warnings = ["odd\x03 text"]
    result.parts[1].name = "Flute\x0c 1"
    result.parts[1].notes = ["read\x1f on page 4"]
    target, _ = export_parts(result, pdf, tmp_path / "out")

    root = ET.parse(target / "export" / f"{target.name}.xml").getroot()
    assert root.findtext("title") == "Song & Dance"
    assert [warning.text for warning in root.findall("warnings/warning")] == ["odd text"]
    flute = root.findall("parts/part")[1]
    assert flute.findtext("name") == "Flute 1"
    assert [note.text for note in flute.findall("notes/note")] == ["read on page 4"]


def test_the_three_export_files_hold_the_same_part_fields(tmp_path: Path) -> None:
    """JSON, CSV and XML list the same fields for a part: none is missing anywhere."""
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out")
    export = target / "export"
    json_fields = set(json.loads((export / f"{target.name}.json").read_text(encoding="utf-8"))["parts"][0])
    with (export / f"{target.name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_fields = set(next(csv.DictReader(handle)))
    xml_fields = {child.tag for child in ET.parse(export / f"{target.name}.xml").getroot().find("parts/part")}
    assert json_fields == xml_fields
    assert json_fields <= csv_fields  # the CSV adds the credits to every row


def test_extra_pages_not_written_are_still_listed_without_a_file(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out", include_extra=False)
    data = json.loads((target / "export" / f"{target.name}.json").read_text(encoding="utf-8"))
    assert [(extra["reason"], extra["file"]) for extra in data["extra"]] == [
        ("cover", ""), ("unassigned", ""), ("blank", ""),
    ]


# Each CSV row is {**part, **credits}: a part's fields merged with the
# fields of the whole result. If Part and Result ever got a field with the
# same name, the result's value would silently replace the part's on every
# row, and the header would have that column twice. These two tests make
# such a clash fail here instead of quietly corrupting the CSV.
# (A test, not an assert in exporter.py: "python -O" turns asserts off.)


def test_part_fields_and_credit_fields_share_no_name() -> None:
    # isdisjoint = "the two have no name in common".
    assert set(PART_FIELDS).isdisjoint(CREDIT_FIELDS)


def test_the_csv_columns_have_no_duplicates() -> None:
    # A set drops duplicates, so the lengths differ when a name is there
    # twice (this also covers the extra "file" column between the two lists).
    assert len(set(CSV_COLUMNS)) == len(CSV_COLUMNS)


# The XML finds the list fields by their VALUE (a list), not by their name,
# so a list field added later becomes child elements by itself, like in the
# JSON. "tags" is a made-up field that Part doesn't have today.


def test_a_new_list_field_becomes_child_elements_in_the_xml(tmp_path: Path) -> None:
    pdf, result = _example(tmp_path)
    target, _ = export_parts(result, pdf, tmp_path / "out")
    data = json.loads((target / "export" / f"{target.name}.json").read_text(encoding="utf-8"))
    data["parts"][0]["tags"] = ["brass", "solo"]  # a list field nobody named in exporter.py
    data["parts"][1]["tags"] = []

    xml_path = tmp_path / "tags.xml"
    _write_xml(data, xml_path)

    parts = ET.parse(xml_path).getroot().findall("parts/part")
    # One <tag> per item ("tags" without its "s"), not one joined "brass | solo".
    assert [tag.text for tag in parts[0].findall("tags/tag")] == ["brass", "solo"]
    # No joined text in <tags> itself (only the indentation ET.indent adds).
    assert parts[0].findtext("tags").strip() == ""
    assert parts[1].find("tags") is not None  # an empty list: an empty <tags/>
    assert parts[1].findall("tags/tag") == []
