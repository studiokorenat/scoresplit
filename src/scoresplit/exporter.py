"""
exporter.py: write one PDF per part, the extra pages and the export files.

Pipeline position:  (result) -> [exporter] -> files on disk

Resulting layout (flat: every part in the main folder):

    <out>/Nuovo Cinema Paradiso - Peter Kleine Schaars/
        01_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Conductor_Score_pages_1-24.pdf
        02_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Flute_1_C_tc_pages_25-27.pdf
        ...
        extra/
            cover_pages_1.pdf
            blank_pages_237-238.pdf
        export/
            Nuovo Cinema Paradiso - Peter Kleine Schaars.json
            Nuovo Cinema Paradiso - Peter Kleine Schaars.csv
            Nuovo Cinema Paradiso - Peter Kleine Schaars.xml
            paradiso.pdf            (the original PDF, the whole score)

A part's file name is built by part_filenames(): its row number among the
exported parts ("01", "02", ...), the title, the arranger, then what
part_filename() gives: the name as printed, the key, the clef ("tc" /
"bc") and the pages. See those functions for the rules.

Pages are copied with PDFium's import_pages(): the page content is copied
as-is (no re-rendering), so the quality is identical to the original.

The three export files hold the SAME data in three formats (the credits,
every field of every part with its PDF file name, the extra pages with
their file names, the warnings), written with Python's own json, csv and
xml modules (no extra libraries). A person deletes what they don't need:

    .json  the whole result (models.Result) plus the file names: it can be
           edited and given back to `scoresplit split --edits`.
    .csv   one row per part and per group of extra pages, for a spreadsheet.
    .xml   the same, for programs that prefer XML.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from .instruments import find_part_name
from .models import (
    CheckReport,
    Extra,
    Part,
    ProgressCallback,
    Result,
    check_result,
    format_page_ranges,
)

# The two subfolders of the main output folder.
EXTRA_FOLDER = "extra"  # pages that are in no part (cover, blank, ...)
EXPORT_FOLDER = "export"  # the result as JSON, CSV and XML

# How each kind of extra page is called in the terminal table. (The file
# names use the reason itself: "cover_pages_1.pdf".)
EXTRA_LABELS = {
    "cover": "Cover",
    "blank": "Blank",
    "unassigned": "Unassigned",
}

# The three export files (JSON, CSV, XML) hold the SAME data; a person can
# delete what they don't need afterwards. The field lists are read from the
# models themselves, so a field added to Part or Result later shows up in
# all three files by itself.
#
# Every field of a part: name, instrument, ..., confidence, notes.
# (Part.model_fields is pydantic's dict of the fields, in the class order.)
PART_FIELDS = list(Part.model_fields)
# Every field about the whole PDF: source, title, composer, ..., warnings.
# "parts" and "extra" are left out: they are lists of their own.
CREDIT_FIELDS = [name for name in Result.model_fields if name not in ("parts", "extra")]

# The columns of the CSV export, in this order: one row per part (and one
# per group of extra pages), then "file" = where that PDF was written,
# relative to the main folder ("" when it wasn't written), then the
# credits, repeated on every row so each row stands on its own.
CSV_COLUMNS = [*PART_FIELDS, "file", *CREDIT_FIELDS]

# How a list (a part's notes, the warnings) is written in ONE CSV cell.
CSV_LIST_SEPARATOR = " | "


@dataclass(frozen=True)
class ExportedFile:
    """
    One file that export_parts() wrote.

    ``part`` is the position of the part in ``result.parts`` (0 = the first
    part) when the file holds a part, and None for the extra pages and the
    three export files (JSON, CSV, XML). With it a caller can say "this
    file is the Clarinet 1 part" without matching file names to part names.
    """

    path: Path  # where the file was written (inside the main output folder)
    part: int | None = None  # index in result.parts, or None


class OutputExistsError(Exception):
    """The output folder already exists and isn't empty (use --force)."""


class InvalidResultError(Exception):
    """The result can't be exported as it is (see check_export)."""


# Most file systems allow at most 255 BYTES per file or folder name. Accented
# letters take 2 bytes in UTF-8, so we count bytes, not characters, and stay
# well below the limit (the key, clef, pages and ".pdf" are added later).
MAX_NAME_BYTES = 200

# The whole file name of a part stays below this many bytes (the limit of
# most file systems is 255). When the title and arranger would make it
# longer, they are shortened: the part's own name and pages matter more.
MAX_FILENAME_BYTES = 250

# A file name lists its pages, e.g. "blank_pages_2_9.pdf". With many
# scattered pages that list gets very long, so above this many characters it
# is shortened to "blank_40_pages_6-236.pdf" (40 pages, from 6 to 236). The
# full page list is still in the export files.
MAX_PAGES_IN_FILENAME = 40

# The clef in a part's file name: the short signs printed on parts
# ("Baritone T.C."). Only treble and bass clef get one; an alto, tenor or
# percussion clef is left out, to keep the names short.
CLEF_SIGNS = {
    "treble": "tc",
    "bass": "bc",
}


def safe_filename(name: str) -> str:
    """
    Make a string safe to use as a file/folder name on Linux, Windows and macOS.
    Spaces are kept: this is used for the main folder ("Title - Arranger").

    "Trombone 1/2 in Bb" -> "Trombone 1-2 in Bb"
    'Title: "Live"?'     -> "Title Live"
    """
    # A slash would make a subfolder, so it becomes a dash: "1/2" -> "1-2"
    # (removing it would turn "1/2" into "12", a different number).
    name = name.replace("/", "-").replace("\\", "-")
    # Remove characters Windows forbids (these files may be
    # opened on Windows) and invisible control characters.
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name)  # tidy up spaces
    # Cut very long names (e.g. OCR garbage). encode() gives the UTF-8 bytes;
    # errors="ignore" drops a letter that was cut in half.
    name = name.encode("utf-8")[:MAX_NAME_BYTES].decode("utf-8", errors="ignore")
    name = name.strip(" .")  # no leading/trailing dots or spaces
    return name or "Untitled"


def _filename_words(text: str) -> str:
    """
    A name made fit for the start of a part's file name: what safe_filename
    does, plus spaces become "_" and dots are dropped.

        "2. klarinet in Es" -> "2_klarinet_in_Es"
        "Trombone 1/2"      -> "Trombone_1-2"
        "Picc."             -> "Picc"
    """
    text = safe_filename(text).replace(".", " ")
    # split() without arguments splits on any run of spaces, so
    # "2  klarinet" gives ["2", "klarinet"]: no "__" in the file name.
    return "_".join(text.split()) or "Untitled"


def _pages_for_filename(pages: list[int]) -> str:
    """
    The pages part of a file name:

        [5, 6, 7]    -> "pages_5-7"
        [5, 6, 7, 9] -> "pages_5-7_9"
        40 scattered pages from 6 to 236 -> "40_pages_6-236"
    """
    # "5-7,9" -> "5-7_9": a comma is allowed in file names, but awkward to type.
    text = format_page_ranges(pages).replace(",", "_")
    if len(text) > MAX_PAGES_IN_FILENAME:
        # pages[0] is the first page, pages[-1] the last one.
        return f"{len(pages)}_pages_{pages[0]}-{pages[-1]}"
    return f"pages_{text}"


def _name_has_key(part: Part) -> bool:
    """
    True when the part's name already says its key: "2. klarinet in Es",
    "Bb Clarinet 1", "Horn in F". Then the file name doesn't repeat it.

    The name is read with the same reader detection uses (find_part_name),
    so every spelling it knows counts ("in Es" = Eb, "in B" = Bb). Only the
    part's own key counts: if a person changed the key to Bb but the name
    still says "in Es", the file name gets "_Bb" so it isn't hidden.
    """
    reading = find_part_name(part.name)
    return reading is not None and reading.key == part.key


def part_filename(part: Part, max_bytes: int = MAX_FILENAME_BYTES) -> str:
    """
    The file name of a part's PDF: <name>_<key>_<clef>_pages_<pages>.pdf

        Euphonium, key C, bass clef, pages 171-174
            -> "Euphonium_C_bc_pages_171-174.pdf"
        "2. klarinet in Es", treble clef, pages 40-42   (key already in the name)
            -> "2_klarinet_in_Es_tc_pages_40-42.pdf"
        Conductor Score (no key, no clef)
            -> "Conductor_Score_pages_1-24.pdf"

    The rules:
      * the name as printed; spaces become "_", characters that aren't
        allowed in file names are removed;
      * the key is left out for the conductor score, for parts without a
        key (drums, piano, guitar: their key is None), and when the name
        already contains it;
      * the clef as a short sign: "tc" (treble clef) or "bc" (bass clef),
        the abbreviations printed on parts ("Baritone T.C."). Any other
        clef (alto, tenor, percussion), none, mixed clefs (piano: None) and
        the conductor score get no sign;
      * the pages: "pages_5-7", or "pages_5-7_9" for several ranges.

    The whole name, ".pdf" included, is at most ``max_bytes`` bytes. When
    it would be longer, the name is shortened (from the end): the key, clef
    and pages tell files apart better than the end of a very long name.
    """
    name = _filename_words(part.name)
    key = ""
    clef_sign = ""
    if part.kind != "score":
        if part.key and not _name_has_key(part):
            key = _filename_words(part.key)
        # .get() gives "" for a clef that isn't in CLEF_SIGNS.
        clef_sign = CLEF_SIGNS.get(part.clef or "", "")
    pages = _pages_for_filename(part.page_list())

    # The end that is always kept whole: clef, pages and ".pdf". It is at
    # most about 55 bytes (the page list is capped by MAX_PAGES_IN_FILENAME).
    # filter(None, ...) drops an empty clef sign: no "__pages".
    ending = "_".join(filter(None, [clef_sign, pages])) + ".pdf"
    # The bytes left for the name and the key, with a "_" after each.
    room = max_bytes - _byte_length(ending)
    # A real key is short ("Bb"), so normally it fits whole. Only a garbage
    # key typed into an edits file can be long; then it is cut too.
    key = _cut_bytes(key, room - 1)
    if key:
        room -= _byte_length(key) + 1
    name = _cut_bytes(name, room - 1)
    return "_".join(filter(None, [name, key, ending]))


def extra_filename(extra: Extra) -> str:
    """
    The file name for a group of extra pages: "blank_pages_237-238.pdf".
    Long page lists are shortened: "blank_40_pages_6-234.pdf".
    """
    return f"{extra.reason}_{_pages_for_filename(extra.page_list())}.pdf"


def _byte_length(text: str) -> int:
    """How many bytes `text` takes in UTF-8 ("Š" takes 2, "S" takes 1)."""
    return len(text.encode("utf-8"))


def _cut_bytes(text: str, limit: int) -> str:
    """
    `text` cut to at most `limit` bytes of UTF-8, without cutting a letter
    in half and without a "_" left at the end. A limit of 0 or less gives "".
    """
    if limit <= 0:
        return ""
    cut = text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    return cut.rstrip("_")


def _credits_for_filename(result: Result) -> str:
    """
    The title and the arranger, for the front of a part's file name:

        "Film Medley", arranged by "Dennis Mariev" -> "Film_Medley_Dennis_Mariev"

    Like the main folder (Result.folder_name), an original composition
    without an arranger uses the composer. What wasn't found is left out
    (no "Untitled" in every file name); with neither, this gives "".
    """
    person = result.arranger or result.composer
    # A list comprehension that keeps only the texts that were found.
    pieces = [_filename_words(text) for text in (result.title, person) if text]
    return "_".join(pieces)


def part_filenames(result: Result) -> dict[int, str]:
    """
    The file name of every exported part, by its position in result.parts:

        {0: "01_Film_Medley_Dennis_Mariev_Conductor_Score_pages_1-24.pdf",
         1: "02_Film_Medley_Dennis_Mariev_Flute_1_C_tc_pages_25-27.pdf", ...}

    In front of part_filename() come:
      * the row number: 1, 2, 3 ... counting only the exported parts, in
        the result's order, written with at least two digits ("01") so the
        files sort in that order in a file manager. With 100 parts or more
        every number gets three digits ("001");
      * the title and the arranger (see _credits_for_filename), shortened
        when the whole name would get too long.

    The row number makes every file name different, even for two parts with
    the same name and pages.
    """
    credits = _credits_for_filename(result)
    # The positions of the exported parts: [0, 1, 3, ...] (2 = a duplicate).
    exported = [index for index, part in enumerate(result.parts) if part.export]
    # len(str(123)) = 3 digits. max() keeps at least 2.
    digits = max(2, len(str(len(exported))))

    names: dict[int, str] = {}
    # enumerate(..., start=1) counts 1, 2, 3 ... instead of 0, 1, 2 ...
    for row, index in enumerate(exported, start=1):
        number = str(row).zfill(digits)  # zfill pads with zeros: "1" -> "01"
        # The part's own name gets the room left after the number and its
        # "_". It is only shortened when it can't fit even without credits.
        rest = part_filename(result.parts[index], MAX_FILENAME_BYTES - len(number) - 1)
        # The bytes left for the credits: the limit, less the number, the
        # rest and the two "_" between the three pieces. When nothing is
        # left, _cut_bytes gives "" and the credits are left out.
        room = MAX_FILENAME_BYTES - len(number) - _byte_length(rest) - 2
        short_credits = _cut_bytes(credits, room)
        # filter(None, ...) drops empty pieces: no "01__Flute" without credits.
        names[index] = "_".join(filter(None, [number, short_credits, rest]))
    return names


def check_export(result: Result, page_count: int) -> CheckReport:
    """
    Check that the result can be exported safely: every problem of
    models.check_result (pages that don't exist, exported parts without
    pages, ...).

    Two parts can't end up in the same file: every file name starts with
    its own row number (see part_filenames).

    ``page_count`` is the number of pages of the PDF that will be cut.
    """
    return check_result(result, page_count)


def _write_pages(source: pdfium.PdfDocument, pages: list[int], target: Path) -> None:
    """Copy the given 1-based pages from `source` into a new PDF file `target`."""
    # Careful: import_pages() with an EMPTY list copies ALL pages, so an empty
    # list must never get this far.
    if not pages:
        raise ValueError(f"no pages to write to {target}")
    new_pdf = pdfium.PdfDocument.new()
    # import_pages wants 0-based indexes, hence p - 1.
    new_pdf.import_pages(source, [p - 1 for p in pages])
    target.parent.mkdir(parents=True, exist_ok=True)
    new_pdf.save(str(target))
    new_pdf.close()


# ---------------------------------------------------------------------------
# The export files: JSON, CSV, XML
# ---------------------------------------------------------------------------


def _export_data(result: Result, files: dict[int, str], extras: list[tuple[Extra, str]]) -> dict:
    """
    Everything the three export files hold, as one plain dict (the JSON
    shape). The CSV and the XML are written from this same dict, so the
    three files can't drift apart:

        {"schema_version": 3, "source": "paradiso.pdf", ..., "warnings": [...],
         "parts": [{"name": "Flute 1", ..., "notes": [...], "file": "02_...pdf"}],
         "extra": [{"reason": "cover", "pages": "1", "file": "extra/cover_pages_1.pdf"}]}

    ``files``  = {position of the part: its PDF file name} (parts written).
    ``extras`` = (group of extra pages, its file or "" when not written).

    "file" is an addition to the Result shape. The JSON can still be given
    back to `scoresplit split --edits`: pydantic ignores keys it doesn't know.
    """
    # mode="json" turns every value into plain JSON types (str, int, list...).
    data = result.model_dump(mode="json")
    for index, part in enumerate(data["parts"]):
        # .get(index, "") = the file name, or "" for a part not exported.
        part["file"] = files.get(index, "")
    # The extra pages as really worked out for this PDF (see export_parts).
    data["extra"] = [{**extra.model_dump(mode="json"), "file": file} for extra, file in extras]
    return data


def _cell(value: object) -> str:
    """
    One value as text, for a CSV cell or an XML element. None becomes "",
    True/False "true"/"false" (as in JSON) and a list one text joined with
    " | ", so the CSV and the XML read the same way.
    """
    if value is None:
        return ""
    if isinstance(value, bool):  # isinstance = PHP's instanceof / is_bool
        return "true" if value else "false"
    if isinstance(value, list):
        return CSV_LIST_SEPARATOR.join(str(item) for item in value)
    return str(value)


def _write_json(data: dict, path: Path) -> None:
    """The export data as JSON: the whole result plus the file names."""
    # ensure_ascii=False keeps letters like "š" as they are (not "š").
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(data: dict, path: Path) -> None:
    """
    One row per part, then one row per group of extra pages, with the
    columns of CSV_COLUMNS. An extra-pages row has its reason as the kind
    ("cover", "blank", "unassigned"), the label as the name ("Cover") and
    "export" = whether its PDF was written.
    """
    credits = {field: data[field] for field in CREDIT_FIELDS}
    rows = [{**part, **credits} for part in data["parts"]]
    for extra in data["extra"]:
        rows.append(
            {
                "name": EXTRA_LABELS.get(extra["reason"], extra["reason"]),
                "kind": extra["reason"],
                "pages": extra["pages"],
                "export": bool(extra["file"]),
                "file": extra["file"],
                **credits,
            }
        )
    # newline="" is what Python's csv module asks for: it writes the line
    # endings itself. "utf-8-sig" adds a small marker at the start that makes
    # Excel read accented letters ("Kazalište") right; other programs ignore it.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            # .get(column) = None (-> "") for a column the row doesn't have.
            writer.writerow({column: _cell(row.get(column)) for column in CSV_COLUMNS})


# The characters XML 1.0 doesn't allow anywhere, not even escaped: the
# control characters except tab (\x09), newline (\x0a) and \x0d; the halves
# of a broken letter (\ud800-\udfff); and ￾, ￿.
XML_FORBIDDEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")


def _xml_safe(text: str) -> str:
    """
    `text` without the characters XML can't hold:

        "Flute\\x0c 1" -> "Flute 1"

    Text read from a PDF is already clean (pdf_reader.tidy_text), but an
    edits file, a note or a warning may still hold one. Without this the
    XML file is written, but no program can read it.
    """
    return XML_FORBIDDEN.sub("", text)


def _write_xml(data: dict, path: Path) -> None:
    """
    The export data as XML:

        <scoresplit schema_version="3">
          <source>paradiso.pdf</source> ... <title>, <composer>, <arranger>
          <warnings><warning>...</warning></warnings>
          <parts>
            <part><name>Flute 1</name>...<notes><note>...</note></notes>
                  <file>02_...pdf</file></part>
          </parts>
          <extra>
            <pages reason="cover" file="extra/cover_pages_1.pdf">1</pages>
          </extra>
        </scoresplit>

    ElementTree escapes "&" and "<" in the texts by itself. Characters XML
    can't hold at all are removed (see _xml_safe): every text and attribute
    goes through it.
    """
    root = ET.Element("scoresplit", schema_version=str(data["schema_version"]))
    for field in CREDIT_FIELDS:
        if field == "schema_version":
            continue  # already an attribute of <scoresplit>
        if field == "warnings":
            warnings_element = ET.SubElement(root, "warnings")
            for warning in data["warnings"]:
                ET.SubElement(warnings_element, "warning").text = _xml_safe(warning)
        else:
            ET.SubElement(root, field).text = _xml_safe(_cell(data[field]))

    parts_element = ET.SubElement(root, "parts")
    for part in data["parts"]:
        part_element = ET.SubElement(parts_element, "part")
        for field, value in part.items():
            if field == "notes":
                # One <note> per note, instead of one joined text.
                notes_element = ET.SubElement(part_element, "notes")
                for note in value:
                    ET.SubElement(notes_element, "note").text = _xml_safe(note)
            else:
                ET.SubElement(part_element, field).text = _xml_safe(_cell(value))

    extra_element = ET.SubElement(root, "extra")
    for extra in data["extra"]:
        pages_element = ET.SubElement(
            extra_element, "pages", reason=_xml_safe(extra["reason"]), file=_xml_safe(extra["file"])
        )
        pages_element.text = _xml_safe(extra["pages"])

    tree = ET.ElementTree(root)
    ET.indent(tree)  # one element per line, indented: readable by a person
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_files(target: Path) -> list[Path]:
    """The three export files of a main output folder: .json, .csv, .xml."""
    # target.name is the main folder's own name ("Song - Arranger").
    return [target / EXPORT_FOLDER / f"{target.name}{suffix}" for suffix in (".json", ".csv", ".xml")]


# ---------------------------------------------------------------------------
# Writing everything
# ---------------------------------------------------------------------------


def output_folder(result: Result, out_dir: Path) -> Path:
    """The main output folder inside `out_dir`: out_dir / "<Title> - <Arranger>"."""
    return out_dir / safe_filename(result.folder_name())


def folder_in_use(folder: Path) -> bool:
    """
    True when `folder` exists and isn't empty: export_parts then refuses to
    write into it (unless force), so the files of an earlier split are never
    mixed with new ones by accident.
    """
    # any(folder.iterdir()) = "the folder has at least one entry".
    return folder.is_dir() and any(folder.iterdir())


def export_parts(
    result: Result,
    pdf_path: Path,
    out_dir: Path,
    force: bool = False,
    include_extra: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[Path, list[ExportedFile]]:
    """
    Write the PDFs described by `result`, and the export files.

    Args:
        result:        what to export (parts with export=True, plus the
                       pages that are in no part).
        pdf_path:      the original PDF.
        out_dir:       where the main "<Title> - <Arranger>" folder is created.
        force:         allow writing into an existing, non-empty folder.
        include_extra: also write cover/blank/unassigned pages to extra/.
        progress:      optional function called as progress(files_done,
                       files_total): once with 0 before the first file, then
                       after every file written.

    Returns:
        (the main output folder, one ExportedFile per file written: the
        parts first, in the result's order, then the extra pages, then the
        JSON, CSV and XML export files, then the copy of the original PDF)

    Raises:
        InvalidResultError: the result has errors (see check_export).
        OutputExistsError:  the output folder isn't empty and force is off.
    """
    target = output_folder(result, out_dir)

    source = pdfium.PdfDocument(str(pdf_path))
    written: list[ExportedFile] = []
    try:
        # Check the result again here, so that nothing is written for a bad
        # result even if the caller forgot to check it first.
        errors = check_export(result, len(source)).errors
        if errors:
            raise InvalidResultError("; ".join(errors))

        if folder_in_use(target) and not force:
            raise OutputExistsError(str(target))

        # Decide every file first, so the total is known before writing:
        # a progress bar needs "of how many" from the start.
        # files = duplicates and parts the user switched off are skipped;
        # extras = every page in no part, worked out afresh from the parts
        # (see Result.extra_groups), for the pages this PDF really has.
        files = part_filenames(result)  # {position: file name}
        # all_extras is listed in the export files even when the extra pages
        # aren't written (include_extra off); then their "file" is "".
        all_extras = result.extra_groups(len(source))
        extras = all_extras if include_extra else []
        exports = export_files(target)
        # The original PDF, the whole score, is copied as it is into export/.
        original_copy = target / EXPORT_FOLDER / pdf_path.name
        total = len(files) + len(extras) + len(exports) + 1  # + 1 = the copy
        if progress:
            progress(0, total)

        def done(exported: ExportedFile) -> None:
            """Remember a written file and report the progress."""
            written.append(exported)
            if progress:
                progress(len(written), total)

        # .items() gives (position, file name) pairs, in the result's order.
        for index, filename in files.items():
            file_path = target / filename
            _write_pages(source, result.parts[index].page_list(), file_path)
            done(ExportedFile(path=file_path, part=index))

        for extra in extras:
            file_path = target / EXTRA_FOLDER / extra_filename(extra)
            _write_pages(source, extra.page_list(), file_path)
            done(ExportedFile(path=file_path))

        # One set of data for all three export files (see _export_data).
        extra_files = [
            (extra, f"{EXTRA_FOLDER}/{extra_filename(extra)}" if include_extra else "")
            for extra in all_extras
        ]
        data = _export_data(result, files, extra_files)
        json_path, csv_path, xml_path = exports
        json_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(data, json_path)
        done(ExportedFile(path=json_path))
        _write_csv(data, csv_path)
        done(ExportedFile(path=csv_path))
        _write_xml(data, xml_path)
        done(ExportedFile(path=xml_path))
        # copyfile copies the bytes exactly: the same PDF, nothing re-saved.
        shutil.copyfile(pdf_path, original_copy)
        done(ExportedFile(path=original_copy))
    finally:
        # "finally" runs whether or not an error happened: always close the PDF.
        source.close()

    return target, written
