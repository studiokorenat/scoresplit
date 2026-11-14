"""
models.py: the data structures that move through the whole program.

Think of this file as the "Models / DTOs" folder of a Laravel app. It holds no
logic, only the *shape* of the data. Every other module imports from here.

Two kinds of classes are used:

1. ``@dataclass`` classes (from Python's standard library) for INTERNAL data
   that never leaves the program: text pieces found on a page, the result of
   analysing one page, and so on. A dataclass is a class where Python writes
   ``__init__`` for you from the listed fields, like a PHP 8 class that only
   uses constructor property promotion.

2. ``pydantic.BaseModel`` classes for the EXTERNAL contract: the RESULT
   (``Result``, with its ``Part`` and ``Extra`` lists). It is the one shape
   used everywhere: what ``inspect`` returns, the JSON that the editor
   saves and that a human or a GUI edits, and what ``scoresplit split
   --edits`` reads back. Pydantic validates the
   data when it is loaded (wrong types, missing fields and so on raise a
   clear error), much like a Laravel FormRequest with rules.

Pipeline overview (which module produces which model):

    pdf_reader / ocr  ->  TextSpan        (pieces of text with a position)
    page_analyzer     ->  PageFacts       (raw facts about one page, cached)
    classify          ->  PageInfo        (what kind of page it is, which part)
    segment           ->  Part, Extra     (the final list of parts + pages)
    pipeline          ->  Result          (parts + title, composer, arranger)
    exporter          ->  PDF files on disk
"""

# "from __future__ import annotations" makes Python treat type hints as plain
# text, so a class can mention a class defined further down the file.
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# A progress callback: a function that receives (done, total) and returns
# nothing, e.g. progress(12, 238) = "12 of 238 pages done". Used by the page
# analyser (pages) and the exporter (files). It lives here, with the other
# shapes shared by several modules, so exporter.py can use it without
# importing the (heavy) page analyser.
ProgressCallback = Callable[[int, int], None]

# ---------------------------------------------------------------------------
# Internal models (dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class TextSpan:
    """
    One piece of text found on a page, together with where it is.

    It comes either from the PDF's embedded text (Sibelius, Finale, MuseScore
    exports) or from OCR on a scanned page. Both paths produce this same class,
    so the rest of the program doesn't care where the text came from.

    Coordinates are *normalised*: 0.0 is the left/top edge of the page and 1.0
    is the right/bottom edge. That way "the top 15% of the page" means
    ``y0 < 0.15`` for every page size and every render resolution.

    Example:
        TextSpan(text="Flute 1", x0=0.07, y0=0.05, x1=0.15, y1=0.07, size=12.0)
    """

    text: str  # the text itself, e.g. "Clarinet in Bb 2"
    x0: float  # left edge   (0.0 = left side of the page)
    y0: float  # top edge    (0.0 = top of the page)
    x1: float  # right edge
    y1: float  # bottom edge
    size: float  # approximate letter height in PDF points (1 pt = 1/72 inch)


@dataclass
class PageFacts:
    """
    Everything we *measured* on one page, before any interpretation.

    These facts are expensive to get (OCR can take a second per page), so they
    are saved to the cache (see ``cache.py``). The interpretation step
    (``classify.py``) is cheap and is re-run every time. That means improving
    the instrument dictionary never forces a new OCR run.
    """

    page_number: int  # 1-based, like a human counts pages
    source: Literal["text", "ocr"]  # where the text came from
    header: list[TextSpan] = field(default_factory=list)  # text in the top strip
    margin: list[TextSpan] = field(default_factory=list)  # text in the left margin
    body: list[TextSpan] = field(default_factory=list)  # whole-page text (only when needed)
    ink_ratio: float = 0.0  # share of dark pixels; ~0 means a blank page
    staff_count: int = 0  # number of 5-line staves detected
    clef: str | None = None  # "treble", "bass" or None when unsure
    first_staff_y: float | None = None  # top of the first staff (0..1), None = no staves
    # Staves found with the page turned 90 degrees. Only measured when no
    # staves are found the normal way: a page printed sideways (Tullamore
    # page 100) has music, but its staff lines run from top to bottom.
    sideways_staff_count: int = 0

    # Note on ``field(default_factory=list)``: in Python you must NOT write
    # ``header: list = []`` in a dataclass, because all objects would share
    # the SAME list. ``default_factory=list`` creates a fresh empty list for
    # each new object.


@dataclass
class PartName:
    """
    A parsed instrument part name, split into its components.

    "Clarinet in Bb 2" (Sibelius style) and "2. klarinet in B" (Croatian
    style) both become:
        PartName(instrument_id="clarinet", instrument_name="Clarinet",
                 number="2", key="Bb")
    and ``display()`` gives the same canonical name: "Clarinet 2 in Bb".
    The canonical name is used to compare parts; the name a person sees is
    ``wording``, the name as printed ("2. klarinet in B").
    """

    instrument_id: str  # stable id from instruments.yaml, e.g. "clarinet"
    instrument_name: str  # display name from instruments.yaml, e.g. "Clarinet"
    family: str = "other"  # woodwind / brass / percussion / ...
    number: str | None = None  # "1", "2", "1&2", "Solo" ... or None
    key: str | None = None  # transposition: "Bb", "Eb", "F", "C" ... or None
    # Clef written in the name ("Baritone (T.C.)" -> "treble"), or None. It is
    # not part of display() nor of `wording`: the clef has its own column in
    # the table.
    clef: str | None = None
    label: str | None = None  # full custom label (e.g. multi-percussion parts)
    score: float = 0.0  # how sure the fuzzy match was, 0..100
    abbreviation: bool = False  # True if matched via an abbreviation ("Picc.")
    # A key read that this instrument is never written in ("Bass Clarinet
    # in Eb"), moved out of `key` because it is probably misread (see
    # classify._set_aside_unlisted_key). segment.py then uses the default
    # key and adds a note.
    unlisted_key: str | None = None
    # The name as printed on the page, cleaned up: only the words of the
    # name (no title or credits around it), OCR noise fixed ("B:" -> "Bb",
    # "Hom" -> "Horn"), ALL CAPITALS calmed ("FLUTE 1" -> "Flute 1").
    # None when the words were too garbled to show ("mTRUMPET" only looked
    # like "Trumpet"): segment.py then falls back to the canonical name.
    wording: str | None = None

    def display(self) -> str:
        """
        The canonical name: "Clarinet 2 in Bb", whatever the page printed.
        Two pages with the same canonical name name the same part.
        """
        # A custom label (e.g. "Shaker, Tambourine") wins over the generated name.
        if self.label:
            return self.label
        # Build the name piece by piece and join with spaces at the end.
        pieces = [self.instrument_name]
        if self.number:
            pieces.append(self.number)
        if self.key:
            pieces.append(f"in {self.key}")  # f"..." is like "in {$key}" in PHP
        return " ".join(pieces)


@dataclass
class PageInfo:
    """
    The interpretation of one page: what kind of page is it, and which part?

    Produced by ``classify.py`` from a ``PageFacts`` object.
    """

    page_number: int
    kind: Literal["blank", "score", "part", "unknown"]
    part: PartName | None = None  # only set when kind == "part"
    is_start: bool = False  # True = first page of a part (title/credits on it)
    clef: str | None = None
    has_staves: bool = False
    # True = music printed sideways (staves only found with the page turned
    # 90 degrees): it can't be read, so it becomes a part of its own.
    sideways: bool = False
    confidence: float = 0.0  # 0..1, how sure we are about this page
    note: str = ""  # free text explaining the decision (for debugging)


# ---------------------------------------------------------------------------
# External contract (pydantic): the result
# ---------------------------------------------------------------------------

# The version of the export's shape: the result's JSON and the CSV/XML
# written from it. Bump it whenever the shape changes in a way older readers
# can't handle. Version 3 was the first simple result below; version 4 is
# the same JSON with a new CSV layout (row_type and reason columns, only
# title/composer/arranger as credits). Versions 1 and 2 (the old "page
# map") can't be read any more.
SCHEMA_VERSION = 4

# The highest page number a result may mention. No real score has this many
# pages; the limit stops an edited file like "1-999999999999" from creating a
# trillion page numbers in memory (which would crash the program). Pages
# beyond the PDF's real page count are reported by check_result.
MAX_PAGE_NUMBER = 100_000

# MAX_PAGE_NUMBER limits ONE page number, not all of them together. Parts may
# share pages, so 300 parts that each say "1-100000" (an 11 KB edits file)
# made 30 million page numbers: 25 seconds and 1.2 GB of memory. Two more
# limits keep any edits file cheap, also one sent to the web tool by anyone:
#
#   MAX_PARTS: the number of parts. The biggest real score in tests/golden
#   has 91 parts.
#
#   MAX_LISTED_PAGES: how many pages the parts list together (a page in two
#   parts counts twice), and the same for the extra groups. A real result
#   lists each page about once, so twice the biggest PDF allowed leaves room
#   for parts that share pages. The pages are counted from the text BEFORE
#   any list of pages is built (see count_listed_pages), so even a file far
#   above the limit is refused in milliseconds.
MAX_PARTS = 1000
MAX_LISTED_PAGES = 2 * MAX_PAGE_NUMBER

# The clefs a part can have. "percussion" is the neutral clef of drum parts.
Clef = Literal["treble", "bass", "alto", "tenor", "percussion"]

# Why a page is in no part: the cover (front matter before the music), a
# blank page, or a page detection couldn't place ("unassigned"). A page that
# an edit took out of every part is "unassigned" too.
ExtraReason = Literal["cover", "blank", "unassigned"]


# Invisible control characters: \x00-\x1f (tab, newline, escape ...), \x7f
# (delete) and \x80-\x9f (the "C1" ones: "\x9b" works like "\x1b[" in some
# terminals). One line of text never needs them, and they do harm: XML can't
# hold most of them, and "\x1b[31m" in a part name would change the
# terminal when the table is printed. Text read from a PDF goes through
# pdf_reader.tidy_text; text from an edits file through _clean_text below.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def without_control_characters(text: str) -> str:
    """
    `text` with every control character turned into a space, and the spaces
    tidied (so "Flute\\x031" gives "Flute 1", never "Flute1"):

        "A.\\x00Person" -> "A. Person"
        "Song\\x07"     -> "Song"

    Text without a control character comes back exactly as it was, so a
    name a person typed with two spaces keeps them.
    """
    if not CONTROL_CHARACTERS.search(text):
        return text
    # split() + join: trims the ends and turns every run of spaces into one.
    return " ".join(CONTROL_CHARACTERS.sub(" ", text).split())


def _clean_text(value: str | list[str] | None) -> str | list[str] | None:
    """without_control_characters for one text, a list of texts, or None."""
    if isinstance(value, str):
        return without_control_characters(value)
    if isinstance(value, list):
        return [without_control_characters(text) for text in value]
    return value  # None stays None


def _check_page_number(page: int) -> int:
    """Raise ValueError if `page` is above MAX_PAGE_NUMBER, else return it."""
    if page > MAX_PAGE_NUMBER:
        raise ValueError(f"page {page} is too high (the limit is {MAX_PAGE_NUMBER})")
    return page


def _page_runs(value: str | list[int]) -> list[tuple[int, int]]:
    """
    Read a page text into its runs of pages, as (first, last) pairs, WITHOUT
    building the list of every page in between:

        _page_runs("3-5,9")   -> [(3, 5), (9, 9)]
        _page_runs([9, 3])    -> [(9, 9), (3, 3)]
        _page_runs("")        -> []

    The runs are in the order written and may overlap ("1-5,3-7").

    Raises ValueError for text that isn't a page range ("3-", "a", "5-3",
    a page above MAX_PAGE_NUMBER), and TypeError for a value that is no
    text or list at all (7, None from an edited JSON file).
    """
    # isinstance(x, list) is like PHP's is_array($x).
    if isinstance(value, list):
        # A pair (page, page) for each page of the list: a run of one page.
        return [(page, page) for page in (_check_page_number(int(p)) for p in value)]
    if not isinstance(value, str):
        # Without this check, value.split() below would crash with an
        # AttributeError, which pydantic doesn't turn into a clear error.
        raise TypeError(f"pages must be text like '3-5,9', not {type(value).__name__}")

    runs: list[tuple[int, int]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue  # skip empty pieces like in "3-5,,9"
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"page range '{chunk}' goes backwards")
            # Check the end here, BEFORE anyone builds the range: that's what
            # keeps a huge range like "1-999999999999" from filling the memory.
            runs.append((start, _check_page_number(end)))
        else:
            page = _check_page_number(int(chunk))
            runs.append((page, page))
    return runs


def parse_page_ranges(value: str | list[int]) -> list[int]:
    """
    Turn a human page-range string into a sorted list of page numbers.

    The result stores pages as a string like "3-47" or "159-162,165", because
    that's easy for a human to read and edit. A plain JSON list [1, 2, 3] is
    also accepted, which is handy for code (e.g. a GUI).

    Examples:
        parse_page_ranges("3-5,9")   -> [3, 4, 5, 9]
        parse_page_ranges([9, 3])    -> [3, 9]
        parse_page_ranges("")        -> []

    Raises ValueError for text that isn't a page range ("3-", "a", "5-3"),
    TypeError for a value that is no text or list (see _page_runs).
    """
    pages: set[int] = set()  # a set ignores duplicates automatically
    for start, end in _page_runs(value):
        # range(a, b) stops BEFORE b, so +1 to include the last page.
        pages.update(range(start, end + 1))
    return sorted(pages)


def count_listed_pages(value: object) -> int:
    """
    How many pages a page text lists, counted WITHOUT building the list, so
    "1-100000" costs one subtraction instead of 100 000 numbers:

        count_listed_pages("3-5,9")    -> 4
        count_listed_pages([9, 3])     -> 2
        count_listed_pages("1-5,3-7")  -> 10  (pages 3-5 are written twice)

    Anything that isn't a valid page text counts 0 ("5-3", "a", None): it
    is reported elsewhere (by check_result, or by pydantic for a value that
    isn't text). Used to refuse an absurdly big result before it is built
    (see MAX_LISTED_PAGES).
    """
    try:
        # end - start + 1: the run 3-5 holds 3 pages (3, 4 and 5).
        return sum(end - start + 1 for start, end in _page_runs(value))
    except (ValueError, TypeError):
        return 0


def format_page_ranges(pages: list[int]) -> str:
    """
    The opposite of ``parse_page_ranges``: [3, 4, 5, 9] -> "3-5,9".
    """
    if not pages:
        return ""
    pages = sorted(pages)
    ranges: list[str] = []
    start = previous = pages[0]
    # pages[1:] means "every element except the first" (a list slice).
    for page in pages[1:]:
        if page == previous + 1:
            previous = page  # still consecutive, extend the current range
            continue
        # The run ended: store it and start a new one.
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _tidy_pages(value: str | list[int]) -> str:
    """
    Write pages the standard way: [5, 6, 7] -> "5-7", "7,5-6" -> "5-7".

    Text that isn't a valid page range is kept exactly as it is, so the
    result can still be loaded and check_result can report the problem
    together with all the others (instead of the first typo stopping
    everything with one error).
    """
    try:
        return format_page_ranges(parse_page_ranges(value))
    except (ValueError, TypeError):
        # A list with something odd in it becomes "a,b,c" text.
        if isinstance(value, list):
            return ",".join(str(page) for page in value)
        return value


class Part(BaseModel):
    """
    One part of the result: a conductor score or one instrument part.

    Every field can be edited by a person (in the saved edits) or a GUI.
    """

    # The name as printed in the PDF, cleaned up ("2. klarinet in Es",
    # "Clarinet in Bb 1"), or the recognised name ("Trumpet 1") when the
    # print was too garbled. Two parts may have the same name (one in
    # treble clef, one in bass clef): their pages tell them apart.
    name: str
    instrument: str | None = None  # instruments.yaml id, e.g. "clarinet"; None = not recognised
    number: str | None = None  # "1", "2", "1&2", "Solo" ... or None
    key: str | None = None  # "Bb", "Eb", "F", "C" ... or None (score, drums, piano, unknown)
    # True when no key was printed and the instrument's usual key was filled
    # in ("Clarinet 1" -> Bb), so a person knows it is a guess they can change.
    key_assumed: bool = False
    clef: Clef | None = None  # None = more than one clef, or not detected
    # True when no clef was printed or detected and the instrument's one clef
    # was filled in (Trumpet -> treble), like key_assumed.
    clef_assumed: bool = False
    kind: Literal["score", "part"] = "part"
    pages: str  # human page ranges, e.g. "171-174"
    export: bool = True  # false = not written when splitting (e.g. a duplicate)
    duplicate_of: str | None = None  # name of the part this is a copy of
    confidence: float = 1.0  # 0..1; low values deserve a human look
    notes: list[str] = Field(default_factory=list)  # warnings shown to the user

    # A "validator" runs automatically when the model is created. This one
    # writes the pages the standard way (see _tidy_pages). mode="before" runs
    # it before pydantic checks the type, so a list of page numbers
    # ([5, 6, 7]) is accepted too and turned into the string "5-7".
    @field_validator("pages", mode="before")
    @classmethod
    def _normalise_pages(cls, value: str | list[int]) -> str:
        return _tidy_pages(value)

    # Every text a person can edit, cleaned of control characters when the
    # model is made (e.g. when an edits file is loaded): see _clean_text.
    # mode="after" runs once pydantic has checked the type is right.
    @field_validator("name", "instrument", "number", "key", "duplicate_of", "notes", mode="after")
    @classmethod
    def _remove_control_characters(cls, value: str | list[str] | None) -> str | list[str] | None:
        return _clean_text(value)

    def page_list(self) -> list[int]:
        """
        The pages of this part as a list of ints, e.g. [171, 172, 173, 174].
        Raises ValueError when the page text is invalid (see check_result).
        """
        return parse_page_ranges(self.pages)


class Extra(BaseModel):
    """Pages that belong to no part, and why: the cover, blank pages, ..."""

    reason: ExtraReason
    pages: str

    # See Part._normalise_pages.
    @field_validator("pages", mode="before")
    @classmethod
    def _normalise_pages(cls, value: str | list[int]) -> str:
        return _tidy_pages(value)

    def page_list(self) -> list[int]:
        return parse_page_ranges(self.pages)


class Result(BaseModel):
    """
    The complete result: the PDF's title and credits, its parts and pages.

    This ONE shape is used everywhere: in memory, in the saved edits that a
    person or the web GUI changes, in the JSON export and in
    ``inspect --json``. Keep it stable, and bump SCHEMA_VERSION when you
    change it.

    The ``extra`` list is WORKED OUT, not kept by hand: every page that is in
    no part is an extra page. Its reason (cover, blank) comes from detection;
    a page without one is "unassigned". So when an edit deletes a part, its
    pages become extra pages by themselves. See ``extra_groups``.
    """

    schema_version: int = SCHEMA_VERSION
    source: str  # file name of the analysed PDF
    sha256: str  # fingerprint of the PDF: saved edits only apply to the same PDF
    # Limited like the page numbers: an edits file can come from anyone, and
    # working out the extra pages walks over every page from 1 to page_count,
    # so "page_count": 1000000000000 would hang the program. Field(ge=0,
    # le=...) means "0 <= page_count <= MAX_PAGE_NUMBER" (ge = greater or
    # equal, le = less or equal). Pydantic checks fields BEFORE the
    # mode="after" validator below runs, so a bad value never gets there.
    page_count: int = Field(ge=0, le=MAX_PAGE_NUMBER)
    ocr_pages: int = 0  # how many pages were read with OCR (0 = a text PDF)
    title: str | None = None
    composer: str | None = None
    arranger: str | None = None
    # max_length=MAX_PARTS: at most that many parts. Pydantic stops reading
    # the list as soon as it is too long, so a file with a million parts is
    # refused without reading them all.
    parts: list[Part] = Field(default_factory=list, max_length=MAX_PARTS)
    extra: list[Extra] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # about the whole PDF

    # mode="before" runs on the raw data, BEFORE pydantic builds a Part or
    # Extra from each item (building one reads its whole page list). So a
    # file that lists millions of pages is refused while it is still cheap.
    @field_validator("parts", "extra", mode="before")
    @classmethod
    def _limit_listed_pages(cls, items: object, info: ValidationInfo) -> object:
        """Refuse a list of parts (or extra groups) above MAX_LISTED_PAGES."""
        listed = _total_listed_pages(items)
        if listed > MAX_LISTED_PAGES:
            # info.field_name is "parts" or "extra": the field being checked.
            raise ValueError(
                f"the {info.field_name} list {listed} pages together, more than "
                f"{MAX_LISTED_PAGES} (a page in two parts counts twice)"
            )
        return items

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        """Refuse a file of another version, with a sentence that says why."""
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {value} is not supported (this scoresplit reads "
                f"version {SCHEMA_VERSION}): run 'scoresplit inspect' again"
            )
        return value

    # See Part._remove_control_characters.
    @field_validator("source", "title", "composer", "arranger", "warnings", mode="after")
    @classmethod
    def _remove_control_characters(cls, value: str | list[str] | None) -> str | list[str] | None:
        return _clean_text(value)

    # mode="after" runs once the whole model is built: the moment to bring
    # the worked-out extra pages up to date with the (maybe edited) parts.
    @model_validator(mode="after")
    def _work_out_extra(self) -> Result:
        self.refresh_extra()
        return self

    def extra_groups(self, page_count: int | None = None) -> list[Extra]:
        """
        Every page that is in no part, grouped by reason, in page order:

            parts on pages 2-9, detection said page 1 is the cover, 10 blank
            -> [Extra(reason="cover", pages="1"), Extra(reason="blank", pages="10")]

        The reasons are taken from the current ``extra`` list (what detection
        said about each page); a page it doesn't mention is "unassigned". A
        page that is in a part again is no extra page any more.

        ``page_count`` defaults to the result's own; split passes the real
        number of pages of the PDF it cuts.
        """
        if page_count is None:
            page_count = self.page_count

        # Which reason detection gave each page: {1: "cover", 10: "blank"}.
        reasons: dict[int, str] = {}
        for extra in self.extra:
            for page in _pages_or_nothing(extra.pages):
                reasons[page] = extra.reason

        # Every page that some part uses (also parts that aren't exported:
        # a duplicate's pages are shown in the table, not in extra/).
        in_a_part: set[int] = set()
        for part in self.parts:
            in_a_part.update(_pages_or_nothing(part.pages))

        # Group the left-over pages by reason. A dict keeps the order in
        # which keys are added, so the groups come in order of their first page.
        by_reason: dict[str, list[int]] = {}
        for page in range(1, page_count + 1):
            if page not in in_a_part:
                reason = reasons.get(page, "unassigned")
                by_reason.setdefault(reason, []).append(page)
        return [Extra(reason=reason, pages=pages) for reason, pages in by_reason.items()]

    def refresh_extra(self) -> None:
        """Bring ``extra`` up to date after the parts changed (see extra_groups)."""
        self.extra = self.extra_groups()

    def folder_name(self) -> str:
        """
        The name of the main output folder: "<Title> - <Arranger>".

        Original compositions have no arranger: then the composer is used
        ("Ross Roy - Jacob de Haan"). With neither, just the title.
        """
        title = self.title or "Untitled"
        person = self.arranger or self.composer
        if person:
            return f"{title} - {person}"
        return title


def _total_listed_pages(items: object) -> int:
    """
    How many pages a list of parts (or of extra groups) lists together, from
    their page texts (see count_listed_pages). The items are dicts when they
    come from a JSON file, Part or Extra objects when code made them. Anything
    else counts 0: pydantic reports a value of the wrong type itself.
    """
    if not isinstance(items, list):
        return 0
    total = 0
    for item in items:
        # dict.get("pages") is like $item['pages'] ?? null in PHP;
        # getattr(item, "pages", None) is the same for an object's field.
        pages = item.get("pages") if isinstance(item, dict) else getattr(item, "pages", None)
        total += count_listed_pages(pages)
    return total


def _pages_or_nothing(pages: str) -> list[int]:
    """parse_page_ranges, but an invalid page text gives [] (check_result reports it)."""
    try:
        return parse_page_ranges(pages)
    except ValueError:
        return []


# ---------------------------------------------------------------------------
# Checking a result before it is split
# ---------------------------------------------------------------------------


@dataclass
class CheckReport:
    """
    What check_result found. ``errors`` must be fixed before splitting;
    ``warnings`` are worth a look, but splitting works.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_result(result: Result, page_count: int | None = None) -> CheckReport:
    """
    Check a result (usually an edited one) and return EVERY problem at once,
    so a person or a GUI can fix them all in one go. Errors:

      * a page text that isn't a page range ("5-3", "a"),
      * a page the PDF doesn't have,
      * an exported part without pages.

    Warning:

      * a page in two exported parts (allowed: that page goes into both
        files). Only pages the PDF has: a page past the end is an error
        already, and is not stored again as a shared page.

    ``page_count`` is the number of pages of the PDF that will be cut; it
    defaults to the result's own.
    """
    if page_count is None:
        page_count = result.page_count
    report = CheckReport()

    # Which exported part each page went to first, as its position in
    # result.parts: {12: 3, ...}. Positions, not names, because two parts may
    # have the same name (one in treble clef, one in bass clef).
    first_owner: dict[int, int] = {}
    # Pages shared by two parts: {(3, 4): [12, 13]}.
    shared: dict[tuple[int, int], list[int]] = {}

    # enumerate() gives (position, part) pairs: (0, first part), (1, ...), ...
    for index, part in enumerate(result.parts):
        try:
            pages = parse_page_ranges(part.pages)
        except ValueError as error:
            report.errors.append(f"'{part.name}': pages '{part.pages}' can't be read ({error})")
            continue

        missing = [page for page in pages if page < 1 or page > page_count]
        if missing:
            report.errors.append(
                f"'{part.name}': page(s) {format_page_ranges(missing)} don't exist "
                f"(PDF has {page_count})"
            )
        if not part.export:
            continue  # a part that isn't written can't clash with another
        if not pages:
            report.errors.append(f"'{part.name}': has no pages (give it pages, or set \"export\": false)")
        for page in pages:
            # Only pages the PDF has. The others are reported as missing
            # above; storing them here too, one by one, is what let parts
            # that all said "1-100000" fill the memory on a 10-page PDF.
            if page < 1 or page > page_count:
                continue
            owner = first_owner.setdefault(page, index)
            if owner != index:
                shared.setdefault((owner, index), []).append(page)

    for (first, second), pages in shared.items():
        first_name, second_name = result.parts[first].name, result.parts[second].name
        report.warnings.append(
            f"'{first_name}' and '{second_name}' share page(s) {format_page_ranges(pages)}"
        )
    return report
