"""
page_analyzer.py: collect the raw facts of every page (text, ink, staves, clef).

Pipeline position:  pdf_reader + ocr + image_analysis -> [page_analyzer] -> classify

For each page this module decides HOW to read it:

  * The page has embedded text (Sibelius/Finale/... export)?
        -> take the text straight from the PDF (fast and exact).
    A scan WITH a text layer (the scanner's own, often garbled OCR) also
    gets its header OCR'd by us; the better reading of each line wins
    (read_header_again, merge_header_readings).
  * The page is a scan (an image, no text)?
        -> OCR the header strip and the left margin with Tesseract.
    A scan whose only text is a stamp on every page (an archive's name in
    the text layer, american.pdf) counts as a scan too (find_stamp_texts).

It also renders every page once to measure ink (blank page?), find the
staves and detect the clef.

Because OCR is slow, the pages are processed IN PARALLEL on half the CPU
cores, at low priority, with a ProcessPoolExecutor, and the results are
cached (see cache.py).

ProcessPoolExecutor in two sentences: it starts N copies ("workers") of
this Python program and hands each one pages to analyse. The results come
back to the main program. That's similar to dispatching jobs to N Laravel
queue workers and collecting their results.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium

from . import cache
from .classify import has_credits, name_lines
from .image_analysis import (
    BLANK_INK_RATIO,
    deskew,
    detect_clef,
    find_staves_on_page,
    find_text_boxes,
    find_text_lines,
    ink_ratio,
    music_fingerprint,
)
from .instruments import contains_score_keyword, find_part_name
# ProgressCallback (pages_done, pages_total) is defined in models.py; it is
# imported here so `page_analyzer.ProgressCallback` keeps working too.
from .models import PageFacts, PartName, ProgressCallback, TextSpan
from .ocr import OcrUnavailableError, ocr_image, ocr_region, tessdata_folder, tesseract_available
from .pdf_reader import (
    MIN_LETTERS_FOR_TEXT_PAGE,
    count_letters,
    extract_text_spans,
    fit_dpi,
    largest_image_share,
    open_pdf,
    render_page_gray,
    render_region_gray,
)

# ---------------------------------------------------------------------------
# Tuning constants. Regions are (x0, y0, x1, y1) as shares of the page.
# ---------------------------------------------------------------------------

# The header: the top strip with the part name, title and composer/arranger.
# Its height ADAPTS to the page: it reaches down to the first staff (the
# title block of a first page can be tall, e.g. a big logo), but is never
# smaller than HEADER_MIN_BOTTOM or larger than HEADER_MAX_BOTTOM (shares of
# the page height).
HEADER_MIN_BOTTOM = 0.15
HEADER_MAX_BOTTOM = 0.32

# The left margin, where a conductor score lists its instruments.
MARGIN_REGION = (0.0, 0.08, 0.22, 0.98)

# Resolution for OCR and pixel analysis. 200 dpi is a good balance: higher is
# slower and not more accurate for printed music headers.
DEFAULT_DPI = 200

# When the first OCR try finds no part name, the header is read again at
# this higher resolution with a different Tesseract mode.
RETRY_DPI = 300

# A margin with at least this many recognised instrument names is probably
# a conductor score; then we don't need to OCR the whole page.
SCORE_MARGIN_MIN_INSTRUMENTS = 3

# A "stamp": text that sits on (almost) every page of a scanned PDF, like the
# archive stamp "ARCHIVOCUCOSAX" in the text layer of american.pdf. It is the
# ONLY text those pages have; the music and the part names are pictures. A
# stamp must not make a page count as a text page (that would stop OCR), so
# its letters are not counted. A text is a stamp when it is on at least this
# share of the pages ...
STAMP_MIN_PAGE_SHARE = 0.8
# ... and the PDF has at least this many pages. In a short PDF (one part of
# two or three pages) a title on every page is normal, not a stamp.
STAMP_MIN_PAGES = 4

# A page whose biggest picture covers at least this share of it is a scan.
# When such a page also has a text layer, that text is the scanner's own OCR
# ("KIARINET", "BARTION" in superstar.pdf), so its header is read with our
# own OCR as well (see read_header_again). Born-digital pages have no
# picture (or a small logo) and are never OCR'd for this.
SCANNED_PAGE_MIN_IMAGE_SHARE = 0.5


@dataclass(frozen=True)
class AnalysisSettings:
    """
    How to analyse the pages. ``frozen=True`` makes objects read-only, which
    is a good habit for settings that are passed around.
    """

    ocr_mode: str = "auto"  # "auto" (only scans), "always", or "never"
    lang: str = "eng"  # Tesseract language string, e.g. "eng+hrv"
    dpi: int = DEFAULT_DPI
    # Can Tesseract be used? A text page's header is OCR'd as well only when
    # it can (see analyze_page), so the results differ, and the cache key
    # must tell them apart (review step 5).
    # Left out, it is asked from the system when the settings are made.
    # field(default_factory=...) calls the function for every new object
    # (a plain "= tesseract_available()" would run once, at import). The
    # lambda looks the function up at that moment, so tests can replace it.
    ocr_available: bool = field(default_factory=lambda: tesseract_available())

    def cache_key(self) -> str:
        """A short text identifying these settings, used in the cache file name."""
        key = f"{self.ocr_mode}-{self.lang}-{self.dpi}"
        # With OCR off, Tesseract is never run, so it doesn't matter.
        if self.ocr_mode == "never":
            return key
        return key + ("-ocr" if self.ocr_available else "-noocr")


# ---------------------------------------------------------------------------
# Analysing ONE page
# ---------------------------------------------------------------------------


def header_bottom(first_staff_y: float | None) -> float:
    """
    Where the header ends: just above the first staff, within the limits.

    Examples: first staff at 0.22 (a page with a big logo) -> 0.215
              first staff at 0.06 (a continuation page)    -> 0.15 (minimum)
              no staves found                              -> 0.15
    """
    if first_staff_y is None:
        return HEADER_MIN_BOTTOM
    return min(HEADER_MAX_BOTTOM, max(HEADER_MIN_BOTTOM, first_staff_y - 0.005))


def _has_part_name(header: list[TextSpan]) -> bool:
    """
    Does the header print a part name where names are printed (the name
    lines, see classify.name_lines)? A cue lower down, inside the music,
    doesn't count: classify ignores it, so it mustn't stop the second
    reading of the header that looks for the real name either.
    """
    return any(find_part_name(span.text) for span in name_lines(header))


def find_stamp_texts(pages: list[list[TextSpan]]) -> frozenset[str]:
    """
    The texts that are a stamp: printed on (almost) every page, and not
    something the detection needs.

    ``pages`` holds the embedded text spans of every page, in page order.

    Example: american.pdf has "ARCHIVOCUCOSAX" on all 84 pages and no other
    text -> frozenset({"ARCHIVOCUCOSAX"}).

    A text on every page that names a part ("Trumpet 1" on every page of a
    one-part PDF), a score ("Conductor Score") or a composer ("arr. J. Smith")
    is kept as real page text: the detection reads it.
    """
    if len(pages) < STAMP_MIN_PAGES:
        return frozenset()

    # How many pages each text is on. A Counter is a dict that counts:
    # counter["x"] += 1 works even when "x" isn't in it yet.
    pages_with_text: Counter[str] = Counter()
    for spans in pages:
        # A set: a text printed twice on one page still counts once for it.
        texts_on_page = {span.text.strip() for span in spans}
        pages_with_text.update(texts_on_page)

    stamps = set()
    for text, page_count in pages_with_text.items():
        if page_count < STAMP_MIN_PAGE_SHARE * len(pages):
            continue  # not on enough pages
        if not text:
            continue
        if find_part_name(text) or contains_score_keyword(text):
            continue  # a part or score name: the detection needs it
        if has_credits([TextSpan(text, 0, 0, 0, 0, 0)]):
            continue  # a composer or arranger: the detection needs it
        stamps.add(text)
    return frozenset(stamps)


def letters_without_stamps(spans: list[TextSpan], stamps: frozenset[str]) -> int:
    """
    How many letters the page's text has, leaving out the stamps.

    Example: page 2 of american.pdf, spans ["ARCHIVOCUCOSAX"], stamps
    {"ARCHIVOCUCOSAX"} -> 0 letters, so the page is a scan and gets OCR'd.
    """
    return count_letters([span for span in spans if span.text.strip() not in stamps])


def _count_instruments(spans: list[TextSpan]) -> int:
    """How many DIFFERENT instruments are named in these spans."""
    ids = set()
    for span in spans:
        part = find_part_name(span.text)
        if part:
            ids.add(part.instrument_id)
    return len(ids)


def _says_more(part: PartName) -> tuple[int, int]:
    """
    How much a part name says, to break a tie between two readings of the
    same line: first how many of number and key it has, then how long the
    printed name is.

    Examples: "1. Cornet Bb"         -> (2, 12)
              "Cornet Bb"            -> (1, 9)
              "Bariton Saxofoon Eb"  -> (1, 19)
              "Saxofoon Eb"          -> (1, 11)
    """
    filled = (1 if part.number else 0) + (1 if part.key else 0)
    printed = part.wording or part.label or part.instrument_name
    return filled, len(printed)


def _reads_better(new: PartName, old: PartName | None) -> bool:
    """True when the reading `new` of a line beats the reading `old`."""
    if old is None:
        return True  # the old reading named no part at all
    if new.score != old.score:
        return new.score > old.score  # the closer match to the dictionary
    return _says_more(new) > _says_more(old)  # a tie: the one that says more


def _same_line(first: TextSpan, second: TextSpan) -> bool:
    """
    True when two spans are (two readings of) the same printed line: they
    overlap side to side, and up and down by at least half the smaller one.
    """
    overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
    overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
    smaller_height = min(first.y1 - first.y0, second.y1 - second.y0)
    return overlap_x > 0 and overlap_y > 0.5 * smaller_height


def merge_header_readings(text_spans: list[TextSpan], ocr_spans: list[TextSpan]) -> list[TextSpan]:
    """
    Merge our own OCR of a scanned page's header into the scanner's text layer.

    Only the OCR lines that name a part are taken. Each one replaces the
    text-layer span on the same line when it reads better (see _reads_better);
    otherwise both stay and classify.best_header_part picks. Everything else in
    the text layer (title, composer) stays as it was: the two readings measure
    the letter size differently, so mixing their titles would let a big OCR'd
    scribble ("Webber-Hautvast %") outvote the real title.

    Example (superstar page 34):
        text layer  "BARTION SAXOFOON Eb"  -> Saxophone in Eb (score 100)
        our OCR     "BARITON SAXOFOON Eb"  -> Baritone Saxophone in Eb (100)
        a tie in score; the OCR reading says more, so it replaces the text one.
    """
    # (span, part) pairs for the OCR lines that name a part.
    ocr_names = []
    for span in ocr_spans:
        part = find_part_name(span.text)
        if part is not None:
            ocr_names.append((span, part))

    kept = []
    for span in text_spans:
        part = find_part_name(span.text)
        beaten = any(
            _same_line(span, ocr_span) and _reads_better(ocr_part, part)
            for ocr_span, ocr_part in ocr_names
        )
        if not beaten:
            kept.append(span)
    return kept + [span for span, _ in ocr_names]


def read_header_again(
    page: pdfium.PdfPage, facts: PageFacts, staves: list, settings: AnalysisSettings
) -> bool:
    """
    Should a text page's header ALSO be read with our own OCR?

    Yes when the page is a scan with a text layer (the scanner's OCR, which
    can be garbled: "KIARINET", "l'it()litsONE"), it has music, and its
    margin doesn't already list the instruments of a conductor score (on a
    score page the header strip can reach a staff label like "Eb Clar.",
    which would turn the page into a part).

    Born-digital pages (exact text, no big picture) never: OCR would be slow
    on every continuation page and worse than their text.
    """
    return (
        settings.ocr_mode != "never"
        and bool(staves)
        and _count_instruments(facts.margin) < SCORE_MARGIN_MIN_INSTRUMENTS
        and largest_image_share(page) >= SCANNED_PAGE_MIN_IMAGE_SHARE
        # The setting, not tesseract_available(): then the cache key and
        # what the analysis did always agree.
        and settings.ocr_available
    )


def analyze_page(
    pdf: pdfium.PdfDocument,
    page_number: int,
    settings: AnalysisSettings,
    stamps: frozenset[str] = frozenset(),
) -> PageFacts:
    """
    Collect all facts about one page. ``page_number`` is 1-based.

    ``stamps`` are the texts on (almost) every page of this PDF (see
    find_stamp_texts); they don't count as real text when deciding on OCR.
    """
    page = pdf[page_number - 1]  # pypdfium2 counts pages from 0

    # 1) Try the embedded text first (unless OCR is forced).
    spans = [] if settings.ocr_mode == "always" else extract_text_spans(page)
    if settings.ocr_mode == "never":
        use_ocr = False
    elif settings.ocr_mode == "always":
        use_ocr = True
    else:  # "auto": OCR only pages without real text (a stamp isn't real text)
        use_ocr = letters_without_stamps(spans, stamps) < MIN_LETTERS_FOR_TEXT_PAGE

    # 2) Render the page once and measure its pixels. Scans are straightened
    #    first (they are often slightly rotated), which helps staff detection
    #    and OCR alike.
    #    fit_dpi lowers the resolution for giant pages so the picture can't
    #    eat all the memory; normal pages keep settings.dpi.
    page_width, page_height = page.get_size()  # in PDF points
    dpi = fit_dpi(page_width, page_height, settings.dpi)
    gray = render_page_gray(page, dpi)
    # Straighten every page, not only the ones we OCR: some "text" PDFs are
    # scans with an OCR text layer added (Funk Attack), and are tilted just
    # the same. A page exported from notation software is perfectly straight,
    # so deskew() leaves it unchanged (it costs ~0.1 s to find that out).
    gray = deskew(gray)
    # On a light scan with grey staff lines, the staves are found with a
    # lighter ink limit; the clef is then measured with that limit too.
    staves, ink_threshold = find_staves_on_page(gray)
    # Top of the first staff, as a share of the page height (None = no staves).
    # find_staves returns the staves top to bottom; staves[0][0][0] is the y
    # of the first line of the first staff.
    first_staff_y = staves[0][0][0] / gray.shape[0] if staves else None
    facts = PageFacts(
        page_number=page_number,
        source="ocr" if use_ocr else "text",
        ink_ratio=ink_ratio(gray),
        staff_count=len(staves),
        clef=detect_clef(gray, staves, ink_threshold),
        first_staff_y=first_staff_y,
    )
    # No staves: maybe the music is printed sideways. np.rot90 turns the
    # picture by 90 degrees, so lines that ran from top to bottom now run
    # from left to right, where find_staves looks for them. (Which way it
    # is turned doesn't matter: a staff is 5 lines either way.)
    # np.ascontiguousarray makes a real copy of the turned picture, laid out
    # row by row in memory, as the other image functions expect.
    # A blank page is not turned: three staves of five lines are far more
    # ink than BLANK_INK_RATIO, so it can't hold sideways music, and the
    # turned copy and second search would be wasted. (Pages of text are
    # still turned: classify checks for sideways music BEFORE it checks for
    # prose, and the search costs only ~0.01 s a page.)
    if not staves and facts.ink_ratio >= BLANK_INK_RATIO:
        turned = np.ascontiguousarray(np.rot90(gray))
        facts.sideways_staff_count = len(find_staves_on_page(turned)[0])
    header_region = (0.0, 0.0, 1.0, header_bottom(first_staff_y))

    # 3a) Text page: just sort the spans into regions. No OCR needed.
    if not use_ocr:
        facts.header = [s for s in spans if s.y0 < header_region[3]]
        facts.margin = [
            s for s in spans if s.x1 <= MARGIN_REGION[2] and s.y0 >= MARGIN_REGION[1]
        ]
        facts.body = spans
        # A first page (credits in the header, music below) whose text has
        # no part name: the name may be printed but missing from the text.
        # Scans with an OCR text layer (Deep Purple page 64) lose words the
        # scanner couldn't read, e.g. "F Horn" next to a handwritten "1º-3º".
        # Then read the header with our own OCR as well. Only first pages:
        # continuation pages normally have no name, so OCR there is wasted.
        if (
            settings.ocr_mode != "never"
            and staves
            and has_credits(facts.header)
            and not _has_part_name(facts.header)
            and settings.ocr_available
        ):
            facts.header += ocr_region(gray, header_region, dpi, psm=11, lang=settings.lang)
        # A scan with a text layer: the scanner's OCR may have garbled the
        # part name ("BARTION SAXOFOON"), or garbled the first of two name
        # lines so that the second one wins ("EARITON I +ll Bb" over "1e
        # TUBA Bb"). We can't tell a garbled name from a right one by its
        # text, so every such page gets its header read by our own OCR, and
        # the better reading of each line wins (merge_header_readings).
        elif read_header_again(page, facts, staves, settings):
            ocr_spans = ocr_region(gray, header_region, dpi, psm=11, lang=settings.lang)
            facts.header = merge_header_readings(facts.header, ocr_spans)
        return facts

    # 3b) Scanned page.
    if facts.ink_ratio < BLANK_INK_RATIO:
        return facts  # blank page: nothing to read

    # ocr_available is False when no language files were found (see
    # ocr.tessdata_folder): the Tesseract language packages are missing, or
    # TESSDATA_PREFIX points to a folder without them. The OCR engine itself
    # comes with scoresplit (the tesserocr package), so it can't be missing.
    if not settings.ocr_available:
        raise OcrUnavailableError(
            f"page {page_number} is a scanned image and needs OCR, but the "
            "OCR language files were not found"
        )

    facts.header = ocr_region(gray, header_region, dpi, psm=11, lang=settings.lang)

    # Second try for the header when no part name was found: higher
    # resolution and another Tesseract mode often read what the first missed.
    if not _has_part_name(facts.header) and not any(
        contains_score_keyword(s.text) for s in facts.header
    ):
        # The retry reads the classic header strip only: psm 6 treats the
        # region as one block of text, which works best on a small strip.
        retry_region = (0.0, 0.0, 1.0, HEADER_MIN_BOTTOM)
        # Only a strip of the page is rendered, so only its size counts.
        retry_dpi = fit_dpi(page_width, page_height * HEADER_MIN_BOTTOM, RETRY_DPI)
        sharper = render_region_gray(page, retry_region, retry_dpi)
        facts.header += ocr_image(sharper, retry_region, retry_dpi, psm=6, lang=settings.lang)
        # A label in a frame ("Score C", cornfield page 2) is misread with
        # its frame lines; read the inside of each frame on its own.
        header_rows = int(header_region[3] * gray.shape[0])
        for inside in find_text_boxes(gray, header_rows, dpi):
            facts.header += ocr_region(gray, inside, dpi, psm=6, lang=settings.lang)
        # Still no name: read each short line of the header on its own, as
        # one line (psm 7). Next to a big title the whole-header pass can
        # drop most of a small name ("F Horn I" read as "F", "H", "1",
        # cornfield page 38); cut out on its own it reads "F Horn 1".
        # Only lines that name a part are added: the other lines (composer,
        # tempo) were read already, and a second reading adds only noise.
        # Not on pages that already have a name: re-reading those gave
        # mixed results ("Eb Horn 1" -> "f}bA Hf)rn 1", round 9.10).
        if not _has_part_name(facts.header) and not any(
            contains_score_keyword(s.text) for s in facts.header
        ):
            for line in find_text_lines(gray, header_rows, dpi):
                line_spans = ocr_region(gray, line, dpi, psm=7, lang=settings.lang)
                facts.header += [s for s in line_spans if find_part_name(s.text)]

    facts.margin = ocr_region(gray, MARGIN_REGION, dpi, psm=11, lang=settings.lang)

    # Whole-page OCR only for pages that are neither a part nor a score and
    # have no staves: typically the cover page with title and credits.
    looks_like_score = _count_instruments(facts.margin) >= SCORE_MARGIN_MIN_INSTRUMENTS
    if not _has_part_name(facts.header) and not looks_like_score and not staves:
        facts.body = ocr_region(gray, (0.0, 0.0, 1.0, 1.0), dpi, psm=3, lang=settings.lang)

    return facts


# ---------------------------------------------------------------------------
# Analysing ALL pages (in parallel, with cache)
# ---------------------------------------------------------------------------

# Every worker process opens the PDF once and keeps it here. ("global"
# variables live per process, so each worker has its own copy.)
_worker_pdf: pdfium.PdfDocument | None = None


# The lowest CPU priority on Linux and macOS (the same as `nice -n 19`).
WORKER_NICENESS = 19

# Up to this many fingerprints are made right here in the main process,
# without a worker pool. Starting a pool costs about half a second (each
# worker starts Python, loads numpy and opens the PDF), while one fingerprint
# takes about 0.13 s, so for a few pages the pool would only slow things down.
FINGERPRINT_IN_PROCESS_MAX_PAGES = 4


def default_worker_count() -> int:
    """
    How many worker processes to start when --workers isn't given: half the
    CPU cores this program may use, and at least 1.

    Why half: OCR keeps every worker at 100%, so with one worker per core the
    whole machine is busy. Half leaves room for everything else: the desktop,
    or on a server the web app, the database and a second job. --workers N
    still sets any number, for full speed.

    sched_getaffinity(0) counts the cores THIS process is allowed to run on,
    which can be fewer than the machine has (a Docker container started with
    --cpuset-cpus, or the `taskset` command). cpu_count() would count them
    all. sched_getaffinity doesn't exist on macOS and Windows, so those fall
    back to cpu_count().
    """
    if hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = os.cpu_count() or 1  # cpu_count() can return None
    # "//" divides and rounds down: 12 // 2 = 6, 3 // 2 = 1, 1 // 2 = 0.
    return max(1, cores // 2)


def _init_worker(pdf_path: str, tessdata: str | None = None) -> None:
    """
    Runs once in each worker process when it starts, before its first page.

    The OCR runs INSIDE this worker: the Tesseract engine (tesserocr) is
    loaded into the worker on its first OCR call (ocr._engine) and kept
    for the next pages. No separate `tesseract` program is started.
    So everything set here applies to the OCR as well.

    ``tessdata`` is the language folder the main process found (see
    ocr.tessdata_folder), or None if it found none.
    """
    # "global": this function sets the module variable, not a local one.
    global _worker_pdf

    # Use the language folder the main process already found. A worker
    # started with "forkserver" or "spawn" (Python 3.14 on Linux, macOS,
    # Windows) doesn't share the main process's remembered answer, so it
    # would otherwise run `tesseract --list-langs` again, once per worker.
    # ocr.tessdata_folder looks at TESSDATA_PREFIX first, so setting it here
    # (in this worker only) also makes sure every worker uses the same folder.
    if tessdata is not None:
        os.environ["TESSDATA_PREFIX"] = tessdata

    # One thread for Tesseract. Each worker already keeps one core busy, so
    # extra threads per worker only fight over the same cores.
    #
    # Tesseract built with OpenMP (a library for running a loop on several
    # threads) reads this setting ONCE, when the library is loaded. That
    # happens on the first OCR call (ocr._engine imports tesserocr there),
    # so setting it here, first thing in the worker, is early enough.
    #
    # The tesserocr 2.13.0 wheels bundle a Tesseract built WITHOUT OpenMP,
    # so today this changes nothing: a worker has one thread either way
    # (measured, see the v3 log). It stays as a safeguard: where no wheel
    # exists (e.g. macOS older than 15), pip builds tesserocr against the
    # system's Tesseract, and that one often IS built with OpenMP (Arch's
    # is). Without the limit it would start a thread for every core.
    os.environ["OMP_THREAD_LIMIT"] = "1"

    # Lower this worker's CPU priority, like starting it with `nice`. The OCR
    # still uses every core that is idle, but whenever another program wants
    # the CPU (the desktop, a web request on a server), that program goes
    # first. The OCR runs in this same process, so it has this low priority
    # too. os.nice doesn't exist on Windows, hence the hasattr check.
    if hasattr(os, "nice"):
        os.nice(WORKER_NICENESS)

    _worker_pdf = open_pdf(pdf_path)


def _analyze_in_worker(
    page_number: int, settings: AnalysisSettings, stamps: frozenset[str]
) -> PageFacts:
    """The job a worker runs for one page."""
    assert _worker_pdf is not None, "worker was not initialised"
    return analyze_page(_worker_pdf, page_number, settings, stamps)


def _run_in_workers(
    pdf_path: Path,
    worker_count: int,
    job: Callable[..., Any],
    jobs: list[tuple],
    on_result: Callable[[Any], None],
) -> None:
    """
    Run `job(*arguments)` for every tuple of arguments in `jobs` in a pool
    of `worker_count` worker processes, and call `on_result(result)` in THIS
    process for each result, in the order they finish (not the order of
    `jobs`). Each worker opens `pdf_path` once (_init_worker).

    Used for the page analysis and for the music fingerprints of the
    duplicate-part check (fingerprint_pages).
    """
    # One thread for numpy's maths library in the workers.
    #
    # numpy comes with OpenBLAS, a library for fast matrix maths (np.dot,
    # np.linalg, ...). When it loads, it starts a helper thread for every
    # core, ready to split a big calculation. scoresplit never asks it for
    # one, so in every worker those threads only sleep (11 per worker on
    # a 12-core laptop, using almost no CPU). They make the workers look
    # as if they used many threads, and if a later change did use matrix
    # maths, 6 workers x 12 threads would fight over 12 cores.
    #
    # Why here and not in _init_worker: OpenBLAS reads this setting once,
    # when numpy is imported. A worker imports numpy while it loads this
    # module, BEFORE _init_worker runs, so setting it there is too late.
    # Workers copy the main process's environment when they start, so
    # setting it here, before the pool starts, reaches them in time.
    # (The main process already has numpy loaded, so nothing changes
    # for it.)
    #
    # setdefault: only when it isn't set yet, so a value the user set
    # themselves is kept.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    # The language folder, found once here (it is remembered, so usually this
    # costs nothing) and handed to every worker.
    tessdata = tessdata_folder()

    with ProcessPoolExecutor(
        max_workers=worker_count, initializer=_init_worker, initargs=(str(pdf_path), tessdata)
    ) as pool:
        # submit() queues one job and returns a "future" (a promise of a
        # result). as_completed() yields the futures as they finish.
        # `*arguments` unpacks the tuple: job(*(5, x)) is job(5, x).
        futures = [pool.submit(job, *arguments) for arguments in jobs]
        try:
            for future in as_completed(futures):
                on_result(future.result())  # result() re-raises any error from the worker
        except BaseException:
            # A job failed (or the user pressed Ctrl+C). Throw away the jobs
            # still waiting in the queue, so we don't OCR hundreds of pages
            # before reporting the error. Only the few jobs already running
            # are waited for.
            # (BaseException also includes Ctrl+C, unlike Exception.)
            pool.shutdown(wait=False, cancel_futures=True)
            raise  # re-raise the same error for the caller


def stamps_of_document(pdf_path: Path) -> frozenset[str]:
    """
    The stamp texts of a whole PDF (see find_stamp_texts). Reading the
    embedded text of every page is fast (no rendering, no OCR).
    """
    pdf = open_pdf(str(pdf_path))
    try:
        pages = [extract_text_spans(pdf[index]) for index in range(len(pdf))]
    finally:
        pdf.close()
    return find_stamp_texts(pages)


def analyze_document(
    pdf_path: Path,
    sha256: str,
    settings: AnalysisSettings,
    progress: ProgressCallback | None = None,
    use_cache: bool = True,
    workers: int | None = None,
) -> list[PageFacts]:
    """
    Analyse every page of a PDF and return the facts in page order.

    Pages already in the cache are not analysed again.
    """
    pdf = open_pdf(str(pdf_path))
    page_count = len(pdf)
    pdf.close()

    known = cache.load_facts(sha256, settings.cache_key()) if use_cache else {}
    todo = [n for n in range(1, page_count + 1) if n not in known]

    if progress:
        progress(page_count - len(todo), page_count)

    if todo:
        worker_count = workers or default_worker_count()
        worker_count = max(1, min(worker_count, len(todo)))
        cached_count = len(known)
        # Stamps need ALL pages, so they are found here, once, before the
        # workers start; each worker gets them with its page.
        # (Not needed when OCR is forced on or off: then text decides nothing.)
        stamps = stamps_of_document(pdf_path) if settings.ocr_mode == "auto" else frozenset()

        def keep(facts: PageFacts) -> None:
            """Called with each analysed page, as soon as a worker is done with it."""
            known[facts.page_number] = facts
            if progress:
                # `known` holds the cached pages plus those analysed so far.
                progress(len(known), page_count)

        try:
            _run_in_workers(
                pdf_path,
                worker_count,
                _analyze_in_worker,
                [(n, settings, stamps) for n in todo],
                keep,
            )
        finally:
            # Runs on success AND on error: cache every page analysed so far,
            # so that the next run doesn't start again from zero.
            if use_cache and len(known) > cached_count:
                cache.save_facts(sha256, settings.cache_key(), known)

    return [known[n] for n in range(1, page_count + 1)]


# ---------------------------------------------------------------------------
# Music fingerprints for the duplicate-part check (in parallel)
# ---------------------------------------------------------------------------


def page_fingerprint(pdf: pdfium.PdfDocument, page_number: int) -> np.ndarray:
    """
    The music fingerprint of one page (image_analysis.music_fingerprint):
    rendered at DEFAULT_DPI and straightened first.
    """
    page = pdf[page_number - 1]
    width, height = page.get_size()
    gray = render_page_gray(page, fit_dpi(width, height, DEFAULT_DPI))
    return music_fingerprint(deskew(gray))


def _fingerprint_in_worker(page_number: int) -> tuple[int, np.ndarray]:
    """The job a worker runs for one page: (page number, its fingerprint)."""
    assert _worker_pdf is not None, "worker was not initialised"
    return page_number, page_fingerprint(_worker_pdf, page_number)


def fingerprint_pages(
    pdf_path: Path,
    page_numbers: list[int],
    progress: ProgressCallback | None = None,
    workers: int | None = None,
) -> dict[int, np.ndarray]:
    """
    The music fingerprints of these pages: {page number: fingerprint}.

    Each page takes a fraction of a second (rendering, straightening, and
    finding the staff lines), but a PDF with many same-name parts needs
    many pages (italiano.pdf: 118), so they are made in parallel, in the
    same kind of worker pool as the page analysis (same worker count, low
    priority). A few pages (FINGERPRINT_IN_PROCESS_MAX_PAGES or fewer) are
    made here, without a pool. Either way the fingerprints are the same.
    `progress(done, total)` counts pages; it is not called at all when
    `page_numbers` is empty.
    """
    fingerprints: dict[int, np.ndarray] = {}
    if not page_numbers:
        return fingerprints
    total = len(page_numbers)
    if progress:
        progress(0, total)

    def keep(result: tuple[int, np.ndarray]) -> None:
        """Called with each fingerprint, as soon as a worker is done with it."""
        page_number, fingerprint = result  # "unpack" the (number, fingerprint) pair
        fingerprints[page_number] = fingerprint
        if progress:
            progress(len(fingerprints), total)

    if total <= FINGERPRINT_IN_PROCESS_MAX_PAGES:
        # Only a few pages: make them here, one after the other.
        pdf = open_pdf(str(pdf_path))
        try:
            for page_number in page_numbers:
                keep((page_number, page_fingerprint(pdf, page_number)))
        finally:
            pdf.close()  # also closed when an error happens above
        return fingerprints

    worker_count = max(1, min(workers or default_worker_count(), total))
    _run_in_workers(
        pdf_path, worker_count, _fingerprint_in_worker, [(n,) for n in page_numbers], keep
    )
    return fingerprints
