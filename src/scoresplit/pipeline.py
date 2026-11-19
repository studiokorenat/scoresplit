"""
pipeline.py: run all the steps in order and produce the result.

This is the "controller" that ties the modules together:

    analyze_document()  pages -> PageFacts        (page_analyzer.py, slow, cached)
    classify_page()     PageFacts -> PageInfo     (classify.py)
    build_parts()       PageInfo list -> parts    (segment.py)
    extract_metadata()  -> title/composer/arranger (metadata.py)
                        => Result                  (models.py)

``inspect_pdf`` is called by api.inspect, which adds the input checks,
the OCR-language warning and the typed errors. To use scoresplit from other
Python code, call the functions in api.py, not this module: api.py is the
public interface (the CLI is one user of it), this is its engine room.
"""

from __future__ import annotations

from pathlib import Path

from .cache import file_sha256
from .classify import classify_page, fix_clef_sign_numbers, mark_title_pages
from .image_analysis import music_similarity
from .metadata import extract_metadata
from .models import PageFacts, PageInfo, Result
from .page_analyzer import AnalysisSettings, ProgressCallback, analyze_document, fingerprint_pages
from .segment import build_parts

# Two same-name parts are copies of each other only when their pages look
# this alike (0..1, see image_analysis.music_similarity). Measured: ~0.95 for
# two scans of the same page, at most ~0.67 for different parts with the
# same layout (Funk Attack: Euphonium treble/bass clef, Bass in Eb/Bb).
DUPLICATE_MIN_SIMILARITY = 0.8

# How many pages of the two parts are compared (page 1 with page 1, ...).
DUPLICATE_PAGES_COMPARED = 2

# A composer or arranger filled in from the parts' first pages (see
# inspect_pdf) must be read on at least this many of them (pages, not vote
# weight: one text page weighs more in the vote, but is still one page).
FILL_CREDITS_MIN_PAGES = 3


class MusicComparer:
    """
    Answers "do these two lists of pages contain the same music?" by looking
    at the pixels, for all same-name parts of a PDF at once (see
    segment.SameMusic).

    Pages are rendered only when asked (a duplicate candidate is rare). A PDF
    with many same-name parts needs many pages, so their fingerprints are
    made in parallel (page_analyzer.fingerprint_pages), reported as the
    "compare" stage through `progress(pages_done, pages_total)`.
    """

    def __init__(
        self,
        pdf_path: Path,
        progress: ProgressCallback | None = None,
        workers: int | None = None,
    ) -> None:
        self.pdf_path = pdf_path
        self.progress = progress
        self.workers = workers

    def same_music(self, comparisons: list[tuple[list[int], list[int]]]) -> list[bool]:
        """
        For each (pages_a, pages_b): True if the first pages of both parts
        look like copies of each other (page 1 with page 1, ...).
        """
        # zip() stops at the shorter list; [:N] = the first N pairs.
        pairs_of = [
            list(zip(pages_a, pages_b, strict=False))[:DUPLICATE_PAGES_COMPARED]
            for pages_a, pages_b in comparisons
        ]
        # Every page any comparison needs, each once ({...} is a set: no
        # repeats), fingerprinted in one go.
        needed = sorted({page for pairs in pairs_of for pair in pairs for page in pair})
        fingerprints = fingerprint_pages(self.pdf_path, needed, self.progress, self.workers)
        return [
            all(
                music_similarity(fingerprints[a], fingerprints[b]) >= DUPLICATE_MIN_SIMILARITY
                for a, b in pairs
            )
            for pairs in pairs_of
        ]


def read_credits(facts: list[PageFacts], infos: list[PageInfo]) -> dict[str, str | None]:
    """
    The title, composer and arranger: {"title": ..., "composer": ...,
    "arranger": ...}. Also marks first pages in `infos` (mark_title_pages).

    Two votes:
      1. On the pages that look like first pages (credits, a cover).
      2. The title (and the names found) then mark more first pages: those
         whose credits OCR couldn't read. The credits are read again on all
         of them, and a name read on at least FILL_CREDITS_MIN_PAGES pages
         fills in a composer or arranger that the first vote
         - didn't find (cornfield: only the cover voted, and its "Jacob de
           Haan" sits in the middle, not on the right), or
         - read on fewer pages than that: one page of OCR noise ("URTI PRN
           S", cornfield with English + Croatian OCR) gives way to the name
           printed on many first pages.
         A name read on fewer pages never counts in the second vote: a line
         read on one page is too often something else (american:
         "CD/Cassette Thunderbirds" on one page).
    """
    meta = extract_metadata(facts, infos)
    # The same first vote, counting only names read on enough pages: a value
    # that is missing there (or different) was read on too few pages.
    confirmed = extract_metadata(facts, infos, min_pages=FILL_CREDITS_MIN_PAGES)
    # The values the second vote may fill in: empty, or read on too few pages.
    weak = [
        key
        for key in ("composer", "arranger")
        if meta[key] is None or meta[key] != confirmed[key]
    ]

    mark_title_pages(facts, infos, meta["title"], (meta["composer"], meta["arranger"]))
    if weak:
        again = extract_metadata(facts, infos, min_pages=FILL_CREDITS_MIN_PAGES)
        for key in weak:
            # "a or b" gives b when a is None: a weak value stays when the
            # second vote found nothing better.
            meta[key] = again[key] or meta[key]
        # A name found just now marks more first pages: those where only the
        # composer was read (cornfield page 48 shows "ROCK", not the title).
        mark_title_pages(facts, infos, meta["title"], (meta["composer"], meta["arranger"]))
    return meta


def inspect_pdf(
    pdf_path: Path,
    settings: AnalysisSettings,
    progress: ProgressCallback | None = None,
    use_cache: bool = True,
    workers: int | None = None,
    compare_progress: ProgressCallback | None = None,
) -> Result:
    """
    Analyse a PDF and return the detected parts as a Result.

    Args:
        pdf_path:  the PDF to analyse.
        settings:  OCR mode, language and resolution.
        progress:  optional function called as progress(done, total).
        use_cache: reuse/save page facts in ~/.cache/scoresplit.
        workers:   number of parallel processes (default: half the CPU cores).
        compare_progress: optional function called as compare_progress(done,
                   total) while same-name parts are compared (pages); never
                   called when there is nothing to compare.
    """
    sha256 = file_sha256(pdf_path)
    facts = analyze_document(pdf_path, sha256, settings, progress, use_cache, workers)

    # A "list comprehension": build a new list by applying classify_page to
    # every element. Same as array_map in PHP.
    infos = [classify_page(page_facts) for page_facts in facts]
    fix_clef_sign_numbers(infos)

    # The title and composer are needed to spot first pages whose credits OCR
    # missed (or that have no "arr." label at all), so metadata comes before
    # grouping the pages into parts.
    meta = read_credits(facts, infos)

    # Same-name parts are only marked as duplicates when their pages really
    # look alike (otherwise, e.g. "Bass" in Eb and in Bb, both are exported).
    parts, extras, warnings = build_parts(
        infos, MusicComparer(pdf_path, compare_progress, workers).same_music
    )

    if not meta["title"]:
        warnings.append("title not found: using the file name")
        meta["title"] = pdf_path.stem  # file name without ".pdf"

    return Result(
        source=pdf_path.name,
        sha256=sha256,
        page_count=len(facts),
        ocr_pages=sum(1 for page_facts in facts if page_facts.source == "ocr"),
        title=meta["title"],
        composer=meta["composer"],
        arranger=meta["arranger"],
        parts=parts,
        extra=extras,
        warnings=warnings,
    )
