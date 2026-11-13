"""
classify.py: decide what each page IS, from its raw facts.

Pipeline position:  page_analyzer -> [classify] -> segment

Input:  PageFacts (text spans, ink, staves, clef) for one page.
Output: PageInfo, which says one of:
    "blank"   - an empty page
    "score"   - a page of the conductor score
    "part"    - a page of an instrument part (with the part name, if readable)
    "unknown" - anything else (cover page, instrument list, unreadable page)

This step is fast and never touches the PDF, so it runs again on every
`inspect` even when the facts come from the cache. You can tweak the rules
below and see the effect right away.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .image_analysis import BLANK_INK_RATIO
from .instruments import (
    contains_score_keyword,
    find_part_name,
    find_two_line_name,
    is_unlisted_key,
)
from .models import PageFacts, PageInfo, PartName, TextSpan
from .pdf_reader import count_letters

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# A score page names at least this many DIFFERENT instruments in its margin...
SCORE_MIN_INSTRUMENTS = 4
# ...from at least this many instrument families (woodwind, brass, ...). This
# stops a multi-percussion PART (Shaker, Tambourine, Chimes...) from looking
# like a score.
SCORE_MIN_FAMILIES = 2

# A page with this many staves is crowded like a score: no part page has
# that many (they have about 8-12). On such a page fewer readable margin
# labels are enough, which helps with badly scanned score pages whose
# margin is cut off or smudged (Gardijska page 1: 20 staves, but only 3
# instrument names were readable).
SCORE_MANY_STAVES = 16
SCORE_MIN_INSTRUMENTS_MANY_STAVES = 3

# A header text this many times larger than the part name is a TITLE, which
# marks the first page of a part (used for PDFs with embedded text only:
# OCR letter heights are too unreliable for this).
TITLE_SIZE_FACTOR = 1.8

# Vertical instrument lists (see _stacked_list_below): items must start at the
# same x (within this share of the page width), have about the same size
# (within this share), and follow each other closely (gap below this many
# letter heights).
# (Loose enough for OCR, whose positions and letter heights wobble a bit.)
STACK_X_TOLERANCE = 0.02
STACK_SIZE_TOLERANCE = 0.25
STACK_MAX_GAP_IN_SIZES = 1.0

# A text with this many words is a sentence (program notes, a biography:
# "...he studied trumpet with professor Josef Pomberger..."), never a part
# name, even if it mentions an instrument...
SENTENCE_MIN_WORDS = 9
# ...but only if at least this many of its words start with a small letter
# ("he", "studied", "with"), as running text does. A header line where the
# scanner merged part name, title and arranger into one long line has
# (almost) none: "Eb Alto Saxophone 1-2 DEEP PURPLE MEDLEY arr. Toshihiko
# Sahashi" is 10 words, but it's not a sentence.
SENTENCE_MIN_LOWERCASE_WORDS = 3

# A page with at least this many letters in such sentences, and at most
# PROSE_PAGE_MAX_STAVES staves, is a TEXT page (biography, program notes,
# instrumentation list): not a part, not the score. (A few "staves" are
# often found in photos or ruled lines on such pages.)
PROSE_PAGE_MIN_LETTERS = 300
PROSE_PAGE_MAX_STAVES = 2

# A page with no staves and at least this many lines of text is a list: a
# publisher's catalogue or discography (italiano pages 2 and 18, 103 and 393
# lines). It names many pieces ("Fighting Trombones", "Celebration (Solo
# Voice)"), but it is no part. A real part whose staves weren't found has
# far fewer lines: at most 63 in the examples (a percussion page in Sedona).
CATALOGUE_MIN_LINES = 80

# OCR'd names are a bit less trustworthy than embedded text.
OCR_CONFIDENCE_FACTOR = 0.9

# Words that appear in the credits printed on the first page of a part or
# score ("arr. Peter Kleine Schaars", "Words & Music by", "Glazba: ...").
# re.IGNORECASE = case doesn't matter. \b = word boundary.
CREDIT_PATTERN = re.compile(
    r"\b(arr\.|arr\b|arranged|arrangement|arranger|arrangiert|bearb\.|bearbeitung|"
    r"aran[žz]man\w*|obrada|obradio|orchestrated|orchestration|composer|composed|"
    r"music\s+by|words|lyrics|glazba|skladatelj|tekst|muzika|commissioned)",
    re.IGNORECASE,
)


def is_label_case(label: str) -> bool:
    """
    Is the label written the way labels are printed? Every word in lower
    case, Capitalised or in CAPITALS: "arr.", "Arr.", "ARR.", "Music by",
    "WORDS & MUSIC BY". Not "aRR": OCR noise read from the notes
    (cornfield page 13 with English + Croatian OCR: "=a=aRR s").
    """
    # word.capitalize(): "aRR" -> "Arr", so "aRR" is none of the three.
    return all(word in (word.lower(), word.upper(), word.capitalize()) for word in label.split())


def find_credit_label(text: str) -> re.Match | None:
    """
    The first credit label in `text` ("arr.", "Music by", ...) that is
    written like a label (is_label_case), or None.
    """
    # finditer gives every match in turn, not just the first one.
    for match in CREDIT_PATTERN.finditer(text):
        if is_label_case(match.group(0)):
            return match
    return None


def has_credits(spans: list[TextSpan]) -> bool:
    """True if any span looks like a composer/arranger credit."""
    return any(find_credit_label(span.text) for span in spans)


def margin_instruments(spans: list[TextSpan]) -> list[PartName]:
    """All instrument names found in the margin spans (one per span at most)."""
    found = []
    for span in spans:
        part = find_part_name(span.text)
        if part:
            found.append(part)
    return found


def is_sentence(text: str) -> bool:
    """True for running text (a sentence), which is never a part name."""
    words = text.split()
    # word[:1] is the first letter ("" for an empty word, so no IndexError).
    lowercase_words = sum(1 for word in words if word[:1].islower())
    return len(words) >= SENTENCE_MIN_WORDS and lowercase_words >= SENTENCE_MIN_LOWERCASE_WORDS


def is_prose_page(facts: PageFacts) -> bool:
    """A page of running text (biography, program notes) with no real music."""
    if facts.staff_count > PROSE_PAGE_MAX_STAVES:
        return False
    spans = facts.header + facts.body
    return count_letters([span for span in spans if is_sentence(span.text)]) >= (
        PROSE_PAGE_MIN_LETTERS
    )


def is_catalogue_page(facts: PageFacts) -> bool:
    """A page of lists (catalogue, discography): many lines of text, no staves."""
    # facts.body holds the text lines of the whole page: always in a text
    # PDF, on a scan only when OCR read the whole page (see page_analyzer).
    return facts.staff_count == 0 and len(facts.body) >= CATALOGUE_MIN_LINES


def header_says_score(spans: list[TextSpan]) -> bool:
    """
    True if a header span says "Score" / "Conductor" / "Partitura" ...

    Sentences don't count: "Commissioned by the City Band and its conductor
    John Smith" is a dedication, not a label on the page.
    """
    return any(
        contains_score_keyword(span.text) and not is_sentence(span.text) for span in spans
    )


# A part name in the header must match the dictionary at least this well
# (0..100) to outvote a score keyword (see _keyword_contradicted).
CLEAR_PART_NAME_MIN_SCORE = 90


def _keyword_contradicted(facts: PageFacts, margin_parts: list[PartName]) -> bool:
    """
    Safety net for a score keyword in the wrong place: is everything ELSE on
    the page saying "this is a part"? That is:

      * the header has a clear part name ("1 Clarinet in Bb"),
      * the margin names no instrument at all (a score labels its staves),
      * and the page isn't crowded with staves like a score.

    Then the keyword was probably part of some other text (a dedication,
    a title), and the page is treated as a part. A "Piano/Conductor" page
    is then the Piano part, which is what the pianist needs anyway.
    """
    if margin_parts or facts.staff_count >= SCORE_MANY_STAVES:
        return False
    found = best_header_part(facts.header)
    return found is not None and found[0].score >= CLEAR_PART_NAME_MIN_SCORE


def is_score_page(facts: PageFacts) -> bool:
    """
    A conductor score page lists many instruments of several families in its
    left margin, or says "Score"/"Conductor"/"Partitura" in its header.
    """
    # A page without music is never a score page, even if it lists all the
    # instruments (the "Instrumentation" page) or says "Full Score" (the
    # score's title page): it is front matter, which goes to extra/.
    if facts.staff_count == 0:
        return False
    found = margin_instruments(facts.margin)
    if header_says_score(facts.header) and not _keyword_contradicted(facts, found):
        return True
    # A "set comprehension" {x for ...} builds a set: duplicates disappear.
    distinct_instruments = {part.instrument_id for part in found}
    distinct_families = {part.family for part in found}
    if facts.staff_count >= SCORE_MANY_STAVES:
        min_instruments = SCORE_MIN_INSTRUMENTS_MANY_STAVES
    else:
        min_instruments = SCORE_MIN_INSTRUMENTS
    return (
        len(distinct_instruments) >= min_instruments
        and len(distinct_families) >= SCORE_MIN_FAMILIES
    )


def _is_next_line(first: TextSpan, last: TextSpan, span: TextSpan) -> bool:
    """
    True when `span` is the next line of a block of lines that starts with
    `first` and (so far) ends with `last`: the same left edge and size as
    `first`, right below `last`.
    """
    return (
        span is not last
        and abs(span.x0 - first.x0) < STACK_X_TOLERANCE
        and abs(span.size - first.size) < STACK_SIZE_TOLERANCE * first.size
        # gap between the two lines, compared with the line height
        # (both in normalised page units, so it works for any paper size)
        and -0.002 <= span.y0 - last.y1 < STACK_MAX_GAP_IN_SIZES * (last.y1 - last.y0)
    )


def _line_below(first: TextSpan, spans: list[TextSpan]) -> TextSpan | None:
    """The line right below `first` (same left edge and size), or None."""
    below = [span for span in spans if _is_next_line(first, first, span)]
    # min(..., key=...) = the one with the smallest y0: the closest line.
    return min(below, key=lambda span: span.y0) if below else None


def _stacked_list_below(first: TextSpan, spans: list[TextSpan]) -> list[TextSpan]:
    """
    Instrument names written as a vertical list under `first`, same left
    edge and same size, e.g. on a percussion title page:

        Tam-tam
        Tambourine
        Wind Chimes

    Returns the spans of the list (including `first`), top to bottom.
    """
    stack = [first]
    while True:
        last = stack[-1]
        following = [
            span
            for span in spans
            if _is_next_line(first, last, span) and find_part_name(span.text) is not None
        ]
        if not following:
            return stack
        stack.append(min(following, key=lambda span: span.y0))


# A line under the part name that says which other instrument the player
# also plays: "doubles Flute", "doubling Piccolo", "double on Oboe". It is a
# note, never part of the name (italiano: "Eb CLARINET" / "doubles Flute"
# is the Eb clarinet part, not "Eb Clarinet, Flute").
# ^\W* = at the start, after any punctuation OCR put there.
DOUBLING_NOTE_PATTERN = re.compile(r"^\W*(?:doubles|doubling|double\s+on)\b", re.IGNORECASE)


def is_doubling_note(text: str) -> bool:
    """True for a "doubles Flute" note under a part name."""
    return DOUBLING_NOTE_PATTERN.search(text) is not None


# A key printed on its own: "C", "Bb", "Eb", "F" (see _join_split_keys).
LONE_KEY_PATTERN = re.compile(r"^[A-G](?:b|♭)?$")

# ...that is joined to a name starting at most this far to its right (share
# of the page width). On cornfield page 43, "C - Trombone I" was read as "C"
# and "Trombone I", 0.03 apart: the dash between them was not read.
LONE_KEY_MAX_GAP = 0.05


def _join_split_keys(spans: list[TextSpan]) -> list[TextSpan]:
    """
    OCR splits a line at a wide gap, so a key printed in front of the name
    with a dash between them ("C - Trombone I") can come out as two spans,
    "C" and "Trombone I". Join a lone key to the part name right of it on
    the same line: "C Trombone I". Other spans are returned unchanged.
    """
    keys = [span for span in spans if LONE_KEY_PATTERN.match(span.text.strip())]
    if not keys:
        return spans
    joined: list[TextSpan] = []
    used_keys: list[TextSpan] = []
    for span in spans:
        if span in keys:
            continue  # added below, joined or on its own
        key = next(
            (
                key
                for key in keys
                if key not in used_keys
                and 0 <= span.x0 - key.x1 < LONE_KEY_MAX_GAP
                # the same line: they overlap up and down by half a line
                and min(key.y1, span.y1) - max(key.y0, span.y0)
                > 0.5 * min(key.y1 - key.y0, span.y1 - span.y0)
                and find_part_name(span.text) is not None
            ),
            None,  # next() returns this when no key fits
        )
        if key is None:
            joined.append(span)
            continue
        used_keys.append(key)
        # replace() copies the span with some fields changed.
        joined.append(
            replace(
                span,
                text=f"{key.text.strip()} {span.text}",
                x0=key.x0,
                y0=min(key.y0, span.y0),
                y1=max(key.y1, span.y1),
            )
        )
    # The keys that joined no name stay as they were.
    return joined + [key for key in keys if key not in used_keys]


def best_header_part(spans: list[TextSpan]) -> tuple[PartName, TextSpan] | None:
    """
    Pick the most likely part name among the header spans.

    Several spans may contain an instrument (the part name, but also e.g. a
    staff label "Shaker" just below it, or "Piano" in "Play Piano" inside
    the music). The winner is the best match; on a tie, the one closest to
    the top (the part name is printed above everything else); then the
    bigger text.

    A name printed over two lines ("Bb BASS" / "CLARINET") is read as one
    (see find_two_line_name). A "doubles Flute" line is never part of it.
    """
    # Leave out the "doubles Flute" notes: they name an instrument, but not
    # the part's own.
    spans = [span for span in spans if not is_doubling_note(span.text)]
    # "C" + "Trombone I" on one line, split by OCR -> "C Trombone I".
    spans = _join_split_keys(spans)

    best: tuple[PartName, TextSpan] | None = None
    best_rank: tuple | None = None
    for span in spans:
        if is_sentence(span.text):
            continue  # "...he studied trumpet with...": not a part name
        part = find_part_name(span.text)
        if part is None:
            continue
        # Abbreviations ("Picc.", "Vl.") belong in score margins. In a part
        # header they are almost always OCR noise ("VL." read from a smudge),
        # unless the same line also prints a key: "COR Eb" is a name line.
        if part.abbreviation and part.key is None:
            continue
        # Tuples compare element by element: score first, then the smallest
        # y (= highest on the page, hence the minus sign; rounded so that
        # spans on the same line count as equally high), then size.
        rank = (part.score, -round(span.y0, 2), round(span.size))
        if best_rank is None or rank > best_rank:
            best, best_rank = (part, span), rank

    if best is None:
        return None

    # A name that goes on in the line below: "1st Bb TENOR" / "SAXOPHONE" is
    # the Tenor Saxophone 1, not the Tenor Horn 1 that the top line alone
    # says. Also when the top line has a number (unlike the list below).
    part, span = best
    below = _line_below(span, spans)
    if below is not None:
        two_lines = find_two_line_name(span.text, below.text)
        if two_lines is not None:
            return two_lines, span

    # A vertical list of instruments is ONE combined part: join the names.
    # Not when the top name has a number ("Percussion 2"): that name is
    # complete, and the list below it just says which instruments it uses.
    if part.number:
        return best
    stack = _stacked_list_below(span, spans)
    if len(stack) > 1:
        combined = find_part_name(", ".join(s.text for s in stack))
        if combined is not None:
            # A key printed on one line only belongs to the whole part:
            # "BARITON" / "1e TUBA Bb (BASSE SOLO)" (superstar) is the
            # baritone in Bb. (A list keeps a key only when all its names
            # print the same one; here the first prints none.)
            if combined.key is None:
                printed = {part.key for part in map(find_part_name, (s.text for s in stack)) if part}
                printed.discard(None)
                if len(printed) == 1:
                    combined = replace(combined, key=printed.pop())
            return combined, span
    return best


def _is_title_size(facts: PageFacts, part_span: TextSpan | None) -> bool:
    """For text PDFs: is there a header text much bigger than the part name?"""
    if facts.source != "text" or part_span is None:
        return False
    for span in facts.header:
        if span is part_span or sum(c.isalpha() for c in span.text) < 4:
            continue
        if span.size >= TITLE_SIZE_FACTOR * part_span.size:
            return True
    return False


def _set_aside_unlisted_key(part: PartName) -> None:
    """
    A key the instrument is never written in (not in its `keys` list) is
    most likely misread: Funk Attack's "Bass Clarinet in Bb" was read as
    "in Eb", and a bass clarinet has no Eb. The key is moved from `key` to
    `unlisted_key`, so the other pages of the part (or the default key) decide
    it; segment.py adds a note.

    This is done for EVERY page, text PDFs too (decided by the user): many
    "text" PDFs are scans whose text layer came from the scanner's own OCR
    (Funk Attack). The price: a truly printed unusual key in a real text PDF
    ("Horn in E") also becomes the default key, with a note to check.

    A part with a label ("Trumpet/Cornet 1 in Eb") is left alone too: the
    key is part of the label text, which would then disagree with it.
    """
    if part.label is None and is_unlisted_key(part.instrument_id, part.key):
        part.unlisted_key = part.key
        part.key = None


def classify_page(facts: PageFacts) -> PageInfo:
    """Turn the raw facts of one page into a PageInfo (see module docstring)."""
    info = PageInfo(
        page_number=facts.page_number,
        kind="unknown",
        clef=facts.clef,
        has_staves=facts.staff_count > 0,
    )
    all_spans = facts.header + facts.margin + facts.body

    # 1) Blank: (almost) no ink and no words.
    if facts.ink_ratio < BLANK_INK_RATIO and count_letters(all_spans) == 0:
        info.kind = "blank"
        info.confidence = 1.0
        return info

    # 2) A text page (biography, program notes): front matter, no music.
    #    has_staves = False, so it goes to extra/ as a cover page, never
    #    into a part (in Funk Attack, "...he studied trumpet..." made a
    #    "Trumpet" part) nor into the score.
    if is_prose_page(facts):
        info.has_staves = False
        info.note = "text page (notes or biography)"
        return info
    #    The same for a catalogue or discography: lists, no music.
    if is_catalogue_page(facts):
        info.note = "text page (catalogue or list)"
        return info

    # 3) Conductor score.
    if is_score_page(facts):
        info.kind = "score"
        info.confidence = 0.9 if facts.source == "ocr" else 1.0
        info.is_start = has_credits(facts.header)
        info.note = "instrument labels in margin"
        return info

    # 4) Instrument part.
    found = best_header_part(facts.header)
    part_span = found[1] if found else None
    info.is_start = has_credits(facts.header) or _is_title_size(facts, part_span)

    if found:
        part = found[0]
        info.kind = "part"
        info.part = part
        info.confidence = part.score / 100
        if facts.source == "ocr":
            info.confidence *= OCR_CONFIDENCE_FACTOR
        _set_aside_unlisted_key(part)
        return info

    # 5) A first page of a part whose name couldn't be read: it has credits
    #    and music on it. The name will come from its following pages.
    if info.is_start and info.has_staves:
        info.kind = "part"
        info.confidence = 0.3
        info.note = "first page, part name not readable"
        return info

    # 6) A cover sheet for one part: no music, and the part name somewhere
    #    on the page (often big, at the bottom: "HORN IN F 1"). It becomes the
    #    first page of that part (segment.py glues it to the music after it).
    cover_part = _cover_sheet_part(facts)
    if cover_part is not None:
        info.kind = "part"
        info.part = cover_part
        info.is_start = True
        info.confidence = cover_part.score / 100
        info.note = "part cover sheet"
        return info

    # 7) Anything else.
    info.note = "no part name found"
    return info


# A cover sheet's part name is a short line on its own (e.g. "HORN IN F 1"),
# not a sentence that happens to contain an instrument word.
COVER_NAME_MAX_LENGTH = 40

# ...printed big and clear, so its name must match the dictionary well
# (0..100). A loose match is usually some other word: on the Ross Roy cover
# OCR read the publisher logo "De Haske" as "de hasker", which is 83% like
# "shaker". ("Hom" for "Horn" still scores 100 via the m -> rn variant.)
COVER_NAME_MIN_SCORE = 90


def _cover_sheet_part(facts: PageFacts) -> PartName | None:
    """
    If the page is a part cover sheet (no staves, exactly one instrument
    named on the whole page), return that part name.
    """
    if facts.staff_count > 0 or not facts.body:
        return None
    # "full score" / "conductor" on the page: it's the SCORE's cover.
    if any(contains_score_keyword(span.text) for span in facts.body):
        return None
    candidates: list[tuple[PartName, TextSpan]] = []
    for span in facts.body:
        if len(span.text) > COVER_NAME_MAX_LENGTH:
            continue
        part = find_part_name(span.text)
        if part is not None and not part.abbreviation and part.score >= COVER_NAME_MIN_SCORE:
            candidates.append((part, span))
    # A list of many instruments is an instrumentation page, not a cover sheet.
    if len({part.instrument_id for part, _ in candidates}) != 1:
        return None
    # The biggest one (it is usually printed large).
    part, _ = max(candidates, key=lambda item: item[1].size)
    return part


# A header text must contain the title at least this well (0..100, rapidfuzz
# partial_ratio) to count as "the title is printed on this page".
TITLE_MATCH_MIN_SCORE = 90

# Titles shorter than this are too generic to search for ("Intro").
TITLE_MIN_LENGTH = 6


def mark_title_pages(
    facts: list[PageFacts],
    infos: list[PageInfo],
    title: str | None,
    people: tuple[str | None, ...] = (),
) -> None:
    """
    Second chance to find the first page of a part: the composition title, or
    the composer/arranger name (`people`), is printed on it. This catches
    first pages whose credits line OCR couldn't read ("arr. ..." garbled)
    but whose big title was read fine. It also catches title blocks with no
    "arr."/"Music by" label at all, e.g. old march cards that print just
    "MARKO POKLEPOVIĆ" on the left and the title in the middle.

    It also rescues "unknown" pages with music and the title on them: the
    first page of a part whose NAME couldn't be read. They become part pages
    without a name; the name is then voted from the pages that follow.

    Changes `infos` in place. Call after the title is known (metadata.py).
    """
    # Imported here rather than at the top, because only this function needs it.
    from rapidfuzz import fuzz

    # Every text to look for, lowercase; short ones ("Intro") are too generic.
    wanted_texts = [
        text.lower() for text in (title, *people) if text and len(text) >= TITLE_MIN_LENGTH
    ]
    if not wanted_texts:
        return
    # strict=True: raise an error if the two lists ever have different lengths.
    for page_facts, info in zip(facts, infos, strict=True):
        is_unnamed_music = info.kind == "unknown" and info.has_staves
        if info.is_start or not (info.kind == "part" or is_unnamed_music):
            continue
        # any(...) with a generator: True as soon as one pair matches.
        # partial_ratio = how well `wanted` matches the best-matching
        # PART of the span text: "nuovo cinema paradiso" inside
        # "bassoon 2 nuovo cinema paradiso" scores 100.
        found = any(
            len(span.text) >= len(wanted) * 0.8
            and fuzz.partial_ratio(wanted, span.text.lower()) >= TITLE_MATCH_MIN_SCORE
            for span in page_facts.header
            for wanted in wanted_texts
        )
        if found:
            info.is_start = True
            info.note = "title or composer printed on page"
            if is_unnamed_music:
                info.kind = "part"
                info.confidence = 0.3


# OCR reads a treble clef sign printed after the part name as this digit:
# "Bb Baritone / Euphonium 𝄞" -> "Bb Baritone / Euphonium 6".
CLEF_SIGN_AS_NUMBER = "6"


def fix_clef_sign_numbers(infos: list[PageInfo]) -> None:
    """
    Turn a part number "6" that is really a misread treble clef sign into
    the clef: "Baritone/Euphonium 6 in Bb" -> "Baritone/Euphonium in Bb"
    with clef "treble" (a clef written in the name, like "(T.C.)").

    A real 6th part has a 5th part beside it, so the "6" is only kept when
    the same instrument also has a part 5 in this PDF.

    Changes `infos` in place. Needs ALL pages, so call it after classify_page.
    """
    numbers_by_instrument: dict[str, set[str]] = {}
    for info in infos:
        if info.part and info.part.number:
            numbers_by_instrument.setdefault(info.part.instrument_id, set()).add(info.part.number)
    for info in infos:
        part = info.part
        if part is None or part.number != CLEF_SIGN_AS_NUMBER:
            continue
        if "5" in numbers_by_instrument.get(part.instrument_id, set()):
            continue  # a real Horn 5 and Horn 6
        # The label and the printed name keep the publisher's wording: remove
        # the "6" from them. count=1: only the first one.
        label, wording = part.label, part.wording
        if label:
            label = re.sub(rf"\s*\b{CLEF_SIGN_AS_NUMBER}\b", "", label, count=1)
        if wording:
            wording = re.sub(rf"\s*\b{CLEF_SIGN_AS_NUMBER}\b", "", wording, count=1)
        # replace() makes a copy of the dataclass with these fields changed.
        info.part = replace(
            part, number=None, clef=part.clef or "treble", label=label, wording=wording
        )
