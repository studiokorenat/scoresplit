"""
segment.py: group the classified pages into parts.

Pipeline position:  classify -> [segment] -> (result) -> exporter

Input:  one PageInfo per page (what kind of page, which part name).
Output: a list of Part (name + pages) and Extra (cover/blank/... pages).

The main difficulty is that OCR is imperfect: one page may say "Hom 2in F",
the next nothing readable, the next "Horn 2 in F". So instead of trusting
every page, pages are first grouped into SEGMENTS (runs of pages that
belong together), and each segment's name is then decided by a VOTE of its
pages.

Two ways to find where a new part begins:

  * "start-page mode" (used when the PDF has title/credit pages): a new part
    starts at every page that carries the credits ("arr. ...", "Words &
    Music by ..."). That's what publishers print on the first page of every
    part. Pages in between belong to the part before them.

  * "name-change mode" (fallback): a new part starts whenever the part name
    in the header changes.

Each part is then named twice:
  * its RECOGNISED name ("Clarinet 2 in Bb"), the same however the PDF
    writes it. It decides which parts are the same part.
  * its PRINTED name ("2. klarinet in B"), which the person sees: the most
    common clean wording of the name on its pages (see _printed_name). When
    no page printed it cleanly, the recognised name is used, with a note.

Finally, parts with the same recognised name are handled:
  * different clefs  -> nothing to do: two parts (the clef column and the
    pages tell them apart)
  * same clef, same length, and the pages LOOK the same (compared by
    pipeline.MusicComparer) -> a duplicate copy: listed but not exported
  * otherwise -> exported, with a note to check it (e.g. a "Bass" in Eb
    and one in Bb, where OCR couldn't read the keys)

A part with no printed key gets its instrument's usual key ("Clarinet 1"
-> Bb), marked key_assumed so the table shows it is a guess. Two kinds of
key get a "please check" note: no key printed for a generic instrument with
keys ("Bass"), and a key the instrument isn't normally written in.

The same for the clef: a part whose clef was neither printed in its name nor
detected on its pages gets its instrument's one clef (Trumpet -> treble,
Bassoon -> bass), marked clef_assumed. Instruments written in either clef
(trombone, tuba) get none.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import pairwise

from .instruments import (
    clef_key_note,
    default_key,
    find_part_name,
    instrument_clef,
    key_from_clef,
    load_dictionary,
    unusual_key_note,
)
from .models import Extra, PageInfo, Part, PartName, format_page_ranges

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# At least this many part pages with credits -> use "start-page mode".
MIN_START_PAGES_FOR_START_MODE = 2

# Share of a segment's decided pages that must agree on the clef.
CLEF_MAJORITY = 0.6
# At least this share of a segment's pages must have a recognised clef at all.
CLEF_MIN_DECIDED_SHARE = 0.5

# Two same-name parts whose page counts differ by at most this many pages
# are considered copies of each other.
DUPLICATE_PAGE_TOLERANCE = 1

# Parts with a confidence below this get a "please check" note.
LOW_CONFIDENCE = 0.6

# A part whose name wasn't printed cleanly on any page (only OCR noise that
# looked like the name, "mTRUMPET") gets its recognised name, and at most
# this confidence, so the table shows it in yellow (see report.py).
UNCLEAR_NAME_CONFIDENCE = 0.8

# How much a page WITHOUT a readable name lowers its part's confidence (0..1),
# in a PDF that normally prints the name on every page. In PDFs that print the
# name only on the first page of a part, unnamed pages are normal and cost
# (almost) nothing: the penalty is scaled by how often names appear.
INHERITED_PAGE_PENALTY = 0.5

# Below this share of named continuation pages, a PDF counts as "names only on
# first pages" (a publisher style, e.g. De Haske).
NAMES_ON_CONTINUATION_PAGES = 0.5

SCORE_NAME = "Conductor Score"

# In start-page mode, a page that is NOT a title page starts a new part only
# if it names another instrument this clearly (match quality 0..100, see
# _starts_new_part), and the next named page agrees.
SAFETY_NET_MIN_SCORE = 90

# A function that answers "do these two page lists hold the same music?"
# (pipeline.MusicComparer.same_music). Without one (unit tests), same-name
# parts of the same length count as copies.
SameMusic = Callable[[list[int], list[int]], bool]


@dataclass
class Segment:
    """A run of pages that belong to one part (or to the score)."""

    kind: str  # "score" or "part"
    pages: list[PageInfo] = field(default_factory=list)
    inherited: list[int] = field(default_factory=list)  # pages without own name
    name: PartName | None = None
    # The canonical name ("Clarinet 2 in Bb"): equal for the same part.
    recognised_name: str = ""
    # The name a person sees: as printed ("Clarinet in Bb 2"), see _printed_name.
    shown_name: str = ""
    confidence: float = 0.0
    clef: str | None = None
    notes: list[str] = field(default_factory=list)
    export: bool = True
    duplicate_of: str | None = None
    # True when the key in `name` wasn't printed but filled in (from the part
    # before it, or the default key after a misread one). See _part_key.
    key_assumed: bool = False

    def page_numbers(self) -> list[int]:
        return [page.page_number for page in self.pages]


# ---------------------------------------------------------------------------
# Step 1: smoothing (fix single misclassified score pages)
# ---------------------------------------------------------------------------


def _smooth_score_pages(infos: list[PageInfo]) -> None:
    """
    A single non-score page sandwiched between two score pages is almost
    certainly a score page whose margin OCR failed. Fix it in place.

    Not when the score page AFTER it is a start page (credits or title on
    it): a conductor score starts only once, so a second "score start" means
    those "score" pages are really the first pages of parts. (Sedona: a
    dedication with "Founder/Conductor" on every part's first page made them
    look like score pages, and smoothing then swallowed every second page.)
    """
    # Ignore blank pages when looking for neighbours.
    content = [info for info in infos if info.kind != "blank"]
    for i in range(1, len(content) - 1):
        page = content[i]
        if page.kind == "score" or page.is_start:
            continue
        following = content[i + 1]
        if content[i - 1].kind == "score" and following.kind == "score" and not following.is_start:
            page.kind = "score"
            page.part = None
            page.confidence = 0.5
            page.note = "between score pages"

    # Unreadable music pages at the very START of the PDF, followed by the
    # score: the score's first page(s), with a margin OCR couldn't read (in
    # the Gardijska scan, page 1 is cut off at the left edge). Only "unknown"
    # pages: a page with a part name before the score stays a part.
    leading: list[PageInfo] = []
    for page in content:
        if page.kind == "unknown" and page.has_staves:
            leading.append(page)
            continue
        if page.kind == "score":
            for unread in leading:
                unread.kind = "score"
                unread.confidence = 0.5
                unread.note = "unreadable page before the score"
        break  # stop at the first page that is not an unreadable music page


def _continue_score(infos: list[PageInfo]) -> None:
    """
    For PDFs whose parts have start pages (start-page mode): after a score
    page, the music pages that are NOT a part's start page belong to the
    score too, up to the first real part start. Fix them in place.

    A hand-copied score has unreadable margins, so many of its pages are not
    recognised as score pages ("unknown"), and a label inside the music can
    read like a part name: "Tub, Bar" on suite page 15 started a Tuba part
    that swallowed the rest of the score.

    Safety net for a part whose start page was missed (title not read): a
    page whose name is confirmed by the next named page (the same
    instrument) still ends the score.

    Pages without music are left alone here; one between two score pages
    joins the score in _smooth_score_pages (suite page 9).
    """
    # Ignore blank pages, like _smooth_score_pages.
    content = [info for info in infos if info.kind != "blank"]
    in_score = False  # True while we are inside the score
    for index, page in enumerate(content):
        if page.kind == "score":
            in_score = True
            continue
        if page.is_start:
            in_score = False  # the first part begins
            continue
        if not in_score or not page.has_staves:
            continue
        if page.kind == "part" and page.part is not None:
            following = _next_part_page(content, index)
            if following and following.part.instrument_id == page.part.instrument_id:
                in_score = False  # a real part, its title page missed
                continue
        page.kind = "score"
        page.part = None
        page.confidence = 0.5
        page.note = "continues the score"


# ---------------------------------------------------------------------------
# Step 2: cutting the page list into segments
# ---------------------------------------------------------------------------


def _segment_identity(segment: Segment) -> PartName | None:
    """The part name the segment has so far (its first named page)."""
    for page in segment.pages:
        if page.part is not None:
            return page.part
    return None


def _names_conflict(a: PartName, b: PartName) -> bool:
    """
    True if two part names clearly describe DIFFERENT parts.

    A missing number/key is not a conflict ("Flute" vs "Flute 1" is fine),
    only two different values are.
    """
    if a.instrument_id != b.instrument_id:
        return True
    if a.number and b.number and a.number != b.number:
        return True
    if a.key and b.key and a.key != b.key:
        return True
    # "Baritone (T.C.)" and "Baritone (B.C.)" are two different parts.
    if a.clef and b.clef and a.clef != b.clef:
        return True
    if (a.label or "") != (b.label or "") and a.label and b.label:
        return True
    return False


def _names_related(a: PartName, b: PartName) -> bool:
    """
    True if one instrument name contains the other: "Clarinet" and
    "Bass Clarinet", "Trombone" and "Bass Trombone", "Saxophone" and
    "Alto Saxophone".
    """
    name_a, name_b = a.instrument_name.lower(), b.instrument_name.lower()
    return name_a in name_b or name_b in name_a


def _segment_clef(segment: Segment) -> str | None:
    """
    The clef of the segment so far: the one printed in its name ("Baritone
    (T.C.)"), else the clef a clear majority of its pages show (image
    detection), else None.
    """
    for page in segment.pages:
        if page.part is not None and page.part.clef:
            return page.part.clef
    clefs = [page.clef for page in segment.pages if page.clef]
    if not clefs:
        return None
    winner, count = Counter(clefs).most_common(1)[0]
    return winner if count >= CLEF_MAJORITY * len(clefs) else None


def _printed_clef_changes(current: Segment, info: PageInfo) -> bool:
    """
    True when page `info` prints a clef in its name that is NOT the clef of
    the part so far. In rhcp.pdf "Euphonium" (treble clef music, no clef in
    the name) is followed by the cover page "Euphonium (B.C.)": same
    instrument, but the bass clef version begins there.
    """
    if info.part is None or not info.part.clef:
        return False
    clef = _segment_clef(current)
    return clef is not None and clef != info.part.clef


def _next_part_page(infos: list[PageInfo], index: int) -> PageInfo | None:
    """The next page after `index` that has a part name (skipping others)."""
    for info in infos[index + 1 :]:
        if info.kind == "part" and info.part is not None:
            return info
        if info.kind == "score":
            return None
    return None


def _starts_new_part(
    current: Segment | None, info: PageInfo, infos: list[PageInfo], index: int, start_mode: bool
) -> bool:
    """Decide whether part page `info` begins a new segment."""
    if current is None or current.kind != "part":
        return True

    # In both modes: a clef printed in the name that differs from the part so
    # far starts the other clef version ("Euphonium" -> "Euphonium (B.C.)").
    # The instrument is the same, so the checks below would miss it.
    if _printed_clef_changes(current, info):
        return True

    identity = _segment_identity(current)
    if start_mode:
        if info.is_start:
            return True
        # Safety net for a missed title page: this page AND the next named
        # page both clearly name another instrument -> new part anyway.
        # Only a clearly read name counts: OCR noise like "I aSs I aS" (86%
        # like "Bass") in the middle of a Baritone part must not cut it.
        if (
            identity
            and info.part
            and info.part.score >= SAFETY_NET_MIN_SCORE
            and info.part.instrument_id != identity.instrument_id
        ):
            # ...except when the names are related ("Clarinet" vs "Bass
            # Clarinet") and only ONE page of the part so far says the
            # shorter name: then OCR probably dropped a word on that page, and
            # the name vote in _resolve_name will settle it.
            pages_with_identity = sum(
                1
                for page in current.pages
                if page.part is not None and page.part.instrument_id == identity.instrument_id
            )
            if pages_with_identity <= 1 and _names_related(identity, info.part):
                return False
            following = _next_part_page(infos, index)
            if following and following.part.instrument_id == info.part.instrument_id:
                return True
        return False

    # name-change mode
    if identity and info.part:
        return _names_conflict(identity, info.part)
    return False


def _cut_into_segments(
    infos: list[PageInfo], start_mode: bool
) -> tuple[list[Segment], list[tuple[int, str]]]:
    """
    Walk through the pages and group them.

    Returns (segments, extras), where extras is a list of
    (page_number, reason) for pages that belong to no part.
    """
    segments: list[Segment] = []
    extras: list[tuple[int, str]] = []
    current: Segment | None = None
    seen_music = False  # becomes True at the first score/part page

    for index, info in enumerate(infos):
        if info.kind == "blank":
            extras.append((info.page_number, "blank"))
            continue

        if info.kind == "score":
            if current is None or current.kind != "score":
                current = Segment(kind="score")
                segments.append(current)
            current.pages.append(info)
            seen_music = True
            continue

        if info.kind == "part":
            if _starts_new_part(current, info, infos, index, start_mode):
                current = Segment(kind="part")
                segments.append(current)
            current.pages.append(info)
            if info.part is None:
                current.inherited.append(info.page_number)
            seen_music = True
            continue

        # kind == "unknown"
        if info.has_staves and current is not None:
            # Music without a readable name: it continues the part before it.
            current.pages.append(info)
            current.inherited.append(info.page_number)
        elif not seen_music and not info.has_staves:
            extras.append((info.page_number, "cover"))  # front matter
        else:
            extras.append((info.page_number, "unassigned"))

    return segments, extras


# ---------------------------------------------------------------------------
# Step 3: naming each segment by vote
# ---------------------------------------------------------------------------


def _agreed_value(values: list[str | None]) -> str | None:
    """
    A number or key that the pages agree on.

    * all pages that have a value have the SAME value -> that value (even if
      only one page could read it);
    * different values -> the most common one, but only if at least 2 pages
      have it (a single page could have read a bar number as part number).
    """
    counts = Counter(v for v in values if v)
    if not counts:
        return None
    value, count = counts.most_common(1)[0]
    if len(counts) == 1 or count >= 2:
        return value
    return None


def _most_common(values: list[str | None]) -> str | None:
    """The most frequent non-empty value, or None."""
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else None


def continuation_name_rate(infos: list[PageInfo]) -> float:
    """
    Share of continuation pages (music pages that are not a part's first page)
    that have a readable part name. ~1.0 = the publisher prints the name on
    every page; ~0.0 = only on the first page of each part.
    """
    continuation = [
        info
        for info in infos
        if info.kind in ("part", "unknown") and info.has_staves and not info.is_start
    ]
    if not continuation:
        return 1.0
    named = sum(1 for info in continuation if info.part is not None)
    return named / len(continuation)


def _resolve_name(segment: Segment, name_rate: float = 1.0) -> None:
    """Decide the segment's part name, clef and confidence."""
    segment.key_assumed = False  # this function may run twice on a segment
    named = [page for page in segment.pages if page.part is not None]
    if not named:
        segment.name = None
        segment.confidence = 0.0
        return

    # Vote on the INSTRUMENT. Start pages have the cleanest header, so their
    # vote counts double.
    votes: Counter = Counter()
    for page in named:
        weight = page.confidence * (2 if page.is_start else 1)
        votes[page.part.instrument_id] += weight
    winner_id = votes.most_common(1)[0][0]
    same = [page for page in named if page.part.instrument_id == winner_id]
    winner_part = same[0].part

    # Pages that name a LESS specific related instrument also agree: OCR
    # read "Z. Horn" for "English Horn", or "Clarinet" for "Bass Clarinet".
    agreeing = [
        page
        for page in named
        if page.part.instrument_id == winner_id
        or page.part.instrument_name.lower() in winner_part.instrument_name.lower()
    ]

    # The first page's name is the most reliable source for number and key.
    start_part = next(
        (p.part for p in same if p.is_start), None
    )  # next(generator, default) = the first match, or None if there is none
    template = start_part or winner_part

    # Fill gaps (e.g. the first page read only "Trumpet in") from the other
    # pages of the same instrument.
    number = template.number or _agreed_value([p.part.number for p in same])
    key = template.key or _agreed_value([p.part.key for p in same])
    clef = template.clef or _agreed_value([p.part.clef for p in same])
    label = template.label or _most_common([p.part.label for p in same])

    # No page read a usable key, but one read a key this instrument is never
    # written in (see classify._set_aside_unlisted_key): use the default key.
    # "Bass Clarinet in Eb" -> "Bass Clarinet in Bb", with a note.
    # A generic instrument has no default key ("Bass in A" -> no key), so
    # the note then only asks to check.
    misread = _agreed_value([p.part.unlisted_key for p in same])
    if key is None and misread:
        key = default_key(template.instrument_id)
        segment.key_assumed = key is not None
        assumed = f"assumed {key}, " if key else ""
        segment.notes.append(
            f"key read as {misread}, but {template.instrument_name} isn't written "
            f"in {misread}: {assumed}please check"
        )

    segment.name = PartName(
        instrument_id=template.instrument_id,
        instrument_name=template.instrument_name,
        family=template.family,
        number=number,
        key=key,
        clef=clef,
        label=label,
        score=template.score,
    )

    # Confidence = (share of NAMED pages that agree)
    #            x (their average match quality)
    #            x (a penalty for pages without a name, scaled by how unusual
    #               unnamed pages are in this PDF: see continuation_name_rate)
    #
    # A disagreeing continuation page counts less in PDFs that rarely print
    # names there (weight = name_rate): such "names" are usually instrument
    # changes inside the music ("Orch. Bells", "Susp. Cym."), not part names.
    disagreeing = [page for page in named if page not in agreeing]
    disagree_weight = sum(1.0 if page.is_start else name_rate for page in disagreeing)
    agreement = len(agreeing) / (len(agreeing) + disagree_weight)
    average_quality = sum(p.confidence for p in agreeing) / len(agreeing)
    unnamed_share = 1 - len(named) / len(segment.pages)
    penalty = 1 - INHERITED_PAGE_PENALTY * unnamed_share * name_rate
    segment.confidence = round(agreement * average_quality * penalty, 2)

    if disagreeing:
        segment.notes.append(
            f"page(s) {format_page_ranges([p.page_number for p in disagreeing])} "
            "looked like another instrument"
        )


def _resolve_clef(segment: Segment) -> None:
    """
    The segment's clef: the one printed in the part name ("Baritone (B.C.)"),
    else a clear majority of its pages (image detection), else None.
    """
    # Printed text beats the image guess (which may fail on a smudged scan).
    if segment.name is not None and segment.name.clef:
        segment.clef = segment.name.clef
        return
    clefs = [page.clef for page in segment.pages if page.clef]
    # Too few pages with a clear clef (e.g. percussion parts): don't guess.
    if len(clefs) < CLEF_MIN_DECIDED_SHARE * len(segment.pages):
        return
    winner, count = Counter(clefs).most_common(1)[0]
    if count >= CLEF_MAJORITY * len(clefs):
        segment.clef = winner


# ---------------------------------------------------------------------------
# Step 4: printed names and duplicates
# ---------------------------------------------------------------------------


def _same_part_groups(segments: list[Segment]) -> list[list[Segment]]:
    """
    The segments grouped by recognised name, in order: [[Flute 1], [Bass,
    Bass], ...]. Same-name segments in DIFFERENT clefs are different parts
    (Euphonium in treble and in bass clef), so they get a group each; a
    segment without a clear clef then gets its own group too.
    """
    by_name: dict[str, list[Segment]] = {}
    for segment in segments:
        by_name.setdefault(segment.recognised_name, []).append(segment)

    groups: list[list[Segment]] = []
    # .values() = the lists only, without the names.
    for group in by_name.values():
        clefs = {segment.clef for segment in group if segment.clef}
        if len(clefs) <= 1:
            groups.append(group)
            continue
        by_clef: dict[str | None, list[Segment]] = {}
        for segment in group:
            by_clef.setdefault(segment.clef, []).append(segment)
        groups.extend(by_clef.values())
    return groups


def _merge_cover_sheets(parts: list[Segment]) -> list[Segment]:
    """
    Some parts start with a cover sheet: a page with only the part name and
    the title, no music. It becomes its own little segment. Glue it to the
    following segment when that one has the same name.

    Returns the new list of segments.
    """
    merged: list[Segment] = []
    for segment in parts:
        previous = merged[-1] if merged else None
        previous_is_cover = previous is not None and not any(p.has_staves for p in previous.pages)
        if (
            previous_is_cover
            and previous.name is not None
            and segment.name is not None
            and not _names_conflict(previous.name, segment.name)
            and previous.page_numbers()[-1] + 1 == segment.page_numbers()[0]
        ):
            # Put the cover pages in front of this segment, drop the cover segment.
            segment.pages = previous.pages + segment.pages
            segment.inherited = previous.inherited + segment.inherited
            merged[-1] = segment
        else:
            merged.append(segment)
    return merged


def _same_series(a: PartName, b: PartName) -> bool:
    """Same instrument, key, clef and no custom label: "Trombone 1" / "Trombone 3"."""
    return (
        a.instrument_id == b.instrument_id
        and a.key == b.key
        and a.clef == b.clef
        and not a.label
        and not b.label
    )


def _fill_missing_numbers(parts: list[Segment]) -> None:
    """
    A part without a number, sitting between two numbered parts of the same
    instrument, gets the number in between:

        Trombone 1, Trombone, Trombone 3  ->  Trombone 1, Trombone 2, Trombone 3

    (In the Gardijska scan OCR read "2nd Trombone" as "an Trombone".) Only
    a gap of exactly one number is filled, so the guess can't be far off.
    """
    # range(1, len - 1): every part that has a neighbour on both sides.
    for i in range(1, len(parts) - 1):
        name, before, after = parts[i].name, parts[i - 1].name, parts[i + 1].name
        if name is None or before is None or after is None:
            continue
        if name.number or name.label or not _same_series(before, after):
            continue
        # The middle part may also have lost its key ("Clarinet" between two
        # "Clarinet ... in Bb"), but it must not have a DIFFERENT one.
        if name.instrument_id != before.instrument_id or name.key not in (None, before.key):
            continue
        if name.clef not in (None, before.clef):
            continue
        if not (before.number or "").isdigit() or not (after.number or "").isdigit():
            continue
        guess = int(before.number) + 1
        if int(after.number) != guess + 1:
            continue
        name.number = str(guess)
        name.key = before.key
        parts[i].recognised_name = name.display()
        parts[i].notes.append(
            f"part number {guess} guessed from the parts before and after it: please check"
        )


def _number_range(number: str | None) -> tuple[int, int] | None:
    """
    The first and last player of a part number, or None when it isn't made
    of plain numbers: "2" -> (2, 2), "3&4" -> (3, 4), "1-3" -> (1, 3),
    "1/2" -> (1, 2), "Solo" -> None.
    """
    pieces = re.split(r"[&/+-]", number or "")
    if not all(piece.isdigit() for piece in pieces):
        return None  # also for "" (no number at all)
    return int(pieces[0]), int(pieces[-1])


def _fill_missing_keys(parts: list[Segment]) -> None:
    """
    A numbered part without a key, right after the previous number of the
    same instrument WITH a key, gets that key:

        Clarinet 1 in Bb, Clarinet 2, Clarinet 3  ->  ... 2 in Bb, 3 in Bb

    (In Funk Attack the flat sign of "B♭ CLARINET 2" was read as "Bo" or
    "m".) Runs front to back, so the key passes along the whole row.

    Not when the two parts are in different clefs (printed in the name or
    detected on the page): in italiano, "1. Bb Trombone" is in treble clef,
    and the "2. Trombone" after it in bass clef is in C, not Bb.
    """
    # pairwise([a, b, c]) gives (a, b), (b, c): each part with the one before it.
    for previous, segment in pairwise(parts):
        name, before = segment.name, previous.name
        if name is None or before is None or name.key or name.label or before.label:
            continue
        if name.instrument_id != before.instrument_id or not before.key:
            continue
        if name.clef not in (None, before.clef):
            continue
        # segment.clef = the clef printed in the name, else the one detected
        # on the pages (see _resolve_clef). An unknown clef doesn't block.
        if segment.clef and previous.clef and segment.clef != previous.clef:
            continue
        # The numbers must follow each other: "2" after "1", and also a pair
        # after a single one ("Horn 3&4" after "Horn 2 in Eb", cornfield).
        first, last = _number_range(name.number), _number_range(before.number)
        if first is None or last is None or first[0] != last[1] + 1:
            continue
        name.key = before.key
        segment.key_assumed = True
        segment.recognised_name = name.display()
        segment.notes.append(f"key {before.key} taken from the part before it: please check")


def _keys_from_clef(parts: list[Segment]) -> None:
    """
    A part with no printed key, of an instrument with a `treble_clef_key` in
    instruments.yaml, is in the key its clef tells (see
    instruments.key_from_clef):

        Euphonium (treble clef) -> Bb, with a note
        Euphonium (bass clef)   -> C (the default key anyway)

    The part's key is filled in later, marked as assumed (see _part_key);
    the name stays as printed (the clef column tells the two apart). This
    only adds the note when the clef gives a key
    other than the default, so a person sees why the key isn't the usual one.
    That goes for a combined part too ("Baritone, Euphonium": the first
    instrument decides).
    """
    for segment in parts:
        name = segment.name
        if name is None:
            continue
        note = clef_key_note(name.instrument_id, name.key, segment.clef)
        if note:
            segment.notes.append(note)


def _note_unknown_keys(parts: list[Segment]) -> None:
    """
    A part of a GENERIC instrument that has keys ("Bass") but no printed key
    gets no key assumed (see instruments.default_key), so a person is asked
    to check it:

        "Bass"        -> note "key not printed: please check"
        "Bass in Eb"  -> no note
        "Bass Guitar" -> no note (no keys at all)
    """
    dictionary = load_dictionary()
    for segment in parts:
        if segment.name is None or segment.name.key:
            continue
        instrument = dictionary.by_id.get(segment.name.instrument_id)
        if instrument is None or not instrument.keys or default_key(instrument.id):
            continue  # no keys at all, or a default key that is used anyway
        # A misread key already has its own note (see _resolve_name).
        if any(note.startswith("key read as") for note in segment.notes):
            continue
        segment.notes.append("key not printed: please check")


def _note_unusual_keys(parts: list[Segment]) -> None:
    """
    A key the instrument isn't normally written in ("Flute in Eb") gets a
    note: the key is kept, but the name may be misread.
    """
    for segment in parts:
        if segment.name is None:
            continue
        note = unusual_key_note(segment.name.instrument_id, segment.name.key)
        if note:
            segment.notes.append(note)


def _handle_same_names(parts: list[Segment], same_music: SameMusic | None = None) -> None:
    """
    Find copies among parts with the same recognised name and clef (see the
    module docstring). Names are not changed: two parts may share a name,
    and their pages tell them apart.
    """
    for group in _same_part_groups(parts):
        if len(group) < 2:
            continue
        original = group[0]
        original_pages = format_page_ranges(original.page_numbers())
        for segment in group[1:]:
            same_length = abs(len(segment.pages) - len(original.pages)) <= DUPLICATE_PAGE_TOLERANCE
            looks_same = same_length and (
                same_music is None or same_music(original.page_numbers(), segment.page_numbers())
            )
            if looks_same:
                segment.export = False
                segment.duplicate_of = original.shown_name
                segment.notes.append(
                    f"duplicate of '{original.shown_name}' (pages {original_pages}), not exported"
                )
            elif same_length:
                # Same name and length, but different notes: a different part
                # whose distinguishing word (key, clef, number) wasn't read.
                # Better one PDF too many than a part silently missing.
                segment.notes.append(
                    f"same name as pages {original_pages} but different music "
                    "(another key or clef?): please rename"
                )
            else:
                segment.notes.append(
                    f"same name as pages {original_pages} but a different length: please check"
                )


def _printed_name(segment: Segment) -> str | None:
    """
    The name of the part as printed: the most common clean wording
    (PartName.wording) among its pages, or None when no page has one.

    Only pages that name exactly this part take part in the vote: the same
    instrument, number, printed key and label. So a continuation page that
    printed only "Bass Tuba" doesn't outvote the first page's "Bass Tuba in
    C", and a page whose key was misread ("Bass Clarinet in Eb") isn't
    shown. As in _resolve_name, a start page's vote counts double.
    """
    name = segment.name
    if name is None:
        return None
    # A filled-in key wasn't printed, so a printed name doesn't have it.
    printed_key = None if segment.key_assumed else name.key
    votes: Counter = Counter()
    for page in segment.pages:
        part = page.part
        if part is None or part.wording is None or part.unlisted_key:
            continue
        names_this_part = (
            part.instrument_id == name.instrument_id
            and part.number == name.number
            and part.key == printed_key
            and part.label == name.label
            and part.clef in (None, name.clef)
        )
        if names_this_part:
            votes[part.wording] += page.confidence * (2 if page.is_start else 1)
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def _choose_shown_names(parts: list[Segment]) -> None:
    """
    Give every part the name a person sees: as printed (see _printed_name),
    else the recognised name without a filled-in key ("Trumpet 1"), with a
    note and a lower confidence so the table shows it in yellow.
    """
    for segment in parts:
        if segment.name is None:
            segment.shown_name = segment.recognised_name  # "Unknown Part (p5-7)"
            continue
        printed = _printed_name(segment)
        if printed is not None:
            segment.shown_name = printed
            continue
        key = None if segment.key_assumed else segment.name.key
        # replace() makes a copy of the dataclass with `key` changed.
        segment.shown_name = replace(segment.name, key=key).display()
        # A misread key ("Bass Clarinet in Eb") already has its own note,
        # which says why the printed name isn't used.
        if not any(note.startswith("key read as") for note in segment.notes):
            segment.notes.append("the name wasn't printed clearly: please check it")
        segment.confidence = min(segment.confidence, UNCLEAR_NAME_CONFIDENCE)


def _part_key(segment: Segment) -> tuple[str | None, bool]:
    """
    The key of a finished part, and whether it was assumed: (key, key_assumed).

        "Clarinet 1 in Eb"         -> ("Eb", False)  printed
        "Clarinet 1"               -> ("Bb", True)   the instrument's usual key
        "Euphonium", treble clef   -> ("Bb", True)   the key its clef tells
        "Clarinet 2" after "1 in A"-> ("A", True)    taken from the part before
        "Snare Drum", "Bass"       -> (None, False)  no keys / no usual key

    The name keeps showing only what was printed ("Clarinet 1"); the key is
    filled in here so every part that has a key says which one.
    """
    name = segment.name
    if name is None:
        return None, False
    if name.key:
        return name.key, segment.key_assumed
    # Not printed: the clef's key first (a treble-clef euphonium is in Bb),
    # else the usual key. "a or b" gives b when a is None.
    key = key_from_clef(name.instrument_id, segment.clef) or default_key(name.instrument_id)
    return key, key is not None


def _part_clef(segment: Segment) -> tuple[str | None, bool]:
    """
    The clef of a finished part, and whether it was assumed: (clef, clef_assumed).

    In this order (see _resolve_clef for the first two):

        "Baritone (B.C.)"              -> ("bass", False)    printed in the name
        Trumpet, treble clef on pages  -> ("treble", False)  detected
        Trumpet, no clef found         -> ("treble", True)   the instrument's clef
        Trombone, no clef found        -> (None, False)      written in either clef
        Snare Drum, Piano              -> (None, False)      no single clef
        "Pauken, Glockenspiel"         -> (None, False)      two instruments, two clefs

    The assumed clef is filled in only here, at the very end, so a guess
    never changes a decision made before it: which same-name parts are
    copies (_same_part_groups), or whether a key is passed on
    (_fill_missing_keys). Only the result shows it.
    """
    if segment.clef is not None:
        return segment.clef, False
    if segment.name is None:
        return None, False
    clef = instrument_clef(segment.name.instrument_id)
    if clef is not None and _names_another_clef(segment, clef):
        return None, False
    return clef, clef is not None


def _names_another_clef(segment: Segment, clef: str) -> bool:
    """
    Does the printed name of the part also list an instrument written in
    another clef than `clef` (the clef of the part's own instrument)?

    "Pauken, Glockenspiel" (superstar page 63): the timpani are in bass
    clef, the glockenspiel in treble, so the part has both and no single
    clef can be assumed. The names in a list are split at "," and "/".
    """
    printed = _printed_name(segment) or ""
    # re.split cuts the text at every "," or "/": "A, B/C" -> ["A", " B", "C"].
    for piece in re.split(r"[,/]", printed):
        other = find_part_name(piece)
        if (
            other is not None
            and segment.name is not None
            and other.instrument_id != segment.name.instrument_id
            and instrument_clef(other.instrument_id) != clef
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def build_parts(
    infos: list[PageInfo], same_music: SameMusic | None = None
) -> tuple[list[Part], list[Extra], list[str]]:
    """
    Turn per-page information into the parts and extra pages of the result.

    `same_music` decides whether two same-name parts are real copies (see
    SameMusic). Returns (parts, extras, warnings).
    """
    warnings: list[str] = []

    # Start-page mode: the parts have start pages (see the module docstring).
    # Counted before the score is fixed up below, which never changes a
    # start page.
    start_pages = sum(1 for info in infos if info.kind == "part" and info.is_start)
    start_mode = start_pages >= MIN_START_PAGES_FOR_START_MODE

    if start_mode:
        _continue_score(infos)
    _smooth_score_pages(infos)

    segments, extra_pages = _cut_into_segments(infos, start_mode)
    name_rate = continuation_name_rate(infos)
    names_on_every_page = name_rate >= NAMES_ON_CONTINUATION_PAGES

    # --- the score: all score segments become ONE conductor score
    score_segments = [s for s in segments if s.kind == "score"]
    part_segments = [s for s in segments if s.kind == "part"]

    # --- name every part (twice around the cover-sheet merge: merging needs
    #     the names, and the merged segments need a fresh vote)
    for segment in part_segments:
        _resolve_name(segment, name_rate)
    part_segments = _merge_cover_sheets(part_segments)

    for segment in part_segments:
        segment.notes = []
        _resolve_name(segment, name_rate)
        _resolve_clef(segment)
        if segment.name is not None:
            segment.recognised_name = segment.name.display()
        else:
            pages = format_page_ranges(segment.page_numbers())
            segment.recognised_name = f"Unknown Part (p{pages})"
            segment.notes.append("no part name could be read: please rename")
        # Only worth a note when unnamed pages are unusual in this PDF.
        if segment.inherited and names_on_every_page:
            segment.notes.append(
                f"page(s) {format_page_ranges(segment.inherited)} had no readable "
                "name and were added to this part"
            )
        if segment.confidence < LOW_CONFIDENCE:
            segment.notes.append("low confidence: please check")

    _fill_missing_numbers(part_segments)
    _fill_missing_keys(part_segments)
    _keys_from_clef(part_segments)
    # Check the keys once they are final.
    _note_unknown_keys(part_segments)
    _note_unusual_keys(part_segments)
    # Once number and key are final: the name as printed, then the copies.
    _choose_shown_names(part_segments)
    _handle_same_names(part_segments, same_music)

    # --- convert to the result's models
    parts: list[Part] = []
    if score_segments:
        score_pages = [n for s in score_segments for n in s.page_numbers()]
        notes = []
        if len(score_segments) > 1:
            notes.append("score pages were not contiguous: please check")
        smoothed = [p.page_number for s in score_segments for p in s.pages if p.confidence < 0.6]
        if smoothed:
            notes.append(f"page(s) {format_page_ranges(smoothed)} assumed to be score")
        parts.append(
            Part(name=SCORE_NAME, kind="score", pages=score_pages, confidence=0.9, notes=notes)
        )

    for segment in part_segments:
        name = segment.name
        key, key_assumed = _part_key(segment)
        clef, clef_assumed = _part_clef(segment)
        parts.append(
            Part(
                name=segment.shown_name,
                kind="part",
                pages=segment.page_numbers(),
                instrument=name.instrument_id if name else None,
                number=name.number if name else None,
                key=key,
                key_assumed=key_assumed,
                clef=clef,
                clef_assumed=clef_assumed,
                confidence=segment.confidence,
                export=segment.export,
                duplicate_of=segment.duplicate_of,
                notes=segment.notes,
            )
        )

    # --- extras grouped by reason: {"blank": [2, 9], "cover": [1]}
    by_reason: dict[str, list[int]] = {}
    for page_number, reason in extra_pages:
        by_reason.setdefault(reason, []).append(page_number)
    extras = [Extra(pages=pages, reason=reason) for reason, pages in by_reason.items()]

    if "unassigned" in by_reason:
        warnings.append(
            f"{len(by_reason['unassigned'])} page(s) could not be assigned to a part: "
            f"{format_page_ranges(by_reason['unassigned'])}"
        )
    if part_segments and not names_on_every_page:
        warnings.append(
            "this PDF prints part names only on the first page of each part: "
            "the following pages were added to the part before them"
        )
    if part_segments and not start_mode:
        warnings.append("no part title pages found: parts were split where the header name changes")

    return parts, extras, warnings
