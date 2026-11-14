"""
metadata.py: find the composition title, composer and arranger.

Pipeline position:  page_analyzer + classify -> [metadata] -> result

The title and credits are printed on the cover, on the first score page and
on the first page of EVERY part. OCR may misread them on some pages, so
we collect a candidate from every such page and take a VOTE: the spelling
that appears most often wins.

    Title    = the biggest text in the header (that isn't a part name or credit)
    Arranger = text after "arr.", "Arranged by", "Arranger:", "Aranžman:", ...
    Composer = text after "Composer:", "Music by", "Glazba:", ... or, when
               there is no such label, the right-aligned line just above the
               arranger (the usual layout: composer above arranger).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .classify import CREDIT_PATTERN, find_credit_label, is_label_case, is_sentence
from .instruments import find_part_name
from .models import PageFacts, PageInfo, TextSpan

# "arr. Peter Kleine Schaars" -> group 1 = "Peter Kleine Schaars".
# (?:...) is a group that is NOT captured; (.*) captures the rest.
ARRANGER_PATTERN = re.compile(
    r"(?:\barr(?:anged)?\b\.?(?:\s+by)?|\barrangement(?:\s+by)?|\barranger|"
    r"\barrangiert(?:\s+von)?|\bbearb(?:eitung|\.)?(?:\s+von)?|\baran[žz]man\w*|"
    r"\bobradio|\bobrada|\borchestrated\s+by)\s*[:\-–]?\s*(.*)$",
    re.IGNORECASE,
)

COMPOSER_PATTERN = re.compile(
    r"(?:\bcomposer|\bcomposed\s+by|\bwords\s*(?:&|and)\s*music\s+by|\bmusic\s+by|"
    r"\bmusic\s*:|\bglazba|\bskladatelj|\bmuzika|\bkomponist|\bmusik\s*:)\s*[:\-–]?\s*(.*)$",
    re.IGNORECASE,
)

# Things that are never a title: tempo marks, copyright lines, publisher names.
NOT_A_TITLE = re.compile(
    r"(copyright|©|\(c\)|edition|publishing|lentamente|moderato|allegro|andante|"
    r"adagio|tempo|rubato|largo|presto|=\s*\d)",
    re.IGNORECASE,
)

# A credit value needs at least this many letters to count (filters OCR noise).
MIN_NAME_LETTERS = 4

# ...and is at most this long: longer text is a sentence (program notes, a
# biography: "Following his senior education, Jacob de Haan completed ..."),
# not a name.
MAX_NAME_LENGTH = 60
MAX_NAME_WORDS = 8

# Embedded text is exact, OCR is not: a title/name read from embedded text
# counts this many times in the vote.
TEXT_VOTE_WEIGHT = 3

# An unlabelled composer name printed on the right side of a title page
# ("Jacob de Haan" under the title) must start right of this (share of width).
UNLABELLED_COMPOSER_MIN_X = 0.55

# Old march cards print the composer on the LEFT ("MARKO POKLEPOVIĆ") and the
# part name on the right. A name there must end left of this.
UNLABELLED_COMPOSER_LEFT_MAX_X = 0.45

# An unlabelled name found only on the left side must be seen on at least
# this many PAGES (not vote weight: one text page weighs TEXT_VOTE_WEIGHT
# but is still one page), so one stray word pair isn't taken as the composer.
LEFT_COMPOSER_MIN_PAGES = 2

# OCR often drops accents ("POKLEPOVIĆ" -> "POKLEPOVIC") but rarely adds
# them. So when both spellings were seen, the accented one wins if it was
# read on at least this many PAGES (not vote weight, as above).
ACCENTED_SPELLING_MIN_PAGES = 2


def nice_case(text: str) -> str:
    """
    Turn SHOUTED words into normal capitalisation; leave the rest alone.

    "HAUSTOR LIVE"                    -> "Haustor Live"
    "Dennis MARIEV"                   -> "Dennis Mariev"
    "Darko RUNDEK and Srdan SACHER"   -> "Darko Rundek and Srdan Sacher"
    "Peter Kleine Schaars"            -> unchanged
    """
    words = []
    for word in text.split():
        if len(word) > 1 and word.isupper():
            word = word.capitalize()
        words.append(word)
    return " ".join(words)


def _clean_value(text: str) -> str:
    """Trim punctuation/noise around a captured name."""
    return text.strip(" .,:;-–|\"'")


def _is_name_sized(text: str) -> bool:
    """Long enough to be a name, short enough not to be a sentence."""
    return (
        _letters(text) >= MIN_NAME_LETTERS
        and len(text) <= MAX_NAME_LENGTH
        and len(text.split()) <= MAX_NAME_WORDS
    )


def _letters(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def _span_below(span: TextSpan, spans: list[TextSpan]) -> TextSpan | None:
    """The closest span directly below `span` that overlaps it horizontally."""
    below = [
        other
        for other in spans
        if other.y0 >= span.y1 - 0.002 and other.x0 < span.x1 and other.x1 > span.x0
        and other is not span
    ]
    # min(..., key=...) picks the one with the smallest vertical distance.
    return min(below, key=lambda other: other.y0 - span.y1, default=None)


def _span_above_right_aligned(span: TextSpan, spans: list[TextSpan]) -> TextSpan | None:
    """The closest span above `span` whose right edge lines up with it."""
    above = [
        other
        for other in spans
        if other.y1 <= span.y0 + 0.002 and abs(other.x1 - span.x1) < 0.05 and other is not span
    ]
    return min(above, key=lambda other: span.y0 - other.y1, default=None)


def _credit_value(spans: list[TextSpan], pattern: re.Pattern) -> tuple[str, TextSpan] | None:
    """
    Find a credit like "arr. NAME" in the spans. If the label stands alone
    ("Words & Music by"), the name is taken from the line below it.
    Returns (name, the span that had the label) or None.
    """
    for span in spans:
        # A credit is a short line. In a sentence of program notes, "...his
        # many compositions in symphonic/modern arrangement (Jazz Waltz..."
        # is not an arranger credit.
        if is_sentence(span.text):
            continue
        match = pattern.search(span.text)
        # The label is the matched text before the name (group 1): it must
        # be written like a label, not "aRR" inside OCR noise.
        if not match or not is_label_case(span.text[match.start() : match.start(1)]):
            continue
        value = _clean_value(match.group(1))
        if _letters(value) < MIN_NAME_LETTERS:
            below = _span_below(span, spans)
            value = _clean_value(below.text) if below else ""
        if _is_name_sized(value):
            return value, span
    return None


def _looks_like_person(text: str) -> bool:
    """
    "Jacob de Haan", "Ennio Morricone": 2-5 words, no digits, at least two
    words starting with a capital letter, and not a title/tempo/part name.
    """
    words = text.split()
    if not 2 <= len(words) <= 5 or any(char.isdigit() for char in text):
        return False
    if sum(1 for word in words if word[:1].isupper()) < 2:
        return False
    # Every word needs a letter: "I N VRN §" (OCR noise from the notes) isn't a name.
    if not all(any(char.isalpha() for char in word) for word in words):
        return False
    if CREDIT_PATTERN.search(text) or NOT_A_TITLE.search(text) or find_part_name(text):
        return False
    return _letters(text) >= 6 and _letters(text) >= 0.8 * len(text.replace(" ", ""))


def _unlabelled_composer(
    spans: list[TextSpan], title: str | None
) -> tuple[str, str] | None:
    """
    A person's name in the title block, without a label: on the right
    ("Jacob de Haan" under "ROSS ROY"), or else on the left (old march cards:
    "MARKO POKLEPOVIĆ" left, title in the middle, part name right).
    The one nearest the top wins.

    Returns (name, side) with side "right" or "left", or None.
    """
    people = [
        span
        for span in spans
        if span.text != title and _looks_like_person(_clean_value(span.text))
    ]
    right = [span for span in people if span.x0 >= UNLABELLED_COMPOSER_MIN_X]
    left = [span for span in people if span.x1 <= UNLABELLED_COMPOSER_LEFT_MAX_X]
    for side, candidates in (("right", right), ("left", left)):
        if candidates:
            return _clean_value(min(candidates, key=lambda span: span.y0).text), side
    return None


def _is_upright(span: TextSpan) -> bool:
    """
    True for text printed upright, along the edge of the page: its box is
    taller than it is wide. On the cornfield cover the publisher's name "de
    haske muziekuitgave" runs from bottom to top. OCR measures the letter size
    along the line, so that name looked 168 pt tall and won the title vote.
    (A normal line of 4+ letters is always wider than tall.)
    """
    return (span.y1 - span.y0) > (span.x1 - span.x0)


def _title_candidate(spans: list[TextSpan]) -> str | None:
    """The biggest text among the spans that could be a title."""
    # (text, size) pairs. The text can be a PART of the span, see below.
    candidates: list[tuple[str, float]] = []
    for span in spans:
        if _is_upright(span):
            continue  # a publisher's name along the page edge, not the title
        # A stray dash or colon at either end is OCR noise ("CORNFIELD ROCK -").
        text = span.text.strip(" -–:,|")
        # The title and the arranger on one line ("DEEP PURPLE MEDLEY arr.
        # Toshihiko Sahashi", Deep Purple): the words before the credit can
        # still be the title. A line that STARTS with the credit ("Music by
        # ...") leaves nothing and is skipped below.
        credit = find_credit_label(text)
        if credit:
            text = text[: credit.start()].strip(" -–:,")
        letters = _letters(text)
        if letters < 4 or letters < 0.6 * len(text.replace(" ", "")):
            continue  # too short, or mostly symbols (OCR noise)
        if NOT_A_TITLE.search(text):
            continue
        if find_part_name(text):
            continue  # that's a part name, not the title
        candidates.append((text, span.size))
    if not candidates:
        return None
    # The biggest text; [0] = its text.
    return max(candidates, key=lambda candidate: candidate[1])[0]


def _vote(votes: list[tuple[str, int]], min_pages: int = 1) -> str | None:
    """
    The value with the most votes. Each vote is (value, weight) and comes
    from one page (a page adds at most one vote to each list, see
    _read_title_page). Values are compared loosely (case and punctuation
    ignored), and the winner is returned in its most-voted original spelling.

    The WEIGHT picks the winner (embedded text is more reliable than OCR),
    but the winner must have been read on at least `min_pages` PAGES to
    count (else None). One text page weighs TEXT_VOTE_WEIGHT, but it is
    still only one page: one noisy line must not count as "read on 3 pages".
    """
    if not votes:
        return None

    totals: Counter = Counter()  # loose value -> total weight
    pages: Counter = Counter()  # loose value -> number of votes (= pages)
    for value, weight in votes:
        totals[_loose(value)] += weight
        pages[_loose(value)] += 1
    winner_key = totals.most_common(1)[0][0]
    if pages[winner_key] < min_pages:
        return None

    spellings: Counter = Counter()  # exact spelling -> total weight
    spelling_pages: Counter = Counter()  # exact spelling -> number of pages
    for value, weight in votes:
        if _loose(value) == winner_key:
            spellings[value] += weight
            spelling_pages[value] += 1
    # OCR drops accents more often than it adds them, so an accented spelling
    # read on several pages beats a more frequent plain one:
    # "POKLEPOVIĆ" (6 pages) beats "POKLEPOVIC" (29 pages).
    # "Several" means PAGES, like min_pages above: one text page weighs 3
    # but is still one page.
    accented = [
        value
        for value, _weight in spellings.most_common()
        if _accent_count(value) > 0 and spelling_pages[value] >= ACCENTED_SPELLING_MIN_PAGES
    ]
    if accented:
        return accented[0]
    return spellings.most_common(1)[0][0]


def _strip_accents(text: str) -> str:
    """'Poklepović' -> 'Poklepovic', 'Đavoli' -> 'Davoli'."""
    # NFKD splits "ć" into "c" + a separate accent mark; the marks are then
    # dropped. "đ"/"Đ" are letters of their own (not c + mark), so they are
    # replaced by hand.
    decomposed = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _accent_count(text: str) -> int:
    """How many letters of `text` carry an accent (ć, š, ž, đ, é ...)."""
    # NFD splits "ć" into "c" + accent mark, but (unlike NFKD) leaves
    # ligatures such as "ﬁ" alone, so they don't count as accents.
    return sum(
        1
        for char in text
        if char in "đĐ"
        or any(unicodedata.combining(part) for part in unicodedata.normalize("NFD", char))
    )


def _loose(value: str) -> str:
    """
    The vote key: case, punctuation and accents ignored, so "POKLEPOVIĆ",
    "Poklepovic" and "poklepović." all count as the same name.
    """
    # [\W_] = everything that is NOT a letter or digit. In Python 3, \w
    # includes letters like Š, Đ and Cyrillic, so "Šuma" stays "suma"
    # (with [^a-z0-9] it would become "uma"). casefold() is a stronger
    # lower() for comparing text in any language.
    return re.sub(r"[\W_]", "", _strip_accents(value).casefold())


@dataclass
class _Votes:
    """All votes collected so far. Each list holds (value, weight) pairs."""

    titles: list[tuple[str, int]] = field(default_factory=list)
    composers: list[tuple[str, int]] = field(default_factory=list)
    arrangers: list[tuple[str, int]] = field(default_factory=list)
    # Names without a label, by where they were printed.
    right_names: list[tuple[str, int]] = field(default_factory=list)
    left_names: list[tuple[str, int]] = field(default_factory=list)


def _read_title_page(page_facts: PageFacts, info: PageInfo, votes: _Votes) -> None:
    """Add the title/composer/arranger candidates of one page to `votes`."""
    # Pages without music (covers) are read in full; music pages only by
    # their header (the title block above the first staff).
    has_music = page_facts.staff_count > 0 or info.kind == "score"
    spans = page_facts.header if has_music else page_facts.body
    weight = TEXT_VOTE_WEIGHT if page_facts.source == "text" else 1

    title = _title_candidate(spans)
    if title:
        votes.titles.append((title, weight))

    arranger = _credit_value(spans, ARRANGER_PATTERN)
    if arranger:
        votes.arrangers.append((arranger[0], weight))

    composer = _credit_value(spans, COMPOSER_PATTERN)
    if composer:
        votes.composers.append((composer[0], weight))
    elif arranger:
        # No "Composer:" label: try the line right above the arranger.
        above = _span_above_right_aligned(arranger[1], spans)
        if above and _is_name_sized(above.text) and not find_part_name(above.text):
            votes.composers.append((_clean_value(above.text), weight))
    else:
        # No credits at all (an original composition): a person's name in
        # the title block is probably the composer.
        found = _unlabelled_composer(spans, title)
        if found:
            name, side = found
            target = votes.right_names if side == "right" else votes.left_names
            target.append((name, weight))


def _confirm_titles(facts: list[PageFacts], votes: _Votes, voted: set[int]) -> None:
    """
    Give a title candidate one more vote for every music page that prints it
    in its header (and did not vote already).

    The covers alone can be a tie: tullamore's front cover's biggest text is
    the publisher's series label "Concert Band", the title page says "LORD
    TULLAMORE" and the back cover "publications", one vote each, and the
    first one won. The real title is the one the music pages print on top
    (here on every part's first page), so those pages settle it.

    Only candidates that already have a vote count, and only a header line
    that IS the candidate (loosely: case and punctuation ignored), so a
    tempo or a part name on a music page never becomes the title.
    """
    candidates: dict[str, str] = {}  # loose key -> one spelling of it
    for value, _weight in votes.titles:
        key = _loose(value)
        # A candidate of symbols only ("* * *") has the key "", which every
        # dash or ornament line in a header has too: it must match nothing.
        if key:
            candidates.setdefault(key, value)
    if not candidates:
        return
    for page_facts in facts:
        if page_facts.page_number in voted or page_facts.staff_count == 0:
            continue  # this page voted already; a page without music isn't one
        # A set: a title printed twice on one page still counts once.
        # (_loose already drops the dashes and colons around a line.)
        printed = {_loose(span.text) for span in page_facts.header}
        for key, spelling in candidates.items():
            if key in printed:
                votes.titles.append((spelling, 1))


def extract_metadata(
    facts: list[PageFacts], infos: list[PageInfo], min_pages: int = 1
) -> dict[str, str | None]:
    """
    Return {"title": ..., "composer": ..., "arranger": ...} (values may be None).

    `min_pages`: a composer or arranger must be read on at least this many
    pages to count (pages, not vote weight: see _vote).

    `facts` and `infos` are the per-page lists, in the same page order.
    """
    votes = _Votes()
    voted: set[int] = set()  # the page numbers read for the votes
    # zip() walks two lists side by side: (facts[0], infos[0]), (facts[1], infos[1]) ...
    # strict=True: raise an error if the two lists ever have different lengths.
    for page_facts, info in zip(facts, infos, strict=True):
        if info.is_start or (info.kind == "unknown" and bool(page_facts.body)):
            _read_title_page(page_facts, info, votes)
            voted.add(page_facts.page_number)

    if not votes.titles:
        # No page looked like a title page: there was no "arr."/"Music by"
        # anywhere. Old march cards are like that: every part is one page
        # with just "MARKO POKLEPOVIĆ", the title and the part name on top.
        # Every part page then IS a first page, so read them all.
        votes = _Votes()
        voted = set()
        for page_facts, info in zip(facts, infos, strict=True):
            if info.kind == "part":
                _read_title_page(page_facts, info, votes)
                voted.add(page_facts.page_number)

    _confirm_titles(facts, votes, voted)
    title = _vote(votes.titles)
    # Unlabelled names from both sides vote together (one per page, the right
    # side preferred), so a stray right-side fragment on ONE page can't beat
    # a name printed on the left of every page. Left-side names count only
    # when seen on a few pages (see LEFT_COMPOSER_MIN_PAGES). Each page adds
    # at most one left name, so the number of votes IS the number of pages.
    unlabelled = list(votes.right_names)
    if len(votes.left_names) >= LEFT_COMPOSER_MIN_PAGES:
        unlabelled += votes.left_names
    composer = _vote(votes.composers, min_pages) or _vote(unlabelled, min_pages)
    arranger = _vote(votes.arrangers, min_pages)
    return {
        "title": nice_case(title) if title else None,
        "composer": nice_case(composer) if composer else None,
        "arranger": nice_case(arranger) if arranger else None,
    }
