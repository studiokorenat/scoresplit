"""
Tests for the page classification rules and the pixel helpers that were
added for real-world edge cases (text pages, tilted/thick-lined scans,
same-name parts with different music).

Pages and images are built by hand, so no PDF is needed.
"""

import warnings

import numpy as np

from scoresplit.classify import classify_page, fix_clef_sign_numbers, is_sentence
from scoresplit.image_analysis import (
    INK_THRESHOLD,
    LIGHT_SCAN_INK_THRESHOLD,
    _long_horizontal_runs,
    _staff_start,
    detect_clef,
    find_staves,
    find_staves_on_page,
    music_fingerprint,
    music_similarity,
)
from scoresplit.models import PageFacts, TextSpan
from scoresplit.segment import build_parts

BIOGRAPHY = (
    "Otto M. Schwarz was born in Neunkirchen, Austria, in 1967. He received his "
    "music education at the music academy in Vienna where he studied trumpet with "
    "professor Josef Pomberger of the Vienna Philharmonic Orchestra"
)


def test_biography_page_is_not_a_trumpet_part() -> None:
    # A text page; a photo on it was taken for 2 "staves".
    facts = PageFacts(
        page_number=3,
        source="text",
        staff_count=2,
        ink_ratio=0.1,
        header=[TextSpan(BIOGRAPHY, 0.2, 0.05, 0.9, 0.08, 9)],
        body=[TextSpan(BIOGRAPHY, 0.2, 0.05, 0.9, 0.08, 9)] * 3,
    )
    info = classify_page(facts)
    assert info.kind == "unknown"
    assert info.part is None
    assert info.has_staves is False  # so it becomes a cover page, not part of a part


def test_a_sentence_in_the_header_is_not_a_part_name() -> None:
    # A music page whose header has a sentence mentioning "trumpet" AND the
    # real part name: the real name wins.
    facts = PageFacts(
        page_number=10,
        source="text",
        staff_count=9,
        ink_ratio=0.1,
        header=[
            TextSpan("He studied trumpet with professor Josef Pomberger of the Vienna", 0.2, 0.02, 0.9, 0.04, 9),
            TextSpan("Oboe", 0.05, 0.06, 0.12, 0.08, 12),
        ],
    )
    assert classify_page(facts).part.display() == "Oboe"


def test_long_horizontal_runs_finds_lines_not_notes() -> None:
    ink = np.zeros((5, 100), dtype=bool)
    ink[1, 5:95] = True  # a staff line
    ink[3, 10:20] = True  # a short blob (a note head)
    runs = _long_horizontal_runs(ink, 60)
    assert runs[1, 5:95].all() and not runs[1, :5].any()
    assert not runs[3].any()


def test_staff_start_skips_a_thin_vertical_line() -> None:
    on_lines = np.zeros(200, dtype=bool)
    on_lines[20:22] = True  # the dark edge of a scan crossing the staff
    on_lines[80:200] = True  # the staff itself
    assert _staff_start(on_lines, 23) == 80


def _fake_music_page(seed: int) -> np.ndarray:
    """A white page with staff lines and random 'note heads'."""
    rng = np.random.default_rng(seed)
    gray = np.full((800, 600), 255, dtype=np.uint8)
    for staff_top in range(100, 700, 120):
        for line in range(5):
            gray[staff_top + line * 12, 40:560] = 0
        for x in rng.integers(60, 540, 12):
            y = staff_top + int(rng.integers(0, 48))
            gray[y : y + 8, x : x + 10] = 0
    return gray


def test_music_similarity_tells_copies_from_different_parts() -> None:
    page = _fake_music_page(1)
    rescanned = np.roll(np.roll(page, 9, axis=0), -6, axis=1)  # shifted on the scanner
    other_part = _fake_music_page(2)  # same layout, different notes
    fingerprint = music_fingerprint(page)
    assert music_similarity(fingerprint, music_fingerprint(rescanned)) > 0.9
    assert music_similarity(fingerprint, music_fingerprint(other_part)) < 0.6


def test_front_matter_without_music_is_not_a_score_page() -> None:
    # The "Instrumentation" page lists every instrument in its margin, and a
    # score's title page says "Full Score", but neither has music on it:
    # they are front matter (-> extra/), not part of the Conductor Score.
    names = ["Full Score", "Piccolo", "Flute 1", "Oboe", "Clarinet 1", "Horn 1", "Trumpet 1", "Timpani"]
    facts = PageFacts(
        page_number=2,
        source="text",
        staff_count=0,
        ink_ratio=0.05,
        header=[TextSpan("Full Score", 0.4, 0.05, 0.6, 0.08, 20)],
        margin=[TextSpan(name, 0.05, 0.2 + i * 0.03, 0.2, 0.22 + i * 0.03, 10) for i, name in enumerate(names)],
    )
    assert classify_page(facts).kind == "unknown"
    facts.staff_count = 20  # the same labels next to music: a score page
    assert classify_page(facts).kind == "score"


def test_score_cover_with_a_publisher_logo_is_not_a_shaker_part() -> None:
    # Ross Roy page 1 (eng+hrv OCR): the "De Haske" logo was read as
    # "de hasker", 83% like "shaker"; the page also says "full score".
    lines = ["ROSS ROY", "Overture for Band", "Jacob de Haan", "full score concert band / harmonie", "7‘ de hasker"]
    facts = PageFacts(
        page_number=1,
        source="ocr",
        staff_count=0,
        ink_ratio=0.05,
        body=[TextSpan(text, 0.1, 0.1 + i * 0.1, 0.6, 0.15 + i * 0.1, 12) for i, text in enumerate(lines)],
    )
    assert classify_page(facts).part is None
    # Even without "full score", a loose match is not a cover sheet's name.
    facts.body = [span for span in facts.body if "score" not in span.text]
    assert classify_page(facts).part is None


# --- Deep Purple: one long header line, clef sign read as "6" -------------


def _first_page_with_header(line: str) -> PageFacts:
    """A scanned first page whose text layer merged name, title and arranger."""
    return PageFacts(
        page_number=52,
        source="text",
        staff_count=9,
        ink_ratio=0.07,
        header=[TextSpan(line, 0.09, 0.07, 0.95, 0.09, 30)],
    )


def test_long_header_line_without_small_words_is_not_a_sentence() -> None:
    # 10 words, but no running text: the part name must still be read.
    line = "Eb Alto Saxophone 1-2 DEEP PURPLE MEDLEY arr. Toshihiko Sahashi"
    assert not is_sentence(line)
    info = classify_page(_first_page_with_header(line))
    assert info.kind == "part"
    assert info.part is not None and info.part.display() == "Alto Saxophone 1-2 in Eb"
    assert info.is_start  # "arr." = first page
    # Real prose is still a sentence.
    assert is_sentence(BIOGRAPHY)


def test_treble_clef_read_as_6_becomes_the_clef() -> None:
    info = classify_page(
        _first_page_with_header("Bb Baritone / Euphonium 6 DEEP PURPLE MEDLEY arr. T. Sahashi")
    )
    fix_clef_sign_numbers([info])
    assert info.part is not None
    assert info.part.display() == "Baritone/Euphonium in Bb"
    assert info.part.wording == "Bb Baritone/Euphonium"
    assert info.part.number is None
    assert info.part.clef == "treble"


def test_a_real_part_6_is_kept_when_part_5_exists() -> None:
    infos = [
        classify_page(_first_page_with_header("Horn 5 in F")),
        classify_page(_first_page_with_header("Horn 6 in F")),
    ]
    fix_clef_sign_numbers(infos)
    assert [info.part.display() for info in infos if info.part] == ["Horn 5 in F", "Horn 6 in F"]


def test_light_scan_staves_are_found_with_the_lighter_limit() -> None:
    # A staff drawn in grey (160): invisible at the normal ink limit.
    page = np.full((400, 400), 255, dtype=np.uint8)
    for y in (100, 110, 120, 130, 140):
        page[y, 20:380] = 160
    assert find_staves(page) == []
    assert len(find_staves(page, threshold=LIGHT_SCAN_INK_THRESHOLD)) == 1


# --- Sedona: "Conductor" in a dedication on every part's first page --------


def _sedona_first_page(dedication: str) -> PageFacts:
    """A part's first page (text PDF) with a dedication line in the header."""
    return PageFacts(
        page_number=29,
        source="text",
        staff_count=12,
        ink_ratio=0.08,
        header=[
            TextSpan("Commissioned by and dedicated to the Kettering (Ohio) Civic Band on its 40th Anniversary,", 0.2, 0.017, 0.8, 0.03, 8),
            TextSpan(dedication, 0.35, 0.03, 0.65, 0.04, 8),
            TextSpan("1 Clarinet in Bb", 0.05, 0.037, 0.25, 0.05, 15),
            TextSpan("SEDONA", 0.45, 0.052, 0.55, 0.07, 17),
        ],
    )


def test_conductor_in_a_dedication_does_not_make_a_score_page() -> None:
    info = classify_page(_sedona_first_page("Paul Sharle - Founder/Conductor"))
    assert info.kind == "part"
    assert info.part.display() == "Clarinet 1 in Bb"
    assert info.is_start is True  # "Commissioned" is a credit


def test_score_keyword_loses_to_a_clear_part_name_without_margin_labels() -> None:
    # Safety net: even a keyword the role rule doesn't catch ("Conductor" on
    # its own line) can't turn a page with a clear part name, no instrument
    # labels in the margin and part-like staves into a score page.
    facts = _sedona_first_page("Conductor")
    assert classify_page(facts).kind == "part"
    # With instrument labels in the margin it IS a score page.
    labels = ["Picc.", "Fl.", "Ob.", "Cl.", "Tpt.", "Hn.", "Tbn."]
    facts.margin = [TextSpan(name, 0.02, 0.2 + i * 0.05, 0.08, 0.22 + i * 0.05, 9) for i, name in enumerate(labels)]
    assert classify_page(facts).kind == "score"


# --- a key the instrument is never written in -----------------------------------


def _part_page(page_number: int, header: str, source: str) -> PageFacts:
    """A page of music with just a part name in its header."""
    return PageFacts(
        page_number=page_number,
        source=source,
        staff_count=9,
        ink_ratio=0.07,
        header=[TextSpan(header, 0.09, 0.07, 0.40, 0.09, 14)],
    )


def test_key_the_instrument_never_has_becomes_the_default_key() -> None:
    # Funk Attack (scanned): "Bass Clarinet in Bb" was read as "in Eb",
    # and a bass clarinet is never written in Eb.
    info = classify_page(_part_page(39, "Bass Clarinet in Eb", "ocr"))
    assert (info.part.key, info.part.unlisted_key) == (None, "Eb")
    parts, _, _ = build_parts([info])
    # The printed name holds the misread key, so the recognised name is
    # shown, without the assumed key (that is in the key column).
    assert parts[0].name == "Bass Clarinet"
    assert (parts[0].key, parts[0].key_assumed) == ("Bb", True)
    assert parts[0].confidence <= 0.8  # yellow in the table
    assert "key read as Eb, but Bass Clarinet isn't written in Eb: assumed Bb, please check" in parts[0].notes


def test_unlisted_key_loses_to_a_key_read_on_another_page() -> None:
    pages = [
        classify_page(_part_page(1, "Bass Clarinet in Eb", "ocr")),
        classify_page(_part_page(2, "Bass Clarinet in A", "ocr")),
    ]
    pages[0].is_start = True
    parts, _, _ = build_parts(pages)
    assert parts[0].name == "Bass Clarinet in A"
    assert not any("key read as" in note for note in parts[0].notes)


def test_text_pdfs_get_the_same_rule() -> None:
    # Many "text" PDFs are scans with the scanner's own OCR text layer
    # (Funk Attack), so the rule applies to them too.
    info = classify_page(_part_page(39, "Bass Clarinet in Eb", "text"))
    assert (info.part.key, info.part.unlisted_key) == (None, "Eb")
    parts, _, _ = build_parts([info])
    assert parts[0].key == "Bb"


def test_an_instrument_without_keys_keeps_a_printed_key() -> None:
    # No `keys` list to compare with: "Violin in Bb" keeps its key.
    info = classify_page(_part_page(1, "Violin in Bb", "text"))
    assert info.part.key == "Bb"
    parts, _, _ = build_parts([info])
    assert parts[0].key == "Bb"
    assert "unusual key for Violin (Bb): please check" in parts[0].notes


def test_a_misread_key_on_a_bass_assumes_no_key() -> None:
    # A bass is never written in A, so "Bass in A" is a misreading; the bass
    # has no default key (it is generic), so the key stays empty.
    info = classify_page(_part_page(1, "Bass in A", "ocr"))
    assert (info.part.key, info.part.unlisted_key) == (None, "A")
    parts, _, _ = build_parts([info])
    assert parts[0].key is None
    assert parts[0].notes == ["key read as A, but Bass isn't written in A: please check"]


# ---------------------------------------------------------------------------
# Round 9.3: part names over two lines, "doubles" notes and abbreviations
# in the header. Positions as on the italiano pages (two lines at x=0.057).
# ---------------------------------------------------------------------------


def _header_page(lines: list[str]) -> PageFacts:
    """A page of music whose header has these lines, one under the other."""
    header = [
        TextSpan(line, 0.057, 0.06 + i * 0.018, 0.25, 0.076 + i * 0.018, 13)
        for i, line in enumerate(lines)
    ]
    return PageFacts(page_number=1, source="ocr", staff_count=9, ink_ratio=0.07, header=header)


def test_a_name_over_two_lines_is_read_as_one() -> None:
    # "1st Bb TENOR" alone is the Tenor Horn; with the line below it is
    # the Tenor Saxophone, also though the top line has a number.
    info = classify_page(_header_page(["1st Bb TENOR", "SAXOPHONE"]))
    assert info.part.instrument_id == "tenor_saxophone"
    assert (info.part.number, info.part.key) == ("1", "Bb")
    info = classify_page(_header_page(["Bb BASS", "CLARINET"]))
    assert info.part.display() == "Bass Clarinet in Bb"


def test_a_doubles_line_is_not_part_of_the_name() -> None:
    info = classify_page(_header_page(["Eb CLARINET", "doubles Flute"]))
    assert info.part.display() == "Clarinet in Eb"
    info = classify_page(_header_page(["Eb TRUMPET /", "CORNET", "doubles 1st Trumpet / Cornet"]))
    assert (info.part.key, info.part.wording) == ("Eb", "Eb Trumpet/Cornet")


def test_a_list_of_instruments_on_two_lines_is_still_a_list() -> None:
    info = classify_page(_header_page(["Mallets", "Xylophone"]))
    assert info.part.label == "Mallets, Xylophone"


def test_an_abbreviation_counts_in_a_header_only_with_a_key() -> None:
    # "COR Eb": a key on the same line makes it a name line...
    info = classify_page(_header_page(["COR Eb"]))
    assert (info.kind, info.part.instrument_id) == ("part", "horn")
    # ...a lone abbreviation is still taken for OCR noise.
    assert classify_page(_header_page(["Cor."])).kind == "unknown"


# ---------------------------------------------------------------------------
# Round 9.4: catalogue pages
# ---------------------------------------------------------------------------


def _text_page(header: str, line_count: int) -> PageFacts:
    """A page without staves: `header` on top, then `line_count` lines of text."""
    lines = [TextSpan(header, 0.04, 0.02, 0.5, 0.04, 12)] + [
        TextSpan(f"EMR {10000 + i} Some Piece", 0.04, 0.05 + i * 0.009, 0.4, 0.058 + i * 0.009, 8)
        for i in range(line_count - 1)
    ]
    return PageFacts(
        page_number=2, source="text", staff_count=0, ink_ratio=0.09, header=lines[:1], body=lines
    )


def test_a_catalogue_page_is_no_part() -> None:
    # Italiano page 2: a discography that names "Fighting Trombones".
    facts = _text_page("Fighting Trombones (Schiltknecht)", 103)
    info = classify_page(facts)
    assert (info.kind, info.has_staves) == ("unknown", False)
    assert info.note == "text page (catalogue or list)"


def test_a_part_without_staves_and_some_text_is_still_a_part() -> None:
    # Sedona page 66: a percussion page whose staves weren't found, 63 lines.
    info = classify_page(_text_page("Percussion", 63))
    assert (info.kind, info.part.instrument_id) == ("part", "percussion")


# ---------------------------------------------------------------------------
# Round 9.6: staves and clefs on light and hand-copied scans
# ---------------------------------------------------------------------------
#
# The pages below are drawn by hand: a white page, staff lines from x=40 to
# x=560 (spacing 12 pixels), and a "clef" made of a few rectangles starting
# at x=50. Gray values: 0 = black ink, 160 = the grey of a light scan (only
# ink at the lighter LIGHT_SCAN_INK_THRESHOLD), 255 = white paper.

SPACING = 12
CLEF_X = 50


def _page(height: int = 400) -> np.ndarray:
    return np.full((height, 600), 255, dtype=np.uint8)


def _draw_staff(page: np.ndarray, top: int, gray: int = 0) -> list[int]:
    """Draw 5 staff lines (2 pixels thick) from row `top`; return their rows."""
    rows = [top + i * SPACING for i in range(5)]
    for row in rows:
        page[row : row + 2, 40:560] = gray
    return rows


def test_light_scan_with_a_few_dark_staves_is_read_with_the_lighter_limit() -> None:
    # Mission pages 56-57: one staff is dark enough for the normal limit,
    # the other three are grey. Retry, and keep the four staves.
    page = _page(800)
    _draw_staff(page, 100, gray=0)
    for top in (250, 400, 550):
        _draw_staff(page, top, gray=160)
    assert len(find_staves(page)) == 1
    staves, threshold = find_staves_on_page(page)
    assert (len(staves), threshold) == (4, LIGHT_SCAN_INK_THRESHOLD)


def test_a_page_with_two_real_staves_keeps_the_normal_limit() -> None:
    # A short coda: two staves, both black. The retry finds nothing more.
    page = _page(800)
    _draw_staff(page, 100)
    _draw_staff(page, 250)
    staves, threshold = find_staves_on_page(page)
    assert (len(staves), threshold) == (2, INK_THRESHOLD)


def test_a_faint_ghost_line_above_the_staff_is_left_out() -> None:
    # Suite page 73: a line shining through from the back of the paper, one
    # space above the staff and evenly spaced with it. Taking the first five
    # lines would shift the whole staff up by one line.
    page = _page()
    page[100 - SPACING : 100 - SPACING + 2, 40:180] = 0  # ghost: 2/3 of the search band
    rows = _draw_staff(page, 100)
    [(lines, _spacing)] = find_staves(page)
    # A 2-pixel line at row r has its centre at r + 0.5.
    assert lines == [row + 0.5 for row in rows]


def _treble(page: np.ndarray, top: int) -> None:
    """A treble clef: one stroke from well above the staff to below it."""
    page[top - int(1.5 * SPACING) : top + 4 * SPACING + int(1.2 * SPACING), CLEF_X : CLEF_X + 3] = 0


def test_a_treble_clef_is_one_stroke_out_of_the_staff() -> None:
    page = _page()
    _draw_staff(page, 100)
    _treble(page, 100)
    staves, threshold = find_staves_on_page(page)
    assert detect_clef(page, staves, threshold) == "treble"


def test_a_staff_at_the_left_edge_of_the_page_gets_its_clef() -> None:
    # Buffalo Dances pages 41 and 104: the scan is cut so tightly that the
    # staff lines start in column 0. The clef area starts a little left of
    # the staff, at column -1, and in Python ink[:, -1:26] means "from the
    # LAST column to 26": an empty slice, whose mean is NaN (with a
    # "Mean of empty slice" warning).
    page = _page()
    top = 100
    for i in range(5):
        page[top + i * SPACING : top + i * SPACING + 2, 0:560] = 0
    # A treble clef that starts right at the edge: a stroke from above the
    # staff into its top two spaces, and its tail below the staff. (One
    # stroke through all four spaces would look like a barline.)
    page[top - int(1.5 * SPACING) : top + 26, 0:3] = 0
    page[top + 4 * SPACING + 5 : top + 4 * SPACING + 19, 0:3] = 0
    staves, threshold = find_staves_on_page(page)
    # catch_warnings() puts the warning settings back afterwards;
    # simplefilter("error") turns every warning inside into an error.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detect_clef(page, staves, threshold) == "treble"


def test_a_rehearsal_mark_above_a_clef_is_no_treble_clef() -> None:
    # Suite page 72: a circled letter above a bass clef and a dynamic below.
    # Ink above AND below, but with white between the mark and the staff.
    page = _page()
    top = 100
    _draw_staff(page, top)
    page[top + 4 : top + 2 * SPACING, CLEF_X : CLEF_X + 8] = 0  # the clef's body
    page[top - 19 : top - 8, CLEF_X : CLEF_X + 20] = 0  # the rehearsal mark
    page[top + 4 * SPACING + 5 : top + 4 * SPACING + 17, CLEF_X + 5 : CLEF_X + 8] = 0  # "f"
    staves, threshold = find_staves_on_page(page)
    assert detect_clef(page, staves, threshold) is None


def _light_bass_page(stroke_above: bool) -> np.ndarray:
    """
    A light scan (grey staff lines) with a hand-drawn bass clef whose two
    dots run into the staff lines: no white gap around them. With
    `stroke_above`, the "dots" reach up above the staff like a sharp (#).
    """
    page = _page()
    top = 100
    _draw_staff(page, top, gray=160)
    page[top + 2 : top + 3 * SPACING, CLEF_X : CLEF_X + 12] = 60  # the clef's body
    dots_x = CLEF_X + int(2.3 * SPACING)
    dots_from = top - int(1.3 * SPACING) if stroke_above else top
    page[dots_from : top + 2 * SPACING, dots_x : dots_x + 4] = 60  # dots touching the lines
    return page


def test_bass_dots_touching_the_lines_on_a_light_scan() -> None:
    page = _light_bass_page(stroke_above=False)
    staves, threshold = find_staves_on_page(page)
    assert threshold == LIGHT_SCAN_INK_THRESHOLD
    assert detect_clef(page, staves, threshold) == "bass"


def test_a_stroke_reaching_above_the_staff_is_no_bass_dot() -> None:
    page = _light_bass_page(stroke_above=True)
    staves, threshold = find_staves_on_page(page)
    assert detect_clef(page, staves, threshold) is None


def test_dots_without_a_white_gap_count_only_on_a_light_scan() -> None:
    # On a normal page the stem of a flat can look like that (Deep Purple
    # page 64), so there the dots must float in their spaces.
    page = _light_bass_page(stroke_above=False)
    staves, _threshold = find_staves_on_page(page)
    assert detect_clef(page, staves, INK_THRESHOLD) != "bass"


def _fat_dots_page(upper_dot_rows: tuple[int, int]) -> np.ndarray:
    """
    A black bass clef whose lower dot is fat and touches the middle line
    above it (superstar page 48). `upper_dot_rows`: where the upper "dot"
    is, counted from the top staff line.
    """
    page = _page()
    top = 100
    _draw_staff(page, top)
    page[top + 2 : top + 3 * SPACING, CLEF_X : CLEF_X + 12] = 0  # the clef's body
    dots_x = CLEF_X + int(2.3 * SPACING)
    first, last = upper_dot_rows
    page[top + first : top + last, dots_x : dots_x + 4] = 0  # upper dot
    page[top + 13 : top + 21, dots_x : dots_x + 4] = 0  # lower dot, touching line 2
    return page


def test_fat_bass_dots_touching_the_line_between_them() -> None:
    # The upper dot fills its space down to the middle line too: only the
    # gaps above the upper dot and below the lower dot are white.
    page = _fat_dots_page(upper_dot_rows=(4, 13))
    staves, threshold = find_staves_on_page(page)
    assert threshold == INK_THRESHOLD
    assert detect_clef(page, staves, threshold) == "bass"


def test_ink_at_the_edge_of_the_top_space_is_no_bass_dot() -> None:
    # Deep Purple page 45: a notehead in the second space, and a slur that
    # only dips into the top edge of the first space. Not a clef's dots.
    page = _fat_dots_page(upper_dot_rows=(3, 5))
    staves, threshold = find_staves_on_page(page)
    assert detect_clef(page, staves, threshold) is None


def test_one_line_percussion_staves_give_no_clef() -> None:
    # Mission page 84: five one-line staves, evenly spaced far apart, are
    # found as one giant "staff". It counts as music, but has no clef.
    page = _page(1200)
    for i in range(5):
        page[300 + i * 100 : 302 + i * 100, 40:560] = 0
    page[200:850, CLEF_X : CLEF_X + 30] = 0  # ink that would read as a treble clef
    staves, threshold = find_staves_on_page(page)
    assert len(staves) == 1
    assert detect_clef(page, staves, threshold) is None


def test_a_key_split_off_by_ocr_joins_the_name_on_its_line() -> None:
    # cornfield page 43: "C - Trombone I" came out as "C" and "Trombone |",
    # 0.03 apart on one line (the dash between them was not read).
    page = PageFacts(
        page_number=43,
        source="ocr",
        staff_count=12,
        header=[
            TextSpan("C", x0=0.08, y0=0.092, x1=0.09, y1=0.099, size=5.8),
            TextSpan("Trombone |", x0=0.12, y0=0.092, x1=0.20, y1=0.099, size=6.5),
            TextSpan("CORNFIELD ROCK", x0=0.30, y0=0.075, x1=0.73, y1=0.099, size=19.4),
        ],
    )
    assert classify_page(page).part.display() == "Trombone 1 in C"
    # A lone letter far away (a rehearsal letter) is not joined.
    page.header[0] = TextSpan("C", x0=0.60, y0=0.092, x1=0.61, y1=0.099, size=5.8)
    assert classify_page(page).part.display() == "Trombone 1"


def _framed_label_page() -> np.ndarray:
    """A white 'page' at 100 dpi with a thin, scanned-looking frame near the top."""
    page = np.full((400, 600), 255, dtype=np.uint8)
    # The frame: 1.0 x 0.25 inch, its lines broken up every 15 pixels.
    top, bottom, left, right = 40, 65, 50, 150
    page[top, left:right] = 90
    page[bottom, left:right] = 90
    page[top:bottom, left] = 90
    page[top:bottom, right - 1] = 90
    page[top, left + 15 : right : 15] = 255  # gaps in the scanned line
    page[bottom, left + 7 : right : 15] = 255
    # Some "text" inside, and a long line far below (like a staff line).
    page[50:56, 70:120] = 0
    page[300, 20:580] = 0
    return page


def test_a_frame_in_the_header_is_found() -> None:
    from scoresplit.image_analysis import find_text_boxes

    page = _framed_label_page()
    boxes = find_text_boxes(page, bottom=200, dpi=100)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    # The inside of the frame, a few pixels in from its lines.
    assert 50 / 600 < x0 < 60 / 600 and 140 / 600 < x1 < 150 / 600
    assert 40 / 400 < y0 < 50 / 400 and 55 / 400 < y1 < 65 / 400


def test_lines_without_a_frame_are_no_box() -> None:
    from scoresplit.image_analysis import find_text_boxes

    page = np.full((400, 600), 255, dtype=np.uint8)
    page[40, 50:150] = 0  # two lines, but no sides joining them
    page[65, 50:150] = 0
    assert find_text_boxes(page, bottom=200, dpi=100) == []


def test_a_small_name_next_to_a_big_title_is_its_own_text_line() -> None:
    # Cornfield page 38: "F Horn I" on the left shares its rows with the big
    # title "CORNFIELD ROCK"; the composer is below the title. At 200 dpi.
    from scoresplit.image_analysis import find_text_lines

    page = np.full((1000, 1600), 255, dtype=np.uint8)
    for left, right in ((100, 115), (130, 190), (205, 210)):  # "F", "Horn", "I"
        page[150:168, left:right] = 0
    page[120:175, 480:1180] = 0  # the title: 55 pixels = 0.28 inch high
    page[185:210, 940:1185] = 0  # "Jacob de Haan"
    page[60:62, 30:32] = 0  # a speck of dust
    lines = find_text_lines(page, bottom=300, dpi=200)
    # As pixels again, to compare (the margin is half a line height).
    boxes = [
        (round(x0 * 1600), round(y0 * 1000), round(x1 * 1600), round(y1 * 1000))
        for x0, y0, x1, y1 in lines
    ]
    assert boxes == [(91, 141, 219, 177), (928, 173, 1197, 222)]


def test_a_key_on_the_second_line_of_two_names_counts() -> None:
    # superstar 77: "BARITON" / "1e TUBA Bb (BASSE SOLO)": the first name is
    # the instrument, the key printed below it is the part's key.
    info = classify_page(_header_page(["BARITON", "1e TUBA Bb (BASSE SOLO)"]))
    assert (info.part.instrument_id, info.part.key) == ("baritone", "Bb")


def test_a_cue_inside_the_music_does_not_name_a_page() -> None:
    # tullamore page 92: one-line percussion staves weren't found, so the
    # header reached down into the music and held the cue "Triangel". The
    # page has no header of its own: it gets no name and continues the part
    # before it.
    page = PageFacts(
        page_number=92,
        source="ocr",
        staff_count=3,
        ink_ratio=0.08,
        header=[
            TextSpan("J", x0=0.2, y0=0.108, x1=0.22, y1=0.12, size=10),
            TextSpan("Tr iange l", x0=0.62, y0=0.258, x1=0.7, y1=0.27, size=9),
        ],
    )
    info = classify_page(page)
    assert info.part is None


def test_a_name_lower_down_counts_on_a_page_with_credits() -> None:
    # A first page with a tall title block: the credits say it IS a header.
    page = PageFacts(
        page_number=1,
        source="ocr",
        staff_count=10,
        ink_ratio=0.08,
        header=[
            TextSpan("Arr. Jan de Vries", x0=0.7, y0=0.12, x1=0.9, y1=0.14, size=9),
            TextSpan("Flute 1", x0=0.05, y0=0.2, x1=0.15, y1=0.22, size=10),
        ],
    )
    info = classify_page(page)
    assert info.part is not None and info.part.instrument_id == "flute"


def test_music_printed_sideways_is_marked_but_a_back_cover_is_not() -> None:
    # Tullamore page 100: no staves upright, 10 with the page turned.
    sideways = classify_page(
        PageFacts(page_number=100, source="ocr", ink_ratio=0.05, sideways_staff_count=10)
    )
    assert (sideways.sideways, sideways.kind) == (True, "unknown")
    # Page 46, the back cover (logo, address): no staves either way. A dark
    # cover may show one "staff" turned (Funk Attack page 1): still no music.
    for turned in (0, 1):
        cover = classify_page(
            PageFacts(page_number=46, source="ocr", ink_ratio=0.05, sideways_staff_count=turned)
        )
        assert not cover.sideways
    # Upright staves win: the turned page isn't even measured then.
    upright = classify_page(
        PageFacts(page_number=5, source="ocr", ink_ratio=0.05, staff_count=9, sideways_staff_count=4)
    )
    assert not upright.sideways
