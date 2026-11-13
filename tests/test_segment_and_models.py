"""
Tests for page-range helpers, segmentation rules, file names, name casing
and the result model (worked-out extra pages, check_result).

These tests build small fake page lists by hand, so they run in milliseconds
and show exactly which rule is being checked.
"""

import pytest
from pydantic import ValidationError

from scoresplit.exporter import safe_filename
from scoresplit.metadata import nice_case
from scoresplit.models import (
    MAX_PAGE_NUMBER,
    Extra,
    PageInfo,
    Part,
    PartName,
    Result,
    check_result,
    format_page_ranges,
    parse_page_ranges,
)
from scoresplit.segment import build_parts


def make_page(number, name=None, kind="part", start=False, clef="treble", number_part=None, key=None):
    """
    Helper: a PageInfo for a fake page. `name` is an instrument id/name.
    The page prints its name cleanly: its wording is the canonical name.
    """
    part = None
    if name:
        part = PartName(
            instrument_id=name.lower(), instrument_name=name, number=number_part, key=key, score=100
        )
        part.wording = part.display()
    return PageInfo(
        page_number=number,
        kind=kind,
        part=part,
        is_start=start,
        clef=clef,
        has_staves=kind in ("part", "score"),
        confidence=1.0 if part or kind == "score" else 0.0,
    )


def test_page_ranges_round_trip() -> None:
    assert parse_page_ranges("3-5,9") == [3, 4, 5, 9]
    assert parse_page_ranges([9, 3, 3]) == [3, 9]
    assert format_page_ranges([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"


def test_start_pages_split_parts_and_unreadable_pages_are_inherited() -> None:
    pages = [
        make_page(1, kind="blank"),
        make_page(2, "Flute", start=True, number_part="1"),
        make_page(3, "Flute"),  # OCR lost the number: still Flute 1
        make_page(4, kind="unknown"),  # unreadable but has music
        make_page(5, "Flute", start=True, number_part="2"),
        make_page(6, "Flute", number_part="2"),
    ]
    pages[3].has_staves = True
    parts, extras, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts] == [("Flute 1", "2-4"), ("Flute 2", "5-6")]
    assert [(e.reason, e.pages) for e in extras] == [("blank", "1")]


def test_same_name_different_clef_makes_two_parts_with_the_same_name() -> None:
    # The clef is in its own column (and later in the file name), so the
    # name stays as printed: no "(Treble Clef)" suffix.
    pages = [
        make_page(1, "Trombone", start=True, number_part="1", key="Bb", clef="treble"),
        make_page(2, "Trombone", number_part="1", key="Bb", clef="treble"),
        make_page(3, "Trombone", start=True, number_part="1", key="Bb", clef="bass"),
        make_page(4, "Trombone", number_part="1", key="Bb", clef="bass"),
    ]
    parts, _, _ = build_parts(pages)
    names = [p.name for p in parts]
    assert names == ["Trombone 1 in Bb", "Trombone 1 in Bb"]
    assert [p.clef for p in parts] == ["treble", "bass"]
    assert all(p.export for p in parts)


def test_true_duplicate_is_listed_but_not_exported() -> None:
    pages = [
        make_page(1, "Oboe", start=True),
        make_page(2, "Oboe"),
        make_page(3, "Oboe", start=True),
        make_page(4, "Oboe"),
    ]
    parts, _, _ = build_parts(pages)
    assert [p.export for p in parts] == [True, False]
    assert parts[1].duplicate_of == "Oboe"


def test_score_pages_are_merged_and_cover_goes_to_extra() -> None:
    pages = [
        make_page(1, kind="unknown"),  # cover: no staves, before any music
        make_page(2, kind="score", start=True),
        make_page(3, kind="score"),
        make_page(4, "Piccolo"),  # one misread score page between score pages
        make_page(5, kind="score"),
        make_page(6, "Piccolo", start=True),
    ]
    pages[0].has_staves = False
    parts, extras, _ = build_parts(pages)
    assert (parts[0].name, parts[0].pages) == ("Conductor Score", "2-5")
    assert (parts[1].name, parts[1].pages) == ("Piccolo", "6")
    assert [(e.reason, e.pages) for e in extras] == [("cover", "1")]


def test_score_pages_that_restart_do_not_swallow_the_parts_between() -> None:
    # Sedona before the fix: every part's FIRST page was taken for a score
    # page (with credits, so a start page), and every second page sat
    # between two of them. A score starts only once, so the named pages in
    # between must stay parts.
    pages = [
        make_page(1, kind="score", start=True),
        make_page(2, kind="score"),
        make_page(3, kind="score", start=True),
        make_page(4, "Piccolo"),
        make_page(5, kind="score", start=True),
        make_page(6, "Oboe"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts][1:] == [("Piccolo", "4"), ("Oboe", "6")]


def test_safe_filename_and_nice_case() -> None:
    assert safe_filename('Trombone 1/2: "Bb"?') == "Trombone 1-2 Bb"
    assert safe_filename("  ..  ") == "Untitled"
    assert nice_case("HAUSTOR LIVE") == "Haustor Live"
    assert nice_case("Dennis MARIEV") == "Dennis Mariev"
    assert nice_case("Peter Kleine Schaars") == "Peter Kleine Schaars"


def test_ocr_dropping_a_word_on_the_first_page_does_not_split_the_part() -> None:
    # Page 1 was read as "Clarinet" (OCR lost "Bass"), the rest say "Bass Clarinet".
    # (Page 5 starts another part, so the PDF has 2 title pages -> start-page mode.)
    pages = [make_page(1, "Clarinet", start=True)] + [make_page(n, "Bass Clarinet") for n in (2, 3, 4)]
    pages.append(make_page(5, "Tuba", start=True))
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts] == [("Bass Clarinet", "1-4"), ("Tuba", "5")]


def test_missed_title_page_still_splits_when_two_pages_agree() -> None:
    # Page 4 is the first page of Bass Trombone, but its credits weren't read.
    pages = [
        make_page(1, "Trombone", start=True),
        make_page(2, "Trombone"),
        make_page(3, "Trombone"),
        make_page(4, "Bass Trombone"),
        make_page(5, "Bass Trombone"),
        make_page(6, "Tuba", start=True),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts] == [
        ("Trombone", "1-3"),
        ("Bass Trombone", "4-5"),
        ("Tuba", "6"),
    ]


def test_clef_printed_in_the_name_makes_two_parts() -> None:
    # "Baritone (T.C.)" and "Baritone (B.C.)": no title pages (name-change
    # mode), and the image clef detection failed on page 3 (clef=None).
    pages = [make_page(n, "Baritone", clef=c) for n, c in ((1, "treble"), (2, "treble"), (3, None))]
    pages[0].part.clef = "treble"
    pages[2].part.clef = "bass"
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages, p.clef) for p in parts] == [
        ("Baritone", "1-2", "treble"),
        ("Baritone", "3", "bass"),
    ]


def test_bass_clef_cover_page_starts_the_bass_clef_part() -> None:
    # rhcp.pdf: "Euphonium" (no clef in the name, treble clef music), then
    # the cover sheet "Euphonium (B.C.)" and the bass clef pages. Same
    # instrument, so only the printed clef shows where the second part starts.
    pages = [
        make_page(1, "Euphonium", clef=None),  # cover sheet: no music
        make_page(2, "Euphonium", start=True),
        make_page(3, "Euphonium"),
        make_page(4, "Euphonium", clef=None),  # cover sheet "Euphonium (B.C.)"
        make_page(5, "Euphonium", start=True, clef="bass"),
        make_page(6, "Euphonium", clef="bass"),
    ]
    for cover in (pages[0], pages[3]):
        cover.has_staves = False
    for page in pages[3:]:
        page.part.clef = "bass"  # "(B.C.)" printed in the name
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages, p.clef) for p in parts] == [
        ("Euphonium", "1-3", "treble"),
        ("Euphonium", "4-6", "bass"),
    ]


def test_treble_clef_euphonium_is_noted_as_in_bb() -> None:
    # No key printed: treble clef is written in Bb, bass clef in C. The name
    # stays as printed; the key is filled in and marked as assumed, and a
    # note says the clef gave the key Bb.
    pages = [
        make_page(1, "Euphonium", start=True),
        make_page(2, "Euphonium", start=True, clef="bass"),
        make_page(3, "Flute", start=True),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.key, p.key_assumed) for p in parts] == [
        ("Euphonium", "Bb", True),
        ("Euphonium", "C", True),
        ("Flute", "C", True),
    ]
    assert parts[0].notes == ["key not printed: Bb, from the treble clef"]
    assert parts[1].notes == []  # C is the default key anyway
    assert parts[2].notes == []  # the clef doesn't change a flute's key


def test_a_combined_baritone_euphonium_part_gets_its_key_from_the_clef() -> None:
    # "Baritone, Euphonium" in treble clef, no key printed: Bb, and the
    # label stays the part name.
    pages = [make_page(1, "Baritone", start=True), make_page(2, "Flute", start=True)]
    pages[0].part.label = pages[0].part.wording = "Baritone, Euphonium"
    parts, _, _ = build_parts(pages)
    assert (parts[0].name, parts[0].key, parts[0].key_assumed) == ("Baritone, Euphonium", "Bb", True)
    assert parts[0].notes == ["key not printed: Bb, from the treble clef"]


def test_a_printed_key_beats_the_clef() -> None:
    pages = [make_page(1, "Euphonium", start=True, key="C"), make_page(2, "Flute", start=True)]
    parts, _, _ = build_parts(pages)
    assert (parts[0].key, parts[0].key_assumed) == ("C", False)
    assert parts[0].notes == []


def test_missing_number_between_two_numbered_parts_is_filled_in() -> None:
    # OCR read "2nd Trombone" as "an Trombone": no number on page 2.
    pages = [
        make_page(1, "Trombone", start=True, number_part="1"),
        make_page(2, "Trombone", start=True),
        make_page(3, "Trombone", start=True, number_part="3"),
        make_page(4, "Trombone", start=True),  # no neighbour after it: left alone
    ]
    parts, _, _ = build_parts(pages)
    assert [p.name for p in parts] == ["Trombone 1", "Trombone 2", "Trombone 3", "Trombone"]
    assert any("guessed" in note for note in parts[1].notes)


def test_unreadable_first_page_before_the_score_is_score() -> None:
    pages = [
        make_page(1, kind="unknown"),  # score page 1 with an unreadable margin
        make_page(2, kind="score"),
        make_page(3, kind="score"),
        make_page(4, "Flute", start=True),
    ]
    pages[0].has_staves = True
    parts, extras, _ = build_parts(pages)
    assert (parts[0].name, parts[0].pages) == ("Conductor Score", "1-3")
    assert extras == []


def test_same_name_but_different_music_is_exported_too() -> None:
    # Funk Attack: "Bass" in Eb and "Bass" in Bb, both read as just "Bass".
    # Same name, same length, but the pages don't look alike.
    pages = [
        make_page(1, "Bass", start=True),
        make_page(2, "Bass"),
        make_page(3, "Bass", start=True),
        make_page(4, "Bass"),
    ]
    parts, _, _ = build_parts(pages, same_music=lambda a, b: False)
    assert [(p.name, p.export) for p in parts] == [("Bass", True), ("Bass", True)]
    assert "same name as pages 1-2" in parts[1].notes[-1]
    assert "please rename" in parts[1].notes[-1]
    # ...and when the pages DO look alike, it is a real copy.
    parts, _, _ = build_parts(pages, same_music=lambda a, b: True)
    assert [(p.name, p.export, p.duplicate_of) for p in parts] == [
        ("Bass", True, None),
        ("Bass", False, "Bass"),
    ]
    assert parts[1].notes[-1] == "duplicate of 'Bass' (pages 1-2), not exported"


def test_missing_key_is_taken_from_the_previous_number() -> None:
    # "B♭ CLARINET 2" read as "Bo CLARINET 2": the key was lost.
    pages = [
        make_page(1, "Clarinet", start=True, number_part="1", key="Bb"),
        make_page(2, "Clarinet", start=True, number_part="2"),
        make_page(3, "Clarinet", start=True, number_part="3"),
        make_page(4, "Clarinet", start=True, key="Eb"),  # not numbered: unchanged
    ]
    parts, _, _ = build_parts(pages)
    # The names stay as printed; the key is filled in and marked assumed.
    assert [(p.name, p.key, p.key_assumed) for p in parts] == [
        ("Clarinet 1 in Bb", "Bb", False),
        ("Clarinet 2", "Bb", True),
        ("Clarinet 3", "Bb", True),
        ("Clarinet in Eb", "Eb", False),
    ]


def test_weak_name_on_a_continuation_page_does_not_cut_the_part() -> None:
    # Ross Roy p162 (eng+hrv): OCR noise "I aSs I aS" matched "Bass" at 86%
    # in the middle of the Baritone part; the real Bass part follows.
    pages = [
        make_page(1, "Baritone", start=True),
        make_page(2, "Bass", start=False),
        make_page(3, "Bass", start=True),
    ]
    pages[1].part.score = 86
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts] == [("Baritone", "1-2"), ("Bass", "3")]


def test_a_key_that_is_not_printed_is_filled_in_and_marked_assumed() -> None:
    # "Clarinet 1" (no key) gets the usual key Bb, marked key_assumed, so
    # the table shows it is a guess. That flag replaced the old "key not
    # printed: assumed Bb" note. A printed key is not assumed.
    pages = [
        make_page(1, "Clarinet", start=True, number_part="1"),
        make_page(2, "Clarinet", start=True, key="Eb"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.key, p.key_assumed) for p in parts] == [("Bb", True), ("Eb", False)]
    assert not any("assumed" in note for p in parts for note in p.notes)


def test_a_key_taken_from_the_part_before_is_marked_assumed() -> None:
    pages = [
        make_page(1, "Clarinet", start=True, number_part="1", key="A"),
        make_page(2, "Clarinet", start=True, number_part="2"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.key, p.key_assumed) for p in parts] == [("A", False), ("A", True)]
    assert "key A taken from the part before it: please check" in parts[1].notes


def test_parts_without_a_usual_key_get_no_key() -> None:
    # Drums have no key; a plain "Bass" may be a bass guitar, so no key is
    # guessed (and nothing is marked assumed).
    pages = [make_page(1, "Snare Drum", start=True), make_page(2, "Bass", start=True)]
    parts, _, _ = build_parts(pages)
    assert [(p.key, p.key_assumed) for p in parts] == [(None, False), (None, False)]


def test_unusual_key_gets_a_note_in_build_parts() -> None:
    parts, _, _ = build_parts([make_page(1, "Flute", start=True, key="Eb")])
    assert parts[0].key == "Eb"  # kept, but worth a look
    assert "unusual key for Flute (Eb): please check" in parts[0].notes


def test_generic_part_gets_no_note() -> None:
    # make_page() uses the name as instrument id, so "Saxophone" -> "saxophone".
    # A generic name is read fine: no note, and the confidence isn't lowered.
    parts, _, _ = build_parts([make_page(1, "Saxophone", start=True, number_part="1")])
    assert parts[0].instrument == "saxophone"
    assert parts[0].notes == []
    assert parts[0].confidence == 1.0


def test_a_bass_without_a_key_gets_no_key_and_a_note() -> None:
    # A plain "Bass" may be a band bass (Bb, Eb, ...) or a bass guitar (no
    # key), so no key is assumed: a person checks it.
    parts, _, _ = build_parts(
        [make_page(1, "Bass", start=True), make_page(2, "Bass", start=True, key="Eb")],
        same_music=lambda a, b: False,
    )
    assert [p.key for p in parts] == [None, "Eb"]
    assert "key not printed: please check" in parts[0].notes
    assert parts[1].notes == []  # Eb is a usual key for a band bass


# --- the result model (models.Result) ------------------------------------------


def make_result(parts: list[Part], extra: list[Extra] | None = None, page_count: int = 6) -> Result:
    """Helper: a Result with just the fields these tests care about."""
    return Result(source="t.pdf", sha256="0" * 64, page_count=page_count, parts=parts, extra=extra or [])


def test_extra_pages_are_every_page_in_no_part() -> None:
    # Detection said page 1 is the cover and page 6 blank; pages 4-5 are in
    # no part and detection said nothing about them: "unassigned".
    result = make_result(
        [Part(name="Flute", pages="2-3")],
        extra=[Extra(reason="cover", pages="1"), Extra(reason="blank", pages="6")],
    )
    assert result.extra == [
        Extra(reason="cover", pages="1"),
        Extra(reason="unassigned", pages="4-5"),
        Extra(reason="blank", pages="6"),
    ]


def test_deleting_a_part_moves_its_pages_to_the_extra_pages() -> None:
    result = make_result([Part(name="Flute", pages="1-3"), Part(name="Oboe", pages="4-6")])
    assert result.extra == []
    del result.parts[1]  # like deleting the row in an editor
    assert result.extra_groups() == [Extra(reason="unassigned", pages="4-6")]
    result.refresh_extra()
    assert result.extra == [Extra(reason="unassigned", pages="4-6")]


def test_a_page_back_in_a_part_is_no_extra_page() -> None:
    # A saved edit gave the cover page to the score: it isn't an extra any more.
    result = make_result(
        [Part(name="Conductor Score", kind="score", pages="1-6")],
        extra=[Extra(reason="cover", pages="1")],
    )
    assert result.extra == []


def test_check_result_reports_every_problem_at_once() -> None:
    result = make_result(
        [
            Part(name="Flute", pages="5-3"),  # backwards: kept as text, reported
            Part(name="Oboe", pages="4-9"),  # pages 7-9 don't exist
            Part(name="Horn", pages=""),  # exported but empty
            Part(name="Horn copy", pages="", export=False),  # fine: not exported
        ]
    )
    assert result.parts[0].pages == "5-3"
    report = check_result(result)
    assert len(report.errors) == 3
    assert "'Flute': pages '5-3' can't be read" in report.errors[0]
    assert "'Oboe': page(s) 7-9 don't exist (PDF has 6)" in report.errors[1]
    assert "'Horn': has no pages" in report.errors[2]
    assert report.warnings == []


def test_check_result_warns_about_a_page_in_two_parts() -> None:
    result = make_result(
        [
            Part(name="Flute", pages="1-3"),
            Part(name="Oboe", pages="3-4"),
            Part(name="Oboe copy", pages="4", export=False),  # not written: no warning
        ]
    )
    report = check_result(result)
    assert report.errors == []
    assert report.warnings == ["'Flute' and 'Oboe' share page(s) 3"]


def test_check_result_warns_about_two_same_name_parts_sharing_a_page() -> None:
    # Two parts may have the same name: the check compares parts, not names.
    result = make_result([Part(name="Euphonium", pages="1-3"), Part(name="Euphonium", pages="3-4")])
    assert check_result(result).warnings == ["'Euphonium' and 'Euphonium' share page(s) 3"]


def test_a_result_of_another_schema_version_is_refused() -> None:
    with pytest.raises(ValidationError, match="schema_version 2 is not supported"):
        Result(schema_version=2, source="t.pdf", sha256="0", page_count=1)



# An edits file can come from anyone (the web tool takes uploads). A huge
# page_count must be refused AT ONCE: working out the extra pages walks over
# every page from 1 to page_count, so "1000000000000" would hang the program.
@pytest.mark.parametrize("page_count", [MAX_PAGE_NUMBER + 1, 1_000_000_000_000, -1])
def test_a_page_count_out_of_range_is_refused(page_count: int) -> None:
    with pytest.raises(ValidationError, match="page_count"):
        make_result([], page_count=page_count)


def test_the_highest_page_count_still_loads() -> None:
    result = make_result([Part(name="Flute", pages="1")], page_count=MAX_PAGE_NUMBER)
    assert result.extra == [Extra(reason="unassigned", pages=f"2-{MAX_PAGE_NUMBER}")]


# --- Control characters in an edits file ----------------------------------------


def test_control_characters_in_an_edits_file_are_cleaned_when_loaded() -> None:
    # "\x1b[31m" would turn the terminal red (or worse) when the table is
    # printed; "\x9b" is the one-character form of "\x1b[". Every control
    # character becomes a space, as when text is read from a PDF.
    data = {
        "source": "a\x1b[2J.pdf",
        "sha256": "0" * 64,
        "page_count": 2,
        "title": "Song\x07",
        "composer": "\x9b31mSomeone",
        "arranger": "A.\x00Person",
        "warnings": ["odd\x0ctext"],
        "parts": [
            {
                "name": "Flute\x1b[31m 1",
                "instrument": "flute\x7f",
                "number": "1\x03",
                "key": "C\x85",
                "duplicate_of": "Oboe\x1b",
                "notes": ["read\x1f on page 4"],
                "pages": "1-2",
            }
        ],
    }
    result = Result.model_validate(data)
    assert (result.source, result.title, result.composer, result.arranger) == (
        "a [2J.pdf", "Song", "31mSomeone", "A. Person"
    )
    assert result.warnings == ["odd text"]
    part = result.parts[0]
    assert (part.name, part.instrument, part.number, part.key, part.duplicate_of) == (
        "Flute [31m 1", "flute", "1", "C", "Oboe"
    )
    assert part.notes == ["read on page 4"]


def test_clean_text_from_an_edits_file_is_kept_exactly() -> None:
    # Only text WITH a control character is touched: a name a person typed
    # with two spaces keeps them.
    part = Part(name="  Flute  1 ", pages="1")
    assert part.name == "  Flute  1 "
    assert Part(name="Flute", key=None, pages="1").key is None

# --- Names as printed ----------------------------------------------------------


def test_the_most_common_printed_name_wins() -> None:
    pages = [make_page(n, "Clarinet", start=n == 1, number_part="1", key="Bb") for n in (1, 2, 3)]
    pages[0].part.wording = "Clarinet in Bb 1"
    pages[1].part.wording = "CLARINET IN BB 1"  # (calmed by instruments.py in real life)
    pages[2].part.wording = "Clarinet in Bb 1"
    parts, _, _ = build_parts(pages)
    assert parts[0].name == "Clarinet in Bb 1"
    assert parts[0].notes == []


def test_a_printed_name_must_name_the_whole_part() -> None:
    # Paradiso: the first page prints "Bass Tuba in C", the next pages only
    # "Bass Tuba". Those don't say the key, so they don't vote.
    pages = [make_page(1, "Tuba", start=True, key="C"), make_page(2, "Tuba"), make_page(3, "Tuba")]
    pages[0].part.wording = "Bass Tuba in C"
    pages[1].part.wording = pages[2].part.wording = "Bass Tuba"
    parts, _, _ = build_parts(pages)
    assert parts[0].name == "Bass Tuba in C"


def test_no_clean_printed_name_falls_back_to_the_recognised_name() -> None:
    # Funk Attack: "mTRUMPETl" was recognised as Trumpet 1, but its words
    # are noise (no wording). The key isn't printed, so it isn't in the name.
    pages = [make_page(1, "Trumpet", start=True, number_part="1"), make_page(2, "Trumpet")]
    for page in pages:
        page.part.wording = None
    parts, _, _ = build_parts(pages)
    assert parts[0].name == "Trumpet 1"
    assert "the name wasn't printed clearly: please check it" in parts[0].notes
    assert parts[0].confidence == 0.8  # yellow in the table


# ---------------------------------------------------------------------------
# Round 9.4: score continuity and keys across clefs
# ---------------------------------------------------------------------------


def test_a_label_inside_the_score_does_not_start_a_part() -> None:
    # Suite: a hand-copied score whose margins can't be read. Its pages are
    # "unknown", and "Tub, Bar" inside the music on page 3 read like a part
    # name. Until the first real part start, it is all score; the page
    # without staves between two score pages (suite page 9) too.
    pages = [
        make_page(1, kind="score", start=True),
        make_page(2, kind="unknown"),
        make_page(3, "Tuba"),  # a label inside the music, no start page
        make_page(4, kind="unknown"),
        make_page(5, kind="unknown"),  # no staves found on this score page
        make_page(6, kind="score"),
        make_page(7, "Flute", start=True),
        make_page(8, "Flute"),
        make_page(9, "Oboe", start=True),
    ]
    pages[1].has_staves = pages[3].has_staves = True
    parts, extras, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts] == [
        ("Conductor Score", "1-6"),
        ("Flute", "7-8"),
        ("Oboe", "9"),
    ]
    assert extras == []


def test_a_part_whose_title_was_missed_still_ends_the_score() -> None:
    # Two pages in a row name the Flute: a real part whose start page
    # wasn't recognised, not a label inside the score.
    pages = [
        make_page(1, kind="score", start=True),
        make_page(2, "Flute"),
        make_page(3, "Flute"),
        make_page(4, "Oboe", start=True),
        make_page(5, "Clarinet", start=True),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts][:2] == [("Conductor Score", "1"), ("Flute", "2-3")]


def test_without_start_pages_a_name_after_the_score_starts_a_part() -> None:
    # Name-change mode (no start pages): the score is not continued.
    pages = [
        make_page(1, kind="score"),
        make_page(2, "Flute"),
        make_page(3, "Oboe"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.pages) for p in parts] == [
        ("Conductor Score", "1"),
        ("Flute", "2"),
        ("Oboe", "3"),
    ]


def test_a_key_is_not_taken_from_a_part_in_another_clef() -> None:
    # Italiano: "1. Bb Trombone" is in treble clef; "2. Trombone" after it
    # is in bass clef, so in C (the usual key), not Bb.
    pages = [
        make_page(1, "Trombone", start=True, number_part="1", key="Bb", clef="treble"),
        make_page(2, "Trombone", start=True, number_part="2", clef="bass"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.key, p.key_assumed) for p in parts] == [("Bb", False), ("C", True)]
    assert not any("taken from the part before" in note for note in parts[1].notes)


def test_a_clef_not_detected_is_assumed_from_the_instrument() -> None:
    # Light scans (mission, suite): the staves are found, the clef is not.
    # A one-clef instrument then gets its clef, marked assumed like a key.
    pages = [
        make_page(1, "Trumpet", start=True, clef=None),
        make_page(2, "Bassoon", start=True, clef=None),
        # Trombone is written in treble OR bass clef: no guess.
        make_page(3, "Trombone", start=True, clef=None),
        # A clef detected on the page wins over the instrument's clef.
        make_page(4, "Clarinet", start=True, clef="bass"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.name, p.clef, p.clef_assumed) for p in parts] == [
        ("Trumpet", "treble", True),
        ("Bassoon", "bass", True),
        ("Trombone", None, False),
        ("Clarinet", "bass", False),
    ]


def test_no_clef_is_assumed_for_two_instruments_in_two_clefs() -> None:
    # superstar page 63: "Pauken, Glockenspiel". Timpani are in bass clef,
    # the glockenspiel in treble: the part has both, so no clef is assumed.
    # Two instruments in the same clef still get it ("Flute/Piccolo").
    pages = [
        make_page(1, "Timpani", start=True, clef=None),
        make_page(2, "Flute", start=True, clef=None),
    ]
    pages[0].part.wording = "Pauken, Glockenspiel"
    pages[1].part.wording = "Flute/Piccolo"
    parts, _, _ = build_parts(pages)
    assert [(p.clef, p.clef_assumed) for p in parts] == [(None, False), ("treble", True)]


def test_a_clef_printed_in_the_name_is_not_assumed() -> None:
    page = make_page(1, "Horn", start=True, clef=None)
    page.part.clef = "bass"  # "Horn (B.C.)"
    parts, _, _ = build_parts([page])
    assert (parts[0].clef, parts[0].clef_assumed) == ("bass", False)


def test_an_assumed_clef_does_not_keep_copies_apart() -> None:
    # Two copies of a horn part: the clef was detected on one copy only (a
    # bass clef, from a low passage). The other copy's assumed treble clef
    # is a guess, so it must not make the two count as different parts.
    pages = [
        make_page(1, "Horn", start=True, clef="bass"),
        make_page(2, "Horn", start=True, clef=None),
    ]
    parts, _, _ = build_parts(pages)
    assert [p.export for p in parts] == [True, False]
    assert [(p.clef, p.clef_assumed) for p in parts] == [("bass", False), ("treble", True)]



def test_a_missing_key_is_also_taken_for_a_pair_of_players() -> None:
    # cornfield: "Eb Horn II", then "Horn III - IV" whose "Eb" OCR lost.
    pages = [
        make_page(1, "Horn", start=True, number_part="2", key="Eb"),
        make_page(2, "Horn", start=True, number_part="3&4"),
    ]
    parts, _, _ = build_parts(pages)
    assert [(p.key, p.key_assumed) for p in parts] == [("Eb", False), ("Eb", True)]


def test_a_page_printed_sideways_is_a_part_of_its_own_at_zero_confidence() -> None:
    # Tullamore: page 100 is Trombone 1-2 printed sideways, between two
    # parts. It is neither added to the part before it, nor an extra page.
    pages = [
        make_page(1, "Trombone", start=True, number_part="1"),
        make_page(2, kind="unknown"),  # sideways
        make_page(3, kind="unknown"),  # sideways too: the same part
        make_page(4, "Trombone", number_part="3"),  # no title: still a new part
        make_page(5, kind="unknown"),  # a back cover without music
    ]
    pages[1].sideways = pages[2].sideways = True
    parts, extras, warnings = build_parts(pages)
    assert [(p.name, p.pages, p.confidence) for p in parts] == [
        ("Trombone 1", "1", 1.0),
        ("Unknown Part (p2-3)", "2-3", 0.0),
        ("Trombone 3", "4", 1.0),
    ]
    assert parts[1].instrument is None
    assert parts[1].notes == [
        "page printed sideways (or unreadable): please name it or join it to a part"
    ]
    assert [(e.reason, e.pages) for e in extras] == [("unassigned", "5")]
