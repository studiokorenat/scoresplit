"""
Tests for the scoring in scripts/precision.py, with small made-up results
(no PDFs needed). The golden tests in test_examples.py trust this scoring,
so it must count right.
"""

import precision

from scoresplit.models import parse_page_ranges


def part(instrument, pages, number=None, key=None, clef="treble", export=True, name=None):
    """Helper: one part in golden-file shape. The name defaults to "flute 1"."""
    return {
        "name": " ".join(filter(None, [instrument, number])) if name is None else name,
        "instrument": instrument,
        "number": number,
        "key": key,
        "clef": clef,
        "pages": pages,
        "export": export,
    }


def result(parts, extra=(), title="Song", composer="Composer", arranger=None):
    """Helper: a whole golden file (or detection result)."""
    return {
        "title": title,
        "composer": composer,
        "arranger": arranger,
        "parts": list(parts),
        "extra": list(extra),
    }


GOLDEN = result(
    [
        part("flute", "2-3", number="1", key="C"),
        part("clarinet", "4-5", number="1", key="Bb"),
        part("tuba", "6-7", key="C", clef="bass"),
    ],
    extra=[{"reason": "cover", "pages": "1"}],
)


def test_a_perfect_result_scores_everything_right() -> None:
    score = precision.score(GOLDEN, GOLDEN)
    assert score.pages_right == score.pages_total == 7
    assert (score.parts_found, score.parts_missed, score.parts_extra) == (3, 0, 0)
    assert score.keys_right == score.clefs_right == score.exports_right == 3
    assert score.names_right == 3
    assert score.credits_right == 3
    assert score.differences == []


def test_a_missed_part_and_its_pages() -> None:
    # The tuba wasn't recognised: its pages went to "unassigned".
    actual = result(
        GOLDEN["parts"][:2],
        extra=[{"reason": "cover", "pages": "1"}, {"reason": "unassigned", "pages": "6-7"}],
    )
    score = precision.score(GOLDEN, actual)
    assert (score.parts_found, score.parts_missed, score.parts_extra) == (2, 1, 0)
    assert score.pages_right == 5
    assert "missed: Tuba in C (bass clef) (pages 6-7)" in score.differences
    assert (
        "pages 6-7: expected Tuba in C (bass clef), got extra page (unassigned)"
        in score.differences
    )


def test_a_part_cut_in_two_gives_one_extra_part() -> None:
    # The clarinet's second page became a separate (wrong) oboe part.
    actual = result(
        [
            GOLDEN["parts"][0],
            part("clarinet", "4", number="1", key="Bb"),
            part("oboe", "5", key="C"),
            GOLDEN["parts"][2],
        ],
        extra=GOLDEN["extra"],
    )
    score = precision.score(GOLDEN, actual)
    # The clarinet still counts as found (it shares page 4 with the golden part).
    assert (score.parts_found, score.parts_missed, score.parts_extra) == (3, 0, 1)
    assert score.pages_right == 6
    assert "extra part: Oboe in C (treble clef) (pages 5)" in score.differences


# Two copies of the same part (same label), like deep_purple's
# "Bass in Bb (bass clef)" at pages 82-83 and again at 84-85.
TWO_COPIES = result(
    [
        part("bass", "2-3", key="Bb", clef="bass"),
        part("bass", "4-5", key="Bb", clef="bass"),
    ],
)


def test_two_copies_of_a_part_that_match_exactly_score_every_page() -> None:
    score = precision.score(TWO_COPIES, TWO_COPIES)
    assert score.pages_right == score.pages_total == 4
    assert score.differences == []


def test_a_page_in_the_wrong_copy_of_a_part_counts_as_wrong() -> None:
    # Page 4 belongs to the second copy, but detection put it in the first.
    actual = result(
        [
            part("bass", "2-4", key="Bb", clef="bass"),
            part("bass", "5", key="Bb", clef="bass"),
        ],
    )
    score = precision.score(TWO_COPIES, actual)
    assert score.pages_right == 3
    # The message tells the two copies apart.
    assert (
        "pages 4: expected Bass in Bb (bass clef), copy 2, "
        "got Bass in Bb (bass clef), copy 1"
    ) in score.differences


def test_wrong_key_clef_export_and_credits_are_counted() -> None:
    actual = result(
        [
            part("flute", "2-3", number="1", key="C"),
            part("clarinet", "4-5", number="1", key="A"),  # wrong key
            part("tuba", "6-7", key="C", clef="treble", export=False),  # wrong clef and export
        ],
        extra=GOLDEN["extra"],
        composer=None,
    )
    score = precision.score(GOLDEN, actual)
    assert score.pages_right == 7  # the pages themselves are all in the right part
    assert (score.keys_right, score.clefs_right, score.exports_right) == (2, 2, 2)
    assert score.credits_right == 2
    assert "composer: expected 'Composer', got None" in score.differences


def test_names_are_counted_ignoring_capitals_and_spacing() -> None:
    actual = result(
        [
            part("flute", "2-3", number="1", key="C", name="FLUTE  1"),  # same name
            part("clarinet", "4-5", number="1", key="Bb", name="Clarinet 1 in Bb"),  # wrong
            GOLDEN["parts"][2],
        ],
        extra=GOLDEN["extra"],
    )
    score = precision.score(GOLDEN, actual)
    assert score.names_right == 2
    assert (
        "Clarinet 1 in Bb (treble clef) (pages 4-5): name expected 'clarinet 1', "
        "got 'Clarinet 1 in Bb'" in score.differences
    )


def test_credits_ignore_capitals_and_spacing() -> None:
    assert precision.same_text("A Tribute to Prljavo", "A TRIBUTE  To prljavo")
    assert not precision.same_text("Composer", None)
    assert precision.same_text(None, None)


def test_same_numbers_written_differently_still_match() -> None:
    golden = result([part("horn", "1-2", number="1&2", key="F")])
    actual = result([part("horn", "1-2", number="I & II", key="F")])
    assert precision.score(golden, actual).parts_found == 1


def test_part_numbers() -> None:
    assert precision.part_numbers("1&2") == ("1", "2")
    assert precision.part_numbers("1/2") == ("1", "2")
    assert precision.part_numbers("1st, 2nd & 3rd") == ("1", "2", "3")
    assert precision.part_numbers("2-4") == ("2", "3", "4")
    assert precision.part_numbers("1º-3º") == ("1", "2", "3")
    assert precision.part_numbers("II") == ("2",)
    assert precision.part_numbers("Solo") == ("solo",)
    assert precision.part_numbers(None) is None


def test_golden_problems_finds_gaps_overlaps_and_typos() -> None:
    golden = result(
        [part("flute", "1-3", name=""), part("klarinet", "3-4", clef="trebel")],
        extra=[{"reason": "covr", "pages": "6"}],
    )
    problems = precision.golden_problems(golden, page_count=5)
    assert any("unknown instrument id 'klarinet'" in p for p in problems)
    assert any("unknown clef 'trebel'" in p for p in problems)
    assert "Flute (treble clef): no name" in problems
    assert any("unknown reason 'covr'" in p for p in problems)
    assert "pages 5 are in no part and no extra" in problems
    assert "pages 3 are listed more than once" in problems
    assert "pages 6 don't exist (the PDF has 5)" in problems
    assert precision.golden_problems(GOLDEN, page_count=7) == []


def test_golden_problems_finds_a_key_typo() -> None:
    # "B♭" isn't how scoresplit writes a key ("Bb" is), so it could never match.
    golden = result([part("clarinet", "1", number="1", key="B♭")])
    problems = precision.golden_problems(golden, page_count=1)
    assert any("Clarinet 1" in p and "unknown key 'B♭'" in p for p in problems)


def test_golden_problems_finds_a_wrong_number_and_export() -> None:
    golden = result(
        [part("flute", "1", number=1, key="C", name="Flute 1"), part("oboe", "2", key="C", export="yes")]
    )
    problems = precision.golden_problems(golden, page_count=2)
    assert any("number must be text" in p for p in problems)
    assert any("export must be true or false" in p and "'yes'" in p for p in problems)


def test_golden_problems_reports_a_missing_field_instead_of_crashing() -> None:
    flute = part("flute", "1", number="1", key="C")
    del flute["number"]
    problems = precision.golden_problems(
        result([flute], extra=[{"reason": "cover"}]), page_count=1
    )
    assert "part 1 ('flute 1'): no 'number' field" in problems
    assert "extra 1: no 'pages' field" in problems


def test_golden_problems_reports_unreadable_pages() -> None:
    problems = precision.golden_problems(result([part("flute", "3-1", key="C")]), page_count=3)
    assert any("Flute" in p and "goes backwards" in p for p in problems)


def test_the_real_golden_files_have_no_problems() -> None:
    # The PDFs aren't read here (that is slow): the page count is taken from
    # the highest page the golden file lists. test_examples.py checks against
    # the real page count.
    for path in precision.golden_files():
        golden = precision.load_golden(path)
        listed = [
            page
            for entry in golden["parts"] + golden["extra"]
            for page in parse_page_ranges(entry["pages"])
        ]
        assert precision.golden_problems(golden, max(listed)) == [], path.name


def test_drops_compares_with_the_baseline() -> None:
    baseline = precision.score(GOLDEN, GOLDEN).counts()
    worse = baseline | {"pages_right": 6, "names_right": 2, "parts_extra": 1}
    assert precision.drops(worse, baseline) == [
        "pages_right went down from 7 to 6",
        "names_right went down from 3 to 2",
        "parts_extra went up from 0 to 1",
    ]
    # A baseline from before a count existed (no "names_right") can't drop.
    old_baseline = {name: value for name, value in baseline.items() if name != "names_right"}
    assert precision.drops(baseline | {"names_right": 0}, old_baseline) == []
    better_or_same = baseline | {"keys_right": 4}
    assert precision.drops(better_or_same, baseline) == []
