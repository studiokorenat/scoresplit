"""
Tests for metadata.py: title and composer of PDFs without credit labels.

The pages are built by hand (no PDF needed), like a march card from the
Gardijska example: composer on the left, title in the middle, part name on
the right, and no "arr." / "Music by" anywhere.
"""

from scoresplit.classify import classify_page
from scoresplit.metadata import extract_metadata
from scoresplit.models import PageFacts, TextSpan
from scoresplit.pdf_reader import tidy_text


def march_card(number: int, part: str, composer: str = "MARKO POKLEPOVIC") -> PageFacts:
    """Helper: the OCR'd header of one march card page."""
    return PageFacts(
        page_number=number,
        source="ocr",
        staff_count=9,
        header=[
            TextSpan(composer, x0=0.08, y0=0.04, x1=0.24, y1=0.06, size=9),
            TextSpan("Gardijska", x0=0.42, y0=0.02, x1=0.57, y1=0.06, size=24),
            TextSpan(part, x0=0.80, y0=0.05, x1=0.94, y1=0.07, size=12),
            TextSpan("March", x0=0.46, y0=0.07, x1=0.52, y1=0.09, size=11),
            # OCR noise from the notes on the right: not a person's name
            TextSpan("I N VRN §", x0=0.63, y0=0.12, x1=0.72, y1=0.13, size=10),
        ],
    )


def test_title_and_left_composer_without_any_credit_label() -> None:
    facts = [march_card(1, "Piccolo"), march_card(2, "1st Flute"), march_card(3, "2nd Flute")]
    infos = [classify_page(page) for page in facts]
    meta = extract_metadata(facts, infos)
    assert meta["title"] == "Gardijska"
    assert meta["composer"] == "Marko Poklepovic"
    assert meta["arranger"] is None


def test_accented_spelling_wins_when_read_on_several_pages() -> None:
    # OCR read the "Ć" on 2 of 5 pages; it dropped it on the others.
    composers = ["MARKO POKLEPOVIC"] * 3 + ["MARKO POKLEPOVIĆ"] * 2
    facts = [march_card(n, "Oboe", name) for n, name in enumerate(composers, start=1)]
    infos = [classify_page(page) for page in facts]
    assert extract_metadata(facts, infos)["composer"] == "Marko Poklepović"


def test_title_and_arranger_on_one_line() -> None:
    # Deep Purple: the scanner glued "arr." to the title ("MEDLEYarr.").
    # tidy_text splits it off; the words before the credit are the title.
    line = tidy_text("DEEP PURPLE MEDLEYarr. Toshihiko Sahashi")
    assert line == "DEEP PURPLE MEDLEY arr. Toshihiko Sahashi"
    assert tidy_text("Barr  CARRY") == "Barr CARRY"  # real words are not cut
    # A control character (a text layer without a proper font map gives
    # "\x03"-like letters) becomes a space, so two words don't stick together.
    assert tidy_text("Flute\x03 1") == "Flute 1"
    assert tidy_text("Flute\x031") == "Flute 1"
    assert tidy_text("\x00Oboe\x7f") == "Oboe"
    page = PageFacts(
        page_number=40,
        source="text",
        staff_count=9,
        header=[
            TextSpan("Oboe", x0=0.09, y0=0.08, x1=0.13, y1=0.09, size=12),
            TextSpan(line, x0=0.32, y0=0.06, x1=0.95, y1=0.09, size=30),
            TextSpan("Burn - Highway Star", x0=0.34, y0=0.09, x1=0.7, y1=0.1, size=23),
        ],
    )
    meta = extract_metadata([page], [classify_page(page)])
    assert meta["title"] == "Deep Purple Medley"
    assert meta["arranger"] == "Toshihiko Sahashi"


def test_a_name_printed_upright_is_not_the_title() -> None:
    # cornfield cover: "de haske muziekuitgave" runs up the page edge. OCR
    # gives it a huge letter size (measured along the line), but it is the
    # publisher, not the title.
    cover = PageFacts(
        page_number=1,
        source="ocr",
        body=[
            TextSpan("CORNFIELD ROCK -", x0=0.33, y0=0.23, x1=0.66, y1=0.26, size=24),
            TextSpan("Jacob de Haan", x0=0.44, y0=0.28, x1=0.60, y1=0.30, size=9.7),
            TextSpan("de muziekuitgave haske", x0=0.78, y0=0.21, x1=0.83, y1=0.70, size=168),
        ],
    )
    meta = extract_metadata([cover], [classify_page(cover)])
    # Also without the stray dash OCR read after it.
    assert meta["title"] == "Cornfield Rock"


def test_a_credit_must_be_read_on_enough_pages_when_asked() -> None:
    # The second credits pass (pipeline.inspect_pdf) only takes a name read
    # on several pages: one line on one page is too often something else.
    facts = [march_card(1, "Piccolo", "CD Cassette"), march_card(2, "Flute", "CD Cassette")]
    infos = [classify_page(page) for page in facts]
    assert extract_metadata(facts, infos)["composer"] == "Cd Cassette"
    assert extract_metadata(facts, infos, min_votes=3)["composer"] is None


def test_a_credit_label_must_be_written_like_one() -> None:
    # cornfield page 13 with English + Croatian OCR: music noise "=a=aRR s"
    # made a nameless first page. "aRR" is no way to print "arr.".
    from scoresplit.classify import has_credits

    def span(text: str) -> TextSpan:
        return TextSpan(text, x0=0.1, y0=0.05, x1=0.5, y1=0.07, size=10)

    assert not has_credits([span("=a=aRR s")])
    for label in ("arr. Jacob de Haan", "Arr. J. de Haan", "ARR. JACOB DE HAAN", "Music by X"):
        assert has_credits([span(label)])
    assert has_credits([span("WORDS & MUSIC BY Freddie Mercury")])


def test_a_credit_read_on_one_page_gives_way_to_one_read_on_many() -> None:
    # cornfield with English + Croatian OCR: the cover's only "name" is noise
    # read on the right ("URTI PRN S", once), while "Jacob de Haan" is printed
    # under the title on every part's first page. Those pages are found by
    # their title (mark_title_pages), and the second vote replaces the noise.
    from scoresplit.pipeline import read_credits

    cover = PageFacts(
        page_number=1,
        source="ocr",
        body=[
            TextSpan("CORNFIELD ROCK", x0=0.30, y0=0.20, x1=0.70, y1=0.25, size=24),
            TextSpan("URTI PRN S", x0=0.60, y0=0.60, x1=0.80, y1=0.62, size=10),
        ],
    )

    def first_page(number: int, part: str) -> PageFacts:
        return PageFacts(
            page_number=number,
            source="ocr",
            staff_count=10,
            header=[
                TextSpan(part, x0=0.06, y0=0.06, x1=0.13, y1=0.07, size=10),
                TextSpan("CORNFIELD ROCK", x0=0.29, y0=0.05, x1=0.71, y1=0.08, size=24),
                TextSpan("Jacob de Haan", x0=0.57, y0=0.08, x1=0.72, y1=0.09, size=12),
            ],
        )

    facts = [cover] + [
        first_page(n, part) for n, part in enumerate(["Flute", "Oboe", "Bassoon"], start=2)
    ]
    infos = [classify_page(page) for page in facts]
    # The first vote alone takes the noise: it is the only name it sees.
    assert extract_metadata(facts, infos)["composer"] == "Urti Prn S"
    assert read_credits(facts, infos)["composer"] == "Jacob de Haan"
