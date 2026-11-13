"""
"Golden" tests: run the real pipeline on the example PDFs and check that
detection is not worse than before.

Each example PDF has a golden file in tests/golden/ (same name, .json) that
holds the TRUTH for that PDF, checked by hand against the pages. scoresplit
isn't perfect yet, so these tests don't ask for an exact match: they score
the result with scripts/precision.py and fail when any score is BELOW the
one stored in tests/golden/baseline.json (the "ratchet": scores may go up,
never down).

When a test fails, see what changed with
    uv run python scripts/precision.py --pdf <name>
When a change makes a score better, store the new scores with
    uv run python scripts/precision.py --save-baseline

The example PDFs are copyrighted and gitignored (never `git add` them): a
test skips itself when its PDF is missing. The scanned examples need
Tesseract and take a while the first time (OCR); later runs read the OCR
cache and are fast.
"""

import pytest

# scripts/precision.py; pyproject.toml puts scripts/ on the import path for pytest.
import precision
from scoresplit.ocr import tesseract_available

# (PDF name, needs OCR) with a short note on what kind of PDF each one is.
CASES = [
    # Text PDF (Sibelius), one part after another, no score, no cover.
    ("Haustor.pdf", False),
    # Copier scan (no text layer, OCR only): a two-page cover, then the
    # score, then the parts; many key and clef versions (Trombone in C, in
    # Bb treble clef, in Bb bass clef; Tuba in Eb/Bb/C).
    ("paradiso.pdf", True),
    # Sibelius pages pasted as images into OpenOffice; logo title; text
    # cover sheets with the part name at the bottom.
    ("metallica.pdf", True),
    # De Haske scan: part names only on the first page of each part,
    # "B: Trumpet/Cornet 1", no arranger (original composition).
    ("rossroy.pdf", True),
    # Old march cards (landscape, one page per part): composer on the left,
    # title in the middle, part name on the right, no "arr." anywhere.
    # "1st Bb Clarinet", "Baritone (T.C.)"/"(B.C.)", "Cassa"; score page 1
    # cut off at the left edge. (English-only OCR can't read the "ć" of
    # Poklepović; with the default eng+hrv it is read.)
    ("gardijska.pdf", True),
    # Scan with an OCR text layer (so a "text" PDF, but tilted and garbled):
    # biography and program-notes pages that mention instruments; Euphonium
    # in treble AND bass clef; "Eb Bass" and "Bb Bass" (tubas) with the keys
    # unread; "mTRUMPETl" = Trumpet 1; "BASS I ELECTRIC BASS" =
    # Bass/Electric Bass; "FLUTE 1, 2 (PICCOLO)".
    ("funkattack.pdf", False),
    # Scan with an OCR text layer, part names only on first pages. The part
    # name, title and arranger are merged into one long line ("Eb Alto
    # Saxophone 1-2 DEEP PURPLE MEDLEYarr. ..."), with "arr." glued to the
    # title; "F Horn 2-4" ranges; a treble clef sign read as "6"
    # ("Baritone / Euphonium 6"); page 64's name is missing from the text
    # layer (own OCR of the header finds "Horn"; the handwritten "1º-3º"
    # isn't readable); page 82 is a light scan with grey staff lines.
    # Pages 58-59 are a handwritten "Sax. Barítono" part, and 82-83/84-85
    # are two scans of the same Bb Bass part (the second is a duplicate).
    ("deep_purple.pdf", True),
    # Text PDF (Sibelius), no score. Title with a long list of songs under
    # it; "Đavoli; arr. KUD "Hartić" Tisno" on one line (composer and
    # arranger); continuation pages carry only a page number.
    ("davoli.pdf", True),
    # Text PDF (Sibelius) in the house style of Haustor: "Words & Music by
    # FILM" (Film is the band, so the composer), Horn in F AND in Eb.
    ("film.pdf", True),
    # Text PDF (Sibelius), same house style; no composer printed (a medley
    # with only "Arranged by"); several percussion parts with lists of
    # instruments ("Bass Drum, Tambourine, Wind Chimes, Shaker").
    ("indexi.pdf", False),
    # Text PDF (Sibelius), same house style; the font has no "č", so
    # "Krajač" is printed as "KrajaÈ"; "Written by" before the composers.
    ("josipa.pdf", True),
    # Text PDF (Sibelius), same house style; small "A TRIBUTE TO" above the
    # title; the band's name is the composer.
    ("prljavo.pdf", False),
    # Text PDF (Sibelius), same house style; Euphonium and Baritone each in
    # treble AND bass clef; composer "R.H.C.P." written with dots.
    ("rhcp.pdf", False),
    # US band publisher, Finale printed through a Windows driver (renamed
    # fonts): a dedication "Paul Sharle - Founder/Conductor" on every page,
    # numbers in front without a dot ("1/2 Flute", "2/3 Clarinet in Bb"),
    # "Mallet Percussion 1 (Glock/Xylo.)".
    ("sedona.pdf", False),
    # Partitura scan (wind band AND brass band set), one page per part: a
    # publisher's catalogue page after the score, many key and clef versions
    # of the same instrument ("1st & 2nd Trombones in C / in Bb / in Bb
    # bass clef"), "Flutes & Oboes" on one part, "2nd & 3rd Bb Flugel Horns".
    ("vangelis.pdf", True),
    # Text PDF (Warner, Finale): part name above the title, "Eb Alto Sax. 1,
    # 2" (the dotted "Sax." is a short form), "Percussion 1" that starts on
    # timpani (bass clef) and moves to vibes (treble clef).
    ("gershwin.pdf", False),
    # Text PDF, one set for wind band, brass band and fanfare: most parts
    # come in several identical copies (duplicates); names on two lines in a
    # box ("1st Bb TENOR" / "SAXOPHONE"), with "doubles Flute" under some;
    # a discography and a catalogue page; trombones and basses in treble
    # AND bass clef; "Fanfare part" and "Special part" sets at the end.
    ("italiano.pdf", False),
    # Light grey scan (OCR only): score first, then one part after another
    # with Roman numerals ("Bb Clarinet II", "Eb Horn III"); drums written
    # on a bass clef; no "by" before the composer "Lalo Schifrin".
    ("mission.pdf", True),
    # Light scan of a hand-copied Dutch fanfare set (OCR only): a short score
    # headed "Directie C", Dutch names and ordinals ("1e FLUIT C", "2e
    # KLARINET Bb", "1e ALT Eb (HORN I)" = alto horn, "STRIJKBAS"), the
    # movement title "I. Aruba" on every page, the composer only on the cover.
    ("suite.pdf", True),
    # Bad scan whose text layer holds only an archive stamp
    # ("ARCHIVOCUCOSAX") on every page, so it looks like a text PDF while
    # the music and names are pictures; odd score pages scanned upside down;
    # a big round library stamp next to most part names.
    ("american.pdf", True),
    # Scan (OCR only), one page per part: a condensed score "Score C",
    # a blank page, small light part names ("Flügel I", "C - Trombone I",
    # "Eb Horn I" / "Solo Horn (b.b.)"), trombones in C and in Bb.
    ("cornfield.pdf", True),
    # Scan with the scanner's own, garbled OCR text layer ("KIARINET",
    # "Webber-Fhuwast"): Dutch fanfare names, two names on two lines ("1e
    # CORNET Bb" / "1e TROMPET Bb"), extra parts in treble AND bass clef
    # after an "ADDITIONAL PARTS" page, a "Piccolo Db".
    ("superstar.pdf", True),
]


@pytest.mark.parametrize(("pdf_name", "needs_ocr"), CASES)
def test_example_is_not_worse_than_baseline(pdf_name: str, needs_ocr: bool) -> None:
    pdf = precision.EXAMPLES / pdf_name
    if not pdf.exists():
        pytest.skip(f"example file missing: {pdf_name}")
    if needs_ocr and not tesseract_available():
        pytest.skip("tesseract not installed")

    golden_path = precision.find_golden(pdf_name)
    golden = precision.load_golden(golden_path)
    actual = precision.detect(pdf)

    # First check the golden file itself (a typo there would skew the score).
    problems = precision.golden_problems(golden, actual["page_count"])
    assert not problems, f"{golden_path.name} has mistakes:\n" + "\n".join(problems)

    score = precision.score(golden, actual)
    baseline = precision.load_baseline()
    assert golden_path.stem in baseline, "no baseline: run scripts/precision.py --save-baseline"
    worse = precision.drops(score.counts(), baseline[golden_path.stem])
    # The message lists what went down, then every difference with the truth.
    assert not worse, "\n".join(["Worse than the baseline:", *worse, "", "Differences:", *score.differences])


def test_every_golden_file_has_a_case_and_a_baseline() -> None:
    """A golden file without a case here (or a baseline) would never be checked."""
    golden_names = {path.stem for path in precision.golden_files()}
    case_names = {pdf_name.removesuffix(".pdf") for pdf_name, _needs_ocr in CASES}
    assert golden_names == case_names
    assert set(precision.load_baseline()) == golden_names
