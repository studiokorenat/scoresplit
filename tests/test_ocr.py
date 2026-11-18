"""
Tests for ocr.py: the OCR engine (tesserocr) and finding the language files.

Every image here is drawn by the test itself with Pillow (no pages from the
copyrighted example PDFs). The font is Pillow's own built-in one, so the
pictures come out the same on every computer.

Three groups:
  * reading words: a clean header must read as the printed words, and a
    noisy header must read EXACTLY as scoresplit 0.2.0 read it. The second
    test catches the "96 dpi trap" (see ocr.py's module docstring): an image
    handed to Tesseract with a resolution reads different words;
  * finding the language folder: TESSDATA_PREFIX, the first line of
    `tesseract --list-langs`, or nothing found;
  * Leptonica (the image library inside tesserocr) must stay quiet.

Run with:  uv run pytest tests/test_ocr.py
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from scoresplit import ocr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The sha256 fingerprint of the English language file these tests' exact
# words were recorded with: the standard eng.traineddata from
# github.com/tesseract-ocr/tessdata 4.1.0 (Arch's tesseract-data-eng, 23 MB).
# Other English files (e.g. the smaller "fast" one some Linux systems ship)
# read noisy images differently, so the exact-words test is skipped there.
STANDARD_ENGLISH_SHA256 = "daa0c97d651c19fba3b25e81317cd697e9908c8208090c94c3905381c23fc047"


def _skip_without_english() -> None:
    """Skip the calling test when Tesseract's English language isn't installed."""
    if "eng" not in ocr.installed_languages():
        pytest.skip("Tesseract's English language file (eng.traineddata) is not installed")


def _skip_without_the_standard_english_file() -> None:
    """Skip the calling test unless eng.traineddata is the one the words were recorded with."""
    _skip_without_english()
    english_file = Path(ocr.tessdata_folder()) / "eng.traineddata"
    fingerprint = hashlib.sha256(english_file.read_bytes()).hexdigest()
    if fingerprint != STANDARD_ENGLISH_SHA256:
        pytest.skip(f"{english_file} is not the standard tessdata 4.1.0 English file")


def _header_image(text_size: int, specks: int, width: int) -> np.ndarray:
    """
    Draw a header strip like the top of a scanned part, as a numpy array.

    Two lines of text (a tempo and a part name, then the arranger), five
    staff lines below them, and `specks` small dots spread over the strip
    like the dirt on a scan. The dots are placed by a fixed calculation (no
    random numbers), so the picture is the same every time.

    Returns a grayscale image: a 2-D numpy array of uint8 (0 = black,
    255 = white), the same kind of image pdf_reader renders.
    """
    height = text_size * 6
    image = Image.new("L", (width, height), 255)  # "L" = 8-bit grayscale, white
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=text_size)  # Pillow's built-in font
    small_font = ImageFont.load_default(size=text_size * 2 // 3)

    draw.text((20, text_size // 2), "Moderato Swing", fill=0, font=font)
    draw.text((width // 2, text_size // 2), "Trumpet in Bb 1", fill=0, font=font)
    draw.text((20, text_size * 2), "Arr. John Smith", fill=0, font=small_font)

    # Five staff lines, a few pixels apart, across the whole width.
    for line in range(5):
        y = int(text_size * 3.6) + line * max(3, text_size // 5)
        draw.line((0, y, width, y), fill=0, width=1)

    # The dirt: dot number i goes to a position computed from i. 7919 and
    # 104729 are prime numbers, so "% width" and "% height" spread the dots
    # all over the strip instead of lining them up. The dot size cycles
    # through 1, 2, 2 and 3 pixels.
    for i in range(specks):
        x = (i * 7919) % width
        y = (i * 104729) % height
        radius = (0, 1, 1, 2)[i % 4]
        draw.ellipse((x, y, x + radius, y + radius), fill=0)

    return np.asarray(image)


def _texts(image: np.ndarray, psm: int = 11) -> list[str]:
    """The texts ocr_image reads in `image` (English only), in its order."""
    # The region (the whole top 10% of a page) and the dpi only change where
    # the spans are placed and their size, not which words are read.
    spans = ocr.ocr_image(image, (0.0, 0.0, 1.0, 0.1), 200, psm, "eng")
    return [span.text for span in spans]


# ---------------------------------------------------------------------------
# Reading words
# ---------------------------------------------------------------------------


def test_a_clean_header_reads_as_printed() -> None:
    _skip_without_english()
    image = _header_image(text_size=28, specks=0, width=1400)
    spans = ocr.ocr_image(image, (0.0, 0.0, 1.0, 0.1), 200, 11, "eng")
    # Compared without spaces: Pillow's font sets "Trumpet in" so tightly
    # that Tesseract may read it as one word, and that's not what this test
    # is about.
    joined = "".join(span.text for span in spans).replace(" ", "")
    assert "ModeratoSwing" in joined
    assert "TrumpetinBb1" in joined
    assert "Arr.JohnSmith" in joined

    # The part name was drawn in the right half of the strip, so its span
    # must start in the right half of the page (the region is the whole
    # width, so page x = image x).
    trumpet = next(span for span in spans if span.text.startswith("Trumpet"))
    assert trumpet.x0 > 0.5
    # ... and within the region: the top 10% of the page.
    assert 0.0 <= trumpet.y0 < trumpet.y1 <= 0.1


def test_a_noisy_header_reads_exactly_as_in_0_2_0() -> None:
    # The 96 dpi trap. Tesseract must get the raw pixels with NO resolution,
    # as 0.2.0 did (it saved a PNG without one, and the `tesseract` program
    # estimated it). Handed over WITH a resolution, e.g. as a Pillow image
    # through engine.SetImage(...), which tesserocr turns into a 96 dpi BMP,
    # Tesseract reads other words on noisy images like this one.
    #
    # Checked when this test was written (2026-11-18):
    #   * the `tesseract` program 5.5.3 (the way 0.2.0 read it: a PNG, no
    #     resolution, --psm 11 -l eng) gives exactly these words, with the
    #     same confidences and boxes;
    #   * with SetImage(Image.fromarray(image)) in _read_words, the middle
    #     line becomes "Bb 1", so this test fails.
    #
    # The words look odd ("l.3b", "Johi") because the dots and staff lines
    # confuse Tesseract, as on a real scan. What matters is that they stay
    # the same as in 0.2.0.
    _skip_without_the_standard_english_file()
    image = _header_image(text_size=22, specks=150, width=1400)
    assert _texts(image) == ["Moderato swing", "...Trum.pet.in l.3b 1", "Johi Smith"]


def test_the_page_segmentation_mode_can_change_between_images() -> None:
    # One engine reads every region of a worker, with psm 11, 6 and 7 mixed.
    # A psm used for one image must not stick to the next one: reading the
    # same image with psm 11, then 7, then 11 again gives the first result
    # again.
    _skip_without_english()
    image = _header_image(text_size=28, specks=0, width=1400)
    first = _texts(image, psm=11)
    _texts(image, psm=7)
    assert _texts(image, psm=11) == first


def test_only_grayscale_uint8_images_are_read() -> None:
    # A colour image (3 numbers per pixel) would be read as garbage, so
    # _read_words refuses it. (The check runs before the engine is made, so
    # no language files are needed.)
    colour = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D grayscale uint8"):
        ocr._read_words(colour, 11, "eng")


# ---------------------------------------------------------------------------
# Finding the language folder
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_folder_search():
    """
    Forget the folder and languages found so far, before AND after a test.

    tessdata_folder() and installed_languages() remember their answer for
    the whole process (lru_cache). A test that changes TESSDATA_PREFIX or
    the `tesseract` program must make them look again, and afterwards the
    other tests must not keep the test's made-up answer.
    """
    ocr.tessdata_folder.cache_clear()
    ocr.installed_languages.cache_clear()
    yield  # the test runs here
    ocr.tessdata_folder.cache_clear()
    ocr.installed_languages.cache_clear()


def _fake_tesseract(monkeypatch: pytest.MonkeyPatch, stdout: str, stderr: str = "") -> None:
    """
    Pretend the `tesseract` program is installed and prints this list.

    monkeypatch.setattr swaps a function for the length of one test:
    shutil.which then finds "tesseract", and subprocess.run returns our text
    instead of starting the program.
    """

    def fake_which(name: str) -> str | None:
        return "/usr/bin/tesseract" if name == "tesseract" else None

    def fake_run(command, **kwargs):
        assert command == ["tesseract", "--list-langs"]
        # CompletedProcess is what subprocess.run normally returns.
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(ocr.shutil, "which", fake_which)
    monkeypatch.setattr(ocr.subprocess, "run", fake_run)


def test_tessdata_prefix_wins(fresh_folder_search, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESSDATA_PREFIX", "/my/tessdata")
    # Even with the program installed, the variable is used first.
    _fake_tesseract(monkeypatch, 'List of available languages in "/usr/share/tessdata/" (1):\neng\n')
    assert ocr.tessdata_folder() == "/my/tessdata"


def test_the_folder_comes_from_the_first_line_of_list_langs(
    fresh_folder_search, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)  # make sure it isn't set
    _fake_tesseract(
        monkeypatch, 'List of available languages in "/usr/share/tessdata/" (3):\neng\nhrv\nosd\n'
    )
    assert ocr.tessdata_folder() == "/usr/share/tessdata/"


def test_list_langs_on_stderr_is_read_too(fresh_folder_search, monkeypatch: pytest.MonkeyPatch) -> None:
    # Some Tesseract versions print the list on stderr instead of stdout.
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    _fake_tesseract(monkeypatch, "", stderr='List of available languages in "/opt/tessdata" (1):\neng\n')
    assert ocr.tessdata_folder() == "/opt/tessdata"


def test_an_empty_tessdata_prefix_counts_as_not_set(
    fresh_folder_search, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TESSDATA_PREFIX", "")
    _fake_tesseract(monkeypatch, 'List of available languages in "/usr/share/tessdata/" (1):\neng\n')
    assert ocr.tessdata_folder() == "/usr/share/tessdata/"


def test_nothing_found_without_the_variable_and_the_program(
    fresh_folder_search, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    # shutil.which returns None: "no such program".
    monkeypatch.setattr(ocr.shutil, "which", lambda name: None)
    assert ocr.tessdata_folder() is None
    assert ocr.installed_languages() == set()
    assert ocr.tesseract_available() is False
    # Making an engine then fails with a clear message (the CLI turns it
    # into "OCR missing", exit code 3).
    ocr._engines.clear()
    with pytest.raises(ocr.OcrUnavailableError, match="language files were not found"):
        ocr._engine("eng")


def test_a_list_without_a_folder_finds_nothing(
    fresh_folder_search, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The program is there, but its output has no folder in quotes.
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    _fake_tesseract(monkeypatch, "something unexpected\n")
    assert ocr.tessdata_folder() is None


def test_installed_languages_are_the_traineddata_files(
    fresh_folder_search, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A made-up language folder: two languages, one script in a subfolder,
    # and a file that isn't a language.
    (tmp_path / "eng.traineddata").write_bytes(b"")
    (tmp_path / "hrv.traineddata").write_bytes(b"")
    (tmp_path / "script").mkdir()
    (tmp_path / "script" / "Latin.traineddata").write_bytes(b"")
    (tmp_path / "pdf.ttf").write_bytes(b"")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    assert ocr.installed_languages() == {"eng", "hrv", "script/Latin"}
    assert ocr.resolve_languages("eng+hrv+deu") == ("eng+hrv", ["deu"])


def test_a_folder_that_does_not_exist_has_no_languages(
    fresh_folder_search, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "missing"))
    assert ocr.installed_languages() == set()
    assert ocr.tesseract_available() is False


# ---------------------------------------------------------------------------
# Leptonica stays quiet
# ---------------------------------------------------------------------------


def test_making_an_engine_silences_leptonica(monkeypatch: pytest.MonkeyPatch) -> None:
    # Leptonica writes messages like "Error in boxClipToRectangle: box
    # outside rectangle" straight to stderr (american.pdf printed 108 lines
    # during `inspect` before this was fixed). _engine must switch them off.
    #
    # Why we check the call instead of making Leptonica talk: on generated
    # images we found no way to make it print a message that doesn't also
    # crash Tesseract (the messages on american.pdf come from Tesseract's
    # layout analysis of one particular scanned margin). The before/after
    # count on american.pdf is in the v3 log.
    _skip_without_english()
    import tesserocr

    levels = []  # every log level _engine sets, in order
    real_setter = tesserocr.set_leptonica_log_level

    def recording_setter(level):
        levels.append(level)
        real_setter(level)  # still really set it

    # _engine does "import tesserocr" and then uses tesserocr.<name>, so it
    # finds our recording function on the module.
    monkeypatch.setattr(tesserocr, "set_leptonica_log_level", recording_setter)
    ocr._engines.clear()  # make _engine create a new engine
    ocr._engine("eng")
    assert levels == [tesserocr.LeptLogLevel.NONE]
