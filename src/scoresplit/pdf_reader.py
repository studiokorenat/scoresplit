"""
pdf_reader.py: read text and pixels from PDF pages with pypdfium2.

Pipeline position:  [pdf_reader] -> page_analyzer -> classify -> segment -> exporter

pypdfium2 is a Python wrapper around PDFium, Google Chrome's PDF engine. It is
licensed Apache-2.0/BSD, so it's fine for commercial use (unlike PyMuPDF,
which is AGPL).

This module does two things:

1. ``extract_text_spans(page)`` reads the EMBEDDED text of a page (PDFs
   exported from Sibelius, Finale, Dorico, MuseScore have real text) and
   groups the characters into ``TextSpan`` pieces with positions.

2. ``render_page_gray(page, dpi)`` turns a page into a grayscale picture
   (a numpy array of pixels). It's used for OCR, blank detection and clef
   detection.

Music notation programs draw the notes themselves with special MUSIC FONTS
(Opus, Bravura, Petrucci...). For PDFium those notes are "text" too: a
quarter note may come out as the letter "œ". We drop every character drawn
with a known music font, or taken from Unicode's Private Use Area (where
music fonts keep their symbols), so only real words remain.
"""

from __future__ import annotations

import ctypes
import re

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c  # the low-level C functions of PDFium

from .models import TextSpan

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Font names (or parts of them) used for music symbols. A character drawn in a
# font whose name contains one of these words is ignored. Compared in lowercase.
MUSIC_FONT_MARKERS = (
    "opus",  # Sibelius (Opus, OpusText, OpusSpecial, ...)
    "bravura",  # Dorico / MuseScore 3+
    "petrucci",  # Finale
    "maestro",  # Finale
    "engraver",  # Finale
    "jazz",  # Finale Jazz / Sibelius Inkpen
    "reprise",  # Sibelius handwritten style
    "inkpen",  # Sibelius handwritten style
    "leland",  # MuseScore 4
    "emmentaler",  # MuseScore 2 / LilyPond
    "feta",  # LilyPond
    "sonata",  # old Adobe music font
    "finale",  # Finale Ash, Finale Broadway, ...
    "musescore",  # MuseScore text-symbol fonts
    "gonville",  # MuseScore
    "mtf-",  # MuseScore
)

# Unicode's "Private Use Area" (U+E000 to U+F8FF): code points without an
# official meaning, where every font may put its own symbols. Music fonts
# put their notes, rests and clefs there (SMuFL fonts like Bravura use
# U+E000.., old Windows "symbol" fonts like Maestro come out as U+F0xx).
# Real words never use these characters, so a character from this range is
# treated as a music symbol even when the font has an anonymous name: PDFs
# printed through a Windows printer driver rename every font to something
# like "TTE19D1C00t00" (Sedona), so MUSIC_FONT_MARKERS can't recognise it.
# range(a, b) includes a but not b, hence 0xF900.
PRIVATE_USE_AREA = range(0xE000, 0xF900)

# The largest picture we ever render, in pixels (width x height). A normal A4
# page at 200 dpi is about 4 million pixels, A2 about 31 million. Without a
# limit, a malicious PDF with a giant page size (PDFs allow ~200 x 200 inches)
# would become a 1.6 GB picture at 200 dpi, in every worker process at once,
# and could crash the machine. 40 million grayscale pixels = 40 MB.
MAX_RENDER_PIXELS = 40_000_000

# Two characters on the same line further apart than this many letter heights
# start a new span (e.g. a page number far left and the part name in the middle).
MAX_GAP_IN_LETTER_HEIGHTS = 1.5

# Music-font characters that are accidentals (flat/sharp), mapped to the plain
# text we use instead. "b"/"#" are the classic Opus/Sonata codes; ♭/♯
# are the Unicode ♭/♯; / are the SMuFL codes (Bravura, Leland...).
MUSIC_FONT_ACCIDENTALS = {
    "b": "b", "♭": "b", "": "b",
    "#": "#", "♯": "#", "": "#",
}

# "arr." glued to the end of a title in CAPITALS: "DEEP PURPLE MEDLEYarr."
# (the scanner's OCR text layer of Deep Purple lost the space). Case matters
# here (no IGNORECASE): two capitals, then a small "arr", so names like
# "Barr" or "CARRY" are never cut.
GLUED_CREDIT_PATTERN = re.compile(r"(?<=[A-Z]{2})(?=arr(?:\.|\b))")


def tidy_text(text: str) -> str:
    """
    Clean up the text of one span, for embedded text and OCR alike.

    Examples:
        "  Flute   1 "                  -> "Flute 1"
        "DEEP PURPLE MEDLEYarr. Sahashi" -> "DEEP PURPLE MEDLEY arr. Sahashi"
    """
    text = GLUED_CREDIT_PATTERN.sub(" ", text)
    return " ".join(text.split())  # trims + collapses spaces


# Minimum number of real letters a page must have to be treated as a "text"
# page. Below this the page is considered scanned and needs OCR.
MIN_LETTERS_FOR_TEXT_PAGE = 5


def open_pdf(path: str) -> pdfium.PdfDocument:
    """Open a PDF file. Raises pypdfium2.PdfiumError if it's not a valid PDF."""
    return pdfium.PdfDocument(path)


def _font_name(textpage: pdfium.PdfTextPage, index: int, buffer: ctypes.Array) -> str:
    """
    Ask PDFium for the font name of character number ``index``.

    pypdfium2 has no comfortable Python wrapper for this, so we call the C
    function directly. ``buffer`` is a pre-allocated block of memory that the
    C function writes the name into (we reuse one buffer for speed).

    The C function returns the length the name needs (in bytes, including
    the closing zero byte), or 0 on failure. It ONLY writes into the buffer
    when the name fits. So we must check that number: otherwise the buffer
    still holds the previous character's font name.
    """
    flags = ctypes.c_int()  # an output parameter we don't need
    needed = pdfium_c.FPDFText_GetFontInfo(textpage.raw, index, buffer, len(buffer), ctypes.byref(flags))
    if needed == 0:
        return ""  # no font info for this character
    if needed > len(buffer):
        # The name doesn't fit: ask again with a buffer that is big enough.
        buffer = ctypes.create_string_buffer(needed)
        pdfium_c.FPDFText_GetFontInfo(textpage.raw, index, buffer, len(buffer), ctypes.byref(flags))
    return buffer.value.decode("utf-8", errors="replace").lower()


def _is_music_font(font_name: str) -> bool:
    # any(...) is True when at least one marker is part of the font name.
    return any(marker in font_name for marker in MUSIC_FONT_MARKERS)


def extract_text_spans(page: pdfium.PdfPage) -> list[TextSpan]:
    """
    Read the embedded text of a page and group it into spans.

    How it works: PDFium gives us the page's characters one by one, in the
    order they were written into the PDF, each with a bounding box. Characters
    written together (one "text object", e.g. the header "Clarinet in Bb 2")
    come in a row. PDFium also inserts "generated" line breaks (\\r\\n) between
    separate text objects. We walk the characters and start a new span at:
      * a generated line break,
      * a music-font character (a note, a dynamic),
      * a big jump in position (different line, or a large gap).

    Returns spans with normalised coordinates (0..1, origin top-left).
    """
    page_width, page_height = page.get_size()  # in PDF points
    textpage = page.get_textpage()
    font_buffer = ctypes.create_string_buffer(256)

    spans: list[TextSpan] = []

    # The span we're currently building. Using a small dict here keeps the
    # loop below easy to read. None = "no span open right now".
    current: dict | None = None

    def close_current() -> None:
        """Turn the span being built into a TextSpan and add it to the list."""
        # "nonlocal" lets this inner function change the variable `current`
        # that belongs to the outer function.
        nonlocal current
        if current is not None:
            text = tidy_text(current["text"])
            if text:
                # PDF coordinates start at the BOTTOM-left, ours at the
                # TOP-left, hence "page_height - top".
                spans.append(
                    TextSpan(
                        text=text,
                        x0=current["left"] / page_width,
                        y0=(page_height - current["top"]) / page_height,
                        x1=current["right"] / page_width,
                        y1=(page_height - current["bottom"]) / page_height,
                        size=current["size"],
                    )
                )
        current = None

    for index in range(textpage.count_chars()):
        code = pdfium_c.FPDFText_GetUnicode(textpage.raw, index)
        char = chr(code) if code else ""
        generated = bool(pdfium_c.FPDFText_IsGenerated(textpage.raw, index))

        # A generated line break means "the previous text object ended".
        if generated and char in ("\r", "\n"):
            close_current()
            continue
        # A generated space separates words: keep it, it's real spacing.
        if generated and char == " ":
            if current is not None:
                current["text"] += " "
            continue
        if not char.strip():  # a real space or other whitespace in the PDF
            if current is not None:
                current["text"] += " "
            continue

        # A music symbol: drawn with a known music font, or a character from
        # the Private Use Area (see PRIVATE_USE_AREA). The cheap range test
        # comes first: when it's True, Python skips the font lookup ("or"
        # stops at the first True). Without this, a symbol glued to text,
        # like "(end solo)" + a fermata, made one span 35 points tall, which
        # looked like a big TITLE and wrongly started a new part.
        if code in PRIVATE_USE_AREA or _is_music_font(_font_name(textpage, index, font_buffer)):
            # Exception: an accidental right after a note letter is part of
            # the text. Sibelius writes "Clarinet in B♭" as "Clarinet in B"
            # (text font) + "b" (music font, where "b" draws a flat sign).
            accidental = MUSIC_FONT_ACCIDENTALS.get(char)
            if accidental and current is not None and current["text"].rstrip()[-1:] in "ABCDEFG":
                current["text"] = current["text"].rstrip() + accidental
                accidental_right = textpage.get_charbox(index, loose=True)[2]
                current["right"] = max(current["right"], accidental_right)
                continue
            close_current()  # a note glyph ends the current piece of words
            continue

        # loose=True gives the box of the whole font line (ascender to
        # descender), which is a stable measure of the font size.
        left, bottom, right, top = textpage.get_charbox(index, loose=True)
        height = top - bottom

        if current is not None:
            same_line = abs(bottom - current["bottom"]) < 0.5 * height
            gap = left - current["right"]
            too_far = gap > MAX_GAP_IN_LETTER_HEIGHTS * height or gap < -height
            if not same_line or too_far:
                close_current()

        if current is None:
            current = {"text": "", "left": left, "right": right, "bottom": bottom,
                       "top": top, "size": height}
        current["text"] += char
        current["right"] = max(current["right"], right)
        current["top"] = max(current["top"], top)
        current["bottom"] = min(current["bottom"], bottom)
        current["size"] = max(current["size"], height)

    close_current()
    textpage.close()  # free PDFium memory
    return spans


def largest_image_share(page: pdfium.PdfPage) -> float:
    """
    How much of the page its biggest picture covers, from 0.0 to 1.0.

    A scanned page is one big picture: about 1.0. A page exported from
    notation software draws its music as lines and letters, so it has no
    picture at all (0.0), or a small logo. That tells a scan WITH a text
    layer (the scanner's own OCR, often garbled) apart from a real text page.

    Only the picture's box is measured (no pixels are decoded), so this is
    fast: a few milliseconds per page.
    """
    page_width, page_height = page.get_size()
    if page_width <= 0 or page_height <= 0:
        return 0.0
    largest = 0.0
    # filter=[...IMAGE] lists only the pictures; max_depth=2 also looks one
    # level inside "form XObjects" (groups of objects some PDF tools use).
    for image in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=2):
        left, bottom, right, top = image.get_bounds()  # in PDF points
        share = max(0.0, right - left) * max(0.0, top - bottom) / (page_width * page_height)
        largest = max(largest, share)
    # min(1.0, ...): a picture can stick out past the page edge.
    return min(1.0, largest)


def count_letters(spans: list[TextSpan]) -> int:
    """How many alphabetic characters these spans contain in total."""
    # A "generator expression" inside sum(): for every span, for every
    # character, count 1 if it's a letter. Like nested foreach loops in PHP.
    return sum(1 for span in spans for char in span.text if char.isalpha())


def fit_dpi(width_pt: float, height_pt: float, dpi: int) -> int:
    """
    The resolution to render an area of `width_pt` x `height_pt` points at:
    `dpi` itself, or less if the picture would be bigger than MAX_RENDER_PIXELS.

    Normal pages always get the full `dpi`; only oversized pages are lowered.
    Use the returned value everywhere the picture's dpi matters (OCR uses it
    to turn pixel sizes back into font sizes).

    Examples (MAX_RENDER_PIXELS = 40 million):
        fit_dpi(595, 842, 200)       -> 200   (A4: about 4 million pixels)
        fit_dpi(14400, 14400, 200)   -> 31    (200 x 200 inch page)
    """
    # Points -> inches: divide by 72. Pixels = inches x dpi, in both directions.
    pixels = (width_pt / 72 * dpi) * (height_pt / 72 * dpi)
    if pixels <= MAX_RENDER_PIXELS:
        return dpi
    # The pixel count grows with dpi squared, so shrink dpi by the square root
    # of how much too big the picture is. int() rounds down: stay under the limit.
    # max(1, ...) keeps at least 1 dpi, because a dpi of 0 would make an empty picture.
    return max(1, int(dpi * (MAX_RENDER_PIXELS / pixels) ** 0.5))


def render_page_gray(page: pdfium.PdfPage, dpi: int) -> np.ndarray:
    """
    Render a page as a grayscale image at the given resolution.

    Returns a 2D numpy array (rows x columns) of numbers 0..255, where 0 is
    black and 255 is white. numpy arrays are fast "grids of numbers": you can
    do maths on all pixels at once instead of looping in Python.
    """
    # PDF size is in points (72 per inch), so scale = dpi / 72.
    bitmap = page.render(scale=dpi / 72, grayscale=True)
    pixels = bitmap.to_numpy()
    # Some PDFium builds return shape (rows, cols, 1) for grayscale. Make it 2D.
    if pixels.ndim == 3:
        pixels = pixels[:, :, 0]
    return pixels.copy()  # copy so the array stays valid after the bitmap is freed


def render_region_gray(
    page: pdfium.PdfPage, region: tuple[float, float, float, float], dpi: int
) -> np.ndarray:
    """
    Render only part of a page, e.g. the header strip at high resolution.

    ``region`` is (x0, y0, x1, y1) in normalised coordinates (0..1, top-left
    origin). PDFium's ``crop`` wants the amount to cut away from each side
    (left, bottom, right, top) in points, so we convert.
    """
    width, height = page.get_size()
    x0, y0, x1, y1 = region
    crop = (x0 * width, (1 - y1) * height, (1 - x1) * width, y0 * height)
    bitmap = page.render(scale=dpi / 72, grayscale=True, crop=crop)
    pixels = bitmap.to_numpy()
    if pixels.ndim == 3:
        pixels = pixels[:, :, 0]
    return pixels.copy()
