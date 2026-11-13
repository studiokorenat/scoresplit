"""
ocr.py: read text from scanned pages with Tesseract OCR.

Pipeline position:  pdf_reader (pixels) -> [ocr] -> page_analyzer

Tesseract is a separate program (installed with the system package manager:
``tesseract`` + ``tesseract-data-eng`` on Arch, ``tesseract-ocr`` on Ubuntu).
The ``pytesseract`` library just runs that program and parses its output.

We never OCR a whole page unless we must: it's slow, and staff lines and
notes produce lots of garbage "letters". Instead we OCR *regions*: the top
strip (where the part name is) and the left margin (where a conductor score
has its instrument labels).

Tesseract "page segmentation modes" (PSM) used here:
    --psm 11  "sparse text": find as much text as possible in no particular
              order. Best for headers, where title, part name and
              composer are scattered around.
    --psm 6   "a single uniform block of text": a different algorithm, used
              as a second try when psm 11 found no part name.
    --psm 3   "fully automatic": for whole pages (cover pages).
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache

import numpy as np
import pytesseract
from PIL import Image

from .models import TextSpan
from .pdf_reader import tidy_text

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Tesseract gives each word a confidence 0..100. Words below this are
# ignored (they're usually bits of notes or staff lines read as letters).
MIN_WORD_CONFIDENCE = 30

# Two words on the same OCR line further apart than this many word heights
# are split into separate spans (e.g. the title and the composer, when
# Tesseract reads them as one line).
MAX_WORD_GAP_IN_HEIGHTS = 2.5


class OcrUnavailableError(RuntimeError):
    """Raised when a scanned page needs OCR but Tesseract isn't installed."""


def tesseract_available() -> bool:
    """True if the ``tesseract`` program can be found on the PATH."""
    # shutil.which is like the shell command `which tesseract`.
    return shutil.which("tesseract") is not None


@lru_cache(maxsize=1)
def installed_languages() -> set[str]:
    """
    Which Tesseract language packs are installed, e.g. {"eng", "hrv", "osd"}.

    Runs ``tesseract --list-langs`` once and remembers the answer.
    """
    if not tesseract_available():
        return set()
    result = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True, check=False
    )
    # The first output line is a header ("List of available languages ..."),
    # the rest are language codes, one per line.
    lines = result.stdout.strip().splitlines()[1:]
    return {line.strip() for line in lines if line.strip()}


def resolve_languages(requested: str) -> tuple[str, list[str]]:
    """
    Keep only the requested languages that are actually installed.

    ``requested`` uses Tesseract's syntax: "eng+hrv". Returns the usable
    language string plus a list of the missing ones (so the CLI can warn).

    Example (only English installed):
        resolve_languages("eng+hrv") -> ("eng", ["hrv"])
    """
    wanted = [code for code in requested.split("+") if code]
    available = installed_languages()
    usable = [code for code in wanted if code in available]
    missing = [code for code in wanted if code not in available]
    if not usable:
        # Fall back to English if it exists, otherwise to whatever is there.
        usable = ["eng"] if "eng" in available else sorted(available - {"osd"})[:1]
    return "+".join(usable), missing


def ocr_region(
    gray: np.ndarray,
    region: tuple[float, float, float, float],
    dpi: int,
    psm: int,
    lang: str,
) -> list[TextSpan]:
    """
    OCR one rectangular region of a rendered page.

    Args:
        gray:   the WHOLE page as a grayscale numpy array (see pdf_reader).
        region: (x0, y0, x1, y1) in normalised page coordinates, e.g.
                (0, 0, 1, 0.15) = the top 15% of the page.
        dpi:    the resolution `gray` was rendered at (to convert pixel
                heights back to points).
        psm:    Tesseract page segmentation mode (see module docstring).
        lang:   Tesseract language string, e.g. "eng+hrv".

    Returns:
        TextSpans with coordinates normalised to the WHOLE page, so they can
        be mixed with spans from other regions.
    """
    page_h, page_w = gray.shape  # numpy shape is (rows, columns) = (height, width)
    x0, y0, x1, y1 = region
    # Array slicing [rows, columns] cuts out the region, like cropping.
    crop = gray[int(y0 * page_h) : int(y1 * page_h), int(x0 * page_w) : int(x1 * page_w)]
    return ocr_image(crop, region, dpi, psm, lang)


def ocr_image(
    image_gray: np.ndarray,
    region: tuple[float, float, float, float],
    dpi: int,
    psm: int,
    lang: str,
) -> list[TextSpan]:
    """
    OCR an image that shows `region` of a page (already cut out).

    Same as ocr_region, but for when the region was rendered on its own
    (e.g. the header strip re-rendered at a higher resolution).
    """
    if image_gray.size == 0:  # an empty crop: nothing to read
        return []
    image = Image.fromarray(image_gray)

    # image_to_data returns a dict of parallel lists: data["text"][i] is word
    # number i, data["left"][i] its x position, and so on.
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )

    # Group the words by the line Tesseract put them on.
    # Key = (block, paragraph, line) numbers; value = list of word dicts.
    lines: dict[tuple[int, int, int], list[dict]] = {}
    for i, word in enumerate(data["text"]):  # enumerate gives (index, value) pairs
        word = word.strip()
        if not word or float(data["conf"][i]) < MIN_WORD_CONFIDENCE:
            continue
        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        # setdefault returns the existing list, or stores and returns a new one.
        lines.setdefault(line_key, []).append(
            {
                "text": word,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )

    image_h, image_w = image_gray.shape
    spans: list[TextSpan] = []
    for words in lines.values():
        # Sort the words left to right. "key=lambda w: w['left']" means "sort
        # by each word's left value" (lambda = a tiny inline function, like
        # PHP's fn($w) => $w['left']).
        words.sort(key=lambda w: w["left"])
        for group in _split_far_apart_words(words):
            spans.append(_words_to_span(group, region, image_w, image_h, dpi))
    return spans


def _split_far_apart_words(words: list[dict]) -> list[list[dict]]:
    """
    Split one OCR line into groups where words are close together.

    "NUOVO CINEMA PARADISO            Ennio Morricone" becomes two groups.
    """
    groups: list[list[dict]] = [[words[0]]]
    for word in words[1:]:
        previous = groups[-1][-1]  # [-1] = last element
        gap = word["left"] - (previous["left"] + previous["width"])
        height = max(word["height"], previous["height"])
        if gap > MAX_WORD_GAP_IN_HEIGHTS * height:
            groups.append([word])  # too far: start a new group
        else:
            groups[-1].append(word)
    return groups


def _words_to_span(
    words: list[dict],
    region: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    dpi: int,
) -> TextSpan:
    """
    Combine OCR words into one TextSpan.

    Word positions are in pixels of the region image; they are converted to
    normalised WHOLE-page coordinates: region start + (pixel / image size)
    * region size.
    """
    rx0, ry0, rx1, ry1 = region
    left = min(w["left"] for w in words)
    top = min(w["top"] for w in words)
    right = max(w["left"] + w["width"] for w in words)
    bottom = max(w["top"] + w["height"] for w in words)
    # The tallest word height in pixels, converted to points (72 per inch).
    size = max(w["height"] for w in words) * 72 / dpi
    return TextSpan(
        text=tidy_text(" ".join(w["text"] for w in words)),
        x0=rx0 + left / image_w * (rx1 - rx0),
        y0=ry0 + top / image_h * (ry1 - ry0),
        x1=rx0 + right / image_w * (rx1 - rx0),
        y1=ry0 + bottom / image_h * (ry1 - ry0),
        size=size,
    )
