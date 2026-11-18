"""
ocr.py: read text from scanned pages with Tesseract OCR.

Pipeline position:  pdf_reader (pixels) -> [ocr] -> page_analyzer

The OCR engine comes from the ``tesserocr`` package: a Python wrapper around
the Tesseract C library, with its own copy of Tesseract inside. It runs
inside our own process, so no separate ``tesseract`` program is started for
each image. The engine is created once per worker process (loading the
language models takes ~1 s) and then reused for every region.

The language files (``eng.traineddata``, ``hrv.traineddata``, ...) still come
from the system package manager: ``tesseract`` + ``tesseract-data-eng`` on
Arch, ``tesseract-ocr`` on Ubuntu. See tessdata_folder() for how we find them.

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
    --psm 7   "a single text line": for one line of the header at a time.
    --psm 3   "fully automatic": for whole pages (cover pages).

IMPORTANT (the "96 dpi trap"): Tesseract must get the raw pixels with NO
resolution, exactly as 0.2.0 did (it saved a PNG without a resolution, so
Tesseract estimated one). Never hand it a Pillow image with
``api.SetImage(image)``: tesserocr then saves a BMP, which says 96 dpi, and
Tesseract reads different words. See _read_words().
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

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

# The columns of Tesseract's TSV output ("tab-separated values": one line per
# page, block, paragraph, line and word, with the values separated by tabs).
# This is the header line the `tesseract` program writes; GetTSVText leaves
# the header out, so we name the columns ourselves. "text" is the last one.
TSV_COLUMNS = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)

# Finds the folder in the first line of `tesseract --list-langs`, e.g.
#     List of available languages in "/usr/share/tessdata/" (3):
# The brackets (...) "capture" what is between the double quotes, and
# [^"]+ means "one or more characters that are not a double quote".
LIST_LANGS_FOLDER_PATTERN = re.compile(r'"([^"]+)"')


class OcrUnavailableError(RuntimeError):
    """Raised when a scanned page needs OCR but Tesseract isn't installed."""


@lru_cache(maxsize=1)
def tessdata_folder() -> str | None:
    """
    The folder with Tesseract's language files, or None if none is found.

    tesserocr can't find this folder on its own, so we look, in this order:
      1. the TESSDATA_PREFIX environment variable, if it is set (the
         `tesseract` program uses the same variable);
      2. the folder that `tesseract --list-langs` prints in its first line,
         when the `tesseract` program is installed.

    Runs at most once per process; lru_cache remembers the answer.
    """
    # 1) The environment variable. os.environ.get returns None when it isn't
    #    set; "or None" also turns an empty value ("") into None.
    from_environment = os.environ.get("TESSDATA_PREFIX") or None
    if from_environment is not None:
        return from_environment

    # 2) Ask the `tesseract` program. shutil.which is like the shell command
    #    `which tesseract`: it returns None when the program isn't installed.
    if shutil.which("tesseract") is None:
        return None
    result = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True, check=False
    )
    # Some Tesseract versions print the list on stderr instead of stdout, so
    # look in both. The folder is in the first line that has double quotes.
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        match = LIST_LANGS_FOLDER_PATTERN.search(line)
        if match:
            return match.group(1)  # group(1) = the text inside the quotes
    return None


@lru_cache(maxsize=1)
def installed_languages() -> set[str]:
    """
    Which Tesseract language packs are installed, e.g. {"eng", "hrv", "osd"}.

    Every ``<name>.traineddata`` file in the language folder is one language,
    the same list ``tesseract --list-langs`` prints. (Files in a subfolder,
    like ``script/Latin.traineddata``, are named "script/Latin", as Tesseract
    names them.)

    We list the files ourselves instead of asking tesserocr, so the main
    process never loads tesserocr: only the worker processes do (see
    _engine).
    """
    folder = tessdata_folder()
    if folder is None or not Path(folder).is_dir():
        return set()
    languages = set()
    # rglob finds files in the folder AND in all its subfolders.
    for file in Path(folder).rglob("*.traineddata"):
        # The path inside the folder, without ".traineddata":
        # /usr/share/tessdata/script/Latin.traineddata -> "script/Latin".
        # as_posix() writes it with "/", also on Windows.
        languages.add(file.relative_to(folder).with_suffix("").as_posix())
    return languages


def tesseract_available() -> bool:
    """True if OCR can run: the language folder exists and has languages."""
    # bool() of a set is False when the set is empty.
    return bool(installed_languages())


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

    # _read_words returns a dict of parallel lists: data["text"][i] is word
    # number i, data["left"][i] its x position, and so on.
    data = _read_words(image_gray, psm, lang)

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


# ---------------------------------------------------------------------------
# The OCR engine
# ---------------------------------------------------------------------------

# The engines made so far in this process, one per language string:
# {"eng+hrv": <tesserocr engine>}. A module-level variable like this lives
# as long as the process, so every page a worker analyses reuses the engine.
_engines: dict[str, object] = {}

# The process the engines in _engines were made in (os.getpid() = this
# process's id number). See _engine for why this matters.
_engines_pid: int | None = None


def _engine(lang: str):
    """
    The Tesseract engine for ``lang`` (e.g. "eng+hrv"), made on first use.

    Making an engine loads the language models, which takes about a second;
    reading a small image with a loaded engine takes a few hundredths of a
    second. So each process makes one engine per language and keeps it.

    One engine per PROCESS: an engine must never be used by two processes
    or two threads at once. A worker process started with "fork" gets a copy
    of its parent's variables, so if the parent had made an engine, the
    worker would see it in _engines. Checking the process id catches that,
    and the worker then makes its own.

    Raises:
        OcrUnavailableError: the language files can't be found or loaded.
    """
    global _engines_pid  # "global": this function changes the module variable
    if _engines_pid != os.getpid():
        # A new process (or the first call): start with no engines. The old
        # ones belong to the parent process, so we leave them alone.
        _engines.clear()
        _engines_pid = os.getpid()

    if lang in _engines:
        return _engines[lang]

    folder = tessdata_folder()
    if folder is None:
        raise OcrUnavailableError(
            "Tesseract's language files were not found: the 'tesseract' "
            "program is not installed and TESSDATA_PREFIX is not set"
        )

    # Imported here, not at the top of the file: loading tesserocr also loads
    # the Tesseract library, which reads settings like OMP_THREAD_LIMIT at
    # that moment. Importing it only when the first engine is made means the
    # main process never loads it, and each worker sets its settings first
    # (see page_analyzer._init_worker).
    import tesserocr

    # Silence Leptonica, the image library inside tesserocr. It writes
    # messages like "Error in boxClipToRectangle: box outside rectangle"
    # straight to stderr, which is the user's terminal (and where the web
    # tool reads the progress). They are harmless: Tesseract carries on and
    # the words don't change. 0.2.0 never showed them, because the separate
    # `tesseract` program's messages were caught by pytesseract. The setting
    # is for the whole process, so setting it again for a second engine is
    # fine (and cheap).
    tesserocr.set_leptonica_log_level(tesserocr.LeptLogLevel.NONE)

    try:
        # path = the language folder, lang = e.g. "eng+hrv". The engine mode
        # (oem) stays at Tesseract's default, the same as the `tesseract`
        # program used in 0.2.0.
        engine = tesserocr.PyTessBaseAPI(path=folder, lang=lang)
    except RuntimeError as error:
        # tesserocr raises RuntimeError when it can't load the languages,
        # e.g. "Failed to init API, possibly an invalid tessdata path".
        raise OcrUnavailableError(
            f"Tesseract could not load the language(s) '{lang}' from {folder}: {error}"
        ) from error
    _engines[lang] = engine
    return engine


def _read_words(image_gray: np.ndarray, psm: int, lang: str) -> dict[str, list]:
    """
    Let Tesseract read an image; return its words as parallel lists.

    The result looks exactly like what pytesseract.image_to_data(...,
    output_type=DICT) gave in 0.2.0, so ocr_image didn't have to change:
        {"text": ["Trumpet", "1"], "left": [120, 410], "conf": [91, 88], ...}
    Every column of TSV_COLUMNS is a key; "text" holds strings, every other
    column whole numbers.
    """
    # Tesseract wants one byte per pixel (0 = black, 255 = white) in rows of
    # `width` bytes. Every image we OCR is rendered that way (8-bit grayscale,
    # numpy dtype uint8), but check, because anything else would be read as
    # garbage.
    if image_gray.ndim != 2 or image_gray.dtype != np.uint8:
        raise ValueError(
            f"OCR needs a 2-D grayscale uint8 image, got shape {image_gray.shape} "
            f"and dtype {image_gray.dtype}"
        )
    height, width = image_gray.shape  # numpy shape is (rows, columns)

    # tobytes() copies the pixels into one plain block of bytes, row after
    # row. (A crop made by slicing is only a "view" into the whole page, so
    # its rows aren't next to each other in memory; tobytes() fixes that.)
    # Keep the bytes in a variable: tesserocr does NOT copy them, so they
    # must stay alive until Tesseract has read them (in GetTSVText below).
    pixels = image_gray.tobytes()

    engine = _engine(lang)
    # The page segmentation mode can change for every image; changing it
    # doesn't reload the language models.
    engine.SetPageSegMode(psm)
    # The raw pixels, with NO resolution: Tesseract then estimates one, as it
    # did in 0.2.0. (The 96 dpi trap: never use engine.SetImage(pil_image).)
    # Arguments: pixels, width, height, bytes per pixel (1 = grayscale),
    # bytes per row (= width, since each pixel is one byte).
    engine.SetImageBytes(pixels, width, height, 1, width)
    # GetTSVText runs the recognition and returns the words as TSV text.
    # The 0 is the page number (0-based); we read one image at a time.
    tsv = engine.GetTSVText(0)

    # Turn the TSV text into parallel lists, the way pytesseract did:
    # numbers with int(float(value)), so a confidence of "91.5" becomes 91.
    data: dict[str, list] = {column: [] for column in TSV_COLUMNS}
    for line in tsv.splitlines():
        if not line.strip():
            continue  # skip empty lines
        values = line.split("\t")
        # A row for a block, paragraph or line has an empty text at the end,
        # which may be missing. Pad the row with "" up to the full length.
        values += [""] * (len(TSV_COLUMNS) - len(values))
        # zip pairs each column name with its value: ("level", "5"), ...
        for column, value in zip(TSV_COLUMNS, values):
            if column == "text":
                data[column].append(value)
            else:
                data[column].append(int(float(value)))
    return data


# ---------------------------------------------------------------------------
# Turning words into text spans
# ---------------------------------------------------------------------------


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
