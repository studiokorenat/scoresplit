"""
api.py: the public Python interface of scoresplit; the CLI is one user of it.

Everything scoresplit DOES lives here, with plain inputs and outputs:

    inspect()      PDF -> Result                (detect the parts, slow)
    save_edits()   Result -> JSON file
    load_edits()   JSON file -> Result          (checks it belongs to the PDF)
    check_edits()  Result -> CheckReport        (no errors = ready to split)
    split()        PDF + Result -> SplitResult  (writes one PDF per part)

and for the saved edits that the editor keeps next to the PDF
(score.pdf -> score.scoresplit.json):

    saved_edits_path()  PDF -> where its saved edits are
    load_saved_edits()  PDF -> Result, or None when nothing was saved

The Result (models.py) is the one shape of the detection: in memory, in the
saved JSON that a person or a GUI edits, and in the --json output.

Nothing here prints, asks questions or exits the program. When something
goes wrong, a ScoresplitError is raised; each kind carries the exit code
the CLI uses for it, so a caller can tell "bad input" from "Tesseract is
missing" without reading the message. (Like a Laravel service class that
a controller and an Artisan command can both call.)

Example, from other Python code:

    from pathlib import Path
    from scoresplit import api

    result = api.inspect(Path("score.pdf"))
    written = api.split(Path("score.pdf"), result, Path("out"))
    for file in written.files:
        if file.part is not None:  # a part, not extra pages or an export file
            part = result.parts[file.part]
            print(file.relative_path, part.instrument, part.key)

cli.py turns these functions into the `scoresplit` terminal command: it
adds the progress bar, the parts table, the editor (editor.py), --json
output and the exit codes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c  # PDFium's own constants (error codes)
from pydantic import BaseModel, ValidationError

from . import exporter
from .cache import file_sha256
from .exporter import ExportedFile, InvalidResultError, check_export, export_parts
from .models import SCHEMA_VERSION, CheckReport, ProgressCallback, Result
from .ocr import OcrUnavailableError, resolve_languages
from .page_analyzer import AnalysisSettings
from .pipeline import inspect_pdf

# ProgressCallback is imported from models only to be offered here: a
# caller can write `api.ProgressCallback` without knowing where it lives.
# __all__ lists the names that make up this module's public interface.
__all__ = [
    "BadInputError",
    "CheckReport",
    "InspectOptions",
    "InvalidEditsError",
    "OcrMissingError",
    "OutputExistsError",
    "OutputWriteError",
    "ProgressCallback",
    "Result",
    "ScoresplitError",
    "SplitResult",
    "WrittenFile",
    "check_edits",
    "inspect",
    "load_edits",
    "load_saved_edits",
    "output_folder",
    "require_file",
    "save_edits",
    "saved_edits_path",
    "split",
]

# ---------------------------------------------------------------------------
# Exit codes. The `scoresplit` command ends with one of these, and every
# ScoresplitError below carries one. They are part of the contract with
# programs that run scoresplit: never change what a number means.
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_CANCELLED = 1  # the user answered "no" (only the CLI asks)
EXIT_BAD_INPUT = 2
EXIT_OCR_MISSING = 3
EXIT_OUTPUT_EXISTS = 4
EXIT_ERROR = 5

# The saved edits of "score.pdf" are kept next to it in
# "score.scoresplit.json" (see saved_edits_path).
SAVED_EDITS_SUFFIX = ".scoresplit.json"

# Tesseract languages used when the caller doesn't choose any.
DEFAULT_LANGUAGES = "eng+hrv"

# The allowed values of InspectOptions.ocr_mode:
#   "auto"   OCR only pages without embedded text (the default)
#   "always" OCR every page, even if it has text
#   "never"  never OCR (scanned pages stay unrecognised)
OCR_MODES = ("auto", "always", "never")

# Added to the "Tesseract isn't installed" error, so the user knows the fix.
TESSERACT_INSTALL_HINT = (
    "Install it: Arch `sudo pacman -S tesseract tesseract-data-eng`, "
    "Ubuntu `sudo apt install tesseract-ocr`."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ScoresplitError(Exception):
    """
    Base class of every error scoresplit raises on purpose.

    ``message`` is a sentence for a person; ``exit_code`` is the number the
    CLI exits with. ``problems`` is a list of single problems behind the
    message (every mistake in an edits file, for example), so a GUI can show
    them one by one; most errors have none. Each subclass only changes
    ``exit_code``. It is a CLASS
    attribute (written in the class body, not in __init__), so it's the
    same for every error of that kind, like a PHP class constant, and can
    be read from the class or from an error object.

    Errors that aren't a ScoresplitError are bugs or surprises; the CLI
    reports those as "unexpected error" with exit code 5.
    """

    exit_code: int = EXIT_ERROR

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        # super().__init__ is the parent's constructor (parent::__construct
        # in PHP); it makes str(error) return the message too.
        super().__init__(message)
        self.message = message
        # "x or []": no list given (None) becomes an empty list.
        self.problems = problems or []


class BadInputError(ScoresplitError):
    """A file is missing or isn't a PDF, an edits file is invalid or for another PDF, ..."""

    exit_code = EXIT_BAD_INPUT


class InvalidEditsError(BadInputError):
    """
    The result can't be split as it is. ``problems`` lists every error
    (see check_edits) at once.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__("the parts can't be split as they are", problems)


class OcrMissingError(ScoresplitError):
    """A page needs OCR, but Tesseract isn't installed."""

    exit_code = EXIT_OCR_MISSING


class OutputExistsError(ScoresplitError):
    """
    The output folder already exists and isn't empty (pass force=True).

    exporter.py has an error with the same name. That one is turned into
    this one in split(), so a caller of this module only needs api's errors.
    """

    exit_code = EXIT_OUTPUT_EXISTS


class OutputWriteError(ScoresplitError):
    """Writing the output files failed (disk full, no permission, ...)."""

    exit_code = EXIT_ERROR


# ---------------------------------------------------------------------------
# Inputs and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InspectOptions:
    """
    How inspect() analyses a PDF. The defaults are the CLI's defaults.

    ``frozen=True`` makes the object read-only after it's created; to change
    one value, make a new one: InspectOptions(lang="eng").
    """

    ocr_mode: str = "auto"  # one of OCR_MODES
    lang: str = DEFAULT_LANGUAGES  # Tesseract languages joined with "+"
    use_cache: bool = True  # reuse/save page facts in ~/.cache/scoresplit
    workers: int | None = None  # parallel processes (None = half the CPU cores)


class WrittenFile(BaseModel):
    """
    One file that split() wrote.

    ``part`` is the position of its part in ``result.parts`` (0 = the first
    part), so a program finds the part's name, instrument, key, ... there
    without matching file names to part names. It is None for the extra
    pages and the export files (JSON, CSV, XML).

    ``file`` is the absolute path. ``relative_path`` is the same file
    relative to SplitResult.output_dir, always with "/" (also on Windows),
    so a program that moves the whole output folder can still find each
    file.
    """

    part: int | None  # index in result.parts, or None
    file: Path  # absolute path
    relative_path: str  # "05_Song_Arranger_Clarinet_1_in_Eb_tc_pages_19-21.pdf", "extra/cover_pages_1.pdf"


class SplitResult(BaseModel):
    """What split() wrote. (A pydantic model, like models.py's, so it can be turned into JSON.)"""

    output_dir: Path  # the main "<Title> - <Arranger>" folder
    # Every file written: the parts first (in the result's order; parts with
    # export: false are left out), then the extra pages, then the export
    # files, then the copy of the original PDF.
    files: list[WrittenFile]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def require_file(path: Path, what: str) -> None:
    """
    Raise BadInputError if ``path`` isn't an existing file.

    ``what`` names the file in the message: "PDF", "edits file".
    """
    if not path.is_file():
        problem = "is a folder, not a file" if path.is_dir() else "does not exist"
        raise BadInputError(f"{what} {path} {problem}")


def _unreadable_pdf(pdf: Path, error: pdfium.PdfiumError) -> BadInputError:
    """
    The BadInputError for a PDF that PDFium can't open. Anyone can upload a
    file to the web tool, so this must be a clear sentence, never a crash.

    PDFium says why in ``err_code``: a password (or another protection) is
    worth its own message, because the person can fix that. Everything else
    (not a PDF at all, a cut-off download, a PDF without pages) is "damaged".
    """
    # getattr(..., None): not every PdfiumError has an err_code.
    if getattr(error, "err_code", None) in (pdfium_c.FPDF_ERR_PASSWORD, pdfium_c.FPDF_ERR_SECURITY):
        return BadInputError(f"{pdf.name} is protected with a password: remove the password and try again")
    return BadInputError(f"cannot read {pdf.name}: it is not a PDF, it is damaged, or it has no pages")


def _count_pages(pdf: Path) -> int:
    """Number of pages in the PDF (BadInputError if it can't be read)."""
    try:
        document = pdfium.PdfDocument(str(pdf))
    except pdfium.PdfiumError as error:
        # "raise ... from error" keeps the original error attached, so a
        # traceback still shows what PDFium itself complained about.
        raise _unreadable_pdf(pdf, error) from error
    count = len(document)
    document.close()
    return count


def _split_result(output_dir: Path, written: list[ExportedFile]) -> SplitResult:
    """Turn export_parts()'s list of written files into a SplitResult."""
    # A "list comprehension": one WrittenFile per written file, like PHP's
    # array_map(fn($f) => new WrittenFile(...), $written).
    files = [
        WrittenFile(
            part=exported.part,
            # resolve() makes the path absolute ("out/x.pdf" -> "/home/.../out/x.pdf").
            file=exported.path.resolve(),
            # as_posix() writes the path with forward slashes, even on Windows:
            # out/Song/extra/cover_pages_1.pdf -> "extra/cover_pages_1.pdf".
            relative_path=exported.path.relative_to(output_dir).as_posix(),
        )
        for exported in written
    ]
    return SplitResult(output_dir=output_dir, files=files)


def _check_options(options: InspectOptions) -> None:
    """
    Raise BadInputError for option values the analysis can't use.

    (The CLI never sends such values, its option parser already refuses
    them; this protects Python callers from a silent typo.)
    """
    if options.ocr_mode not in OCR_MODES:
        raise BadInputError(f"unknown OCR mode '{options.ocr_mode}': use one of {', '.join(OCR_MODES)}")
    if options.workers is not None and options.workers < 1:
        raise BadInputError(f"workers must be 1 or more, not {options.workers}")


# ---------------------------------------------------------------------------
# The public functions
# ---------------------------------------------------------------------------


def inspect(
    pdf: Path,
    options: InspectOptions | None = None,
    progress: ProgressCallback | None = None,
) -> Result:
    """
    Detect which instrument parts are on which pages. Changes nothing on disk
    (apart from the analysis cache).

    Args:
        pdf:      the sheet-music PDF.
        options:  OCR mode, languages, cache, workers (default: InspectOptions()).
        progress: optional function called as progress(pages_done, pages_total)
                  while the pages are analysed (this can take minutes with OCR).

    Raises:
        BadInputError:   the file is missing, isn't a PDF, is damaged or has a
                         password, or an option is invalid.
        OcrMissingError: a page needs OCR, but Tesseract isn't installed.
    """
    # "x or y" gives y when x is None: the caller may leave options out.
    options = options or InspectOptions()
    require_file(pdf, "PDF")
    _check_options(options)

    # Only the installed languages are used; the missing ones become a warning.
    usable_lang, missing_lang = resolve_languages(options.lang)
    settings = AnalysisSettings(ocr_mode=options.ocr_mode, lang=usable_lang)

    try:
        result = inspect_pdf(
            pdf, settings, progress=progress, use_cache=options.use_cache, workers=options.workers
        )
    except pdfium.PdfiumError as error:
        raise _unreadable_pdf(pdf, error) from error
    except OcrUnavailableError as error:
        raise OcrMissingError(f"{error}. {TESSERACT_INSTALL_HINT}") from error

    # The missing languages only matter if OCR was actually used.
    if result.ocr_pages and missing_lang:
        result.warnings.append(
            f"Tesseract language(s) not installed: {', '.join(missing_lang)} "
            f"(used: {usable_lang}). Arch: tesseract-data-<lang>, Ubuntu: tesseract-ocr-<lang>."
        )
    return result


def save_edits(result: Result, edits_file: Path) -> None:
    """
    Write the result as JSON, so a person or a GUI can edit it and give
    it back to load_edits().

    Raises:
        BadInputError: the file can't be written (no such folder, no permission, ...).
    """
    try:
        edits_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as error:
        raise BadInputError(f"cannot write edits file {edits_file}: {error}") from error


def _validation_problems(error: ValidationError) -> list[str]:
    """
    One line per mistake pydantic found in an edits file, saying where it is:

        "parts.2.pages: Input should be a valid string"

    ("parts.2" is the third part: a program counts from 0.)
    """
    problems = []
    # error.errors() is a list of dicts, one per mistake. "loc" is where it
    # is, as a path into the JSON: ("parts", 2, "pages").
    for mistake in error.errors():
        where = ".".join(str(step) for step in mistake["loc"]) or "the file"
        problems.append(f"{where}: {mistake['msg']}")
    return problems


def load_edits(edits_file: Path, pdf: Path, force: bool = False) -> Result:
    """
    Read a result JSON file (made by inspect, save_edits or a GUI, maybe
    edited).

    The file remembers the sha256 fingerprint of its PDF. A file for another
    PDF is refused, because its page numbers would cut the wrong pages;
    force=True accepts it anyway. Files of an older scoresplit (the "page
    map", schema_version 1 or 2) are refused with a clear message.

    Raises:
        BadInputError: a file is missing, the JSON isn't a valid result
                       (``problems`` then lists every mistake at once), it
                       is of an older version, or it was made for another
                       PDF (and force is off).
    """
    require_file(edits_file, "edits file")
    require_file(pdf, "PDF")
    try:
        data = json.loads(edits_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BadInputError(f"cannot read edits file {edits_file.name}: {error}") from error

    # The JSON must be an object ({...}), not a list or a single value.
    if not isinstance(data, dict):
        raise BadInputError(f"edits file {edits_file.name} is not a scoresplit result (not a JSON object)")

    # Checked before the fields, so an old file gives ONE clear sentence
    # instead of a list of field errors. Version 1 had no schema_version.
    version = data.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        raise BadInputError(
            f"edits file {edits_file.name} has schema_version {version}, but this scoresplit "
            f"reads only version {SCHEMA_VERSION}: detect the parts again"
        )
    try:
        result = Result.model_validate(data)
    except ValidationError as error:
        raise BadInputError(
            f"edits file {edits_file.name} is not a valid scoresplit result",
            _validation_problems(error),
        ) from error

    # "not force" comes first, so the (slow) fingerprint of a big PDF is
    # only computed when it's actually checked.
    if not force and result.sha256 != file_sha256(pdf):
        raise BadInputError(
            f"edits file {edits_file.name} was made for a different PDF "
            f"({result.source}); use --force to use it anyway"
        )
    return result


def saved_edits_path(pdf: Path) -> Path:
    """
    Where the editor saves its edits for `pdf`: next to it, with the same
    name. "scores/paradiso.pdf" -> "scores/paradiso.scoresplit.json".
    """
    # pdf.stem = the file name without its extension ("paradiso").
    return pdf.with_name(pdf.stem + SAVED_EDITS_SUFFIX)


def load_saved_edits(pdf: Path) -> Result | None:
    """
    Read the saved edits of `pdf` (see saved_edits_path), or return None when
    there are none.

    Raises:
        BadInputError: the file can't be read, or it was made for another
                       version of the PDF (its sha256 fingerprint differs:
                       the PDF was changed or replaced after the edits).
    """
    path = saved_edits_path(pdf)
    if not path.exists():
        return None
    # force=True: the fingerprint is checked here, for a clearer message than
    # load_edits's "use --force" (which doesn't apply to saved edits).
    result = load_edits(path, pdf, force=True)
    if result.sha256 != file_sha256(pdf):
        raise BadInputError(
            f"the saved edits in {path.name} were made for another version of {pdf.name}"
        )
    return result


def output_folder(pdf: Path, result: Result, out_dir: Path | None = None) -> Path:
    """
    The main folder split() will write to: "<Title> - <Arranger>" inside
    `out_dir`, or next to the PDF when out_dir is None.
    """
    return exporter.output_folder(result, out_dir or pdf.parent)


def check_edits(result: Result, pdf: Path) -> CheckReport:
    """
    Check that ``result`` can be split from ``pdf`` and return EVERY problem
    at once (see models.check_result and exporter.check_export):

        errors:   page texts that can't be read, pages that don't exist,
                  exported parts without pages, two parts in one file.
                  No errors = ready to split.
        warnings: worth a look, but splitting works (a page in two parts).

    Raises:
        BadInputError: the PDF is missing or can't be read.
    """
    require_file(pdf, "PDF")
    return check_export(result, _count_pages(pdf))


def split(
    pdf: Path,
    result: Result,
    out_dir: Path | None = None,
    force: bool = False,
    include_extra: bool = True,
    progress: ProgressCallback | None = None,
) -> SplitResult:
    """
    Write one PDF per part, the extra pages and the export files (JSON,
    CSV, XML) as described by ``result``. See exporter.py for the layout.

    Args:
        pdf:           the original PDF.
        result:        what to write (from inspect() or load_edits()).
        out_dir:       where the "<Title> - <Arranger>" folder is created
                       (default: the PDF's own folder).
        force:         allow writing into an existing, non-empty folder.
        include_extra: also write cover/blank/unassigned pages to extra/.
        progress:      optional function called as progress(files_done,
                       files_total) while the files are written.

    Returns:
        A SplitResult: the output folder and every file written, each with
        the index of the part it holds (see WrittenFile).

    Raises:
        InvalidEditsError: the result has errors (see check_edits); nothing is written.
        BadInputError:     the PDF is missing or can't be read.
        OutputExistsError: the output folder isn't empty and force is off.
        OutputWriteError:  writing failed; some files may already be written.
    """
    errors = check_edits(result, pdf).errors
    if errors:
        raise InvalidEditsError(errors)

    # No out_dir given: write next to the PDF (pdf.parent = the PDF's folder).
    target_parent = out_dir or pdf.parent
    try:
        target, written = export_parts(
            result, pdf, target_parent, force=force, include_extra=include_extra, progress=progress
        )
    except InvalidResultError as error:
        # export_parts checks the result again; after check_edits above this
        # can only happen if the PDF changed in between.
        raise BadInputError(f"the parts can't be split as they are: {error}") from error
    except exporter.OutputExistsError as error:
        raise OutputExistsError(
            f"output folder already exists and is not empty: {error} (use --force)"
        ) from error
    except OSError as error:
        # OSError = any error from the operating system: disk full, no permission, ...
        raise OutputWriteError(f"cannot write the output files: {error}") from error

    return _split_result(target, written)
