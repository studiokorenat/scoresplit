"""
cli.py: the command-line interface (what you type in the terminal).

This file only deals with the USER: reading options, printing tables,
starting the editor (editor.py), choosing exit codes. All the real work
happens in api.py (which uses pipeline.py and exporter.py). A new feature
goes into api.py first; the command here only calls it and prints the result.

It uses Typer: every function decorated with @app.command() becomes a
sub-command, and its parameters become command-line options. Similar to a
Laravel Artisan command, where the signature defines the arguments.

    scoresplit FILE.pdf                (the same as `scoresplit split FILE.pdf`)
    scoresplit inspect FILE.pdf [--json] [--save-edits FILE.json] [--progress json]
    scoresplit split   FILE.pdf [--edits FILE.json] [--out DIR] [--yes] [--redetect]
                                [--json] [--progress json]

`split` detects the parts, shows the table and opens the editor: the person
checks and fixes the parts, saves the edits (next to the PDF, in
FILE.scoresplit.json) and splits. The next run offers to use the saved
edits instead of detecting again. With --yes there are no questions: the
saved edits (if any) or the detection are split straight away.

EXIT CODES (handy for programs that run scoresplit and check the exit code;
the numbers are defined in api.py):
    0  success
    1  cancelled by the user (quit the editor without splitting)
    2  bad input (file missing or not a PDF, invalid edits file, edits for
       a different PDF, unknown option, ...)
    3  OCR needed, but Tesseract's language files were not found (install
       Tesseract with its language data, see the README)
    4  output folder already exists (use --force)
    5  unexpected error (could not write files, OCR crashed, ...)
       Set SCORESPLIT_DEBUG=1 to see the full Python traceback instead.

JSON MODE (--json), for programs such as the web tool: stdout holds exactly
one JSON document, so a program can parse it. Progress and messages go to
stderr. There are never questions (split needs --yes), and the saved edits
next to the PDF are neither read nor written: the program passes its own
with --edits.

    inspect --json   {"status": "ok", "result": {...the Result...}}
    split --json     {"status": "ok", "output_dir": "...", "warnings": [...],
                      "files": [{"part": 0, "file": "...", "relative_path": "..."}, ...],
                      "result": {...}}
                     ("part" = index in result.parts, null for the extra
                     pages and the export files)
    any error        {"status": "error", "code": 2, "message": "...", "problems": [...]}
                     with the same exit code. "problems" lists every single
                     problem behind the message (all mistakes in an edits
                     file at once), and is often empty.

Mistakes in the command itself (a missing argument, an unknown option) are
JSON errors with exit code 2 too; see run(). A PDF whose name starts with
"-" must come after "--", placed after every option
(`scoresplit split --yes --json -- -score.pdf`), or it is read as an option.
An option's own value may start with "-" (`--title -X` works).

PROGRESS (--progress): "bar" draws a progress bar on stderr, "json" writes
one JSON object per line on stderr (JSON Lines) for a program that shows its
own progress bar, "none" shows nothing. The default "auto" is "bar" when
stderr is a terminal and "none" otherwise. A JSON line looks like

    {"event": "progress", "stage": "analyse", "done": 12, "total": 238}

with stage "analyse" (pages read, the slow part), "compare" (pages compared
to find copies among same-name parts; only when there are any) or "export"
(files written). There is no final "done" event: the program ending is the
signal.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn

from . import __version__, api

# The exit codes and default languages live in api.py. All exit codes are
# imported, also those this file doesn't use (the api errors carry them),
# so that code written for older versions (cli.EXIT_BAD_INPUT, ...) keeps working.
from .api import (
    DEFAULT_LANGUAGES,
    EXIT_BAD_INPUT,
    EXIT_CANCELLED,
    EXIT_ERROR,
    EXIT_OCR_MISSING,
    EXIT_OK,
    EXIT_OUTPUT_EXISTS,
    ScoresplitError,
)
from .editor import Editor, ask_yes_no
from .models import ProgressCallback, Result
from .report import print_result

app = typer.Typer(
    help="Split sheet-music PDFs into one PDF per instrument part. "
    "`scoresplit FILE.pdf` is short for `scoresplit split FILE.pdf`.",
    no_args_is_help=True,  # running just `scoresplit` shows the help
    add_completion=False,
)

# Two "consoles": stdout for normal output, stderr for progress/messages.
stdout = Console()
stderr = Console(stderr=True)


class OcrMode(str, Enum):
    """Allowed values for --ocr (the same as api.OCR_MODES). An Enum is a fixed list of choices."""

    auto = "auto"  # OCR only pages without embedded text (default)
    always = "always"  # OCR every page, even if it has text
    never = "never"  # never OCR (scanned pages stay unrecognised)


class ProgressMode(str, Enum):
    """Allowed values for --progress (see the module docstring)."""

    auto = "auto"  # "bar" when stderr is a terminal, else "none" (default)
    bar = "bar"  # a rich progress bar on stderr, for people
    json = "json"  # JSON Lines on stderr, for programs
    none = "none"  # no progress at all


# ---------------------------------------------------------------------------
# Helpers shared by the commands
# ---------------------------------------------------------------------------


def _print_json(data: dict) -> None:
    """
    Print one JSON document to stdout: the whole answer of a --json command.

    json.dumps writes letters like "Š" as \\u0160, so the output is plain
    ASCII and reads right whatever the terminal's or container's encoding.
    """
    print(json.dumps(data, indent=2))


def _error_json(code: int, message: str, problems: list[str] | None = None) -> dict:
    """The JSON document of an error (see the module docstring)."""
    return {"status": "error", "code": code, "message": message, "problems": problems or []}


def _fail(message: str, code: int, json_output: bool, problems: list[str] | None = None) -> None:
    """
    Print an error (as JSON if requested) and stop with an exit code.
    ``problems`` are the single problems behind the message, one per line.
    """
    if json_output:
        _print_json(_error_json(code, message, problems))
    else:
        # escape(): the message can contain file names or library errors with
        # [square brackets], which rich would otherwise read as style tags.
        stderr.print(f"[red]Error:[/] {escape(message)}{':' if problems else ''}")
        for problem in problems or []:
            stderr.print(f"  {escape(problem)}")
    raise typer.Exit(code)  # stops the program with this exit code


@contextmanager
def _catch_errors(json_output: bool) -> Generator[None, None, None]:
    """
    Turn errors into an error message plus an exit code, the same way for
    every command:

      * a ScoresplitError (raised by api.py on purpose) is printed with
        its own message and exits with its own exit code;
      * any error we didn't plan for exits with code 5, instead of a Python
        traceback with exit code 1, which a calling program would read as
        "the user said no".

    In --json mode the error is printed as a JSON object on stdout.

    Use it as:   with _catch_errors(json_output):
                     ... the command's work ...

    @contextmanager turns this generator into something usable with "with":
    the code inside the "with" block runs at the "yield".
    """
    try:
        yield
    except (typer.Exit, typer.Abort, typer.TyperException):
        # Typer's own signals (our _fail() raises typer.Exit, for example):
        # these are planned exits, so let them through unchanged.
        raise
    except ScoresplitError as error:
        _fail(error.message, error.exit_code, json_output, error.problems)
    except Exception as error:
        if os.environ.get("SCORESPLIT_DEBUG"):
            raise  # developers: show the full traceback
        _fail(f"unexpected error: {type(error).__name__}: {error}", EXIT_ERROR, json_output)


class _Stage(NamedTuple):
    """One stage of the work, as --progress shows it."""

    name: str  # the "stage" in a JSON line: "analyse", "compare" or "export"
    description: str  # the text in front of the bar: "Analysing score.pdf"
    unit: str  # what is counted, after the numbers: "pages", "files"


@contextmanager
def _progress_reporter(
    mode: ProgressMode, *stages: _Stage
) -> Iterator[list[ProgressCallback | None]]:
    """
    Give one progress callback (done, total) per stage, in the order the
    stages are given, shown the way --progress asks. Everything goes to
    stderr, so it never mixes with the JSON result on stdout.

    Use it as:
        with _progress_reporter(mode, _Stage("export", "Writing", "files")) as [progress]:
            api.split(..., progress=progress)

    (`as [progress]` takes the one callback out of the list; with two
    stages it is `as [first, second]`.)

    The first stage's bar is shown at once. A later stage's bar appears on
    its first report only, so a stage that has no work (no same-name parts
    to compare) never shows up, in the terminal or in JSON.

    It gives None callbacks for --progress none, so api functions simply
    skip reporting. "with" also makes sure a bar is cleaned up when the
    block ends, also when the work inside raises an error.
    """
    if mode is ProgressMode.auto:
        # is_terminal: True when stderr is shown in a terminal, False when
        # it goes to a file, a pipe or another program.
        mode = ProgressMode.bar if stderr.is_terminal else ProgressMode.none

    if mode is ProgressMode.json:

        # A small "factory": it makes the callback for ONE stage. (A function
        # defined straight inside a loop would see only the loop's LAST
        # stage, because it looks the variable up when it runs, not when it
        # is made. Passing the stage as an argument fixes it for each.)
        def json_reporter(stage: _Stage) -> ProgressCallback:
            def report_json(done: int, total: int) -> None:
                line = json.dumps(
                    {"event": "progress", "stage": stage.name, "done": done, "total": total}
                )
                # flush=True sends the line at once, instead of waiting in a
                # buffer until more output has collected: the program reading
                # stderr must see each step when it happens.
                print(line, file=sys.stderr, flush=True)

            return report_json

        yield [json_reporter(stage) for stage in stages]
    elif mode is ProgressMode.bar:
        # ONE Progress display for all stages: rich can't show two at once.
        # Each stage is a "task" (a bar line) in it; task.fields["unit"] is
        # the extra `unit=` given to add_task.
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} {task.fields[unit]}"),
            TimeRemainingColumn(),
            console=stderr,
            transient=True,  # remove the bars when done
        ) as progress_bar:
            tasks: dict[str, TaskID] = {}  # stage name -> its bar

            def show(stage: _Stage) -> None:
                # total=None: the total isn't known yet; the first report gives it.
                tasks[stage.name] = progress_bar.add_task(
                    stage.description, total=None, unit=stage.unit
                )

            # The first stage at once: its first report can take a moment
            # (reading the whole PDF to recognise it in the cache).
            if stages:
                show(stages[0])

            def bar_reporter(stage: _Stage) -> ProgressCallback:  # a factory, as above
                def report_bar(done: int, total: int) -> None:
                    if stage.name not in tasks:
                        show(stage)
                    progress_bar.update(tasks[stage.name], completed=done, total=total)

                return report_bar

            yield [bar_reporter(stage) for stage in stages]
    else:
        yield [None for _ in stages]


def _run_inspection(
    pdf: Path,
    ocr: OcrMode,
    lang: str,
    no_cache: bool,
    workers: Optional[int],
    progress_mode: ProgressMode = ProgressMode.auto,
) -> Result:
    """Run api.inspect, showing its progress as --progress asks, and return the Result."""
    options = api.InspectOptions(ocr_mode=ocr.value, lang=lang, use_cache=not no_cache, workers=workers)
    stages = (
        _Stage("analyse", f"Analysing {pdf.name}", "pages"),
        # Only shown when the PDF has same-name parts (see _progress_reporter).
        _Stage("compare", "Comparing same-name parts", "pages"),
    )
    with _progress_reporter(progress_mode, *stages) as [progress, compare_progress]:
        return api.inspect(pdf, options, progress=progress, compare_progress=compare_progress)


def with_default_command(args: list[str]) -> list[str]:
    """
    Make `split` the command when none is given, so `scoresplit score.pdf`
    works like `scoresplit split score.pdf`:

        ["score.pdf", "--yes"]    -> ["split", "score.pdf", "--yes"]
        ["inspect", "score.pdf"]  -> unchanged (a command is given)
        ["--version"], []         -> unchanged (options of scoresplit itself)
    """
    # The names of the commands: "inspect", "split". A command without its
    # own name is called after its function.
    commands = {command.name or command.callback.__name__ for command in app.registered_commands}
    if args and args[0] not in commands and not args[0].startswith("-"):
        return ["split", *args]
    return args


def run(args: list[str]) -> None:
    """
    Run the `scoresplit` command with these arguments (what was typed after
    "scoresplit") and end the program with its exit code. main() in
    __init__.py calls this.

    Why not just app(): Typer reads the arguments BEFORE any command runs.
    A mistake there (a missing file name, an unknown option, a bad
    --progress value) never reaches _catch_errors: Typer prints its usage
    box on stderr and exits with code 2, with nothing on stdout. A program
    that called with --json would get no JSON to read. So with --json in
    the arguments, Typer is told not to handle the mistakes itself
    (standalone_mode=False), and they are printed as the usual JSON error.

    "--json" is looked for in the raw arguments, because reading them is
    what failed. A value that happens to be "--json" (a title, say) is a
    harmless false alarm: the output is still one JSON document.
    """
    args = with_default_command(args)
    if "--json" not in args:
        app(args=args)  # Typer shows its own messages and exits
        return

    try:
        # Without standalone_mode, Typer returns the exit code of a
        # typer.Exit (our _fail raises those), and None when the command
        # simply finished.
        code = app(args=args, standalone_mode=False)
    except typer.TyperException as error:
        # Every argument mistake is one of these (Typer's UsageError and
        # its kinds: missing argument, no such option, bad value).
        # format_message() is Typer's sentence: "No such option: --bogus".
        _print_json(_error_json(EXIT_BAD_INPUT, error.format_message()))
        code = EXIT_BAD_INPUT
    except typer.Abort:
        # Typer stopped on purpose (end of input); like quitting.
        code = EXIT_CANCELLED
    # sys.exit ends the program with this exit code (None = 0 = success).
    sys.exit(code or EXIT_OK)


def _can_ask() -> bool:
    """True when a person can answer questions: the input comes from a terminal."""
    return sys.stdin.isatty()


def _ask(prompt: str) -> str:
    """Show `prompt` (rich markup) and return the line the person types."""
    return stdout.input(prompt)


def _starting_result(
    pdf: Path,
    edits_file: Optional[Path],
    force: bool,
    redetect: bool,
    yes: bool,
    json_output: bool,
    detect: Callable[[], Result],
) -> Result:
    """
    The parts to start from, in this order:

      1. --edits FILE: that file;
      2. the saved edits next to the PDF (api.saved_edits_path), unless
         --redetect or --json: used straight away with --yes, else after
         "Use your saved edits from <date>? [Y/n]";
      3. otherwise `detect()` (the slow part: reading every page).

    Saved edits that can't be used (broken, or for another version of the
    PDF) are reported and detection runs instead.
    """
    if edits_file:
        return api.load_edits(edits_file, pdf, force=force)

    # With --json a program is calling: it passes its own edits (--edits), and
    # a file a person once saved must not change what the program gets.
    if not redetect and not json_output:
        try:
            saved = api.load_saved_edits(pdf)
        except ScoresplitError as error:
            stderr.print(f"[yellow]Saved edits not used:[/] {escape(error.message)}")
            saved = None
        if saved is not None:
            path = api.saved_edits_path(pdf)
            # The file's last change ("modification time"), as "2026-11-11 14:05".
            date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            if yes:
                stderr.print(f"Using your saved edits from {date} (--redetect to detect again).")
                return saved
            if ask_yes_no(_ask, f"Use your saved edits from {date}?", default=True):
                return saved
    return detect()


def _version_callback(value: bool) -> None:
    if value:
        print(f"scoresplit {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Options shared by both commands. typer.Option(default, "--name", help=...)
# describes one command-line option.
# ---------------------------------------------------------------------------

# Note: no exists=True here. api.require_file() checks it, so the error is
# reported as JSON in --json mode too (Typer's own message is plain text).
PdfArgument = typer.Argument(..., help="The sheet-music PDF.")
OcrOption = typer.Option(OcrMode.auto, "--ocr", help="When to use OCR.")
LangOption = typer.Option(
    DEFAULT_LANGUAGES, "--lang", help="Tesseract OCR languages, joined with '+'."
)
NoCacheOption = typer.Option(False, "--no-cache", help="Ignore and don't write the OCR cache.")
WorkersOption = typer.Option(
    None, "--workers", min=1, help="Parallel OCR processes (default: half the CPU cores)."
)
JsonOption = typer.Option(False, "--json", help="Print machine-readable JSON to stdout.")
ProgressOption = typer.Option(
    ProgressMode.auto,
    "--progress",
    help="Progress on stderr: bar, json (one JSON object per line, for programs), "
    "none, or auto (a bar in a terminal, else nothing).",
)


@app.callback()
def main_options(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show the version."
    ),
) -> None:
    """Split sheet-music PDFs into one PDF per instrument part."""


# ---------------------------------------------------------------------------
# scoresplit inspect
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    pdf: Path = PdfArgument,
    save_edits: Optional[Path] = typer.Option(
        None, "--save-edits", help="Save the detected parts as JSON (edit them, then split --edits)."
    ),
    json_output: bool = JsonOption,
    ocr: OcrMode = OcrOption,
    lang: str = LangOption,
    no_cache: bool = NoCacheOption,
    workers: Optional[int] = WorkersOption,
    progress: ProgressMode = ProgressOption,
) -> None:
    """Show which instrument parts are on which pages (changes nothing)."""
    with _catch_errors(json_output):
        # Checked before the progress bar appears, so a typo in the file
        # name gives only the error message.
        api.require_file(pdf, "PDF")
        result = _run_inspection(pdf, ocr, lang, no_cache, workers, progress)

        if save_edits:
            api.save_edits(result, save_edits)
            stderr.print(f"Parts saved to [cyan]{escape(str(save_edits))}[/]")

        if json_output:
            # model_dump() turns the pydantic model into a plain dict;
            # mode="json" also turns values JSON can't hold into text.
            _print_json({"status": "ok", "result": result.model_dump(mode="json")})
        else:
            print_result(stdout, result)


# ---------------------------------------------------------------------------
# scoresplit split
# ---------------------------------------------------------------------------


@app.command()
def split(
    pdf: Path = PdfArgument,
    edits_file: Optional[Path] = typer.Option(
        None, "--edits", help="Use these (edited) parts, a JSON file like inspect --json's result, instead of detecting."
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", file_okay=False, help="Where to create the output folder (default: next to the PDF)."
    ),
    title: Optional[str] = typer.Option(None, "--title", help="Override the composition title."),
    arranger: Optional[str] = typer.Option(None, "--arranger", help="Override the arranger."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="No questions and no editor: split straight away (with the saved edits, if any)."
    ),
    redetect: bool = typer.Option(
        False, "--redetect", help="Detect the parts again, ignoring the edits saved next to the PDF."
    ),
    force: bool = typer.Option(
        False, "--force", help="Write into an existing output folder / accept edits made for another PDF."
    ),
    no_extra: bool = typer.Option(False, "--no-extra", help="Don't write cover/blank/unassigned pages to extra/."),
    json_output: bool = JsonOption,
    ocr: OcrMode = OcrOption,
    lang: str = LangOption,
    no_cache: bool = NoCacheOption,
    workers: Optional[int] = WorkersOption,
    progress: ProgressMode = ProgressOption,
) -> None:
    """Detect the parts, check and edit them in the editor, and write one PDF per part."""
    with _catch_errors(json_output):
        api.require_file(pdf, "PDF")
        if edits_file:
            api.require_file(edits_file, "edits file")

        # Without --yes the editor waits for commands. That's impossible in
        # --json mode or when no one is typing (e.g. run from another
        # program), so stop NOW, before spending minutes on OCR.
        if not yes and (json_output or not _can_ask()):
            _fail("refusing to write without confirmation: add --yes", EXIT_BAD_INPUT, json_output)

        def detect() -> Result:
            return _run_inspection(pdf, ocr, lang, no_cache, workers, progress)

        result = _starting_result(pdf, edits_file, force, redetect, yes, json_output, detect)

        # Command-line overrides win over detected/saved values.
        if title:
            result.title = title
        if arranger:
            result.arranger = arranger

        if not yes:
            # The editor shows the table and only returns the result once
            # it can be split ("x"); None means the person quit ("q").
            editor = Editor(result, pdf, stdout, _ask, out_dir=out, force=force)
            edited = editor.run()
            if edited is None:
                stderr.print("Nothing was split.")
                raise typer.Exit(EXIT_CANCELLED)
            result = edited

        # Checked before anything is written. (api.split checks them again.)
        check = api.check_edits(result, pdf)
        if check.errors:
            raise api.InvalidEditsError(check.errors)

        if yes and not json_output:
            print_result(stdout, result, check.warnings)

        with _progress_reporter(progress, _Stage("export", "Writing", "files")) as [report]:
            written = api.split(
                pdf, result, out, force=force, include_extra=not no_extra, progress=report
            )

        if json_output:
            _print_json(
                {
                    "status": "ok",
                    "output_dir": str(written.output_dir),
                    # Worth a look, but the files were written (see check_edits).
                    "warnings": check.warnings,
                    # Each file with the index of its part in "result"
                    # (null for extra pages and export files).
                    "files": [file.model_dump(mode="json") for file in written.files],
                    "result": result.model_dump(mode="json"),
                }
            )
        else:
            stdout.print(
                f"[green]Done:[/] {len(written.files)} file(s) written to "
                f"[cyan]{escape(str(written.output_dir))}[/]"
            )
