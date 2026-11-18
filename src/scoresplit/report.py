"""
report.py: show the result as a nice table in the terminal.

Pipeline position:  (result) -> [report] -> terminal

Uses the "rich" library, which draws coloured tables and panels in the
terminal. Only the CLI uses this module; in --json mode it's not used at all.

IMPORTANT: rich reads square brackets as style tags ("[bold]...[/]"). Any
text that comes from the PDF or the user (names, titles, notes, paths) must
go through escape() first. Otherwise a part called "Perc [a2]" loses its
"[a2]", and a name containing "[/]" crashes the program.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .exporter import EXTRA_FOLDER, EXTRA_LABELS, safe_filename
from .instruments import load_dictionary
from .models import Part, Result

# Confidence colours: green = fine, yellow = have a look, red = probably wrong.
GOOD_CONFIDENCE = 0.85
OK_CONFIDENCE = 0.6


def _confidence_text(value: float) -> str:
    """Format 0.93 as a coloured "93%". [green]...[/] is rich's colour markup."""
    colour = "green" if value >= GOOD_CONFIDENCE else "yellow" if value >= OK_CONFIDENCE else "red"
    return f"[{colour}]{value:.0%}[/]"  # :.0% formats 0.93 as "93%"


def _key_text(part: Part) -> str:
    """
    The key column: "Bb", or "Bb*" in grey when no key was printed and the
    instrument's usual key was filled in (the "*" is explained under the table).
    """
    if part.key is None:
        return "-"
    if part.key_assumed:
        return f"[dim]{escape(part.key)}*[/]"
    return escape(part.key)


def _clef_text(part: Part) -> str:
    """
    The clef column: "treble", or "treble*" in grey when no clef was printed
    or detected and the instrument's clef was filled in (like _key_text).
    """
    if part.clef is None:
        return "-"
    if part.clef_assumed:
        return f"[dim]{escape(part.clef)}*[/]"
    return escape(part.clef)


def _instrument_text(part: Part) -> str:
    """
    The instrument column: the recognised instrument's English name
    ("Clarinet"), "score" for the conductor score, "?" when not recognised.
    """
    if part.kind == "score":
        return "score"
    if part.instrument is None:
        return "?"
    # .get(): an id that isn't in instruments.yaml (an edited file) is shown as it is.
    instrument = load_dictionary().by_id.get(part.instrument)
    return escape(instrument.name if instrument else part.instrument)


def _row_style(part: Part) -> str:
    """
    The colour of a whole row: yellow = have a look, red = probably wrong,
    "" = fine. A part that isn't exported (a duplicate) is grey instead.
    """
    if not part.export:
        return "dim"
    if part.confidence < OK_CONFIDENCE:
        return "red"
    if part.confidence < GOOD_CONFIDENCE:
        return "yellow"
    return ""


def print_result(
    console: Console,
    result: Result,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> None:
    """
    Print the title block, the parts table, the notes, the extra pages and
    the problems: the result's own warnings, plus ``warnings`` and
    ``errors`` (e.g. from check_result). Looks like this:

         #  Name                Instrument  Key  Clef    Pages  Conf.
         1  Conductor Score     score       -    -       1-24    100%
         2  Flute 1             Flute       C*   treble  25-27    98%
        Notes:
          2: key not printed ...
        Extra pages: cover 1 · blank 237-238  (-> extra/)
        * = key not printed, the usual key is assumed
    """
    source_kind = (
        f"scanned, {result.ocr_pages} page(s) read with OCR"
        if result.ocr_pages
        else "text PDF"
    )
    console.print(
        Panel.fit(
            f"[bold]{escape(result.title or '?')}[/]\n"
            f"Composer: {escape(result.composer or '-')}\n"
            f"Arranger: {escape(result.arranger or '-')}\n"
            f"Source:   {escape(result.source)} ({result.page_count} pages, {source_kind})\n"
            f"Folder:   [cyan]{escape(safe_filename(result.folder_name()))}/[/]",
            # The version in the box title ("scoresplit 0.2.0"), so a run in
            # the terminal shows which version made it. (Only this box: the
            # JSON output has no title.)
            title=f"scoresplit {__version__}",
        )
    )

    # The columns that must never be cut short get a minimum width and
    # no_wrap: the number (you type it to edit a part), the key, clef, pages
    # and confidence. In a narrow terminal only the name and the instrument
    # get narrower: rich then wraps them onto a second line. (rich shrinks
    # only columns that may wrap; with no_wrap everywhere it would drop
    # whole columns instead.)
    def widest(texts: list[str], header: str) -> int:
        return max([len(header), *(len(text) for text in texts)])

    parts = result.parts
    # collapse_padding: one space between two columns instead of two.
    table = Table(header_style="bold", box=box.SIMPLE_HEAD, pad_edge=False, collapse_padding=True)
    table.add_column("#", justify="right", min_width=widest([str(len(parts))], "#"))
    table.add_column("Name", min_width=12)
    table.add_column("Instrument", min_width=8)
    table.add_column("Key", no_wrap=True, min_width=widest([f"{p.key}*" for p in parts if p.key], "Key"))
    table.add_column("Clef", no_wrap=True, min_width=widest([f"{p.clef}*" for p in parts if p.clef], "Clef"))
    table.add_column("Pages", justify="right", no_wrap=True, min_width=widest([p.pages for p in parts], "Pages"))
    table.add_column("Conf.", justify="right", no_wrap=True, min_width=5)

    for number, part in enumerate(result.parts, start=1):
        name = escape(part.name)
        if not part.export:
            name = f"[strike]{name}[/]"  # switched off (a duplicate): crossed out
        elif part.kind == "score":
            name = f"[bold magenta]{name}[/]"
        table.add_row(
            str(number),
            name,
            _instrument_text(part),
            _key_text(part),
            _clef_text(part),
            escape(part.pages),
            _confidence_text(part.confidence),
            style=_row_style(part),
        )
    console.print(table)

    # The notes, by part number, under the table (in the table they made
    # every row three lines high).
    noted = [(number, part) for number, part in enumerate(result.parts, start=1) if part.notes]
    if noted:
        console.print("[bold]Notes[/]")
        for number, part in noted:
            for note in part.notes:
                console.print(f"  [yellow]{number}:[/] {escape(note)}")

    # Worked out afresh, so the list is right even after the parts changed.
    extras = result.extra_groups()
    if extras:
        groups = " · ".join(
            f"{escape(EXTRA_LABELS.get(extra.reason, extra.reason).lower())} {escape(extra.pages)}"
            for extra in extras
        )
        console.print(f"[bold]Extra pages:[/] {groups}  (-> {EXTRA_FOLDER}/)")

    # The "*" legend names only what is really assumed in this table.
    keys_assumed = any(part.key_assumed for part in result.parts)
    clefs_assumed = any(part.clef_assumed for part in result.parts)
    if keys_assumed and clefs_assumed:
        console.print("[dim]* = key or clef not printed, the instrument's usual one is assumed[/]")
    elif keys_assumed:
        console.print("[dim]* = key not printed, the usual key is assumed[/]")
    elif clefs_assumed:
        console.print("[dim]* = clef not printed or detected, the instrument's clef is assumed[/]")

    for warning in result.warnings + (warnings or []):
        console.print(f"[yellow]Warning:[/] {escape(warning)}")
    for error in errors or []:
        console.print(f"[red]Error:[/] {escape(error)}")

    exported = sum(1 for part in result.parts if part.export)
    skipped = len(result.parts) - exported
    summary = f"{exported} PDF(s) will be created"
    if skipped:
        summary += f", {skipped} part(s) switched off"
    console.print(f"[bold]{summary}.[/]")
