"""
editor.py: let a person check and fix the detected parts before splitting.

Pipeline position:  (result) -> [editor] -> (edited result) -> split

`scoresplit score.pdf` shows the parts table (report.py) and waits for a
command, typed as one letter or as the word in brackets:

    e N  (edit N)        edit part N
    a    (add)           add a part
    d N  (delete N)      delete part N
    o N  (on N / off N)  switch part N on or off ("o" switches it over)
    t    (title)         change the title, composer and arranger
    s    (save)          save the edits
    x    (split)         split and export
    q    (quit)          quit

"e N" asks for each field in turn (name, pages, key, clef) with the current
value as the default: pressing Enter keeps it. Bad input ("9-3", a page the
PDF doesn't have, "Xb" as a key) is refused right there, with the reason,
and the same question is asked again.

"s" saves the edits next to the PDF (score.pdf -> score.scoresplit.json, see
api.saved_edits_path). The next `scoresplit score.pdf` offers to use them.

The editor never reads the keyboard itself: it gets an `ask` function that
shows a prompt and returns the typed line (the CLI passes rich's
Console.input). The tests pass a fake one that returns prepared answers, so
whole editing sessions can be tested without a person typing. It uses only
rich for the screen: no full-screen terminal library.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from rich.console import Console
from rich.markup import escape

from . import api
from .exporter import folder_in_use
from .instruments import contains_score_keyword, find_part_name, parse_clef, parse_key
from .models import Clef, Part, Result, format_page_ranges, parse_page_ranges
from .report import print_result

# A function that shows a prompt and returns what the person typed (without
# the Enter). Raises EOFError when there is no more input (Ctrl+D).
Ask = Callable[[str], str]

# The commands shown under the table, one per row: what to type, the same
# command as a word, and what it does.
MENU_ROWS = [
    ("e N", "edit N", "edit part N"),
    ("a", "add", "add a part"),
    ("d N", "delete N", "delete part N"),
    ("o N", "on N / off N", "switch part N on or off"),
    ("t", "title", "change the title, composer and arranger"),
    ("s", "save", "save the edits"),
    ("x", "split", "split and export"),
    ("q", "quit", "quit"),
]


def _menu_text() -> str:
    """
    MENU_ROWS as aligned lines, e.g.

        e N  (edit N)        edit part N
        o N  (on N / off N)  switch part N on or off

    [bold]...[/] is rich's markup; :<4 pads a text with spaces to 4 characters.
    """
    lines = ["[bold]Commands[/]"]
    for letter, word, meaning in MENU_ROWS:
        lines.append(f"  [bold]{letter:<4}[/] {'(' + word + ')':<15} {meaning}")
    return "\n".join(lines)


MENU = _menu_text()

# Typed instead of a value to leave a field empty: no key, no clef, no composer.
NO_VALUE = "-"

# Every clef a part can have: ("treble", "bass", "alto", "tenor", "percussion").
# get_args() reads the allowed values out of the Literal type in models.py,
# so this list can't get out of step with it.
CLEFS: tuple[str, ...] = get_args(Clef)

# A command: a letter or a word, then maybe a part number: "e 3", "e3",
# "edit 3", "s". [a-z]+ = one or more letters; (\d+)? = an optional number;
# \s* = any spaces between.
COMMAND_PATTERN = re.compile(r"^([a-z]+)\s*(\d+)?$")

# The words that can be typed instead of a letter: "edit 3" = "e 3".
COMMAND_WORDS = {
    "edit": "e",
    "add": "a",
    "delete": "d",
    "title": "t",
    "save": "s",
    "split": "x",
    "quit": "q",
}

# Every command, and those that need a part number ("e 3", "on 3", ...).
# "o" switches a part over; "on" and "off" say which way.
COMMANDS = {"e", "a", "d", "o", "on", "off", "t", "s", "x", "q"}
COMMANDS_WITH_NUMBER = {"e", "d", "o", "on", "off"}


class FieldError(ValueError):
    """A typed value can't be used. The message says why, for the person."""


def ask_yes_no(ask: Ask, question: str, default: bool) -> bool:
    """
    Ask a yes/no question until the answer is one. Enter gives `default`.

        ask_yes_no(ask, "Use your saved edits?", True)  ->  "Use your saved edits? [Y/n] "
    """
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = ask(f"{question} {escape(hint)} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# Reading one field. Each function turns the typed text into the value to
# store, or raises FieldError with the reason.
# ---------------------------------------------------------------------------


def parse_name(text: str) -> str:
    """A part name: any text that isn't empty."""
    if not text:
        raise FieldError("the name can't be empty")
    return text


def parse_pages(text: str, page_count: int) -> str:
    """
    A page list such as "5-7" or "5-7,9", tidied up ("7,5-6" -> "5-7").
    Every page must exist in the PDF, and at least one page is needed.
    """
    # Each piece between commas must be a page ("9") or a range ("5-7").
    # Checked first, so "a" or "5-" gets a helpful sentence instead of
    # Python's own "invalid literal for int()". Empty pieces ("5,,7") are
    # fine; parse_page_ranges skips them.
    pieces = [piece.strip() for piece in text.split(",")]
    if not all(re.fullmatch(r"\d+(\s*-\s*\d+)?", piece) for piece in pieces if piece):
        raise FieldError(f"'{text}' isn't a page list: write pages like 5-7 or 5-7,9")
    try:
        pages = parse_page_ranges(text)
    except ValueError as error:
        # "page range '9-3' goes backwards", "page 999999 is too high", ...
        raise FieldError(str(error)) from error
    if not pages:
        raise FieldError("give at least one page")
    missing = [page for page in pages if page < 1 or page > page_count]
    if missing:
        raise FieldError(
            f"page(s) {format_page_ranges(missing)} don't exist (the PDF has {page_count} pages)"
        )
    return format_page_ranges(pages)


def parse_key_field(text: str) -> str | None:
    """A key ("Bb", "bb", "Es", "Sib", "F"), as the standard spelling; "-" = no key."""
    if text == NO_VALUE:
        return None
    key = parse_key(text)
    if key is None:
        raise FieldError(f"'{text}' isn't a key: type one like Bb, Eb, F or C, or {NO_VALUE} for none")
    return key


def parse_clef_field(text: str) -> str | None:
    """A clef ("treble", "bass", ..., or the short "tc"/"bc"); "-" = none."""
    if text == NO_VALUE:
        return None
    if text.lower() in CLEFS:
        return text.lower()
    clef = parse_clef(text)  # "tc" -> "treble", "bc" -> "bass"
    if clef is None:
        raise FieldError(f"'{text}' isn't a clef: type {', '.join(CLEFS)}, or {NO_VALUE} for none")
    return clef


def parse_optional_text(text: str) -> str | None:
    """Free text such as a composer; "-" = none."""
    return None if text == NO_VALUE else text


def recognise_name(part: Part, name: str) -> str | None:
    """
    Give `part` the new `name`, and update what is read from a name: the
    instrument, its number, and whether it is the conductor score.

    Returns the key written in the new name ("2. klarinet in Es" -> "Eb"),
    or None, so the key question can offer it.

    A name scoresplit doesn't recognise ("Perc. 1") keeps the instrument the
    part had: a typed name is often only a small fix of the old one.
    """
    part.name = name
    parsed = find_part_name(name)
    if parsed is not None:
        part.kind = "part"
        part.instrument = parsed.instrument_id
        part.number = parsed.number
        return parsed.key
    if contains_score_keyword(name):
        # "Conductor Score", "Partitura": a score has no instrument or key.
        part.kind = "score"
        part.instrument = None
        part.number = None
    return None


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------


class Editor:
    """
    One editing session for one PDF. Use it as:

        editor = Editor(result, pdf, console, ask)
        edited = editor.run()   # the result to split, or None (quit)

    Like a small controller: run() reads a command, calls the method for
    it, and shows the table again after every change.
    """

    def __init__(
        self,
        result: Result,
        pdf: Path,
        console: Console,
        ask: Ask,
        out_dir: Path | None = None,
        force: bool = False,
    ) -> None:
        self.result = result
        self.pdf = pdf
        self.console = console
        self.ask = ask
        self.out_dir = out_dir  # where split will write (None = next to the PDF)
        self.force = force  # split may write into a folder that isn't empty
        # True when there are edits that aren't saved yet: "q" asks first.
        self.unsaved = False

    # --- the main loop ---------------------------------------------------

    def run(self) -> Result | None:
        """
        Show the table and handle commands until the person splits ("x":
        returns the result to split) or quits ("q": returns None).
        """
        try:
            return self.command_loop()
        except EOFError:
            # No more input (Ctrl+D, or input from a file that ended), at
            # any question: nobody is left to answer "quit anyway?", so stop.
            return None

    def command_loop(self) -> Result | None:
        """The work of run(): read and handle one command after another."""
        self.show()
        while True:
            line = self.ask("Command: ").strip().lower()
            if not line:
                continue  # just Enter: ask again
            match = COMMAND_PATTERN.match(line)
            # group(1) = the letter or word, group(2) = the number or None.
            # A word becomes its letter ("edit" -> "e"); .get(word, word)
            # leaves letters, "on" and "off" as they are.
            word = match.group(1) if match else ""
            command = COMMAND_WORDS.get(word, word)
            if command not in COMMANDS:
                self.console.print(f"[red]Unknown command '{escape(line)}'.[/]")
                self.console.print(MENU)
                continue

            number_text = match.group(2)
            index = 0  # the part's list position, for e/d/o/on/off (set just below)
            if command in COMMANDS_WITH_NUMBER:
                index = self.part_index(word, number_text)
                if index is None:
                    continue  # part_index already said what was wrong
            elif number_text is not None:
                self.console.print(f"[red]'{word}' doesn't take a part number.[/]")
                continue

            if command == "e":
                self.edit_part(index)
            elif command == "a":
                self.add_part()
            elif command == "d":
                self.delete_part(index)
            elif command == "o":
                self.switch_part(index)
            elif command == "on":
                self.switch_part(index, export=True)
            elif command == "off":
                self.switch_part(index, export=False)
            elif command == "t":
                self.edit_titles()
            elif command == "s":
                self.save()
                continue  # nothing changed: no need to show the table again
            elif command == "x":
                if self.ready_to_split():
                    return self.result
                continue
            elif command == "q":
                if not self.unsaved or ask_yes_no(
                    self.ask, "You have edits that aren't saved. Quit anyway?", default=False
                ):
                    return None
                continue
            # Something changed: show the table again.
            self.show()

    def show(self) -> None:
        """Print the table, the problems found by check_edits, and the menu."""
        check = api.check_edits(self.result, self.pdf)
        self.console.print()
        print_result(self.console, self.result, check.warnings, check.errors)
        self.console.print(MENU)

    def part_index(self, command: str, number_text: str | None) -> int | None:
        """
        Turn the typed part number (1, 2, ... as in the table) into a list
        position (0, 1, ...), or None after saying what's wrong.
        """
        count = len(self.result.parts)
        if number_text is None:
            self.console.print(f"[red]Which part? Type its number, e.g. '{command} 1'.[/]")
            return None
        number = int(number_text)
        if not 1 <= number <= count:
            self.console.print(f"[red]There is no part {number} (the parts are 1-{count}).[/]")
            return None
        return number - 1

    def ask_field(self, label: str, default: str, parse: Callable[[str], object]) -> object:
        """
        Ask for one field until the answer can be used. Enter takes
        `default` (shown in brackets); an empty default means a value must
        be typed. Returns what `parse` made of the text.
        """
        while True:
            # escape(): the prompt is rich markup, where "[Flute 1]" would
            # be read as a style tag and disappear.
            shown = escape(f" [{default}]") if default else ""
            text = self.ask(f"  {label}{shown}: ").strip()
            if not text:
                text = default
            try:
                return parse(text)
            except FieldError as error:
                self.console.print(f"  [red]{escape(str(error))}[/]")

    def changed(self) -> None:
        """Call after every edit: bring the extra pages up to date, remember to save."""
        self.result.refresh_extra()
        self.unsaved = True

    # --- the commands ----------------------------------------------------

    def edit_part(self, index: int) -> None:
        """e N: ask for the name, pages, key and clef of part N."""
        part = self.result.parts[index]
        self.console.print(f"Editing part {index + 1}. Enter keeps the value shown in brackets.")

        name = self.ask_field("Name", part.name, parse_name)
        # The key written in the new name, offered as the key below.
        name_key = recognise_name(part, name) if name != part.name else None

        page_count = self.result.page_count
        part.pages = self.ask_field("Pages", part.pages, lambda text: parse_pages(text, page_count))

        old_key = part.key
        key_default = name_key or part.key or NO_VALUE
        part.key = self.ask_field("Key", key_default, parse_key_field)
        # An assumed key (none printed, the usual one filled in) stays marked
        # only while it is unchanged; a key the person gives is not a guess.
        part.key_assumed = part.key_assumed and part.key == old_key and name_key is None

        old_clef = part.clef
        part.clef = self.ask_field("Clef", part.clef or NO_VALUE, parse_clef_field)
        # The same for an assumed clef: Enter keeps the mark, a change drops it.
        part.clef_assumed = part.clef_assumed and part.clef == old_clef

        # A person has checked this part: it is no longer yellow or red, and
        # the notes about what detection wasn't sure of no longer apply.
        part.confidence = 1.0
        part.notes = []
        self.changed()

    def add_part(self) -> None:
        """a: ask for a new part's name, pages, key and clef, and add it."""
        self.console.print("New part. Type - to leave the key or clef empty.")
        part = Part(name="?", pages="", confidence=1.0)
        name = self.ask_field("Name", "", parse_name)
        name_key = recognise_name(part, name)
        page_count = self.result.page_count
        part.pages = self.ask_field("Pages", "", lambda text: parse_pages(text, page_count))
        part.key = self.ask_field("Key", name_key or NO_VALUE, parse_key_field)
        part.clef = self.ask_field("Clef", NO_VALUE, parse_clef_field)

        # Keep the parts in page order: the new part goes before the first
        # part that starts on a later page.
        first_page = part.page_list()[0]
        position = len(self.result.parts)  # at the end, unless a later part is found
        for i, other in enumerate(self.result.parts):
            other_pages = other.page_list() if other.pages else []
            if other_pages and other_pages[0] > first_page:
                position = i
                break
        self.result.parts.insert(position, part)
        self.console.print(f"Added as part {position + 1}.")
        self.changed()

    def delete_part(self, index: int) -> None:
        """d N: delete part N. Its pages become extra pages (unless another part has them)."""
        # pop() removes the element at `index` from the list and returns it.
        part = self.result.parts.pop(index)
        self.console.print(f"Deleted '{escape(part.name)}' (pages {escape(part.pages)}).")
        self.changed()

    def switch_part(self, index: int, export: bool | None = None) -> None:
        """
        o N / on N / off N: switch part N on or off. A part that is off
        isn't written (e.g. a duplicate). `export` None = the other way
        round from now (o N); True = on, False = off.
        """
        part = self.result.parts[index]
        part.export = (not part.export) if export is None else export
        state = "on: it will be written" if part.export else "off: it won't be written"
        self.console.print(f"'{escape(part.name)}' is {state}.")
        self.changed()

    def edit_titles(self) -> None:
        """t: ask for the title, composer and arranger ("-" = none)."""
        self.console.print("Enter keeps the value shown in brackets; - leaves it empty.")
        result = self.result
        result.title = self.ask_field("Title", result.title or NO_VALUE, parse_optional_text)
        result.composer = self.ask_field("Composer", result.composer or NO_VALUE, parse_optional_text)
        result.arranger = self.ask_field("Arranger", result.arranger or NO_VALUE, parse_optional_text)
        self.changed()

    def save(self) -> None:
        """s: save the edits next to the PDF, for the next run."""
        path = api.saved_edits_path(self.pdf)
        try:
            api.save_edits(self.result, path)
        except api.BadInputError as error:
            self.console.print(f"[red]Not saved: {escape(error.message)}[/]")
            return
        self.unsaved = False
        self.console.print(f"Saved to [cyan]{escape(str(path))}[/]")

    def ready_to_split(self) -> bool:
        """
        x: True when the parts can be split. Otherwise say why and return
        False, so the person stays in the editor and keeps their edits.
        """
        errors = api.check_edits(self.result, self.pdf).errors
        if errors:
            for error in errors:
                self.console.print(f"[red]Error:[/] {escape(error)}")
            self.console.print("[red]Fix these first, then type x again.[/]")
            return False
        folder = api.output_folder(self.pdf, self.result, self.out_dir)
        if folder_in_use(folder) and not self.force:
            self.console.print(
                f"[red]The folder {escape(str(folder))} already exists and isn't empty.[/] "
                "Change the title or arranger (t), move that folder away, "
                "or start again with --force."
            )
            return False
        return True
