"""
Tests for the editor (editor.py) and the interactive `scoresplit split`.

The editor never reads the keyboard itself: it gets an `ask` function. Here
that is a FakeKeyboard that hands out prepared answers, one per question,
and remembers every prompt it was shown. So a whole editing session runs
without a person typing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from rich.console import Console
from typer.testing import CliRunner

from scoresplit import api, cache, cli
from scoresplit.editor import (
    Editor,
    FieldError,
    parse_clef_field,
    parse_key_field,
    parse_pages,
)
from scoresplit.models import CONTROL_CHARACTERS, Extra, Part, Result

runner = CliRunner()


class FakeKeyboard:
    """
    Pretends to be a person typing: each call returns the next prepared
    answer. When the answers run out it raises EOFError, like input() does
    at the end of the input (Ctrl+D).
    """

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []  # every question shown, in order

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)  # pop(0) takes the first answer off the list


def make_pdf(path: Path, pages: int) -> Path:
    """Write an empty PDF with `pages` blank pages."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(595, 842)  # A4 in PDF points
    pdf.save(str(path))
    pdf.close()
    return path


def detected(pdf: Path) -> Result:
    """What detection might find in a 6-page PDF: a cover, two parts, a blank page."""
    return Result(
        source=pdf.name,
        sha256=cache.file_sha256(pdf),  # the saved edits must match THIS pdf
        page_count=6,
        title="Song",
        composer="Someone",
        parts=[
            Part(
                name="Flute 1", instrument="flute", number="1", key="C", key_assumed=True,
                clef="treble", pages="2-3", confidence=0.95,
            ),
            Part(
                name="Trumpet", instrument="trumpet", key="Bb", key_assumed=True, clef="treble",
                clef_assumed=True, pages="4-5", confidence=0.7, notes=["the name wasn't printed clearly: please check it"],
            ),
        ],
        extra=[Extra(reason="cover", pages="1"), Extra(reason="blank", pages="6")],
    )


def start_editor(tmp_path: Path, *answers: str, **options) -> tuple[Editor, FakeKeyboard, Console]:
    """An editor on a fresh 6-page PDF, with these answers typed."""
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    keyboard = FakeKeyboard(*answers)
    # record=True keeps everything printed, for export_text() below.
    console = Console(record=True, width=120)
    return Editor(detected(pdf), pdf, console, keyboard, **options), keyboard, console


# --- one field at a time --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("9-3", "goes backwards"),
        ("7-8", "page(s) 7-8 don't exist (the PDF has 6 pages)"),
        ("0", "page(s) 0 don't exist"),
        ("a", "isn't a page list"),
        ("5-", "isn't a page list"),
        (",", "give at least one page"),
    ],
)
def test_bad_pages_are_refused_with_the_reason(text: str, reason: str) -> None:
    # re.escape(): match the reason as plain text ("(" is special in a regex).
    with pytest.raises(FieldError, match=re.escape(reason)):
        parse_pages(text, page_count=6)


def test_pages_are_tidied() -> None:
    assert parse_pages("6, 2-3,4", page_count=6) == "2-4,6"


def test_keys_and_clefs_accept_the_usual_spellings() -> None:
    assert parse_key_field("es") == "Eb"
    assert parse_key_field("Bb") == "Bb"
    assert parse_key_field("-") is None
    assert parse_clef_field("Bass") == "bass"
    assert parse_clef_field("tc") == "treble"
    assert parse_clef_field("-") is None
    with pytest.raises(FieldError, match="isn't a key"):
        parse_key_field("Xb")
    with pytest.raises(FieldError, match="isn't a clef"):
        parse_clef_field("soprano")


# --- the commands -----------------------------------------------------------------


def test_edit_asks_each_field_and_refuses_bad_input_right_there(tmp_path: Path) -> None:
    editor, keyboard, console = start_editor(
        tmp_path,
        "e 2",
        "Trumpet 2 in Bb",  # name
        "5-4",  # pages: refused (backwards), asked again
        "4-6",  # pages
        "",  # key: Enter keeps Bb (from the new name)
        "bc",  # clef
        "q", "y",  # quit, and yes, without saving
    )
    assert editor.run() is None
    part = editor.result.parts[1]
    assert (part.name, part.instrument, part.number) == ("Trumpet 2 in Bb", "trumpet", "2")
    assert (part.pages, part.key, part.key_assumed, part.clef) == ("4-6", "Bb", False, "bass")
    # A person checked it: no longer yellow, the old notes are gone.
    assert (part.confidence, part.notes) == (1.0, [])
    # Page 6 was a blank extra page; now it's in the part.
    assert [(e.reason, e.pages) for e in editor.result.extra] == [("cover", "1")]

    text = console.export_text()
    assert "goes backwards" in text
    # The current value is offered in brackets, and the pages were asked twice.
    assert "  Name [Trumpet]: " in [prompt.replace("\\", "") for prompt in keyboard.prompts]
    assert sum(prompt.startswith("  Pages") for prompt in keyboard.prompts) == 2


def test_enter_keeps_every_field_and_an_assumed_key_stays_assumed(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "e 1", "", "", "", "", "q", "y")
    editor.run()
    part = editor.result.parts[0]
    assert (part.name, part.pages, part.key, part.key_assumed, part.clef) == (
        "Flute 1", "2-3", "C", True, "treble",
    )


def test_a_changed_key_is_no_longer_assumed(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "e1", "", "", "Eb", "", "q", "y")
    editor.run()
    assert (editor.result.parts[0].key, editor.result.parts[0].key_assumed) == ("Eb", False)


def test_an_assumed_clef_stays_assumed_until_it_is_changed(tmp_path: Path) -> None:
    # Part 2 (Trumpet) has an assumed treble clef. Enter keeps the mark...
    editor, _, _ = start_editor(tmp_path, "e 2", "", "", "", "", "q", "y")
    editor.run()
    assert (editor.result.parts[1].clef, editor.result.parts[1].clef_assumed) == ("treble", True)
    # ... a clef the person types is not a guess.
    editor, _, _ = start_editor(tmp_path, "e 2", "", "", "", "bc", "q", "y")
    editor.run()
    assert (editor.result.parts[1].clef, editor.result.parts[1].clef_assumed) == ("bass", False)


def test_a_score_name_makes_the_part_a_score(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "e 1", "Conductor Score", "", "-", "-", "q", "y")
    editor.run()
    part = editor.result.parts[0]
    assert (part.kind, part.instrument, part.number, part.key, part.clef) == (
        "score", None, None, None, None,
    )


def test_add_puts_the_part_in_page_order(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "a", "Conductor Score", "1", "", "", "q", "y")
    editor.run()
    names = [part.name for part in editor.result.parts]
    assert names == ["Conductor Score", "Flute 1", "Trumpet"]
    assert editor.result.parts[0].kind == "score"
    # The cover page is now in the score.
    assert [(e.reason, e.pages) for e in editor.result.extra] == [("blank", "6")]


def test_add_needs_a_name_and_pages(tmp_path: Path) -> None:
    # Enter without a value is refused for a new part's name and pages.
    editor, _, console = start_editor(
        tmp_path, "a", "", "Tuba in Eb", "", "6", "", "", "q", "y"
    )
    editor.run()
    tuba = editor.result.parts[-1]
    assert (tuba.name, tuba.instrument, tuba.key, tuba.pages) == ("Tuba in Eb", "tuba", "Eb", "6")
    text = console.export_text()
    assert "the name can't be empty" in text
    assert "give at least one page" in text


def test_delete_makes_the_pages_extra_pages(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "d 2", "q", "y")
    editor.run()
    assert [part.name for part in editor.result.parts] == ["Flute 1"]
    assert [(e.reason, e.pages) for e in editor.result.extra] == [
        ("cover", "1"), ("unassigned", "4-5"), ("blank", "6"),
    ]


def test_switch_off_and_on(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "o 1", "q", "y")
    editor.run()
    assert editor.result.parts[0].export is False

    editor, _, _ = start_editor(tmp_path, "o 1", "o 1", "q", "y")
    editor.run()
    assert editor.result.parts[0].export is True


def test_words_work_like_letters(tmp_path: Path) -> None:
    editor, _, _ = start_editor(
        tmp_path,
        "edit 1", "Flute 2", "", "", "",  # = e 1
        "off 2", "off 2",  # off stays off (o would switch it back on)
        "delete1",  # = d 1, also without a space
        "on 1",
        "Save",  # capitals don't matter
        "split",  # = x
    )
    edited = editor.run()
    assert edited is not None
    assert [(part.name, part.export) for part in edited.parts] == [("Trumpet", True)]
    assert api.saved_edits_path(editor.pdf).is_file()


def test_title_composer_and_arranger(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "t", "", "-", "Peter Kleine Schaars", "q", "y")
    editor.run()
    result = editor.result
    assert (result.title, result.composer, result.arranger) == ("Song", None, "Peter Kleine Schaars")


def has_control_characters(text: str) -> bool:
    """True when `text` holds a control character other than a line end."""
    return bool(CONTROL_CHARACTERS.search(text.replace("\r\n", "").replace("\n", "")))


def test_names_and_credits_typed_in_the_editor_are_cleaned(tmp_path: Path) -> None:
    # The editor sets the fields directly; they are checked (and cleaned of
    # control characters) on assignment, like when a file is loaded.
    editor, _, console = start_editor(
        tmp_path,
        "e 1", "Flute\x1b[31m 1", "", "", "",  # name, then Enter for pages, key, clef
        "t", "Song\x1b[2J", "", "",  # title, then Enter for composer and arranger
        "q", "y",
    )
    editor.run()
    assert editor.result.parts[0].name == "Flute [31m 1"
    assert editor.result.title == "Song [2J"
    assert not has_control_characters(console.export_text())


def test_title_and_arranger_from_the_command_line_are_cleaned(tmp_path: Path, terminal: list[Path]) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app,
        ["split", str(pdf), "--yes", "--title", "x\x1b[2J", "--arranger", "\x9b31mA", "--out", str(out)],
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert not has_control_characters(result.output)
    # Not in the folder and file names, nor in the CSV.
    assert (out / "x [2J - 31mA").is_dir()
    assert not any(has_control_characters(str(path)) for path in out.rglob("*"))
    csv_text = next(out.rglob("*.csv")).read_text(encoding="utf-8")
    assert "x [2J" in csv_text
    assert not has_control_characters(csv_text)


def test_wrong_commands_and_part_numbers_say_what_is_wrong(tmp_path: Path) -> None:
    editor, _, console = start_editor(tmp_path, "z", "edit now", "e", "e 9", "s 1", "save 1", "q")
    assert editor.run() is None
    text = console.export_text()
    assert "Unknown command 'z'" in text
    assert "Unknown command 'edit now'" in text
    assert "'save' doesn't take a part number" in text
    assert "Which part?" in text
    assert "There is no part 9 (the parts are 1-2)" in text
    assert "'s' doesn't take a part number" in text


def test_quit_with_unsaved_edits_asks_first(tmp_path: Path) -> None:
    # "n": don't quit; then save, and quit asks nothing any more.
    editor, keyboard, _ = start_editor(tmp_path, "d 1", "q", "n", "s", "q")
    assert editor.run() is None
    assert sum("Quit anyway?" in prompt for prompt in keyboard.prompts) == 1
    saved = Result.model_validate_json(api.saved_edits_path(editor.pdf).read_text(encoding="utf-8"))
    assert [part.name for part in saved.parts] == ["Trumpet"]


def test_end_of_input_quits(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "d 1")  # then no more input (Ctrl+D)
    assert editor.run() is None
    # Also in the middle of editing a part.
    editor, _, _ = start_editor(tmp_path, "e 1", "Flute 2")
    assert editor.run() is None


def test_x_returns_the_edited_result(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "d 2", "x")
    edited = editor.run()
    assert edited is not None
    assert [part.name for part in edited.parts] == ["Flute 1"]


def test_x_with_errors_stays_in_the_editor(tmp_path: Path) -> None:
    # Part 2 is switched off and has a page the PDF doesn't have. Switched
    # on, that is an error, so x is refused and the editor goes on.
    editor, _, console = start_editor(tmp_path, "o 2", "x", "q", "y")
    editor.result.parts[1] = editor.result.parts[1].model_copy(update={"export": False, "pages": "9"})
    assert editor.run() is None
    text = console.export_text()
    assert "Fix these first, then type x again" in text


def test_x_refuses_a_folder_in_use_without_force(tmp_path: Path) -> None:
    editor, _, console = start_editor(tmp_path, "x", "q")
    folder = api.output_folder(editor.pdf, editor.result)
    folder.mkdir()
    (folder / "old.pdf").write_text("from an earlier split")
    assert editor.run() is None
    # " ".join(text.split()) undoes rich's line wrapping of the long path.
    assert "already exists and isn't empty" in " ".join(console.export_text().split())

    editor, _, _ = start_editor(tmp_path, "x", force=True)
    assert editor.run() is not None


# --- the whole session through the command line ------------------------------


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """
    Make `scoresplit split` believe a person is typing (CliRunner's input is
    not a terminal), and replace the slow detection by detected(). Returns
    the list of PDFs detection was run on.
    """
    detections: list[Path] = []

    def fake_detection(pdf: Path, *args, **kwargs) -> Result:
        detections.append(pdf)
        return detected(pdf)

    monkeypatch.setattr(cli, "_can_ask", lambda: True)
    monkeypatch.setattr(cli, "_run_inspection", fake_detection)
    return detections


def lines(*answers: str) -> str:
    """The answers as typed: one per line."""
    return "".join(answer + "\n" for answer in answers)


def test_whole_session_edit_save_quit_then_use_the_saved_edits(
    tmp_path: Path, terminal: list[Path]
) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    out = tmp_path / "out"

    # 1. Detect, rename part 2, save, quit.
    first = runner.invoke(
        cli.app,
        ["split", str(pdf), "--out", str(out)],
        input=lines("e 2", "Trumpet 2 in Bb", "", "", "", "s", "q"),
    )
    assert first.exit_code == cli.EXIT_CANCELLED, first.output
    assert "Nothing was split" in first.output
    saved_file = tmp_path / "score.scoresplit.json"
    assert saved_file.is_file()
    assert not out.exists()

    # 2. Start again: the saved edits are offered (Enter = yes), then split.
    second = runner.invoke(
        cli.app, ["split", str(pdf), "--out", str(out)], input=lines("", "x")
    )
    assert second.exit_code == cli.EXIT_OK, second.output
    assert "Use your saved edits from" in second.output
    assert terminal == [pdf]  # detection ran only the first time

    folder = out / "Song - Someone"
    assert sorted(path.name for path in folder.glob("*.pdf")) == [
        "01_Song_Someone_Flute_1_C_tc_pages_2-3.pdf",
        "02_Song_Someone_Trumpet_2_in_Bb_tc_pages_4-5.pdf",
    ]
    # The export holds the edited name too.
    exported = json.loads((folder / "export" / "Song - Someone.json").read_text(encoding="utf-8"))
    assert [part["name"] for part in exported["parts"]] == ["Flute 1", "Trumpet 2 in Bb"]


def test_saying_no_to_the_saved_edits_detects_again(tmp_path: Path, terminal: list[Path]) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    api.save_edits(detected(pdf), api.saved_edits_path(pdf))
    result = runner.invoke(cli.app, ["split", str(pdf)], input=lines("n", "q"))
    assert result.exit_code == cli.EXIT_CANCELLED, result.output
    assert terminal == [pdf]


def test_redetect_ignores_the_saved_edits(tmp_path: Path, terminal: list[Path]) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    api.save_edits(detected(pdf), api.saved_edits_path(pdf))
    result = runner.invoke(cli.app, ["split", str(pdf), "--redetect"], input=lines("q"))
    assert result.exit_code == cli.EXIT_CANCELLED, result.output
    assert "saved edits" not in result.output
    assert terminal == [pdf]


def test_yes_splits_the_saved_edits_without_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("detection ran although there were saved edits")

    monkeypatch.setattr(cli, "_run_inspection", must_not_run)
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    edits = detected(pdf)
    edits.parts[1].name = "Trumpet 2"
    api.save_edits(edits, api.saved_edits_path(pdf))

    result = runner.invoke(cli.app, ["split", str(pdf), "--yes", "--out", str(tmp_path / "out")])
    assert result.exit_code == cli.EXIT_OK, result.output
    assert (tmp_path / "out" / "Song - Someone" / "02_Song_Someone_Trumpet_2_Bb_tc_pages_4-5.pdf").is_file()


def test_saved_edits_for_another_pdf_are_not_used(tmp_path: Path, terminal: list[Path]) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    edits = detected(pdf)
    edits.sha256 = "0" * 64  # made for another version of the PDF
    api.save_edits(edits, api.saved_edits_path(pdf))

    result = runner.invoke(cli.app, ["split", str(pdf)], input=lines("q"))
    assert result.exit_code == cli.EXIT_CANCELLED, result.output
    assert "Saved edits not used" in result.output
    assert "another version of score.pdf" in result.output
    assert terminal == [pdf]  # detected again instead


def test_json_mode_never_reads_the_saved_edits(tmp_path: Path, terminal: list[Path]) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 6)
    edits = detected(pdf)
    edits.parts[1].name = "Trumpet 2"
    api.save_edits(edits, api.saved_edits_path(pdf))

    result = runner.invoke(
        cli.app, ["split", str(pdf), "--yes", "--json", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert [part["name"] for part in json.loads(result.stdout)["result"]["parts"]] == ["Flute 1", "Trumpet"]
    assert terminal == [pdf]
    # ... nor writes them: the file next to the PDF is still the one saved above.
    assert api.load_saved_edits(pdf) == edits


@pytest.mark.parametrize(
    ("typed", "run"),
    [
        (["score.pdf"], ["split", "score.pdf"]),
        (["score.pdf", "--yes"], ["split", "score.pdf", "--yes"]),
        (["inspect", "score.pdf"], ["inspect", "score.pdf"]),
        (["split", "score.pdf"], ["split", "score.pdf"]),
        (["--version"], ["--version"]),
        ([], []),
    ],
)
def test_split_is_the_command_when_none_is_given(typed: list[str], run: list[str]) -> None:
    assert cli.with_default_command(typed) == run


# --- a part printed sideways: no name, confidence 0 -------------------------------


def with_sideways_part(editor: Editor) -> None:
    """Page 6 holds music printed sideways: a part of its own at 0% (segment.py)."""
    editor.result.extra = [Extra(reason="cover", pages="1")]
    editor.result.parts.append(
        Part(
            name="Unknown Part (p6)", pages="6", confidence=0.0,
            notes=["page printed sideways (or unreadable): please name it or join it to a part"],
        )
    )


def test_a_sideways_part_can_be_named(tmp_path: Path) -> None:
    editor, _, _ = start_editor(tmp_path, "e 3", "Trumpet 2 in Bb", "", "", "tc", "x")
    with_sideways_part(editor)
    edited = editor.run()
    assert edited is not None
    part = edited.parts[2]
    assert (part.name, part.instrument, part.key, part.pages) == ("Trumpet 2 in Bb", "trumpet", "Bb", "6")
    assert (part.confidence, part.notes) == (1.0, [])


def test_a_sideways_part_can_be_joined_to_another_part(tmp_path: Path) -> None:
    # Give the Trumpet part page 6 as well, then delete the unknown part.
    editor, _, _ = start_editor(tmp_path, "e 2", "", "4-6", "", "", "d 3", "x")
    with_sideways_part(editor)
    edited = editor.run()
    assert edited is not None
    assert [(part.name, part.pages) for part in edited.parts] == [("Flute 1", "2-3"), ("Trumpet", "4-6")]
    assert [(e.reason, e.pages) for e in edited.extra] == [("cover", "1")]
