"""
Tests for progress reporting: `--progress json|none|bar|auto` on inspect and
split, and the progress callback of the exporter.

A GUI runs scoresplit as a separate process and reads progress from stderr
(one JSON object per line) while the result arrives on stdout. So the two
streams are checked SEPARATELY here: Typer's CliRunner keeps them apart
(result.stdout / result.stderr).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from scoresplit import cache, cli
from scoresplit.exporter import export_parts
from scoresplit.models import Extra, Part, Result

# The same small helpers the export tests use (pytest puts the tests folder
# on the import path, so a test file can import another one by name).
from test_export_and_cli import make_map, make_pdf

runner = CliRunner()

# Options that keep the inspection quick and self-contained: blank generated
# pages need no OCR, one worker process, and nothing read from or written to
# the user's OCR cache.
FAST_INSPECT = ["--ocr", "never", "--workers", "1", "--no-cache"]


def progress_lines(stderr: str) -> list[dict]:
    """Helper: parse every stderr line as JSON (fails the test if one isn't)."""
    # splitlines() splits the text at line ends; empty lines are skipped.
    return [json.loads(line) for line in stderr.splitlines() if line.strip()]


def map_file_for(pdf: Path, tmp_path: Path) -> Path:
    """Helper: a Result for `pdf` (3 pages): two parts, a duplicate, an extra."""
    page_map = make_map(
        [
            Part(name="Flute", pages="1", instrument="flute"),
            Part(name="Oboe", pages="2", instrument="oboe"),
            Part(name="Oboe (copy)", pages="2", instrument="oboe", export=False, duplicate_of="Oboe"),
        ],
        extra=[Extra(reason="blank", pages="3")],
        page_count=3,
    )
    page_map.title = "Song"
    page_map.sha256 = cache.file_sha256(pdf)  # a map for THIS pdf, so no --force needed
    map_file = tmp_path / "map.json"
    map_file.write_text(page_map.model_dump_json(), encoding="utf-8")
    return map_file


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_progress_json_goes_to_stderr_line_by_line(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 4)
    result = runner.invoke(
        cli.app, ["inspect", str(pdf), "--json", "--progress", "json", *FAST_INSPECT]
    )
    assert result.exit_code == cli.EXIT_OK, result.output

    # stdout: exactly one JSON document, the Result (no progress mixed in).
    output = json.loads(result.stdout)
    assert output["result"]["page_count"] == 4

    # stderr: JSON Lines, one per page plus the one before the first page.
    events = progress_lines(result.stderr)
    assert events, "no progress reported"
    for event in events:
        assert event["event"] == "progress"
        assert event["stage"] == "analyse"
        assert event["total"] == 4
    assert events[0]["done"] == 0
    assert events[-1]["done"] == events[-1]["total"]
    assert len(events) == 5


def test_inspect_progress_none_prints_nothing_on_stderr(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    result = runner.invoke(
        cli.app, ["inspect", str(pdf), "--json", "--progress", "none", *FAST_INSPECT]
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert result.stderr == ""
    json.loads(result.stdout)


def test_progress_auto_is_quiet_when_stderr_is_not_a_terminal(tmp_path: Path) -> None:
    # In the test, stderr is captured (not a terminal), like when another
    # program runs scoresplit: the default "auto" then shows nothing.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    result = runner.invoke(cli.app, ["inspect", str(pdf), "--json", *FAST_INSPECT])
    assert result.exit_code == cli.EXIT_OK, result.output
    assert result.stderr == ""


def test_progress_bar_can_be_forced(tmp_path: Path) -> None:
    # "bar" draws rich's bar even when stderr isn't a terminal; it must not
    # break the run or touch stdout.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    result = runner.invoke(
        cli.app, ["inspect", str(pdf), "--json", "--progress", "bar", *FAST_INSPECT]
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    json.loads(result.stdout)


def test_unknown_progress_mode_is_refused(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    result = runner.invoke(cli.app, ["inspect", str(pdf), "--progress", "loud"])
    assert result.exit_code == 2  # Typer's own "invalid value" error


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_progress_json_reports_each_file(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    map_file = map_file_for(pdf, tmp_path)
    result = runner.invoke(
        cli.app,
        [
            "split", str(pdf), "--edits", str(map_file), "--out", str(tmp_path / "out"),
            "--yes", "--json", "--progress", "json",
        ],
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    output = json.loads(result.stdout)
    written = len(output["files"])
    # Flute, Oboe, the blank page (not the duplicate), the JSON, CSV and
    # XML export files and the copy of the original PDF.
    assert written == 7

    # With --edits nothing is analysed, so only the export stage reports.
    events = progress_lines(result.stderr)
    assert [event["stage"] for event in events] == ["export"] * (written + 1)
    assert [event["done"] for event in events] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert all(event["total"] == written for event in events)


def test_split_without_edits_reports_both_stages(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    result = runner.invoke(
        cli.app,
        [
            "split", str(pdf), "--out", str(tmp_path / "out"), "--yes", "--json",
            "--progress", "json", *FAST_INSPECT,
        ],
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    stages = [event["stage"] for event in progress_lines(result.stderr)]
    # First every "analyse" line, then every "export" line.
    assert stages[0] == "analyse"
    assert stages[-1] == "export"
    assert stages == sorted(stages)  # "analyse" < "export": no mixing


def test_a_later_stage_reports_only_when_it_has_work(capsys) -> None:
    # The "compare" stage (same-name parts) has a callback on every inspect,
    # but most PDFs never call it: then not a single line for it appears.
    stages = (
        cli._Stage("analyse", "Analysing score.pdf", "pages"),
        cli._Stage("compare", "Comparing same-name parts", "pages"),
    )
    with cli._progress_reporter(cli.ProgressMode.json, *stages) as [analyse, compare]:
        analyse(0, 2)
        analyse(2, 2)
    events = progress_lines(capsys.readouterr().err)
    assert [event["stage"] for event in events] == ["analyse", "analyse"]

    # When it IS called, its lines carry its own stage name.
    with cli._progress_reporter(cli.ProgressMode.json, *stages) as [analyse, compare]:
        analyse(2, 2)
        compare(0, 118)
        compare(1, 118)
    events = progress_lines(capsys.readouterr().err)
    assert [(event["stage"], event["done"], event["total"]) for event in events] == [
        ("analyse", 2, 2),
        ("compare", 0, 118),
        ("compare", 1, 118),
    ]


def test_progress_none_gives_no_callbacks() -> None:
    stages = (cli._Stage("analyse", "Analysing", "pages"), cli._Stage("compare", "Comparing", "pages"))
    with cli._progress_reporter(cli.ProgressMode.none, *stages) as callbacks:
        assert callbacks == [None, None]


# ---------------------------------------------------------------------------
# export_parts' callback (for Python callers)
# ---------------------------------------------------------------------------


def test_export_parts_calls_progress_for_every_file(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    page_map: Result = make_map(
        [Part(name="Flute", pages="1"), Part(name="Oboe", pages="2", export=False)],
        extra=[Extra(reason="blank", pages="3"), Extra(reason="cover", pages="")],
        page_count=3,
    )
    calls: list[tuple[int, int]] = []
    # A lambda is a one-line function without a name, like PHP's fn($a, $b) => ...
    _, written = export_parts(
        page_map, pdf, tmp_path / "out", progress=lambda done, total: calls.append((done, total))
    )
    # Flute, the blank page, the 3 export files and the copy of the original
    # PDF are written: the switched-off part and the empty extra are not counted.
    assert len(written) == 6
    assert calls == [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


def test_export_parts_without_extras_counts_parts_and_export_files(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    page_map = make_map(
        [Part(name="Flute", pages="1")], extra=[Extra(reason="blank", pages="3")], page_count=3
    )
    calls: list[tuple[int, int]] = []
    export_parts(
        page_map, pdf, tmp_path / "out", include_extra=False,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
