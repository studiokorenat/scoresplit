"""
Tests for the small machine interface that the web tool uses:

    scoresplit inspect FILE.pdf --json
    scoresplit split FILE.pdf --edits EDITS.json --yes --json --out DIR
    --progress json            (tested in test_progress.py)

and for every error a program can get back, always as ONE JSON document on
stdout with the same exit code:

    {"status": "error", "code": 2, "message": "...", "problems": [...]}

The web tool is public, so anyone can upload any file: broken, empty and
password-protected PDFs, and files that aren't PDFs at all, must give a
clean error (exit code 2), never a Python traceback.

Exit code 1 (cancelled) has no test here: only a person quitting the editor
causes it, and with --json there is never an editor or a question.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scoresplit import api, cache, cli
from scoresplit.models import Extra, Part
from scoresplit.ocr import OcrUnavailableError

# The tiny-PDF and result helpers of the exporter tests (pytest puts the
# tests folder on the import path, so a test file can import another one).
from test_export_and_cli import make_map, make_pdf

runner = CliRunner()

# Quick and self-contained: blank generated pages need no OCR, one worker
# process, nothing read from or written to the user's OCR cache.
FAST_INSPECT = ["--ocr", "never", "--workers", "1", "--no-cache"]

# The keys of every answer, so an added or missing key is noticed.
ERROR_KEYS = {"status", "code", "message", "problems"}


def edits_file_for(pdf: Path, tmp_path: Path, parts: list[Part] | None = None) -> Path:
    """
    Helper: an edits file for `pdf` (3 pages), as the web tool would send it.
    By default: a score, a clarinet, a switched-off duplicate and a blank page.
    """
    result = make_map(
        parts
        or [
            Part(name="Conductor Score", kind="score", pages="1"),
            Part(name="Clarinet 1 in Eb", pages="2", instrument="clarinet", number="1", key="Eb"),
            Part(name="Clarinet 1 in Eb (copy)", pages="2", export=False, duplicate_of="Clarinet 1 in Eb"),
        ],
        extra=[Extra(reason="blank", pages="3")],
        page_count=3,
    )
    result.title = "Song"
    result.sha256 = cache.file_sha256(pdf)  # made for THIS pdf, so no --force needed
    edits = tmp_path / "edits.json"
    edits.write_text(result.model_dump_json(), encoding="utf-8")
    return edits


def json_error(output: str, code: int) -> dict:
    """Helper: read stdout as an error document and check its shape and code."""
    error = json.loads(output)  # fails the test if stdout isn't exactly one JSON document
    assert set(error) == ERROR_KEYS
    assert error["status"] == "error"
    assert error["code"] == code
    assert isinstance(error["problems"], list)
    return error


# ---------------------------------------------------------------------------
# The two commands
# ---------------------------------------------------------------------------


def test_inspect_json_gives_the_result(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    result = runner.invoke(cli.app, ["inspect", str(pdf), "--json", *FAST_INSPECT])
    assert result.exit_code == cli.EXIT_OK, result.output

    output = json.loads(result.stdout)
    assert set(output) == {"status", "result"}
    assert output["status"] == "ok"
    assert output["result"]["page_count"] == 3
    assert output["result"]["sha256"] == cache.file_sha256(pdf)
    # The result is what split --edits reads back.
    assert api.Result.model_validate(output["result"]).source == "score.pdf"


def test_split_json_lists_each_file_with_the_index_of_its_part(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    edits = edits_file_for(pdf, tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json", "--out", str(out)]
    )
    assert result.exit_code == cli.EXIT_OK, result.output

    output = json.loads(result.stdout)
    assert set(output) == {"status", "output_dir", "warnings", "files", "result"}
    assert output["status"] == "ok"
    assert output["output_dir"] == str(out / "Song")
    assert output["warnings"] == []
    # The whole result, also the switched-off duplicate.
    assert len(output["result"]["parts"]) == 3

    files = output["files"]
    assert [set(file) for file in files] == [{"part", "file", "relative_path"}] * len(files)
    # "part" = index in result.parts: the score (0) and the clarinet (1);
    # the duplicate (2) isn't written. null for the blank page and the
    # three export files and the copy of the original PDF.
    assert [file["part"] for file in files] == [0, 1, None, None, None, None, None]
    assert [file["relative_path"] for file in files] == [
        "01_Song_Conductor_Score_pages_1.pdf",
        "02_Song_Clarinet_1_in_Eb_pages_2.pdf",
        "extra/blank_pages_3.pdf",
        "export/Song.json",
        "export/Song.csv",
        "export/Song.xml",
        f"export/{pdf.name}",
    ]
    for file in files:
        assert Path(file["file"]).is_absolute()
        assert Path(file["file"]).is_file()
        # relative_path leads to the same file from output_dir.
        assert (Path(output["output_dir"]) / file["relative_path"]).resolve() == Path(file["file"])
    # The part behind a file, found by its index.
    assert output["result"]["parts"][files[1]["part"]]["key"] == "Eb"


def test_split_json_passes_on_the_warnings(tmp_path: Path) -> None:
    # A page in two parts can be split (it goes into both files), but is
    # worth a look: it comes back in "warnings".
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    edits = edits_file_for(pdf, tmp_path, [Part(name="Flute", pages="1-2"), Part(name="Oboe", pages="2-3")])
    result = runner.invoke(
        cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == cli.EXIT_OK, result.output
    assert json.loads(result.stdout)["warnings"] == ["'Flute' and 'Oboe' share page(s) 2"]


# ---------------------------------------------------------------------------
# Errors inside a command, one per exit code
# ---------------------------------------------------------------------------


def test_split_json_without_yes_is_bad_input(tmp_path: Path) -> None:
    # --json never asks a question, so split needs --yes.
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    result = runner.invoke(cli.app, ["split", str(pdf), "--edits", str(edits_file_for(pdf, tmp_path)), "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    assert "--yes" in json_error(result.stdout, cli.EXIT_BAD_INPUT)["message"]


def test_invalid_edits_list_every_problem_at_once(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    edits = edits_file_for(
        pdf,
        tmp_path,
        [
            Part(name="Flute", pages="9-3"),  # can't be read
            Part(name="Oboe", pages="2-7"),  # pages that don't exist
            Part(name="Horn", pages=""),  # no pages
        ],
    )
    result = runner.invoke(
        cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json", "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == cli.EXIT_BAD_INPUT
    error = json_error(result.stdout, cli.EXIT_BAD_INPUT)
    assert error["message"] == "the parts can't be split as they are"
    assert len(error["problems"]) == 3
    assert "'Flute'" in error["problems"][0]
    assert "'Oboe'" in error["problems"][1]
    assert "'Horn': has no pages" in error["problems"][2]
    assert not (tmp_path / "out").exists()  # nothing written


def test_edits_file_with_wrong_fields_lists_every_mistake(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    edits = edits_file_for(pdf, tmp_path)
    data = json.loads(edits.read_text(encoding="utf-8"))
    data["parts"][0]["kind"] = "solo"  # not "score" or "part"
    data["parts"][1]["export"] = "maybe"  # not true or false
    edits.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    error = json_error(result.stdout, cli.EXIT_BAD_INPUT)
    assert error["message"] == "edits file edits.json is not a valid scoresplit result"
    # Where each mistake is: "parts.0.kind" = the kind of the first part.
    assert [problem.split(":")[0] for problem in error["problems"]] == ["parts.0.kind", "parts.1.export"]



def test_edits_file_with_a_huge_page_count_is_bad_input_not_a_hang(tmp_path: Path) -> None:
    # Without a limit, loading this file would walk over a trillion pages.
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    edits = edits_file_for(pdf, tmp_path)
    data = json.loads(edits.read_text(encoding="utf-8"))
    data["page_count"] = 1_000_000_000_000
    edits.write_text(json.dumps(data), encoding="utf-8")

    result = runner.invoke(cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json"])
    error = json_error(result.stdout, cli.EXIT_BAD_INPUT)
    assert error["message"] == "edits file edits.json is not a valid scoresplit result"
    assert [problem.split(":")[0] for problem in error["problems"]] == ["page_count"]

@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "cannot read edits file edits.json"),
        ("[1, 2]", "is not a scoresplit result (not a JSON object)"),
        ('{"schema_version": 2}', "has schema_version 2, but this scoresplit reads only version 4"),
    ],
)
def test_unreadable_edits_file_is_bad_input(tmp_path: Path, content: str, message: str) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    edits = tmp_path / "edits.json"
    edits.write_text(content, encoding="utf-8")
    result = runner.invoke(cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    assert message in json_error(result.stdout, cli.EXIT_BAD_INPUT)["message"]


def test_missing_edits_file_and_edits_for_another_pdf_are_bad_input(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    result = runner.invoke(cli.app, ["split", str(pdf), "--edits", str(tmp_path / "nope.json"), "--yes", "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    assert "edits file" in json_error(result.stdout, cli.EXIT_BAD_INPUT)["message"]

    other = make_pdf(tmp_path / "other.pdf", 4)  # 4 pages: another fingerprint
    edits = edits_file_for(other, tmp_path)
    result = runner.invoke(cli.app, ["split", str(pdf), "--edits", str(edits), "--yes", "--json"])
    assert result.exit_code == cli.EXIT_BAD_INPUT
    assert "made for a different PDF" in json_error(result.stdout, cli.EXIT_BAD_INPUT)["message"]


def test_missing_tesseract_is_exit_code_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_tesseract(*args, **kwargs):
        raise OcrUnavailableError("Tesseract is not installed")

    monkeypatch.setattr(api, "inspect_pdf", no_tesseract)
    pdf = make_pdf(tmp_path / "score.pdf", 1)
    result = runner.invoke(cli.app, ["inspect", str(pdf), "--json"])
    assert result.exit_code == cli.EXIT_OCR_MISSING
    assert "Tesseract" in json_error(result.stdout, cli.EXIT_OCR_MISSING)["message"]


def test_existing_output_folder_is_exit_code_4(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    args = ["split", str(pdf), "--edits", str(edits_file_for(pdf, tmp_path)), "--yes", "--json", "--out", str(tmp_path / "out")]
    assert runner.invoke(cli.app, args).exit_code == cli.EXIT_OK

    result = runner.invoke(cli.app, args)  # the same folder again
    assert result.exit_code == cli.EXIT_OUTPUT_EXISTS
    assert "--force" in json_error(result.stdout, cli.EXIT_OUTPUT_EXISTS)["message"]
    # --force writes into it anyway.
    assert runner.invoke(cli.app, [*args, "--force"]).exit_code == cli.EXIT_OK


def test_failed_write_is_exit_code_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(api, "export_parts", disk_full)
    pdf = make_pdf(tmp_path / "score.pdf", 3)
    result = runner.invoke(
        cli.app, ["split", str(pdf), "--edits", str(edits_file_for(pdf, tmp_path)), "--yes", "--json"]
    )
    assert result.exit_code == cli.EXIT_ERROR
    assert "No space left" in json_error(result.stdout, cli.EXIT_ERROR)["message"]


# ---------------------------------------------------------------------------
# Untrusted PDFs: anyone can upload anything to the web tool
# ---------------------------------------------------------------------------


def write_raw_pdf(path: Path, objects: list[bytes], trailer: bytes = b"") -> Path:
    """
    Helper: write a PDF by hand, object by object, for PDFs that PDFium
    itself would never write (no pages, a password). Object 1 must be the
    catalog. The cross-reference table ("xref") lists where each object
    starts, in bytes, which is how a PDF reader finds them.
    """
    data = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_start = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    data += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    data += b"trailer\n<< /Size %d /Root 1 0 R %s>>\n" % (len(objects) + 1, trailer)
    data += b"startxref\n%d\n%%%%EOF\n" % xref_start
    path.write_bytes(data)
    return path


CATALOG = b"<< /Type /Catalog /Pages 2 0 R >>"
ONE_PAGE = [CATALOG, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>", b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] >>"]


def empty_file(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def text_file(path: Path) -> Path:
    path.write_text("Dear conductor, here are the parts.", encoding="utf-8")
    return path


def cut_off_pdf(path: Path) -> Path:
    """A real PDF with its second half missing (an interrupted upload)."""
    data = make_pdf(path, 3).read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


def pdf_without_pages(path: Path) -> Path:
    return write_raw_pdf(path, [CATALOG, b"<< /Type /Pages /Kids [] /Count 0 >>"])


def pdf_with_password(path: Path) -> Path:
    """
    A PDF encrypted with a password we don't know. /O and /U are the
    owner's and user's password checks; random bytes match no password, not
    even the empty one, so a reader must ask for one.
    """
    encrypt = b"<< /Filter /Standard /V 1 /R 2 /O <" + b"11" * 32 + b"> /U <" + b"22" * 32 + b"> /P -4 >>"
    return write_raw_pdf(path, [*ONE_PAGE, encrypt], trailer=b"/Encrypt 4 0 R /ID [<" + b"ab" * 16 + b"> <" + b"ab" * 16 + b">] ")


DAMAGED = "cannot read upload.pdf: it is not a PDF, it is damaged, or it has no pages"
PROTECTED = "upload.pdf is protected with a password: remove the password and try again"


def command_line(command: str, pdf: Path, tmp_path: Path) -> list[str]:
    """The three ways the web tool can hand a PDF to scoresplit."""
    if command == "inspect":
        return ["inspect", str(pdf), "--json", *FAST_INSPECT]
    if command == "split":  # detects the parts itself
        return ["split", str(pdf), "--yes", "--json", "--out", str(tmp_path / "out"), *FAST_INSPECT]
    # split with edits made for a good PDF (--force: for another PDF).
    good = make_pdf(tmp_path / "good.pdf", 3)
    edits = edits_file_for(good, tmp_path)
    return ["split", str(pdf), "--edits", str(edits), "--force", "--yes", "--json", "--out", str(tmp_path / "out")]


@pytest.mark.parametrize("command", ["inspect", "split", "split --edits"])
@pytest.mark.parametrize(
    ("make_upload", "message"),
    [
        (empty_file, DAMAGED),
        (text_file, DAMAGED),
        (cut_off_pdf, DAMAGED),
        (pdf_without_pages, DAMAGED),
        (pdf_with_password, PROTECTED),
    ],
)
def test_untrusted_pdf_gives_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, make_upload, message: str
) -> None:
    monkeypatch.delenv("SCORESPLIT_DEBUG", raising=False)  # no traceback on purpose
    pdf = make_upload(tmp_path / "upload.pdf")
    result = runner.invoke(cli.app, command_line(command, pdf, tmp_path))
    assert result.exit_code == cli.EXIT_BAD_INPUT, result.output
    assert json_error(result.stdout, cli.EXIT_BAD_INPUT)["message"] == message
    assert "Traceback" not in result.output
    assert not (tmp_path / "out").exists()


def test_pdf_with_a_damaged_page_still_gives_a_result(tmp_path: Path) -> None:
    # The file opens, but the page's content is garbage: PDFium draws what
    # it can (nothing), and the page is simply read as empty.
    garbage = b"<< /Length 9 /Filter /FlateDecode >>\nstream\n\x00\x01garbage\nendstream"
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>"
    pdf = write_raw_pdf(tmp_path / "upload.pdf", [*ONE_PAGE[:2], page, garbage])
    result = runner.invoke(cli.app, ["inspect", str(pdf), "--json", *FAST_INSPECT])
    assert result.exit_code == cli.EXIT_OK, result.output
    assert json.loads(result.stdout)["result"]["page_count"] == 1


# ---------------------------------------------------------------------------
# Mistakes in the command line itself (cli.run, what main() calls)
# ---------------------------------------------------------------------------


def run_command(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """
    Helper: run `scoresplit <args>` the way the installed command does, and
    return (exit code, stdout, stderr). capsys is pytest's catcher of
    everything printed.
    """
    with pytest.raises(SystemExit) as caught:  # cli.run always ends with sys.exit
        cli.run(args)
    printed = capsys.readouterr()
    return caught.value.code, printed.out, printed.err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["inspect", "--json"], "Missing argument 'pdf'"),
        (["split", "--json", "--yes"], "Missing argument 'pdf'"),
        (["inspect", "x.pdf", "--json", "--bogus"], "No such option: --bogus"),
        (["inspect", "x.pdf", "--json", "--progress", "loud"], "Invalid value for '--progress'"),
        (["inspect", "x.pdf", "--json", "--workers", "0"], "Invalid value for '--workers'"),
        (["split", "x.pdf", "--json", "--edits"], "Option '--edits' requires an argument"),
        (["--json"], "No such option: --json"),
    ],
)
def test_usage_errors_are_json_too(capsys: pytest.CaptureFixture[str], args: list[str], message: str) -> None:
    code, out, err = run_command(args, capsys)
    assert code == cli.EXIT_BAD_INPUT
    assert message in json_error(out, cli.EXIT_BAD_INPUT)["message"]
    assert err == ""  # no usage box


def test_usage_errors_without_json_show_the_usage_box(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run_command(["inspect", "x.pdf", "--bogus"], capsys)
    assert code == cli.EXIT_BAD_INPUT
    assert out == ""
    assert "No such option" in err
    assert "Usage:" in err


def test_run_keeps_the_exit_codes_of_the_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # In JSON mode run() handles Typer's exits itself; the codes must stay.
    pdf = make_pdf(tmp_path / "score.pdf", 2)
    code, out, _ = run_command(["inspect", str(pdf), "--json", *FAST_INSPECT], capsys)
    assert code == cli.EXIT_OK
    assert json.loads(out)["status"] == "ok"

    # `scoresplit FILE.pdf` is still `scoresplit split FILE.pdf`.
    code, out, _ = run_command([str(tmp_path / "nope.pdf"), "--json"], capsys)
    assert code == cli.EXIT_BAD_INPUT
    assert "does not exist" in json_error(out, cli.EXIT_BAD_INPUT)["message"]

    code, out, _ = run_command(["inspect", "--help", "--json"], capsys)
    assert code == cli.EXIT_OK
    assert "Usage:" in out
