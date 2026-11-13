"""
Measure how precisely scoresplit detects parts, using the example PDFs.

This is a developer tool, not part of the installed app. Each example PDF in
example/ has a "golden" file in tests/golden/ (same name, .json) that holds
the TRUTH for that PDF, checked by hand against the pages: which pages
belong to which part, each part's name (as printed, cleaned up),
instrument, number, key and clef, and the title, composer and arranger. This script runs the detection on each PDF and
compares the result with its golden file.

    uv run python scripts/precision.py                  # score table, every PDF
    uv run python scripts/precision.py --pdf haustor    # one PDF, every difference
    uv run python scripts/precision.py --save-baseline  # store the scores as the new minimum
    uv run python scripts/precision.py --draft NAME     # write a golden draft for a new PDF

The table has one row per PDF:

    pages    % of pages assigned to the right part (or the right kind of extra page)
    parts    parts found correctly / missed / extra parts that shouldn't exist
    key      % of the correctly found parts that have the right key
    clef     % of the correctly found parts that have the right clef
    names    % of the correctly found parts that have the right name
             (as printed; capitals and spacing don't count)
    export   % of the correctly found parts with the right export flag
             (a duplicate copy of a part is detected and not exported)
    credits  title, composer and arranger right (0 to 3)

tests/golden/baseline.json stores the scores of the last accepted run, and
tests/test_examples.py fails when any score drops below it. So a change to
the detection code can make the scores better, never worse.

Like the golden tests, this script reads the PDFs with English OCR only
(lang="eng"). The app's default is English + Croatian, which gives slightly
different OCR noise, so the scores describe the "eng" run.
"""

# "from __future__ import annotations" lets us write types like "list[str]"
# in function signatures without Python evaluating them at import time.
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scoresplit.instruments import CANONICAL_KEYS, load_dictionary
from scoresplit.models import Result, format_page_ranges, parse_page_ranges
from scoresplit.ocr import tesseract_available
from scoresplit.page_analyzer import AnalysisSettings
from scoresplit.pipeline import inspect_pdf

# Path(__file__) is this script; .parent is scripts/, its .parent the project.
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "example"
GOLDEN = ROOT / "tests" / "golden"
BASELINE = GOLDEN / "baseline.json"

# The same settings as the golden tests: English OCR only (see the docstring).
SETTINGS = AnalysisSettings(lang="eng")

# In a golden file the conductor score is a part with this "instrument".
SCORE_ID = "score"

# The values a golden file may use (checked by golden_problems).
CLEFS = {"treble", "bass", "alto", "tenor", "percussion", None}
EXTRA_REASONS = {"cover", "blank", "unassigned"}

# The fields every part and every extra in a golden file must have: the
# scoring reads all of them (checked by golden_problems).
GOLDEN_PART_FIELDS = ("name", "instrument", "number", "key", "clef", "pages", "export")
GOLDEN_EXTRA_FIELDS = ("reason", "pages")

# The counts stored in the baseline. For all of them higher is better,
# except parts_extra (parts that shouldn't exist), where lower is better.
HIGHER_IS_BETTER = [
    "pages_right",
    "parts_found",
    "keys_right",
    "clefs_right",
    "names_right",
    "exports_right",
    "credits_right",
]
LOWER_IS_BETTER = ["parts_extra"]


# ---------------------------------------------------------------------------
# Finding and reading the golden files
# ---------------------------------------------------------------------------


def golden_files() -> list[Path]:
    """Every golden file in tests/golden/, sorted by name (not baseline.json)."""
    return sorted(path for path in GOLDEN.glob("*.json") if path != BASELINE)


def find_golden(name: str) -> Path:
    """
    The golden file for a PDF name typed on the command line. Accepts
    "haustor", "Haustor" or "Haustor.pdf" (case doesn't matter).
    """
    wanted = Path(name).stem.lower()
    for path in golden_files():
        if path.stem.lower() == wanted:
            return path
    names = ", ".join(path.stem for path in golden_files())
    raise SystemExit(f"No golden file for {name!r}. Known PDFs: {names}")


def pdf_for(golden_path: Path) -> Path:
    """The example PDF a golden file belongs to: tests/golden/X.json -> example/X.pdf."""
    return EXAMPLES / f"{golden_path.stem}.pdf"


def draft_pdf(name: str) -> Path:
    """
    The example PDF that `--draft NAME` means: "haustor" or "haustor.pdf"
    -> example/haustor.pdf. ".pdf" is added only when it isn't there yet,
    so a name with a dot in it keeps its dot: "my.score" -> my.score.pdf.
    (Path.with_suffix would REPLACE ".score" and give my.pdf.)
    """
    # Path(name).name drops any folders typed in front: the PDF must be in example/.
    file_name = Path(name).name
    if not file_name.lower().endswith(".pdf"):
        file_name += ".pdf"
    return EXAMPLES / file_name


def load_golden(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def golden_problems(golden: dict, page_count: int) -> list[str]:
    """
    Mistakes in a (hand-edited) golden file, as readable sentences, or an
    empty list. Every part and extra must have all its fields, every page
    of the PDF must be in exactly one part or extra, and the instrument
    ids, keys, clefs and extra reasons must be ones scoresplit knows, so a
    typo can't silently lower a score (or crash the scoring).
    """
    problems = []
    instruments = load_dictionary().by_id

    owners: dict[int, int] = defaultdict(int)  # page -> how many places list it

    def count_pages(label: str, pages: str) -> None:
        """Count each page of ``pages`` in owners, or report pages that can't be read."""
        try:
            for page in parse_page_ranges(pages):
                owners[page] += 1
        except (ValueError, TypeError) as error:
            # "3-1", "x" or a number instead of text: parse_page_ranges refuses it.
            problems.append(f"{label}: pages {pages!r} can't be read ({error})")

    # enumerate(..., start=1) counts the parts 1, 2, 3 ... as a person would.
    for position, part in enumerate(golden["parts"], start=1):
        # First check that every field is there: everything below (and the
        # scoring) reads them, and a missing one would crash with a KeyError.
        missing = [name for name in GOLDEN_PART_FIELDS if name not in part]
        if missing:
            for name in missing:
                problems.append(f"part {position} ({part.get('name')!r}): no {name!r} field")
            continue  # skip the other checks for this part
        label = part_label(part)
        if not part["name"]:
            problems.append(f"{label}: no name")
        if part["instrument"] != SCORE_ID and part["instrument"] not in instruments:
            problems.append(f"{label}: unknown instrument id {part['instrument']!r}")
        # The number is text ("1", "1&2", "Solo"), like scoresplit writes it.
        if part["number"] is not None and (not isinstance(part["number"], str) or not part["number"].strip()):
            problems.append(f"{label}: number must be text like \"1\" or null, not {part['number']!r}")
        # A key must be spelled the way scoresplit spells it ("Bb", not "B♭").
        if part["key"] is not None and part["key"] not in CANONICAL_KEYS:
            problems.append(f"{label}: unknown key {part['key']!r} (known: {', '.join(sorted(CANONICAL_KEYS))})")
        if part["clef"] not in CLEFS:
            problems.append(f"{label}: unknown clef {part['clef']!r}")
        # isinstance(x, bool) is True only for true and false, not for "yes" or 1.
        if not isinstance(part["export"], bool):
            problems.append(f"{label}: export must be true or false, not {part['export']!r}")
        count_pages(label, part["pages"])

    for position, extra in enumerate(golden["extra"], start=1):
        missing = [name for name in GOLDEN_EXTRA_FIELDS if name not in extra]
        if missing:
            for name in missing:
                problems.append(f"extra {position}: no {name!r} field")
            continue
        label = f"extra pages {extra['pages']}"
        if extra["reason"] not in EXTRA_REASONS:
            problems.append(f"{label}: unknown reason {extra['reason']!r}")
        count_pages(label, extra["pages"])

    all_pages = set(range(1, page_count + 1))
    missing = sorted(all_pages - set(owners))
    if missing:
        problems.append(f"pages {format_page_ranges(missing)} are in no part and no extra")
    twice = sorted(page for page, count in owners.items() if count > 1)
    if twice:
        problems.append(f"pages {format_page_ranges(twice)} are listed more than once")
    beyond = sorted(set(owners) - all_pages)
    if beyond:
        problems.append(f"pages {format_page_ranges(beyond)} don't exist (the PDF has {page_count})")
    return problems


# ---------------------------------------------------------------------------
# Running the detection
# ---------------------------------------------------------------------------


def result_as_golden(result: Result) -> dict:
    """
    Turn scoresplit's result into the shape of a golden file, so the two
    can be compared field by field. ``page_count`` is added for
    golden_problems; a golden file itself doesn't store it.
    """
    parts = []
    for part in result.parts:
        instrument = SCORE_ID if part.kind == "score" else part.instrument
        parts.append(
            {
                "name": part.name,
                "instrument": instrument,
                "number": part.number,
                # A key that wasn't printed is filled in by scoresplit
                # (key_assumed): a golden file holds the key the part is
                # written in, printed or not.
                "key": part.key,
                "clef": part.clef,
                "pages": part.pages,
                "export": part.export,
            }
        )
    return {
        "title": result.title,
        "composer": result.composer,
        "arranger": result.arranger,
        "parts": parts,
        "extra": [{"reason": extra.reason, "pages": extra.pages} for extra in result.extra],
        "page_count": result.page_count,
    }


def detect(pdf: Path) -> dict:
    """Run scoresplit on one PDF and return the result in golden-file shape."""
    return result_as_golden(inspect_pdf(pdf, SETTINGS))


# ---------------------------------------------------------------------------
# Scoring: compare a detection result with the golden file
# ---------------------------------------------------------------------------


@dataclass
class Score:
    """The counts behind one row of the score table, plus every difference."""

    pages_total: int = 0
    pages_right: int = 0
    parts_expected: int = 0
    parts_found: int = 0
    parts_extra: int = 0
    keys_right: int = 0
    clefs_right: int = 0
    names_right: int = 0
    exports_right: int = 0
    credits_right: int = 0
    # The differences as readable sentences (printed with --pdf).
    differences: list[str] = field(default_factory=list)

    @property
    def parts_missed(self) -> int:
        return self.parts_expected - self.parts_found

    def counts(self) -> dict[str, int]:
        """Every count, without the differences: what the baseline stores."""
        counts = asdict(self)
        del counts["differences"]
        return counts

    def add(self, other: Score) -> None:
        """Add another score's counts to this one (for the total line)."""
        for name, value in other.counts().items():
            setattr(self, name, getattr(self, name) + value)


def part_label(part: dict) -> str:
    """
    A readable name for a part in golden-file shape, used in the
    differences: "Horn 1 in Eb (treble clef)", "Conductor Score".
    """
    if part["instrument"] == SCORE_ID:
        return "Conductor Score"
    instrument = load_dictionary().by_id.get(part["instrument"] or "")
    if instrument is not None:
        label = instrument.name
    else:
        label = part["instrument"] or "unknown part"
    if part["number"]:
        label += f" {part['number']}"
    if part["key"]:
        label += f" in {part['key']}"
    if part["clef"]:
        label += f" ({part['clef']} clef)"
    return label


# Roman numerals as printed in part numbers ("Horns I & II").
ROMAN_NUMERALS = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}


def part_numbers(number: str | None) -> tuple[str, ...] | None:
    """
    Every single number a part number stands for, so that the same numbers
    written in different ways count as the same:

        "1&2", "1/2", "1-2", "1, 2", "I & II", "1st & 2nd" -> ("1", "2")
        "2-4"                                              -> ("2", "3", "4")
        "Solo"                                             -> ("solo",)
        None                                               -> None
    """
    if number is None:
        return None
    found: set[str] = set()
    # Split at "&", "/", ",", "+" and the word "and": "1st & 2nd" -> ["1st", "2nd"].
    for piece in re.split(r"\s*(?:[&/,+]|\band\b)\s*", number.strip().lower()):
        # Drop ordinal endings and dots after a digit: "1st" -> "1",
        # "2." -> "2", "1º-3º" -> "1-3". ((?<=\d) means "right after a digit".)
        piece = re.sub(r"(?<=\d)(st|nd|rd|th|º|°|\.)", "", piece)
        piece = ROMAN_NUMERALS.get(piece, piece)
        # A range of numbers: "2-4" -> 2, 3, 4.
        range_match = re.fullmatch(r"(\d+)-(\d+)", piece)
        if range_match:
            first, last = int(range_match.group(1)), int(range_match.group(2))
            found.update(str(n) for n in range(first, last + 1))
        elif piece:
            found.add(piece)
    return tuple(sorted(found))


def same_text(expected: str | None, got: str | None) -> bool:
    """
    True when two credits are the same apart from capitals and spacing:
    "A Tribute to X" and "A TRIBUTE  TO X" are the same; None only equals None.
    """
    if expected is None or got is None:
        return expected is got
    # .split() cuts at any run of spaces; " ".join puts single spaces back.
    return " ".join(expected.lower().split()) == " ".join(got.lower().split())


def match_parts(expected: list[dict], actual: list[dict]) -> dict[int, int]:
    """
    Pair every golden part with the detected part that is the same part.

    Two parts can be a pair when they have the same instrument and the same
    numbers (see part_numbers()) and share at least one page. The pairs that
    share the most pages are taken first, and each part is used at most
    once. (Key and clef are not used for pairing: they are scored
    separately.)

    Returns {index in expected: index in actual}; a golden part that isn't
    in the result was MISSED, a detected part that isn't in the result is
    an EXTRA part.
    """
    candidates = []  # (pages shared, golden index, detected index)
    for golden_index, golden_part in enumerate(expected):
        golden_pages = set(parse_page_ranges(golden_part["pages"]))
        for actual_index, actual_part in enumerate(actual):
            if actual_part["instrument"] != golden_part["instrument"]:
                continue
            if part_numbers(actual_part["number"]) != part_numbers(golden_part["number"]):
                continue
            shared = len(golden_pages & set(parse_page_ranges(actual_part["pages"])))
            if shared:
                candidates.append((shared, golden_index, actual_index))

    # Most shared pages first; on a tie, the earlier parts first.
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    pairs: dict[int, int] = {}
    used: set[int] = set()
    for _shared, golden_index, actual_index in candidates:
        if golden_index in pairs or actual_index in used:
            continue
        pairs[golden_index] = actual_index
        used.add(actual_index)
    return pairs


def score(golden: dict, actual: dict) -> Score:
    """Compare a detection result with its golden file (both in golden shape)."""
    result = Score()
    expected_parts, actual_parts = golden["parts"], actual["parts"]
    pairs = match_parts(expected_parts, actual_parts)
    result.parts_expected = len(expected_parts)
    result.parts_found = len(pairs)
    result.parts_extra = len(actual_parts) - len(pairs)

    # --- credits (capitals and spacing don't count: a title printed in ALL
    #     CAPITALS can't tell whether "to" in it is written "to" or "To")
    for field_name in ("title", "composer", "arranger"):
        if same_text(golden[field_name], actual[field_name]):
            result.credits_right += 1
        else:
            result.differences.append(
                f"{field_name}: expected {golden[field_name]!r}, got {actual[field_name]!r}"
            )

    # --- parts: missed, extra, and key / clef / export of the found ones
    for golden_index, golden_part in enumerate(expected_parts):
        if golden_index not in pairs:
            result.differences.append(
                f"missed: {part_label(golden_part)} (pages {golden_part['pages']})"
            )
            continue
        actual_part = actual_parts[pairs[golden_index]]
        where = f"{part_label(golden_part)} (pages {golden_part['pages']})"
        for field_name in ("key", "clef", "export"):
            if golden_part[field_name] != actual_part[field_name]:
                result.differences.append(
                    f"{where}: {field_name} expected {golden_part[field_name]!r}, "
                    f"got {actual_part[field_name]!r}"
                )
        if golden_part["key"] == actual_part["key"]:
            result.keys_right += 1
        if golden_part["clef"] == actual_part["clef"]:
            result.clefs_right += 1
        if golden_part["export"] == actual_part["export"]:
            result.exports_right += 1
        # The name as printed. Capitals and spacing don't count (like the
        # credits): "Horn in F 1" and "HORN IN F 1" are the same name.
        if same_text(golden_part["name"], actual_part["name"]):
            result.names_right += 1
        else:
            result.differences.append(
                f"{where}: name expected {golden_part['name']!r}, got {actual_part['name']!r}"
            )
    paired_actual = set(pairs.values())
    for actual_index, actual_part in enumerate(actual_parts):
        if actual_index not in paired_actual:
            result.differences.append(
                f"extra part: {part_label(actual_part)} (pages {actual_part['pages']})"
            )

    # --- pages: does every page end up where the golden file says?
    # "Where" is a short readable text: the golden part's label, or the kind
    # of extra page. A detected part that pairs with a golden part gets that
    # golden part's label, so its pages count as right.
    expected_where = _where_each_page_goes(golden, {i: i for i in range(len(expected_parts))}, golden)
    actual_where = _where_each_page_goes(actual, {a: g for g, a in pairs.items()}, golden)
    all_pages = sorted(set(expected_where) | set(actual_where))
    result.pages_total = len(all_pages)

    wrong: dict[tuple[str, str], list[int]] = defaultdict(list)
    for page in all_pages:
        expected = expected_where.get(page, "nothing")
        got = actual_where.get(page, "nothing")
        if expected == got:
            result.pages_right += 1
        else:
            wrong[(expected, got)].append(page)
    # One line per (expected, got) pair, in page order.
    for (expected, got), pages in sorted(wrong.items(), key=lambda item: item[1][0]):
        result.differences.append(
            f"pages {format_page_ranges(pages)}: expected {expected}, got {got}"
        )
    return result


def golden_part_names(golden: dict) -> list[str]:
    """
    One name per golden part (in the same order) that no other part has.

    Usually this is just part_label(). But a golden file can hold two copies
    of the same part (deep_purple has "Bass in Bb (bass clef)" at pages
    82-83 and again at 84-85). Those would get the same label, so a page put
    in the wrong copy would still look right. Labels that appear more than
    once get ", copy 1", ", copy 2", ... added, in golden-file order.
    """
    labels = [part_label(part) for part in golden["parts"]]
    # How often each label appears: Counter(["a", "b", "a"]) -> {"a": 2, "b": 1}.
    how_often = Counter(labels)
    copies_so_far: dict[str, int] = defaultdict(int)
    names = []
    for label in labels:
        if how_often[label] == 1:
            names.append(label)
        else:
            copies_so_far[label] += 1
            names.append(f"{label}, copy {copies_so_far[label]}")
    return names


def _where_each_page_goes(data: dict, to_golden: dict[int, int], golden: dict) -> dict[int, str]:
    """
    {page number: where it goes} for a golden file or a detection result.
    `to_golden` maps a part's index to the index of its golden part; such a
    part gets that golden part's unique name (see golden_part_names()), so
    two copies of the same part are told apart. A part without a golden part
    gets its own label with "(not in golden file)".
    """
    names = golden_part_names(golden)
    where: dict[int, str] = {}
    for index, part in enumerate(data["parts"]):
        if index in to_golden:
            label = names[to_golden[index]]
        else:
            label = f"{part_label(part)} (not in golden file)"
        for page in parse_page_ranges(part["pages"]):
            where[page] = label
    for extra in data["extra"]:
        for page in parse_page_ranges(extra["pages"]):
            where[page] = f"extra page ({extra['reason']})"
    return where


# ---------------------------------------------------------------------------
# The baseline (the ratchet)
# ---------------------------------------------------------------------------


def load_baseline() -> dict[str, dict[str, int]]:
    """{PDF name without .pdf: the counts of its last accepted score}."""
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def drops(counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """
    What got worse than the baseline, as readable sentences, or an empty list:
    ["pages_right went down from 135 to 133"].
    """
    worse = []
    for name in HIGHER_IS_BETTER:
        # .get(name, 0): a count the baseline doesn't have yet (a new column)
        # counts as 0, and nothing can go down from 0, so it can't drop.
        before = baseline.get(name, 0)
        if counts[name] < before:
            worse.append(f"{name} went down from {before} to {counts[name]}")
    for name in LOWER_IS_BETTER:
        # A count the baseline doesn't have yet is skipped: with nothing to
        # compare against, it can't "go up". (baseline[name] would crash here.)
        if name not in baseline:
            continue
        before = baseline[name]
        if counts[name] > before:
            worse.append(f"{name} went up from {before} to {counts[name]}")
    return worse


def save_baseline(scores: dict[str, Score]) -> None:
    """Write the counts of `scores` to the baseline, keeping PDFs that weren't run."""
    baseline = load_baseline()
    for name, pdf_score in scores.items():
        baseline[name] = pdf_score.counts()
    text = json.dumps(dict(sorted(baseline.items())), indent=2)
    BASELINE.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Writing a golden draft for a new example PDF
# ---------------------------------------------------------------------------


def golden_text(golden: dict) -> str:
    """
    A golden file as JSON text, with one line per part and per extra so it
    is easy to read and correct by hand.
    """

    def one_line(item: dict) -> str:
        # ensure_ascii=False keeps letters like "ć" readable instead of "ć".
        return json.dumps(item, ensure_ascii=False)

    lines = ["{"]
    for field_name in ("title", "composer", "arranger"):
        lines.append(f'  "{field_name}": {one_line(golden[field_name])},')
    lines.append('  "parts": [')
    lines.append(",\n".join(f"    {one_line(part)}" for part in golden["parts"]))
    lines.append("  ],")
    lines.append('  "extra": [')
    if golden["extra"]:
        lines.append(",\n".join(f"    {one_line(extra)}" for extra in golden["extra"]))
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_draft(pdf: Path) -> Path:
    """
    Write tests/golden/<name>.json from today's detection, as a starting
    point. It is only a DRAFT: correct it by hand against the PDF, because a
    golden file must hold the truth, not what scoresplit happens to detect.
    """
    target = GOLDEN / f"{pdf.stem}.json"
    if target.exists():
        raise SystemExit(f"{target} already exists: not overwritten.")
    draft = detect(pdf)
    del draft["page_count"]  # not stored in golden files
    target.write_text(golden_text(draft), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# The score table
# ---------------------------------------------------------------------------


def percent(right: int, total: int) -> str:
    """ "98.5%", or "-" when there is nothing to count."""
    return f"{100 * right / total:.1f}%" if total else "-"


def score_table(scores: dict[str, Score], baseline: dict[str, dict[str, int]]) -> Table:
    table = Table(title="Detection precision (English OCR)")
    columns = ("PDF", "pages", "parts found/missed/extra", "key", "clef", "names", "export", "credits", "vs baseline")
    for column in columns:
        # Every column but the PDF name is right-aligned, so numbers line up.
        table.add_column(column, justify="left" if column in ("PDF", "vs baseline") else "right")

    def row(name: str, pdf_score: Score, compare: str) -> list[str]:
        return [
            name,
            percent(pdf_score.pages_right, pdf_score.pages_total),
            f"{pdf_score.parts_found}/{pdf_score.parts_missed}/{pdf_score.parts_extra}",
            percent(pdf_score.keys_right, pdf_score.parts_found),
            percent(pdf_score.clefs_right, pdf_score.parts_found),
            percent(pdf_score.names_right, pdf_score.parts_found),
            percent(pdf_score.exports_right, pdf_score.parts_found),
            f"{pdf_score.credits_right}/3",
            compare,
        ]

    total = Score()
    for name, pdf_score in scores.items():
        total.add(pdf_score)
        if name not in baseline:
            compare = "no baseline"
        elif worse := drops(pdf_score.counts(), baseline[name]):
            compare = "[red]WORSE: " + "; ".join(worse) + "[/red]"
        elif pdf_score.counts() == baseline[name]:
            compare = "same"
        else:
            compare = "[green]better[/green]"
        table.add_row(*row(name, pdf_score, compare))
    table.add_section()
    credits_total = total.credits_right
    total_row = row("TOTAL", total, "")
    total_row[7] = f"{credits_total}/{3 * len(scores)}"
    table.add_row(*total_row, style="bold")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Score scoresplit's detection against the example PDFs.")
    parser.add_argument("--pdf", help="score only this PDF and list every difference")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="store the scores as the new minimum (refused when a score went down)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --save-baseline: save even when a score went down "
        "(only after correcting a golden file)",
    )
    parser.add_argument("--draft", metavar="PDF", help="write a golden draft for example/PDF")
    args = parser.parse_args()
    console = Console()

    # Check the draft's PDF first: a typo in the name should get a clear
    # message, not a traceback from deep inside the detection.
    if args.draft:
        pdf = draft_pdf(args.draft)
        if not pdf.exists():
            console.print(f"[red]{pdf.name}: missing in example/, no draft written[/red]")
            return 4

    if not tesseract_available():
        console.print("[red]Tesseract is not installed; the scanned examples need it.[/red]")
        return 3

    if args.draft:
        target = write_draft(draft_pdf(args.draft))
        console.print(f"Draft written to {target}. Now correct it by hand against the PDF.")
        return 0

    paths = [find_golden(args.pdf)] if args.pdf else golden_files()
    scores: dict[str, Score] = {}
    for path in paths:
        pdf = pdf_for(path)
        if not pdf.exists():
            console.print(f"[yellow]{pdf.name}: missing in example/, skipped[/yellow]")
            continue
        with console.status(f"Detecting {pdf.name} ..."):
            actual = detect(pdf)
        golden = load_golden(path)
        problems = golden_problems(golden, actual["page_count"])
        if problems:
            console.print(f"[red]{path.name} has mistakes:[/red]")
            for problem in problems:
                console.print(f"  {problem}")
            return 2
        scores[path.stem] = score(golden, actual)

    baseline = load_baseline()
    console.print(score_table(scores, baseline))

    if args.pdf:
        for pdf_score in scores.values():
            console.print("\n[bold]Differences:[/bold]" if pdf_score.differences else "\nNo differences.")
            for difference in pdf_score.differences:
                # markup=False: print brackets in names as they are.
                console.print(f"  {difference}", markup=False)

    if args.save_baseline:
        worse = {
            name: drops(pdf_score.counts(), baseline[name])
            for name, pdf_score in scores.items()
            if name in baseline and drops(pdf_score.counts(), baseline[name])
        }
        if worse and not args.force:
            console.print("[red]Not saved: these scores went down (use --force only after correcting a golden file):[/red]")
            for name, sentences in worse.items():
                console.print(f"  {name}: " + "; ".join(sentences))
            return 1
        save_baseline(scores)
        console.print(f"Baseline saved to {BASELINE.relative_to(ROOT)}.")
    return 0


# The "main guard": run main() only when this file is started as a script,
# not when tests import it. It is also needed because the page analysis
# starts worker processes, which import this file again.
if __name__ == "__main__":
    sys.exit(main())
