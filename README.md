# scoresplit

Split a sheet-music PDF (a conductor score and every part in one file) into
**one PDF per instrument part**.

scoresplit reads every page and works out:

- the **title, composer and arranger**,
- which pages belong to which part, and for each part its **name** (as
  printed on the page), **key** and **clef**,
- which pages are not music (covers, blank pages) and which parts are
  **duplicate copies**.

It shows all of that in a table, lets you correct it, and then writes the
files:

```
Nuovo Cinema Paradiso - Peter Kleine Schaars/
    01_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Conductor_Score_pages_3-47.pdf
    02_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Piccolo_C_tc_pages_48-50.pdf
    ...
    11_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Clarinet_1_in_Bb_tc_pages_82-85.pdf
    ...
    51_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Timpani_bc_pages_237-238.pdf
    extra/
        cover_pages_1-2.pdf
    export/
        Nuovo Cinema Paradiso - Peter Kleine Schaars.json
        Nuovo Cinema Paradiso - Peter Kleine Schaars.csv
        Nuovo Cinema Paradiso - Peter Kleine Schaars.xml
        paradiso.pdf
```

It works with both kinds of PDF:

- **PDFs exported from notation software** (Sibelius, Finale, Dorico,
  MuseScore): it reads the text in the PDF, which is fast and exact.
- **Scanned PDFs**: it reads the page headers with Tesseract OCR.

Part names are understood in English, German, Italian, French, Dutch,
Spanish and Croatian/Serbian (Latin script): "2. klarinet in Es", "Trompete
in B", "Clarinetto in Sib", "1e Bugel Bb" ...

scoresplit is free software under the [MIT license](LICENSE).

---

## Installation

You need three things:

1. **Tesseract OCR**, the program that reads scanned pages (PDFs exported
   from notation software don't need it),
2. **uv**, the Python package manager (it downloads Python 3.11+ by itself,
   so you don't have to install Python),
3. **scoresplit** itself, installed with uv.

Tesseract reads English (`eng`) and Croatian (`hrv`, which also covers
Serbian and Bosnian in Latin script). Without `hrv` scoresplit still works,
with English only, and prints a warning.

### Arch Linux

```bash
sudo pacman -S tesseract tesseract-data-eng tesseract-data-hrv uv git
```

### Ubuntu / Debian

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hrv git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Fedora

```bash
sudo dnf install tesseract tesseract-langpack-eng tesseract-langpack-hrv uv git
```

### macOS (with [Homebrew](https://brew.sh))

```bash
brew install tesseract tesseract-lang uv git
```

### Windows 10 / 11

Open **PowerShell** and install the tools with `winget`:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
winget install --id astral-sh.uv -e
winget install --id Git.Git -e
```

Then put Tesseract on the PATH (the installer doesn't):

```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\Program Files\Tesseract-OCR", "User")
```

For Croatian names (optional), download
[`hrv.traineddata`](https://github.com/tesseract-ocr/tessdata/raw/main/hrv.traineddata)
into `C:\Program Files\Tesseract-OCR\tessdata`.

Close PowerShell and open a new one (PATH changes only show up in a new
window), then check: `tesseract --list-langs` should list `eng` (and `hrv`).

### Then, on every system: scoresplit

```bash
uv tool install git+https://github.com/studiokorenat/scoresplit.git
scoresplit --version
```

If the shell says "command not found" (Windows: "not recognized"), run
`uv tool update-shell` and open a new terminal.

Update to the latest version, or uninstall:

```bash
uv tool install --reinstall git+https://github.com/studiokorenat/scoresplit.git
uv tool uninstall scoresplit
```

---

## Usage

```bash
scoresplit "my score.pdf"
```

(the same as `scoresplit split "my score.pdf"`). scoresplit reads every page
(a scan takes a while the first time; a progress bar shows how far it is),
then shows what it found:

```
╭────────────────────────────── scoresplit ──────────────────────────────╮
│ Nuovo Cinema Paradiso                                                  │
│ Composer: Ennio Morricone                                              │
│ Arranger: Peter Kleine Schaars                                         │
│ Source:   paradiso.pdf (238 pages, scanned, 238 page(s) read with OCR) │
│ Folder:   Nuovo Cinema Paradiso - Peter Kleine Schaars/                │
╰────────────────────────────────────────────────────────────────────────╯

  #  Name                 Instrument          Key  Clef       Pages  Conf.
 ──────────────────────────────────────────────────────────────────────────
  1  Conductor Score      score               -    -           3-47    90%
  2  Piccolo              Piccolo             C*   treble     48-50    90%
 ...
 15  Alto Saxophone 1     Alto Saxophone      Eb*  treble    98-101    79%
 ...
 51  Timpani              Timpani             -    bass     237-238    90%

Notes
  15: page(s) 101 had no readable name and were added to this part
Extra pages: cover 1-2  (-> extra/)
* = key not printed, the usual key is assumed
```

How to read it:

- **Name** is the part name as printed on the page, cleaned up.
- **Key** and **Clef** with a `*`: nothing was printed or detected, so the
  instrument's usual one is assumed (a clarinet in Bb, a trumpet in treble
  clef). Check those if it matters.
- A **yellow** row deserves a look, a **red** one is probably wrong. The
  **Notes** under the table say why.
- A grey, crossed-out row is a **duplicate** copy of another part: it is
  not written (switch it back on with `o N`).
- **Extra pages** are pages in no part (covers, blank pages). They go to
  the `extra/` folder.

### The editor

Under the table scoresplit waits for a command. Type the letter or the
word:

| Command | What it does |
|---|---|
| `e N` (`edit N`) | Edit part N: name, pages, key and clef, one after the other. Enter keeps the current value. |
| `a` (`add`) | Add a part. |
| `d N` (`delete N`) | Delete part N. Its pages become extra pages. |
| `o N` (`on N` / `off N`) | Switch part N on or off (`o` switches it over). |
| `t` (`title`) | Change the title, composer and arranger. |
| `s` (`save`) | Save your edits. |
| `x` (`split`) | Split the PDF and write the files. |
| `q` (`quit`) | Quit without splitting. |

Pages are written like `3-47` or `159-162,165`. A wrong value (pages the PDF
doesn't have, `9-3`, an unknown key) is refused right away, with the reason.

### Saved edits

`s` saves your edits next to the PDF, in `my score.scoresplit.json`. The next
time you run `scoresplit "my score.pdf"`, it asks
*"Use your saved edits from …?"*, so you don't have to fix the same things
twice. To detect again from scratch, add `--redetect`.

### The output

The files are written **next to the PDF** (or into `--out DIR`), in a folder
named `<Title> - <Arranger>` (the composer when there is no arranger):

- **One PDF per part.** The file name is the row number, the title, the
  arranger, the part name, the key, the clef (`tc` treble, `bc` bass) and the
  pages: `11_Nuovo_Cinema_Paradiso_Peter_Kleine_Schaars_Clarinet_1_in_Bb_tc_pages_82-85.pdf`.
  The key is left out when the name already contains it, or the part has no
  key (score, drums, piano).
- **`extra/`**: the pages that are in no part, named by their reason:
  `cover_pages_1-2.pdf`, `blank_pages_237-238.pdf`.
- **`export/`**: the parts as **JSON**, **CSV** (opens in a spreadsheet)
  and **XML**, and a copy of the original PDF. The JSON and XML hold
  everything: the credits, every part (name, instrument, key, clef, pages,
  confidence, notes, ...) with its PDF file name, the extra pages, the
  warnings and the PDF's fingerprint. The CSV is the readable table (see
  below). Delete what you don't need. The export always describes the PDF
  that was really split: its `page_count` and extra pages come from that
  PDF, even when the edits were made for another one (`--force`).

The CSV has one row per part, then one row per group of extra pages, with
these columns (find a column by its **name** in the header row, not by its
position):

| Column | What it holds |
| --- | --- |
| `row_type` | `part`, or `extra` for a group of extra pages |
| `name` … `notes` | the part's fields, as in [the result](#the-result): `name`, `instrument`, `number`, `key`, `key_assumed`, `clef`, `clef_assumed`, `kind`, `pages`, `export`, `duplicate_of`, `confidence`, `notes`. On an `extra` row only `name` (a label: `Cover`), `pages` and `export` are filled in |
| `reason` | on an `extra` row: `cover`, `blank` or `unassigned`; empty on a part row |
| `file` | where that PDF was written, relative to the main folder (empty when it wasn't written) |
| `title`, `composer`, `arranger` | the credits, repeated on every row |

The fingerprint, the page counts and the warnings are only in the JSON and
XML. A text that starts with `=`, `+`, `-` or `@` gets a `'` in front
(`'=1+1`), so a spreadsheet shows it as text instead of running it as a
formula. A list (a part's notes) is one cell with the items joined by
` | `; a `|` inside an item is written as `/` there, so the cell always
splits back into the same items. The JSON and XML keep the text exactly.
A change to these columns raises `schema_version`.

The pages are copied as they are, so the quality is the same as the
original. If the folder already exists and is not empty, scoresplit stops
rather than overwrite your files: delete or rename it, or add `--force`.

### Without questions

```bash
scoresplit split "my score.pdf" --yes          # no table, no editor: split straight away
scoresplit inspect "my score.pdf"              # only show the table, write nothing
```

`--yes` uses your saved edits if there are any, the detection otherwise.

### Editing the parts in a text editor

The editor covers most fixes. You can also save the detection as a JSON
file, edit it, and split with it:

```bash
scoresplit inspect "my score.pdf" --save-edits parts.json
# edit parts.json
scoresplit split "my score.pdf" --edits parts.json
```

The file format is described under [The result](#the-result) below. An edits
file belongs to one PDF (it holds the PDF's fingerprint), so it can't cut
the wrong file by accident; `--force` accepts it anyway.

### All options

| Option | Commands | Meaning |
|---|---|---|
| `--out DIR` | split | Where to create the output folder (default: next to the PDF) |
| `--title`, `--arranger` | split | Use this title or arranger instead of the detected one |
| `--yes`, `-y` | split | No questions and no editor: split straight away |
| `--redetect` | split | Ignore the saved edits next to the PDF |
| `--edits FILE` | split | Use these (edited) parts instead of detecting |
| `--save-edits FILE` | inspect | Save the detected parts as JSON |
| `--force` | split | Write into an existing folder; accept edits made for another PDF |
| `--no-extra` | split | Don't write the `extra/` folder |
| `--ocr auto\|always\|never` | both | `auto` (default): OCR only pages without text |
| `--lang eng+hrv` | both | Tesseract languages (default `eng+hrv`) |
| `--no-cache` | both | Don't use or write the cache |
| `--workers N` | both | Parallel OCR processes (default: half the CPU cores, at low priority) |
| `--progress auto\|bar\|json\|none` | both | How progress is shown on stderr |
| `--json` | both | Machine-readable output (see below) |

The page analysis is cached in `~/.cache/scoresplit` (delete it any time),
so a second run on the same PDF is instant. `scoresplit COMMAND --help` shows
the options of each command.

A PDF whose name starts with `-` must come after `--`, placed after every
option: `scoresplit split --yes -- -score.pdf`.

---

## Using scoresplit from another program

scoresplit can run behind a GUI or a web page. The program runs scoresplit
as a separate process and reads its answer as JSON.

```bash
# 1. Detect: stdout holds the result.
scoresplit inspect score.pdf --json --progress json > detected.json

# 2. The person checks and corrects the parts; the program writes the
#    corrected result (the "result" object of step 1) to edits.json.

# 3. Split with those edits, into a fresh folder.
scoresplit split score.pdf --edits edits.json --yes --json --progress json --out /tmp/job-123
```

The rules:

- With `--json`, **stdout holds exactly one JSON document** and nothing
  else. Progress and messages go to stderr: read the two separately.
- **No questions** with `--json`: `split` needs `--yes` (without it: exit
  code 2), and the saved edits next to the PDF are neither read nor
  written. The program passes its own with `--edits`.
- Mistakes in the command itself (a missing argument, an unknown option)
  are JSON errors too.
- New fields may be added; ignore the ones you don't know.

### `inspect --json`

```jsonc
{"status": "ok", "result": { ... the result, see below ... }}
```

### `split --edits FILE --yes --json`

```jsonc
{
  "status": "ok",
  "output_dir": "/tmp/job-123/Nuovo Cinema Paradiso - Peter Kleine Schaars",
  "warnings": [],            // e.g. "'Flute 1' and 'Flute 2' share page(s) 27"
  "files": [
    {
      "part": 0,             // index in result.parts; null for extra pages and export files
      "file": "/tmp/job-123/Nuovo Cinema Paradiso - .../01_..._Conductor_Score_pages_3-47.pdf",
      "relative_path": "01_..._Conductor_Score_pages_3-47.pdf"   // to output_dir, always "/"
    },
    {"part": null, "file": "...", "relative_path": "extra/cover_pages_1-2.pdf"},
    {"part": null, "file": "...", "relative_path": "export/Nuovo Cinema Paradiso - Peter Kleine Schaars.json"}
  ],
  "result": { ... the result that was split ... }
}
```

Parts with `"export": false` are not written, so no file names them.

### Errors

```jsonc
{
  "status": "error",
  "code": 2,
  "message": "the parts can't be split as they are",
  "problems": ["'Piccolo': page(s) 239-240 don't exist (PDF has 238)", "..."]
}
```

`code` is the same as the exit code. `problems` lists every single mistake
behind the message (all mistakes of an edits file at once), and is often
empty. A broken, empty, password-protected or not-a-PDF upload is a clean
error with code 2, never a crash.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Cancelled by the user (quit the editor without splitting) |
| 2 | Bad input: file missing or not a PDF, invalid edits file, edits for another PDF, unknown option, `--json` without `--yes` |
| 3 | OCR needed but Tesseract isn't installed |
| 4 | The output folder already exists (use `--force`) |
| 5 | Unexpected error (could not write the files, OCR crashed, ...). Set `SCORESPLIT_DEBUG=1` to see the Python traceback |

### Progress (`--progress json`)

One JSON object per line on stderr, sent the moment it happens:

```json
{"event": "progress", "stage": "analyse", "done": 12, "total": 238}
{"event": "progress", "stage": "export", "done": 3, "total": 56}
```

`analyse` is the reading of the pages (the slow part: minutes for a large
scan), `export` the writing of the files (`split --edits` skips the
analysis). Each stage starts with a line that gives the total. There is no
final line: the process ending is the signal.

### The result

The one shape of the detection: what `inspect --json` returns, what the
editor saves, what `--edits` reads, and the JSON file in `export/`.

```jsonc
{
  "schema_version": 4,
  "source": "paradiso.pdf",
  "sha256": "6087…",          // fingerprint: edits only apply to the same PDF
  "page_count": 238,
  "ocr_pages": 238,           // 0 = a text PDF
  "title": "Nuovo Cinema Paradiso",
  "composer": "Ennio Morricone",
  "arranger": "Peter Kleine Schaars",
  "parts": [
    {
      "name": "Clarinet 1 in Bb",   // as printed; also the file name
      "instrument": "clarinet",     // recognised instrument, or null
      "number": "1",                // "1", "2", "1&2", "Solo" ... or null
      "key": "Bb",                  // null for the score, drums, piano ...
      "key_assumed": false,         // true = not printed, the usual key is filled in
      "clef": "treble",             // treble / bass / alto / tenor / percussion / null
      "clef_assumed": false,        // true = not detected, the instrument's clef is filled in
      "kind": "part",               // or "score"
      "pages": "82-85",             // also "159-162,165"
      "export": true,               // false = not written (a duplicate copy)
      "duplicate_of": null,         // the name of the part this is a copy of
      "confidence": 0.9,            // 0..1; below 0.85 yellow, below 0.6 red
      "notes": []                   // what to check, for a person
    }
  ],
  "extra": [{"reason": "cover", "pages": "1-2"}],   // cover / blank / unassigned
  "warnings": []                                    // about the whole PDF
}
```

To change a part, change its fields. The extra pages are worked out again
every time a result is loaded: every page that is in no part is an extra
page, so deleting a part moves its pages to `extra/` by itself. Results from
older versions (`schema_version` 1 to 3) are refused: detect the parts
again. (Version 3 had the same JSON; version 4 changed the CSV columns.)

---

## Development

```bash
git clone https://github.com/studiokorenat/scoresplit.git
cd scoresplit
uv run scoresplit "my score.pdf"     # runs the code from this folder, nothing to install
uv run pytest                        # all tests
```

**Precision comes first.** `tests/golden/` holds the hand-checked truth for
a set of example PDFs, and `scripts/precision.py` scores the detection
against it (pages, parts, keys, clefs, names, credits). A change to the
detection may raise the scores, never lower them: the golden tests fail
when a score drops below `tests/golden/baseline.json`.

```bash
uv run python scripts/precision.py               # the score of every example
uv run python scripts/precision.py --pdf paradiso
```

The example PDFs themselves are copyrighted, so they are not in the
repository: the golden tests skip a PDF that isn't in `example/`.

New instrument names and spellings go into
`src/scoresplit/data/instruments.yaml` (no Python needed); check one with
`uv run python -c "from scoresplit.instruments import find_part_name; print(find_part_name('2. truba in B'))"`.
Every module starts with a docstring that explains its step; start reading
at `cli.py`, then `api.py` and `pipeline.py`.

All dependencies are permissively licensed (MIT, BSD, Apache): pypdfium2,
numpy, pillow, pytesseract, rapidfuzz, pydantic, pyyaml, typer, rich.

## License

MIT, see [LICENSE](LICENSE).
