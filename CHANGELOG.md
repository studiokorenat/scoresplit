# Changelog

What changed in each release of scoresplit. The newest release is at the
top. Version numbers follow [Semantic Versioning](https://semver.org/):
below 1.0, a breaking change raises the middle number (0.1 → 0.2).

## 0.3.0 (2026-11-19)

**Scanned PDFs are about twice as fast.** The OCR now runs inside
scoresplit instead of starting the `tesseract` program for every piece of a
page. Measured on scanned examples: 45% to 55% less CPU time, and the
same parts as 0.2.0 on every example. With the default languages
(`eng+hrv`), a few stray OCR characters on a page (in the header or the
left margin) can read slightly differently than in 0.2.0; on the examples
this never changed a part.

Nothing to do when you upgrade: the install commands, options, output files
and exit codes are the same, and the cache from 0.2.0 is still used.

### Changed

- **No more long pause after reading a PDF with many same-name parts.**
  Finding which parts are copies of each other compares their pages; this
  now runs on several CPU cores and needs about half the CPU, with the
  same answers. italiano.pdf (38 comparisons, 118 pages): 40 s → 9 s.
  A comparison of only a few pages no longer starts extra processes.
- A progress bar "Comparing same-name parts" (and a new `--progress json`
  stage `compare`) shows that comparison. It appears only when there is
  something to compare.
- Every page's tilt is measured about twice as fast (the same result), so
  every PDF needs a little less CPU.
- **The OCR engine comes with scoresplit**: the `tesserocr` package (it
  replaces `pytesseract`), which bundles Tesseract 5.5.4. The system's
  Tesseract package still provides the language files; scoresplit asks the
  `tesseract` program where they are, or uses the folder in
  `TESSDATA_PREFIX` if that is set.
- On macOS older than 15 there is no ready-made `tesserocr` package, so it
  is built from source during the install (see the README).
- Each worker keeps the language models loaded, so it uses about 75 MB more
  memory.
- The box title of the terminal report shows the version
  ("scoresplit 0.3.0").
- The "OCR needed" error (exit code 3) now says that Tesseract's language
  files were not found, and how to install them or set `TESSDATA_PREFIX`.

### Fixed

- Harmless messages from Leptonica (the image library inside the OCR
  engine), such as "Error in boxClipToRectangle", are no longer printed.
- If the `tesseract` program doesn't answer when scoresplit asks where the
  language files are, scoresplit stops waiting after 20 seconds and reports
  them as not found (exit code 3), instead of hanging.

## 0.2.0 (2026-11-14)

Fixes from three code reviews: safer handling of files from anyone, and
better detection on a few kinds of pages.

### Breaking

- **Result files from 0.1.0 are refused.** `schema_version` is now 4.
  Run `scoresplit inspect` on the PDF again to make a new result file.
- **The CSV export has new columns.** If a script or spreadsheet reads the
  CSV, check it against the column list in the README ("The CSV has one row
  per part..."):
  - new `row_type` column: `part`, or `extra` for a group of extra pages
  - new `reason` column on extra rows: `cover`, `blank` or `unassigned`
    (`kind` is now empty on extra rows)
  - only `title`, `composer` and `arranger` are repeated on every row; the
    fingerprint, page counts and warnings are only in the JSON and XML
  - a `|` inside a list item is written as `/`, so a cell always splits
    back at ` | ` into the same items
- **Edits files have limits**: at most 100 000 for a page number and
  `page_count`, 1 000 parts, and 200 000 listed pages. A file over a limit
  is refused (exit code 2). Real scores are far below these.

### Security

- An edits file can no longer make scoresplit use gigabytes of memory by
  listing the same pages over and over.
- Control characters (such as terminal escape codes) are removed from text
  read from a PDF, from loaded result files, from `--title` and
  `--arranger`, and from names typed in the editor.
- A CSV cell that starts with `=`, `+`, `-` or `@` gets a `'` in front, so
  a spreadsheet shows it as text instead of running it as a formula.
- The XML export no longer breaks on characters XML doesn't allow.

### Fixed: detection

- A page printed sideways becomes its own part, "Unknown Part (pN)", at 0%
  confidence, and no longer cuts the part it interrupts in two.
- "E. Bass" and "E. Bass/Tuba" are read as E-flat bass (the flat sign read
  as a dot), not as bass guitar.
- A part name is read only from the top of the page. A cue inside the
  music ("Triangel") no longer counts as the page's part name.
- The title printed on the music pages beats a series name on the cover.
- A credit (composer, arranger) needs to be seen on enough *pages*; one
  page with a text layer no longer counts as three.
- A clef at the very left edge of the page is measured, not missed.

### Fixed: other

- Part file names longer than the file system allows are shortened (the
  key, clef and pages are kept), instead of `split` stopping halfway.
- The export describes the PDF that was really split, also with
  `--edits ... --force`; its JSON then loads back without `--force`, and a
  warning names the other PDF.
- The cache knows whether Tesseract was installed, so installing it later
  makes scoresplit read the pages again.
- Blank pages are no longer turned to look for sideways music (a faster
  first reading).

## 0.1.0 (2026-11-13)

First release.
