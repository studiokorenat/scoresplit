# Changelog

What changed in each release of scoresplit. The newest release is at the
top. Version numbers follow [Semantic Versioning](https://semver.org/):
below 1.0, a breaking change raises the middle number (0.1 → 0.2).

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
