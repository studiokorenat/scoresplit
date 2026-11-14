"""
cache.py: remember expensive page analysis (OCR) between runs.

Pipeline position:  page_analyzer <-> [cache]

OCR of a 238-page scan takes a minute or more. When you run ``inspect`` and
then ``split`` on the same file, the second run should be instant. So the
``PageFacts`` of every page are saved as JSON in:

    ~/.cache/scoresplit/<first 16 chars of the PDF's sha256>/<settings>.json

(or ``$XDG_CACHE_HOME/scoresplit/...`` if that environment variable is set,
the Linux standard for cache folders).

The file name includes ANALYSIS_VERSION and the OCR settings (including
whether Tesseract was installed), so changing the analysis code (bump the
version!), the OCR language or installing Tesseract automatically ignores
old cache files. Deleting the folder is always safe.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from .models import PageFacts, TextSpan

# Bump this whenever page_analyzer / ocr / pdf_reader / image_analysis change
# in a way that produces different PageFacts. Old caches are then ignored.
# A change to the part names (instruments.py, data/instruments.yaml) counts
# too: page_analyzer runs an extra OCR pass on a header without a part name.
# 10: part names in seven languages (simplify step 8).
# 11: alto horn, cowbell and other names (simplify round 9.2).
# 12: staves and clefs on light and hand-copied scans (simplify round 9.6).
# 13: a stamp on every page is not page text; "1st" misread (round 9.8).
# 14: a scan with a text layer gets its header OCR'd too (round 9.9).
# 15: text in a frame is read on its own; OCR-tolerant names ("Oboc",
#     "Flügel", "Horn 111 - 1V") change the OCR retries (round 9.10).
# 16: Dutch names and keys: "Solo + 1e", a key C after the name, a dash
#     between two names (round 9.11).
# 17: fat bass clef dots touching the line between them (round 9.12).
# 18: each short header line of a nameless scanned page is read on its
#     own (round 9.12).
# 19: a credit label must be written like one ("arr", "Arr", "ARR", not
#     "aRR"): has_credits decides which text pages get their header OCR'd.
# 20: control characters in read text become a space (tidy_text), so a
#     name never holds a "\x03" (review step 3).
# 21: the C1 control characters (\x80-\x9f) too.
# 22: a clef at the very left edge of the page (staff starting in column 0)
#     is measured, not missed (Buffalo Dances pages 41 and 104).
# 23: pages without staves are also measured turned by 90 degrees
#     (PageFacts.sideways_staff_count): Tullamore page 100 is printed sideways.
# 24: a cue low in a scan's header (inside the music) no longer stops the
#     second, sharper reading of the header (review step 23).
ANALYSIS_VERSION = 24


def file_sha256(path: Path) -> str:
    """
    Fingerprint of a file's content (identical files = identical hash).

    Reads the file in 1 MB chunks so even huge PDFs don't fill the memory.
    """
    digest = hashlib.sha256()
    # "with open(...) as f" opens the file and closes it automatically at the
    # end of the block, even if an error happens (like try/finally in PHP).
    with open(path, "rb") as f:  # "rb" = read bytes
        # iter(callable, b"") keeps calling f.read until it returns b"" (end of file).
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_dir() -> Path:
    """The root folder for scoresplit's cache."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    # pathlib's "/" operator joins path pieces: Path("a") / "b" -> a/b
    return Path(base) / "scoresplit"


def _cache_file(sha256: str, settings_key: str) -> Path:
    return cache_dir() / sha256[:16] / f"v{ANALYSIS_VERSION}-{settings_key}.json"


def load_facts(sha256: str, settings_key: str) -> dict[int, PageFacts]:
    """
    Load cached PageFacts for a PDF. Returns {page_number: PageFacts};
    an empty dict if nothing is cached (or the cache file is unreadable).
    """
    path = _cache_file(sha256, settings_key)
    if not path.exists():
        return {}
    # A broken cache file is simply ignored (it's only a cache). "Broken"
    # includes valid JSON with the wrong shape, for example after someone
    # changed PageFacts and forgot to bump ANALYSIS_VERSION:
    #   ValueError     - not valid JSON
    #   TypeError      - unknown/missing field, or not a list of dicts
    #   KeyError       - a region ("header", ...) is missing
    #   AttributeError - a page that isn't a dict
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        facts: dict[int, PageFacts] = {}
        for item in raw:
            # ** "unpacks" a dict into keyword arguments: TextSpan(**{"text": "x", ...})
            # is the same as TextSpan(text="x", ...). Like PHP's spread with named args.
            for region in ("header", "margin", "body"):
                item[region] = [TextSpan(**span) for span in item[region]]
            page = PageFacts(**item)
            facts[page.page_number] = page
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}
    return facts


def save_facts(sha256: str, settings_key: str, facts: dict[int, PageFacts]) -> None:
    """Write all PageFacts of a PDF to the cache (errors are ignored: it's only a cache)."""
    path = _cache_file(sha256, settings_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)  # like mkdir -p
        # asdict() turns a dataclass (and nested dataclasses) into plain dicts.
        data = [asdict(facts[number]) for number in sorted(facts)]
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
