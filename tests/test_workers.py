"""
Tests for how many OCR worker processes run, and at what priority.

"monkeypatch" is a pytest helper that swaps a function or variable for the
length of one test and puts the real one back afterwards. Here it fakes how
many CPU cores the machine has, so the tests don't depend on the computer
they run on.
"""

from __future__ import annotations

import os

from scoresplit import page_analyzer


def _fake_cores(monkeypatch, count: int) -> None:
    """Pretend this process may run on `count` CPU cores."""
    # A function that ignores its argument (the process id, 0 = this process)
    # and returns a set of `count` core numbers, like the real one.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(count)), raising=False)


def test_default_is_half_the_cores(monkeypatch):
    _fake_cores(monkeypatch, 12)
    assert page_analyzer.default_worker_count() == 6


def test_odd_core_count_rounds_down(monkeypatch):
    _fake_cores(monkeypatch, 3)
    assert page_analyzer.default_worker_count() == 1


def test_one_core_still_gets_one_worker(monkeypatch):
    # A small VPS with a single core: half of 1 is 0, but at least 1 must run.
    _fake_cores(monkeypatch, 1)
    assert page_analyzer.default_worker_count() == 1


def test_falls_back_to_cpu_count_without_sched_getaffinity(monkeypatch):
    # macOS and Windows have no sched_getaffinity.
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert page_analyzer.default_worker_count() == 4


def test_unknown_cpu_count_still_gets_one_worker(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert page_analyzer.default_worker_count() == 1


def test_worker_lowers_its_priority_and_tesseract_threads(monkeypatch):
    # Record the os.nice() call instead of really lowering the priority of
    # the test run, and don't really open a PDF.
    nice_calls: list[int] = []
    monkeypatch.setattr(os, "nice", nice_calls.append, raising=False)
    monkeypatch.setattr(page_analyzer, "open_pdf", lambda path: "fake pdf")
    # setenv/delenv through monkeypatch: the real value comes back after the test.
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)

    page_analyzer._init_worker("score.pdf")

    assert nice_calls == [page_analyzer.WORKER_NICENESS]
    assert os.environ["OMP_THREAD_LIMIT"] == "1"
    # Put the module's global back as it was (the real workers set it).
    monkeypatch.setattr(page_analyzer, "_worker_pdf", None)


# ---------------------------------------------------------------------------
# Music fingerprints in workers (the duplicate-part check)
# ---------------------------------------------------------------------------


def _pdf_with_different_pages(path, count: int):
    """Helper: a PDF whose pages each hold a different pattern of black blocks."""
    import numpy as np
    from PIL import Image

    pictures = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        gray = np.full((1100, 850), 255, dtype=np.uint8)  # 8.5 x 11 inch at 100 dpi
        for x, y in zip(rng.integers(50, 750, 40), rng.integers(50, 1000, 40), strict=True):
            gray[y : y + 20, x : x + 30] = 0
        pictures.append(Image.fromarray(gray))
    # PIL writes a multi-page PDF: the first picture, then the others.
    pictures[0].save(path, save_all=True, append_images=pictures[1:], resolution=100)
    return path


def test_fingerprints_from_workers_match_the_ones_made_here(tmp_path):
    import numpy as np

    pdf_path = _pdf_with_different_pages(tmp_path / "parts.pdf", 4)
    reports = []
    fingerprints = page_analyzer.fingerprint_pages(
        pdf_path, [4, 2, 3], progress=lambda done, total: reports.append((done, total)), workers=2
    )
    # Each page's own fingerprint (not another page's), the same as without workers.
    pdf = page_analyzer.open_pdf(str(pdf_path))
    try:
        for number in (2, 3, 4):
            assert np.array_equal(fingerprints[number], page_analyzer.page_fingerprint(pdf, number))
    finally:
        pdf.close()
    assert sorted(fingerprints) == [2, 3, 4]
    # Counted in pages: a line with the total first, then one per page.
    assert reports == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_no_pages_to_fingerprint_reports_nothing(tmp_path):
    reports = []
    fingerprints = page_analyzer.fingerprint_pages(
        tmp_path / "never-opened.pdf", [], progress=lambda done, total: reports.append((done, total))
    )
    assert fingerprints == {}
    assert reports == []  # no "0/0" stage


def test_music_comparer_fingerprints_each_needed_page_once(monkeypatch, tmp_path):
    import numpy as np

    from scoresplit import pipeline

    asked = []

    def fake_fingerprint_pages(pdf_path, page_numbers, progress, workers):
        asked.append(page_numbers)
        # Pages 1-4 look alike, page 9 is different.
        return {n: np.eye(8) if n != 9 else np.ones((8, 8)) - np.eye(8) for n in page_numbers}

    monkeypatch.setattr(pipeline, "fingerprint_pages", fake_fingerprint_pages)
    comparer = pipeline.MusicComparer(tmp_path / "x.pdf")
    # Only the first DUPLICATE_PAGES_COMPARED (2) pages of each part count.
    answers = comparer.same_music([([1, 2, 5], [3, 4, 6]), ([1, 2], [9, 4])])
    assert answers == [True, False]
    assert asked == [[1, 2, 3, 4, 9]]  # one call, every page once, in order
