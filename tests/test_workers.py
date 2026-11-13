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
