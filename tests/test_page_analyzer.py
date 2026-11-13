"""
Tests for page_analyzer.py: which pages are read from their embedded text
and which are OCR'd.

The stamp: american.pdf is a scan whose text layer holds only an archive
stamp, "ARCHIVOCUCOSAX", on every page. That text alone made every page
count as a text page, so nothing was OCR'd and no part was found.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from scoresplit import page_analyzer
from scoresplit.models import TextSpan
from scoresplit.page_analyzer import (
    SCANNED_PAGE_MIN_IMAGE_SHARE,
    AnalysisSettings,
    analyze_page,
    find_stamp_texts,
    letters_without_stamps,
    merge_header_readings,
)
from scoresplit.pdf_reader import largest_image_share


def _span(text: str, y0: float = 0.03) -> TextSpan:
    return TextSpan(text=text, x0=0.1, y0=y0, x1=0.3, y1=y0 + 0.01, size=12.0)


STAMP = _span("ARCHIVOCUCOSAX")


def test_text_on_every_page_is_a_stamp() -> None:
    pages = [[STAMP] for _ in range(10)]
    assert find_stamp_texts(pages) == {"ARCHIVOCUCOSAX"}
    # ... so a page with only the stamp has no letters left: it gets OCR'd.
    assert letters_without_stamps([STAMP], find_stamp_texts(pages)) == 0


def test_a_stamp_may_miss_a_few_pages() -> None:
    # On 8 of 10 pages (80%) it is still a stamp.
    pages = [[STAMP] for _ in range(8)] + [[], []]
    assert find_stamp_texts(pages) == {"ARCHIVOCUCOSAX"}


def test_text_on_some_pages_is_no_stamp() -> None:
    # On 7 of 10 pages: real page text (a running title, say).
    pages = [[STAMP] for _ in range(7)] + [[], [], []]
    assert find_stamp_texts(pages) == frozenset()


def test_a_short_pdf_has_no_stamps() -> None:
    # Three pages of one part: a title on every page is normal there.
    pages = [[_span("Funky Tune")] for _ in range(3)]
    assert find_stamp_texts(pages) == frozenset()


def test_a_part_name_score_word_or_credit_on_every_page_is_no_stamp() -> None:
    # A one-part PDF prints its part name on every page; the detection needs it.
    pages = [
        [_span("Trumpet 1"), _span("Conductor Score"), _span("arr. J. Smith")]
        for _ in range(10)
    ]
    assert find_stamp_texts(pages) == frozenset()


def test_other_text_on_the_page_still_counts() -> None:
    # A text page with a stamp stays a text page: only the stamp is left out.
    spans = [STAMP, _span("Flute 1", y0=0.06)]
    assert letters_without_stamps(spans, frozenset({"ARCHIVOCUCOSAX"})) == 5


def _blank_pdf(tmp_path: Path) -> pdfium.PdfDocument:
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(595, 842)  # A4 in PDF points
    path = tmp_path / "blank.pdf"
    pdf.save(str(path))
    pdf.close()
    return pdfium.PdfDocument(str(path))


def test_a_page_with_only_a_stamp_is_read_as_a_scan(tmp_path: Path, monkeypatch) -> None:
    # The page's text layer is only the stamp (faked here: making a real text
    # layer needs a PDF writer). Without stamps it is a text page; with the
    # stamps of its PDF it is a scan, so it would be OCR'd. (The page itself
    # is blank, so no OCR runs and Tesseract isn't needed for this test.)
    monkeypatch.setattr(page_analyzer, "extract_text_spans", lambda page: [STAMP])
    pdf = _blank_pdf(tmp_path)
    settings = AnalysisSettings(ocr_mode="auto")

    assert analyze_page(pdf, 1, settings).source == "text"
    stamps = frozenset({"ARCHIVOCUCOSAX"})
    assert analyze_page(pdf, 1, settings, stamps).source == "ocr"


# ---------------------------------------------------------------------------
# A scan with a text layer: our own OCR of the header (round 9.9)
#
# superstar.pdf is a scan whose text layer is the scanner's own, garbled OCR
# ("KIARINET", "BARTION SAXOFOON"). Our OCR of the header is merged in, and
# the better reading of each line wins.
# ---------------------------------------------------------------------------


def _line(text: str, y0: float, x0: float = 0.4, size: float = 12.0) -> TextSpan:
    return TextSpan(text=text, x0=x0, y0=y0, x1=x0 + 0.2, y1=y0 + 0.012, size=size)


def test_a_garbled_name_is_replaced_by_our_reading() -> None:
    # Same score (100), but our reading says more: Baritone Saxophone.
    text = [_line("BARTION SAXOFOON Eb", 0.086)]
    ocr = [_line("BARITON SAXOFOON Eb", 0.088, size=7.9)]
    assert [s.text for s in merge_header_readings(text, ocr)] == ["BARITON SAXOFOON Eb"]


def test_a_name_the_text_layer_lost_is_added() -> None:
    # "Eb KIARINET" is a weak match; our "Eb KLARINET" is a full one.
    text = [_line("Eb KIARINET", 0.088)]
    ocr = [_line("Eb KLARINET", 0.089)]
    assert [s.text for s in merge_header_readings(text, ocr)] == ["Eb KLARINET"]


def test_a_better_text_reading_stays() -> None:
    # funkattack: the text layer has the key, our OCR lost it. Both stay
    # (best_header_part then picks the text one, which says more).
    text = [_line("Bb Clarinet 1", 0.081)]
    ocr = [_line("Clarinet 1", 0.084)]
    assert [s.text for s in merge_header_readings(text, ocr)] == ["Bb Clarinet 1", "Clarinet 1"]


def test_ocr_noise_and_the_title_are_not_mixed_in() -> None:
    # Only OCR lines that name a part are taken: the text layer's title stays
    # the only title (our OCR measures letter sizes differently).
    title = _line("JESUS CHRIST SUPERSTAR Webber-Hauwast", 0.048, x0=0.1, size=21.0)
    text = [title, _line("I{OBO C", 0.079)]
    ocr = [
        _line("JESUS CHRIST SUPERSTAR", 0.050, x0=0.1, size=13.0),
        _line("Webber-Hautvast %", 0.052, x0=0.6, size=27.4),
        _line("HOBO C", 0.084),
        _line("- A A", 0.120),
    ]
    merged = [s.text for s in merge_header_readings(text, ocr)]
    assert merged == ["JESUS CHRIST SUPERSTAR Webber-Hauwast", "HOBO C"]


def test_a_reading_of_another_line_does_not_replace() -> None:
    # Two name lines: our OCR of the second must not remove the first.
    text = [_line("2e CORNET Bb", 0.083)]
    ocr = [_line("2e TROMPET Bb", 0.099)]
    assert [s.text for s in merge_header_readings(text, ocr)] == ["2e CORNET Bb", "2e TROMPET Bb"]


def _pdf_with_picture(tmp_path: Path, width: float, height: float) -> pdfium.PdfDocument:
    """A one-page A4 PDF with a white picture of `width` x `height` points."""
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(595, 842)
    picture = pdfium.PdfImage.new(pdf)
    bitmap = pdfium.PdfBitmap.new_native(10, 10, pdfium_c.FPDFBitmap_Gray)
    picture.set_bitmap(bitmap)
    # An image object is 1 x 1 point until it is scaled to its size.
    picture.set_matrix(pdfium.PdfMatrix().scale(width, height))
    page.insert_obj(picture)
    page.gen_content()
    path = tmp_path / "picture.pdf"
    pdf.save(str(path))
    pdf.close()
    return pdfium.PdfDocument(str(path))


def test_a_scanned_page_is_one_big_picture(tmp_path: Path) -> None:
    scan = _pdf_with_picture(tmp_path, 595, 842)
    assert largest_image_share(scan[0]) > 0.99
    logo = _pdf_with_picture(tmp_path, 100, 50)  # a logo on an exported page
    assert largest_image_share(logo[0]) < SCANNED_PAGE_MIN_IMAGE_SHARE
    assert largest_image_share(_blank_pdf(tmp_path)[0]) == 0.0
