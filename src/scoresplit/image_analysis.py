"""
image_analysis.py: look at a page's PIXELS (not its text).

Pipeline position:  pdf_reader (pixels) -> [image_analysis] -> page_analyzer

Three questions are answered here:
  * Is the page blank?               -> ink_ratio()
  * Where are the 5-line staves?      -> find_staves()
  * Treble or bass clef?              -> detect_clef()
                                         (bass = the two dots; treble = sticks
                                         out above and below the staff)

Why the clef matters: publishers often print the SAME part twice, once in
treble clef and once in bass clef (e.g. "Trombone 1 in Bb" for musicians
who read treble or bass clef). Same name, different music. The clef tells them apart.

All functions take a grayscale numpy array (0 = black, 255 = white), as made
by ``pdf_reader.render_page_gray``.

Scanned pages are often slightly rotated (1-2 degrees). That breaks staff
detection (a tilted line is spread over many pixel rows), so scans are first
straightened with deskew().

A short numpy primer for this file:
    ink = gray < 128        -> an array of True/False, True where a pixel is dark
    ink.mean()              -> share of True values (True counts as 1)
    ink.mean(axis=1)        -> the same, computed per ROW
    ink[10:20, 5:50]        -> rows 10..19 and columns 5..49 (a rectangle)
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# A pixel darker than this (0..255) counts as ink.
INK_THRESHOLD = 128

# Some scans are so light that their staff lines are grey (130-180), not
# black (Deep Purple page 82, all of mission.pdf and suite.pdf). When fewer
# than LIGHT_SCAN_RETRY_BELOW staves are found with INK_THRESHOLD,
# find_staves is tried again with this lighter limit (see
# find_staves_on_page), and the clef is then measured with it too: at 128
# the grey staff lines are invisible, so the clef can't be found at all.
LIGHT_SCAN_INK_THRESHOLD = 180
# Not only when NO staff is found: on a light scan a few dark spots of the
# lines often make 1 or 2 "staves" at the normal limit (mission pages 56-57,
# suite pages with 1-4 of their ~12 staves).
LIGHT_SCAN_RETRY_BELOW = 3

# A page with less than this share of ink pixels is blank.
# (0.002 = 0.2% of the page; scan noise and a page number stay below it.)
BLANK_INK_RATIO = 0.002

# Staff lines are searched for in this horizontal band of the page (share of
# the width). Using only the left part makes it robust against slightly
# rotated scans, and avoids long horizontal text-free areas on the right.
STAFF_SEARCH_X = (0.05, 0.40)

# A pixel row is a "staff line row" when this share of the band is ink.
STAFF_LINE_MIN_FILL = 0.5

# The 5 lines of a staff must be evenly spaced: the largest and smallest gap
# may differ by at most this share of the average gap.
STAFF_SPACING_TOLERANCE = 0.25

# Clef detection. All distances are in "spaces" (the distance between two
# staff lines) and measured from where the clef glyph STARTS (see
# _find_clef_start: a barline or bracket at the staff start is skipped).
CLEF_X_RANGE = (-0.1, 2.0)  # the clef glyph itself
CLEF_ABOVE = (1.8, 0.4)  # from 1.8 spaces above the top line to 0.4 above
CLEF_BELOW = (0.4, 1.6)  # from 0.4 spaces below the bottom line to 1.6 below

# A treble clef sticks out above AND below the staff.
TREBLE_MIN_INK_ABOVE = 0.04
TREBLE_MIN_INK_BELOW = 0.02
# ...and above the staff it is ONE unbroken stroke that leaves the top line:
# every pixel row from the top line up to this many spaces above it has ink
# somewhere in the clef's columns. A rehearsal mark (a circled letter) above
# a bass clef has white between it and the staff (suite pages 72, 78, 106).
# Only above: below the staff the thin tail of a scanned treble clef often
# breaks up into pieces (Deep Purple), and above, a hand-drawn clef may end
# just over the top line (Deep Purple page 58), so 0.6 spaces, not more.
TREBLE_UNBROKEN_ABOVE = 0.6
# The columns for that test, from the clef start (in spaces). Wider than
# CLEF_X_RANGE: the stroke leaving the staff is at the RIGHT of the glyph,
# up to ~2.4 spaces from where the clef starts.
TREBLE_UNBROKEN_X = 3.0

# Staves taller than this share of the page height are not real staves but
# five one-line percussion staves that happen to be evenly spaced (mission
# pages 84-88, rhcp page 172). Real staves are below 1.1% of the page height,
# those fake ones above 7%. They still count as staves (the page has music),
# but they get no vote in the clef detection.
CLEF_MAX_SPACING_SHARE = 0.03

# A bass clef has two dots right of it, in the top two spaces of the staff.
# They are searched for in this horizontal range... (Not wider: the two
# slanted bars of a sharp in the key signature right after it look like dots.)
BASS_DOTS_X_RANGE = (1.9, 3.1)
# ...and a "dot" is ink in those two spaces (within this distance of the
# space's centre) with NO ink in the two lower spaces, for a few columns
# but not wider than this (wider = not a dot).
BASS_DOT_HALF_HEIGHT = 0.15
BASS_DOT_MAX_WIDTH = 0.9
# Between a dot and the staff lines above and below it there must be a white
# gap: at least one white row this far (in spaces) from the dot's centre.
# A band, not one fixed row, because on scans with thick staff lines the
# line itself already reaches ~0.37 spaces from the space's centre.
BASS_DOT_GAP = (0.26, 0.40)
# On a LIGHT scan that white gap is gone: at LIGHT_SCAN_INK_THRESHOLD the
# lines are thick and the (often hand-drawn) dots run into them (suite.pdf).
# There a column also counts as a dot when nothing is above the staff in
# this band (from 1.4 to 0.8 spaces above the top line): the stroke of a
# sharp (#) reaches up there, a bass clef's dot never does. Not closer to
# the line, which is very thick on a light scan. Only on light scans: on
# normal pages the stem of a flat can pass this test (Deep Purple page 64,
# a key signature after a double barline).
BASS_DOT_NOTHING_ABOVE = (1.4, 0.8)
# On a sharp, thick print (superstar pages 48-53) the two dots are fat and
# run into the staff line BETWEEN them, so the two white gaps next to that
# line are gone. The dots still count when the two OUTER gaps are white (a
# sharp's or flat's stroke runs through them) and the upper dot sits in the
# middle of its space: ink within this distance (in spaces) of the space's
# centre. A notehead in the second space with a slur dipping into the top
# space (Deep Purple page 45) has the same gaps, but the slur only touches
# the edge of the top space.
BASS_DOT_CENTRE_HALF_HEIGHT = 0.05

# How far to the right of the staff start the clef may begin (a bracket or
# barline can push it right).
CLEF_START_SEARCH = 3.0

# The clef starts at the first column with ink within this many spaces of
# the middle of a staff space (0.5 would reach the staff lines themselves).
CLEF_START_SPACE_HALF = 0.2

# The staff starts where its 5 lines run on for at least this many spaces.
# Shorter runs are vertical lines that merely CROSS the staff lines, e.g. the
# dark edge or binding shadow of a scanned page.
STAFF_START_MIN_RUN = 2.0


# Deskewing: the largest tilt we look for (as a slope: 0.05 = about 2.9 degrees),
# the step between tried slopes, and into how many vertical strips the page is
# cut to measure the tilt.
DESKEW_MAX_SLOPE = 0.05
DESKEW_SLOPE_STEP = 0.0005
DESKEW_STRIPS = 12
DESKEW_X_RANGE = (0.05, 0.65)  # part of the page width used for measuring


def estimate_skew_slope(gray: np.ndarray) -> float:
    """
    How much the page is tilted, as a slope (pixels down per pixel right).

    How: cut the page into vertical strips and make a "row profile" of each
    (how much ink is in every pixel row). On a straight page, the staff
    lines are at the same rows in every strip, so adding the profiles gives
    very sharp peaks. On a tilted page they are shifted a bit from strip to
    strip. We try many slopes, shift each strip's profile back by that
    slope, and keep the slope that gives the sharpest combined profile
    (largest sum of squares).
    """
    ink = gray < INK_THRESHOLD
    width = ink.shape[1]
    left, right = int(width * DESKEW_X_RANGE[0]), int(width * DESKEW_X_RANGE[1])
    # np.linspace gives evenly spaced numbers: the strip borders.
    edges = np.linspace(left, right, DESKEW_STRIPS + 1).astype(int)
    profiles = [ink[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    centres = (edges[:-1] + edges[1:]) / 2 - left  # x of each strip, from the left edge

    best_score, best_slope = -1.0, 0.0
    for slope in np.arange(-DESKEW_MAX_SLOPE, DESKEW_MAX_SLOPE + DESKEW_SLOPE_STEP / 2, DESKEW_SLOPE_STEP):
        # np.roll shifts a profile up/down by the number of rows the tilt
        # moved it at that strip.
        combined = sum(
            np.roll(profile, -int(round(slope * centre)))
            for profile, centre in zip(profiles, centres, strict=True)
        )
        score = float((combined**2).sum())
        if score > best_score:
            best_score, best_slope = score, float(slope)
    return best_slope


def deskew(gray: np.ndarray) -> np.ndarray:
    """Return a straightened copy of a (scanned) page image."""
    slope = estimate_skew_slope(gray)
    if abs(slope) < DESKEW_SLOPE_STEP:
        return gray  # already straight: nothing to do
    degrees = math.degrees(math.atan(slope))
    # PIL rotates counter-clockwise for positive angles; fillcolor=255 paints
    # the uncovered corners white.
    rotated = Image.fromarray(gray).rotate(degrees, resample=Image.BILINEAR, fillcolor=255)
    return np.asarray(rotated)


def ink_ratio(gray: np.ndarray) -> float:
    """Share of dark pixels on the page (0.0 = white page, 1.0 = all black)."""
    return float((gray < INK_THRESHOLD).mean())


def is_blank(gray: np.ndarray) -> bool:
    """True if the page has (almost) no ink."""
    return ink_ratio(gray) < BLANK_INK_RATIO


def find_staves(
    gray: np.ndarray, threshold: int = INK_THRESHOLD
) -> list[tuple[list[float], float]]:
    """
    Find the 5-line music staves on a page. Pixels darker than `threshold`
    count as ink.

    Returns a list of staves. Each staff is (line_y_positions, spacing):
    the 5 y-coordinates (pixel rows) of its lines, and the average distance
    between them.

    How: count the ink in every pixel row of the search band. Staff lines
    are rows that are mostly ink. Neighbouring ink rows (a line can be 2-3
    pixels thick) are merged into one line. Then we look for 5 evenly spaced
    lines in a row.

    Six evenly spaced lines in a row: on a light scan a faint "ghost" line
    (the staff on the back of the paper, shining through) can sit one space
    above or below the real staff (suite page 73). Then we keep the five
    whose faintest line is the darkest, so the ghost is left out. Taking the
    first five would shift every line and space of the staff by one, and
    the clef could no longer be read.
    """
    ink = gray < threshold
    height, width = ink.shape
    band_left, band_right = int(width * STAFF_SEARCH_X[0]), int(width * STAFF_SEARCH_X[1])

    row_fill = ink[:, band_left:band_right].mean(axis=1)  # one number per row
    line_rows = np.where(row_fill > STAFF_LINE_MIN_FILL)[0]  # indexes of "line" rows

    # Merge neighbouring rows into lines: [[first_row, last_row], ...]
    lines: list[list[int]] = []
    for row in line_rows:
        if lines and row - lines[-1][1] <= 2:
            lines[-1][1] = row  # touching the previous line: extend it
        else:
            lines.append([row, row])
    centres = [(first + last) / 2 for first, last in lines]
    # How dark each line is: the fullest row of its rows (1.0 = all ink).
    darkness = [float(row_fill[first : last + 1].max()) for first, last in lines]

    def is_staff(start: int) -> bool:
        """Are the 5 lines from index `start` on an evenly spaced staff?"""
        if start + 5 > len(centres):
            return False
        gaps = np.diff(centres[start : start + 5])  # the 4 distances between the 5 lines
        evenly_spaced = gaps.max() - gaps.min() <= STAFF_SPACING_TOLERANCE * gaps.mean()
        return bool(gaps.min() > 3 and evenly_spaced)

    staves: list[tuple[list[float], float]] = []
    i = 0
    while i <= len(centres) - 5:
        if is_staff(i):
            # Lines i..i+4 are a staff. If i+1..i+5 are one too, line i may
            # be a ghost above the real staff: keep the darker five.
            if is_staff(i + 1) and min(darkness[i + 1 : i + 6]) > min(darkness[i : i + 5]):
                i += 1
            five = centres[i : i + 5]
            staves.append((five, float(np.diff(five).mean())))
            i += 5  # these 5 lines are used up
        else:
            i += 1  # try starting one line later
    return staves


def find_staves_on_page(gray: np.ndarray) -> tuple[list[tuple[list[float], float]], int]:
    """
    Find the staves of a page, also on a light scan.

    Returns (staves, threshold): the staves as find_staves gives them, and
    the ink limit that found them. Pass that limit on to detect_clef: the
    clef must be measured with the same limit as the staff lines around it.

    First with the normal INK_THRESHOLD. When that finds fewer than
    LIGHT_SCAN_RETRY_BELOW staves, the page may be a light scan with grey
    staff lines: look again with LIGHT_SCAN_INK_THRESHOLD and keep whichever
    found MORE staves. (A page with one or two real staves, e.g. a short
    coda, finds the same staves both times and keeps the normal limit.)
    """
    staves = find_staves(gray)
    if len(staves) >= LIGHT_SCAN_RETRY_BELOW:
        return staves, INK_THRESHOLD
    light_staves = find_staves(gray, threshold=LIGHT_SCAN_INK_THRESHOLD)
    if len(light_staves) > len(staves):
        return light_staves, LIGHT_SCAN_INK_THRESHOLD
    return staves, INK_THRESHOLD


def _space_has_ink(ink: np.ndarray, centre: float, half: float, spacing: float, column: int) -> bool:
    """Is there ink in `column` within `half` spaces of the row `centre`?"""
    top = int(centre - half * spacing)
    bottom = int(centre + half * spacing) + 1
    return bool(ink[top:bottom, column].any())


def _find_clef_start(ink: np.ndarray, spaces: list[float], staff_start: int, spacing: float) -> int:
    """
    The column where the clef glyph begins.

    Usually right after the staff starts, but a staff can begin with a
    barline or a bracket ([), which pushes the clef further right:

        staff start -> |barline| -> empty gap -> clef

    We skip columns where ALL four spaces have ink (a barline), then empty
    columns (no ink in any space), and stop at the first ink after that.
    """
    width = ink.shape[1]
    limit = min(width - 1, int(staff_start + CLEF_START_SEARCH * spacing))
    column = staff_start

    def filled(c: int, half: float) -> list[bool]:
        return [_space_has_ink(ink, y, half, spacing, c) for y in spaces]

    while column < limit and all(filled(column, 0.3)):
        column += 1  # inside a barline
    # Only the middle of each space counts here: on scans with thick (or
    # ragged) staff lines, the lines reach far into the spaces and would
    # look like the start of the clef.
    while column < limit and not any(filled(column, CLEF_START_SPACE_HALF)):
        column += 1  # the gap before the clef
    return column


def _has_white_gap(
    ink: np.ndarray, centre: float, direction: int, spacing: float, column: int
) -> bool:
    """
    Is there at least one white pixel in `column` between BASS_DOT_GAP[0]
    and BASS_DOT_GAP[1] spaces above (direction -1) or below (+1) `centre`?
    """
    near = round(centre + direction * BASS_DOT_GAP[0] * spacing)
    far = round(centre + direction * BASS_DOT_GAP[1] * spacing)
    top, bottom = min(near, far), max(near, far) + 1
    return not bool(ink[top:bottom, column].all())


def _nothing_above_staff(ink: np.ndarray, top_line: float, spacing: float, column: int) -> bool:
    """Is `column` white in the BASS_DOT_NOTHING_ABOVE band above the staff?"""
    row_from = max(0, int(top_line - BASS_DOT_NOTHING_ABOVE[0] * spacing))
    row_to = max(0, int(top_line - BASS_DOT_NOTHING_ABOVE[1] * spacing))
    return not bool(ink[row_from:row_to, column].any())


def _has_bass_dots(
    ink: np.ndarray,
    lines: list[float],
    spaces: list[float],
    clef_start: int,
    spacing: float,
    light_scan: bool,
) -> bool:
    """
    Look for the two dots of a bass clef: a narrow column of ink in BOTH top
    spaces, with nothing in the two lower spaces.
    """
    col_from = int(clef_start + BASS_DOTS_X_RANGE[0] * spacing)
    col_to = min(ink.shape[1], int(clef_start + BASS_DOTS_X_RANGE[1] * spacing))
    run = longest = 0
    for column in range(col_from, col_to):
        # White just above and below each dot? A dot is a small blob floating
        # in its space; the vertical stroke of a sharp (#) in a key signature
        # runs through the whole space instead.
        floating = all(
            _has_white_gap(ink, centre, direction, spacing, column)
            for centre in (spaces[0], spaces[1])
            for direction in (-1, 1)  # -1 = above, 1 = below
        )
        # Fat dots that touch the line between them: only the outer gaps
        # (above the upper dot, below the lower dot) must be white, and the
        # upper dot must cover the middle of its space.
        touching_the_middle_line = (
            _has_white_gap(ink, spaces[0], -1, spacing, column)
            and _has_white_gap(ink, spaces[1], 1, spacing, column)
            and _space_has_ink(ink, spaces[0], BASS_DOT_CENTRE_HALF_HEIGHT, spacing, column)
        )
        is_dot_column = (
            # ink in the middle of both top spaces...
            _space_has_ink(ink, spaces[0], BASS_DOT_HALF_HEIGHT, spacing, column)
            and _space_has_ink(ink, spaces[1], BASS_DOT_HALF_HEIGHT, spacing, column)
            # ...nothing in the two lower spaces...
            and not _space_has_ink(ink, spaces[2], 0.3, spacing, column)
            and not _space_has_ink(ink, spaces[3], 0.3, spacing, column)
            # ...and not the stroke of a sharp: the dots float in their
            # spaces, or touch only the line between them, or (light scans,
            # where dots touch the thick lines) nothing reaches up above the
            # staff.
            and (
                floating
                or touching_the_middle_line
                or (light_scan and _nothing_above_staff(ink, lines[0], spacing, column))
            )
        )
        run = run + 1 if is_dot_column else 0
        longest = max(longest, run)
    return 2 <= longest <= BASS_DOT_MAX_WIDTH * spacing


def _staff_start(on_lines: np.ndarray, min_run: float) -> int | None:
    """
    The first column of the first run of at least `min_run` columns where
    `on_lines` is True (= the staff lines are there). None if there is none.

    A thin vertical line crossing the staff (page edge, binding shadow) is a
    run of only 1-3 columns, so it is skipped.
    """
    run_start: int | None = None
    # enumerate() gives (index, value) pairs, like foreach ($a as $i => $v).
    for column, on in enumerate(on_lines):
        if not on:
            run_start = None
            continue
        if run_start is None:
            run_start = column
        if column - run_start + 1 >= min_run:
            return run_start
    return None


def _clef_of_staff(
    ink: np.ndarray, lines: list[float], spacing: float, light_scan: bool = False
) -> str | None:
    """
    Decide treble/bass for one staff. `light_scan`: the ink was measured
    with LIGHT_SCAN_INK_THRESHOLD (see _has_bass_dots).

    * two dots in the top spaces right of the clef     -> "bass"
    * ink sticking out above AND below the clef, and
      an unbroken stroke leaving the staff upwards     -> "treble"
    * otherwise (e.g. a percussion clef: two bars)     -> None
    """
    height = ink.shape[0]
    rows = [int(round(y)) for y in lines]
    # The centres of the 4 spaces between the 5 lines.
    spaces = [(lines[i] + lines[i + 1]) / 2 for i in range(4)]

    # Find where the staff starts: at least 4 of the 5 lines have ink (±1
    # pixel, for slightly tilted scans), for a long enough run of columns.
    line_hits = np.stack([ink[max(0, r - 1) : r + 2].any(axis=0) for r in rows]).sum(axis=0)
    staff_start = _staff_start(line_hits >= 4, STAFF_START_MIN_RUN * spacing)
    if staff_start is None:
        return None
    clef_start = _find_clef_start(ink, spaces, staff_start, spacing)

    if _has_bass_dots(ink, lines, spaces, clef_start, spacing, light_scan):
        return "bass"

    # The clef's columns start a little LEFT of clef_start. Keep them on the
    # page: when a staff starts in column 0 (a tightly cut scan), col_from
    # would be -1, and ink[:, -1:26] means "from the LAST column to 26",
    # an empty slice (its mean is NaN, and the clef was missed).
    col_from = max(0, int(clef_start + CLEF_X_RANGE[0] * spacing))
    col_to = min(ink.shape[1], int(clef_start + CLEF_X_RANGE[1] * spacing))
    top_line, bottom_line = lines[0], lines[4]

    def ink_between(row_from: float, row_to: float) -> float:
        row_from, row_to = max(0, int(row_from)), min(height, int(row_to))
        if row_to <= row_from:
            return 0.0
        return float(ink[row_from:row_to, col_from:col_to].mean())

    above = ink_between(top_line - CLEF_ABOVE[0] * spacing, top_line - CLEF_ABOVE[1] * spacing)
    below = ink_between(bottom_line + CLEF_BELOW[0] * spacing, bottom_line + CLEF_BELOW[1] * spacing)
    if above <= TREBLE_MIN_INK_ABOVE or below <= TREBLE_MIN_INK_BELOW:
        return None  # unclear: percussion clef, text in the way, ...

    # Ink above and below. But is it the clef, or a rehearsal mark above a
    # bass clef? The treble clef leaves the staff in one unbroken stroke:
    # every row from the top line up has ink in the clef's columns.
    stroke_to = int(clef_start + TREBLE_UNBROKEN_X * spacing)
    row_from = max(0, int(top_line - TREBLE_UNBROKEN_ABOVE * spacing))
    rows_with_ink = ink[row_from : int(top_line), col_from:stroke_to].any(axis=1)
    if rows_with_ink.all():
        return "treble"
    return None


def detect_clef(
    gray: np.ndarray, staves: list[tuple[list[float], float]], threshold: int = INK_THRESHOLD
) -> str | None:
    """
    The clef of the page: a majority vote over all its staves.

    `threshold` is the ink limit that found the staves (find_staves_on_page):
    on a light scan the clef must be measured with the same lighter limit,
    or the grey staff lines around it are invisible.

    Returns "treble", "bass", or None if there's no clear majority.
    """
    ink = gray < threshold
    light_scan = threshold != INK_THRESHOLD
    votes = Counter()  # Counter = a dict that counts: votes["bass"] += 1
    for lines, spacing in staves:
        if spacing > CLEF_MAX_SPACING_SHARE * gray.shape[0]:
            continue  # one-line percussion staves, not a real staff
        clef = _clef_of_staff(ink, lines, spacing, light_scan)
        if clef:
            votes[clef] += 1
    if not votes:
        return None
    # most_common(1) returns a list with one (value, count) pair: [("bass", 9)].
    winner, winner_votes = votes.most_common(1)[0]
    # Require a clear majority: at least 2/3 of the decided staves.
    if winner_votes >= 2 / 3 * sum(votes.values()):
        return winner
    return None


# ---------------------------------------------------------------------------
# Comparing the MUSIC of two pages (are two same-name parts real copies?)
# ---------------------------------------------------------------------------

# Horizontal ink runs at least this long (in pixels at 200 dpi, ~7.6 mm) are
# staff lines or long beams. They are removed before comparing, because every
# page of music has staff lines in the same places: with them, two DIFFERENT
# parts with the same layout would look alike.
LONG_RUN_PIXELS = 60

# Pages are compared as a coarse map of "notes ink": one cell per this many
# pixels in each direction (4 -> 1/16 of the pixels, fast to compare).
FINGERPRINT_CELL = 4

# Two scans of the same page are never placed exactly alike on the scanner:
# shifts of up to this many cells (x FINGERPRINT_CELL pixels, i.e. ~6 mm at
# 200 dpi) are tried, and the best match counts.
MAX_SHIFT_CELLS = 12


def _long_horizontal_runs(ink: np.ndarray, length: int) -> np.ndarray:
    """
    True for every ink pixel that is part of a horizontal run of at least
    `length` ink pixels (staff lines, beams, long slurs drawn flat).

    How (all rows at once, with running sums instead of loops):
      1. `window_full[y, x]` = the `length` pixels ending at x are all ink.
      2. A pixel is in a long run if some full window ending at x, x+1, ...,
         x+length-1 covers it.
    """
    height, width = ink.shape
    if width < length:
        return np.zeros_like(ink)
    # sums[:, x] = number of ink pixels in columns 0..x-1 of each row.
    sums = np.zeros((height, width + 1), dtype=np.int32)
    sums[:, 1:] = np.cumsum(ink, axis=1)
    window_full = np.zeros((height, width), dtype=bool)
    window_full[:, length - 1 :] = (sums[:, length:] - sums[:, : width - length + 1]) == length
    # Running sum of window ends, to ask "any full window ending in x..x+L-1?".
    ends = np.zeros((height, width + 1), dtype=np.int32)
    ends[:, 1:] = np.cumsum(window_full, axis=1)
    columns = np.arange(width)
    upto = np.minimum(columns + length, width)
    return (ends[:, upto] - ends[:, columns]) > 0


def music_fingerprint(gray: np.ndarray) -> np.ndarray:
    """
    A small map of where the NOTES are on a (straightened) page: ink without
    staff lines, averaged over FINGERPRINT_CELL x FINGERPRINT_CELL blocks and
    lightly blurred, so a one-cell wobble doesn't matter.
    """
    ink = gray < INK_THRESHOLD
    ink &= ~_long_horizontal_runs(ink, LONG_RUN_PIXELS)  # &= ~ : "and not"
    cell = FINGERPRINT_CELL
    rows, cols = ink.shape[0] // cell, ink.shape[1] // cell
    # reshape trick: cut the picture into cell x cell blocks and average each.
    blocks = ink[: rows * cell, : cols * cell].reshape(rows, cell, cols, cell).mean(axis=(1, 3))
    padded = np.pad(blocks, 1)  # a border of zeros, so the 3x3 blur fits
    blurred = sum(padded[dy : dy + rows, dx : dx + cols] for dy in range(3) for dx in range(3))
    return blurred / 9


def music_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    How alike two fingerprints are: about 0.95 for two scans of the same
    page, 0.3-0.7 for different parts with a similar layout (e.g. the same
    part in treble and bass clef, or in Eb and Bb).

    This is the best "correlation" over all shifts up to MAX_SHIFT_CELLS. The
    correlation for every shift at once comes from the FFT (a standard trick:
    multiplying the Fourier transforms = sliding one picture over the other).
    """
    height, width = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a = a[:height, :width] - a[:height, :width].mean()
    b = b[:height, :width] - b[:height, :width].mean()
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if norm == 0:
        return 0.0  # an empty page: nothing to compare
    # Zero padding by MAX_SHIFT_CELLS so shifted copies don't wrap around.
    size = (height + MAX_SHIFT_CELLS, width + MAX_SHIFT_CELLS)
    correlation = np.fft.irfft2(np.fft.rfft2(a, size) * np.conj(np.fft.rfft2(b, size)), size)
    # Shifts 0..+MAX are at the start of each axis, -MAX..-1 at the end.
    near = list(range(MAX_SHIFT_CELLS + 1)) + list(range(-MAX_SHIFT_CELLS, 0))
    return float(correlation[np.ix_(near, near)].max() / norm)


# ---------------------------------------------------------------------------
# Text in a box ("Score C" in a frame, cornfield page 2)
# ---------------------------------------------------------------------------

# Tesseract reads a word printed inside a thin frame badly: the frame's side
# lines run into the letters ("Score C" came out as "C", or "[serec |").
# Inside the frame, without its lines, the same word reads fine. So boxes in
# the header are found and their inside is read on its own.

# A pixel darker than this counts as ink for the frame lines. Lighter than
# INK_THRESHOLD: a scanned thin line is grey and broken up.
BOX_INK_THRESHOLD = 160

# A frame is at least this wide and between these heights (inches): a label
# like "Score C", not a staff (wider) or a letter (smaller).
BOX_MIN_WIDTH_INCHES = 0.4
BOX_MAX_WIDTH_SHARE = 0.5  # ...and at most half the page wide
BOX_HEIGHT_INCHES = (0.1, 0.6)

# Gaps of up to this many pixels in a scanned line are closed before
# measuring it, and its two ends may differ by this much (pixels).
BOX_LINE_GAP = 3
BOX_END_TOLERANCE = 6

# The left and right sides must cover at least this share of the height.
BOX_SIDE_MIN_FILL = 0.8

# The inside of the frame is read this many pixels in from the lines.
BOX_INSET = 4


def _shifted(ink: np.ndarray, step: int, axis: int) -> np.ndarray:
    """`ink` moved by `step` pixels along `axis` (0 = down, 1 = right), padded with False."""
    if step == 0:
        return ink.copy()
    # np.roll moves the array around; the pixels that wrap around the edge
    # are then cleared.
    moved = np.roll(ink, step, axis=axis)
    edge = slice(0, step) if step > 0 else slice(step, None)
    if axis == 0:
        moved[edge, :] = False
    else:
        moved[:, edge] = False
    return moved


def _close_gaps(ink: np.ndarray, axis: int, gap: int) -> np.ndarray:
    """
    Fill holes of up to 2 x `gap` pixels in lines along `axis` (1 = rows,
    0 = columns): a scanned line is broken up into dashes. (First grow the
    ink by `gap` on both sides, then shrink it back: holes stay filled.)
    """
    grown = ink.copy()
    for step in range(-gap, gap + 1):
        grown |= _shifted(ink, step, axis)
    closed = grown.copy()
    for step in range(-gap, gap + 1):
        closed &= _shifted(grown, step, axis)
    return closed


def _horizontal_lines(ink: np.ndarray, min_length: int) -> list[tuple[int, int, int]]:
    """Every horizontal run of ink at least `min_length` long, as (row, start, end)."""
    lines = []
    for row in range(ink.shape[0]):
        # The run starts and ends are where the row changes from False to
        # True and back (np.diff of 0/1 values, with a False added at both ends).
        padded = np.concatenate(([False], ink[row], [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for start, end in zip(starts, ends, strict=True):
            if end - start >= min_length:
                lines.append((row, int(start), int(end)))
    return lines


def find_text_boxes(
    gray: np.ndarray, bottom: int, dpi: int
) -> list[tuple[float, float, float, float]]:
    """
    The insides of the rectangular frames above row `bottom` of the page
    (the header), as (x0, y0, x1, y1) shares of the page, top to bottom.

    A frame = a top and a bottom line with the same ends, joined by a left
    and a right side. Its size must fit a short label (see BOX_*).
    """
    height, width = gray.shape
    ink = gray[:bottom] < BOX_INK_THRESHOLD
    # A slightly tilted line spreads over two rows (or columns): each pixel
    # also counts its two neighbours across the line. Then the gaps of a
    # broken scanned line are closed.
    across_rows = ink | _shifted(ink, 1, 0) | _shifted(ink, -1, 0)
    across_columns = ink | _shifted(ink, 1, 1) | _shifted(ink, -1, 1)
    rows = _close_gaps(across_rows, axis=1, gap=BOX_LINE_GAP)
    columns = _close_gaps(across_columns, axis=0, gap=BOX_LINE_GAP)

    min_height, max_height = (int(inches * dpi) for inches in BOX_HEIGHT_INCHES)
    lines = [
        line
        for line in _horizontal_lines(rows, int(BOX_MIN_WIDTH_INCHES * dpi))
        if line[2] - line[1] <= BOX_MAX_WIDTH_SHARE * width
    ]
    boxes: list[tuple[int, int, int, int]] = []
    for index, (top, left, right) in enumerate(lines):
        # A thick line is found on several rows: only its first row starts a box.
        if any(
            abs(top - box[1]) < min_height and abs(left - box[0]) < BOX_END_TOLERANCE
            for box in boxes
        ):
            continue
        for below, other_left, other_right in lines[index + 1 :]:
            if not min_height <= below - top <= max_height:
                continue
            ends_differ = max(abs(other_left - left), abs(other_right - right))
            if ends_differ > BOX_END_TOLERANCE:
                continue
            # The sides: the share of rows between the two lines that have
            # ink at the left (and right) end, within a few pixels.
            left_side = columns[top:below, max(0, left - 2) : left + 3].any(axis=1).mean()
            right_side = columns[top:below, right - 3 : right + 2].any(axis=1).mean()
            if left_side >= BOX_SIDE_MIN_FILL and right_side >= BOX_SIDE_MIN_FILL:
                boxes.append((left, top, right, below))
                break  # the nearest bottom line closes this box
    return [
        (
            (left + BOX_INSET) / width,
            (top + BOX_INSET) / height,
            (right - BOX_INSET) / width,
            (below - BOX_INSET) / height,
        )
        for left, top, right, below in boxes
    ]


# ---------------------------------------------------------------------------
# Short text lines in the header ("F Horn I", cornfield page 38)
# ---------------------------------------------------------------------------

# When Tesseract reads the whole header at once (psm 11), it can drop most
# of a small name next to a big title: "F Horn I" came out as "F", "H" and
# "1". The same line cut out on its own and read as ONE line (psm 7) reads
# "F Horn 1". So the header is cut into its text lines, and each short one
# is read on its own.

# A text line is between these heights (inches): a part name or a composer,
# not the big title (taller) or a speck of dust (smaller).
TEXT_LINE_HEIGHT_INCHES = (0.04, 0.25)
# ...and at most this share of the page wide: a name, not a staff line.
TEXT_LINE_MAX_WIDTH_SHARE = 0.5
# Two pieces of ink on the same row band belong to the same line when the
# white gap between them is at most this many line heights (a space between
# words), but at least TEXT_LINE_MIN_GAP_INCHES. The gap between a name on
# the left and the title in the middle is much wider.
TEXT_LINE_WORD_GAP = 1.5
TEXT_LINE_MIN_GAP_INCHES = 0.08
# The line is cut out with this much white around it (in line heights):
# Tesseract needs a margin, and a tight cut loses the edge of a letter.
TEXT_LINE_MARGIN = 0.5
# How often a piece is cut again (rows, then columns, then rows, ...).
TEXT_LINE_MAX_CUTS = 4


def _ink_runs(has_ink: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """
    The runs of True in a row of booleans, as (start, end) with `end` one
    past the last True. Runs with at most `max_gap` False values between
    them are joined into one.

    Example: [F, T, T, F, T, F, F, F, T], max_gap=1 -> [(1, 5), (8, 9)]
    """
    runs: list[list[int]] = []
    for index in np.flatnonzero(has_ink):
        if runs and index - runs[-1][1] <= max_gap:
            runs[-1][1] = int(index) + 1  # close enough: extend the last run
        else:
            runs.append([int(index), int(index) + 1])
    return [(start, end) for start, end in runs]


def find_text_lines(
    gray: np.ndarray, bottom: int, dpi: int
) -> list[tuple[float, float, float, float]]:
    """
    The short text lines above row `bottom` of the page (the header), as
    (x0, y0, x1, y1) shares of the page, each with a margin around it.

    How (a "cut" into bands, again and again):
      1. Cut the area into row bands: rows with ink, split at white rows.
      2. Cut each band into pieces: columns with ink, split at wide white
         gaps (wider than the space between two words).
      3. A piece that is smaller than its band (a small name next to a big
         title shares the title's rows) is cut again, from step 1.
    A piece that can't be cut any further is a text line, when its height
    and width fit TEXT_LINE_*.
    """
    height, width = gray.shape
    ink = gray[:bottom] < INK_THRESHOLD
    min_height, max_height = (inches * dpi for inches in TEXT_LINE_HEIGHT_INCHES)
    min_gap = int(TEXT_LINE_MIN_GAP_INCHES * dpi)
    found: list[tuple[int, int, int, int]] = []  # (left, top, right, bottom) in pixels

    def cut(top: int, low: int, left: int, right: int, cuts_left: int) -> None:
        """Cut the area rows top..low, columns left..right (see above)."""
        area = ink[top:low, left:right]
        # .any(axis=1): one True/False per row: does the row have ink?
        for band_top, band_low in _ink_runs(area.any(axis=1), max_gap=1):
            band = area[band_top:band_low]
            word_gap = max(min_gap, int(TEXT_LINE_WORD_GAP * (band_low - band_top)))
            for piece_left, piece_right in _ink_runs(band.any(axis=0), word_gap):
                # The rows this piece really uses (it may be lower than its band).
                rows = np.flatnonzero(band[:, piece_left:piece_right].any(axis=1))
                piece = (
                    left + piece_left,
                    top + band_top + int(rows[0]),
                    left + piece_right,
                    top + band_top + int(rows[-1]) + 1,
                )
                if piece != (left, top, right, low) and cuts_left > 0:
                    cut(piece[1], piece[3], piece[0], piece[2], cuts_left - 1)
                    continue
                line_height = piece[3] - piece[1]
                line_width = piece[2] - piece[0]
                if (
                    min_height <= line_height <= max_height
                    and line_height <= line_width <= TEXT_LINE_MAX_WIDTH_SHARE * width
                ):
                    found.append(piece)

    cut(0, bottom, 0, width, TEXT_LINE_MAX_CUTS)

    lines = []
    for line_left, line_top, line_right, line_bottom in sorted(found, key=lambda box: box[1]):
        margin = int(TEXT_LINE_MARGIN * (line_bottom - line_top))
        lines.append(
            (
                max(0, line_left - margin) / width,
                max(0, line_top - margin) / height,
                min(width, line_right + margin) / width,
                min(bottom, line_bottom + margin) / height,
            )
        )
    return lines
