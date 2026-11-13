"""
Tests for instruments.py: does the part-name parser understand real-world
headers, including OCR typos?

Run with:  uv run pytest            (all tests)
           uv run pytest -k horn    (only tests whose name contains "horn")

pytest finds every function whose name starts with "test_" and runs it. A
test fails when an `assert` is false. That's the whole idea, much simpler than
PHPUnit classes.
"""

import pytest

from scoresplit.instruments import (
    _build_dictionary,
    _parse_instrument,
    clef_key_note,
    contains_score_keyword,
    default_key,
    find_part_name,
    find_two_line_name,
    instrument_clef,
    is_unlisted_key,
    key_from_clef,
    normalise_text,
    parse_key,
    parse_number,
    unusual_key_note,
)


# @pytest.mark.parametrize runs the same test once for every (input, expected)
# pair in the list. Each pair shows up as a separate test in the output.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Sibelius-style headers (embedded text)
        ("Clarinet in Bb 2", "Clarinet 2 in Bb"),
        ("Horn in Eb 4", "Horn 4 in Eb"),
        ("Bass Clarinet in Bb", "Bass Clarinet in Bb"),
        ("Baritone Saxophone", "Baritone Saxophone"),
        ("Baritone", "Baritone"),  # the brass instrument, not the saxophone
        ("Flute 1&2", "Flute 1&2"),
        ("Horn in Eb 1&2", "Horn 1&2 in Eb"),
        # OCR typos seen in the scanned example
        ("Hom 2in F", "Horn 2 in F"),
        ("Horn3inF", "Horn 3 in F"),
        ("Homn 1inF Dixie style", "Horn 1 in F"),
        ("Buphonium 1 in Bb", "Euphonium 1 in Bb"),
        ("Bass Tubain C", "Tuba in C"),  # Bass Tuba is the tuba
        ("Trumpet 1 in By", "Trumpet 1 in Bb"),
        ("Clarinet in B> opt tacet play", "Clarinet in Bb"),
        ("le tacet Bassoon 1", "Bassoon 1"),
        ("Flute 1 86", "Flute 1"),  # 86 is a bar number, not a part number
        # Croatian / key-before-name styles
        ("2. truba in B", "Trumpet 2 in Bb"),
        ("Bb Trumpet 3", "Trumpet 3 in Bb"),
        ("1. klarinet in B", "Clarinet 1 in Bb"),
        ("2. klarinet in Es", "Clarinet 2 in Eb"),
        ("Rog 2 in F", "Horn 2 in F"),
        # score margin abbreviations
        ("Picc.", "Piccolo"),
        ("B. Cl.", "Bass Clarinet"),
        ("Euph.", "Euphonium"),
        # multi-instrument percussion parts keep their wording
        ("Vibraphone, Xylophone", "Vibraphone, Xylophone"),
        ("Glockenspiel &, Xylophone", "Glockenspiel, Xylophone"),
        # flat sign read as ":" / "." by OCR, "/"-joined alternatives
        ("B: Trumpet/Cornet 1", "Trumpet/Cornet 1 in Bb"),
        ("B. Trumpet/Cornet 2", "Trumpet/Cornet 2 in Bb"),
        ("Trombone 3/Bass Trombone", "Trombone 3/Bass Trombone"),
        ("Horn 1/2 in F", "Horn 1/2 in F"),
        # key in front of the name, glued OCR words
        ("F Horn 4", "Horn 4 in F"),
        ("HominF 1", "Horn 1 in F"),
        ("Hormn in F 4", "Horn 4 in F"),
        # march cards (Gardijska): ordinal + key in front of the name
        ("1st Bb Clarinet", "Clarinet 1 in Bb"),
        ("2nd Eb Alto Saxophone", "Alto Saxophone 2 in Eb"),
        ("2nd_Eb Alto Saxophone", "Alto Saxophone 2 in Eb"),
        ("1st Bb Tenor Saxophone", "Tenor Saxophone 1 in Bb"),
        ("4th Bb Trumpet", "Trumpet 4 in Bb"),
        ("Eb Clarinet", "Clarinet in Eb"),
        # raised "st"/"nd" read by OCR as a symbol
        ("1* Bugle", "Flugelhorn 1"),
        ('2" Bugle', "Flugelhorn 2"),
        ("1% Bugle", "Flugelhorn 1"),
        # Italian percussion names
        ("Cassa", "Bass Drum"),
        ("Gran Cassa", "Bass Drum"),
        ("Cassa chiara", "Snare Drum"),
        # Latin timbales (Deep Purple p89-90, handwritten), not the timpani
        ("TIMBALES", "Timbales"),
        ("TIMB.", "Timbales"),
        ("Timpani", "Timpani"),
        # Funk Attack (OCR text layer): "1" read as "l", "1, 2" = one part
        # for two musicians, "/" read as "I", a doubling in brackets
        ("mTRUMPETl", "Trumpet 1"),
        ("FHORN 1,2", "Horn 1&2"),
        ("m ALTO SAXOPHONE 1, 2", "Alto Saxophone 1&2"),
        ("Trumpet 1, 2, 3", "Trumpet 1&2&3"),
        ("BASS I ELECTRIC BASS", "Bass/Electric Bass"),
        ("FLUTE 1, 2 (PICCOLO)", "Flute 1&2"),
        ("TIMPANI", "Timpani"),  # a capital I at the end is NOT a "1"
        ("SHAKER, TAMBOURINE", "Shaker, Tambourine"),
        # Ross Roy with eng+hrv OCR: flat read as "+", noise in front of a name
        ("E+ Clarinet", "Clarinet in Eb"),
        ("}V( Ccolo", "Piccolo"),  # a lone "V" is not the Roman numeral 5
        ("II Horn", "Horn 2"),
        # Sedona (US publisher): number in front without a dot, "N/M" for two
        # musicians, key after the name without "in"
        ("1 Clarinet in Bb", "Clarinet 1 in Bb"),
        ("2/3 Clarinet in Bb", "Clarinet 2/3 in Bb"),
        ("1/2 Flute", "Flute 1/2"),
        ("1/2 Horn in F", "Horn 1/2 in F"),
        ("2/3 Trombone", "Trombone 2/3"),
        ("Tuba Bb", "Tuba in Bb"),
        ("Allegro 1/2 Flute", "Flute 1/2"),  # "N/M" is never a page number
        ("Allegro 2 Flute", "Flute"),  # a bare digit only counts at the start
        # ...and only on a clean line: Deep Purple p64, OCR of "F Horn" + a
        # handwritten "1º-3º" is noise, not Horn 2
        ("2 Homf{— 39", "Horn"),
    ],
)
def test_find_part_name(text: str, expected: str) -> None:
    part = find_part_name(text)
    assert part is not None, f"no part found in {text!r}"
    assert part.display() == expected


@pytest.mark.parametrize(
    "text",
    [
        "Lentamente molto rubato",
        "NUOVO CINEMA PARADISO",
        "arr. Peter Kleine Schaars",
        "Words & Music by",
        "A Medley Including:",
        "Allegro agitato",
        "cresc.",
        "rit.",
        "Dixie style",
        "Yellow",  # matches no alias, not even fuzzily
    ],
)
def test_non_instrument_text_is_ignored(text: str) -> None:
    assert find_part_name(text) is None


def test_a_look_alike_word_is_a_weak_match() -> None:
    # Fuzzy matching (FUZZY_MIN_SCORE = 80) accepts a word that merely looks
    # like an instrument: "hello" is Cello, but with a low score, so the part
    # gets a low confidence and a person has a look.
    part = find_part_name("hello")
    assert part.instrument_id == "cello"
    assert part.score == 80


def test_normalise_text_fixes_glued_words() -> None:
    assert normalise_text("Horn3inF") == "Horn 3 in F"
    assert normalise_text("2nd Trumpet") == "2. Trumpet"
    assert normalise_text("Violin E") == "Violin E"  # "violin" must not be split


def test_parse_key_and_number() -> None:
    assert parse_key("B♭") == "Bb"
    assert parse_key("Bb") == "Bb"
    assert parse_key("Flute") is None
    assert parse_key("Es") == "Eb"
    assert parse_number("II") == "2"
    assert parse_number("Solo&1") == "Solo&1"
    assert parse_number("89") is None
    # Ranges of part numbers ("Eb Alto Saxophone 1-2", "F Horn 2-4")
    assert parse_number("1-2") == "1-2"
    assert parse_number("2-4") == "2-4"
    assert parse_number("4-2") is None  # not a range
    assert parse_number("12-16") is None  # bar numbers
    assert find_part_name("F Horn 2-4").display() == "Horn 2-4 in F"


@pytest.mark.parametrize(
    ("text", "clef"),
    [
        ("Baritone (T.C.)", "treble"),
        ("Baritone (B.C.)", "bass"),
        ("Euphonium TC", "treble"),
        ("Trombone 1 in Bb (Bass Clef)", "bass"),  # scoresplit's own output names
        ("Baritone (Treble Clef)", "treble"),
        ("Baritone", None),
    ],
)
def test_clef_written_in_the_name(text: str, clef: str | None) -> None:
    part = find_part_name(text)
    assert part.clef == clef
    # The clef is not part of the plain name; segment.py adds it when needed.
    assert "Clef" not in part.display() and "T.C" not in part.display()


def test_lowercase_f_in_front_is_a_dynamic_not_a_key() -> None:
    assert find_part_name("f Horn").display() == "Horn"


def test_abbreviations_are_flagged() -> None:
    # Header detection ignores abbreviation matches (OCR noise like "VL.").
    assert find_part_name("VL.").abbreviation is True
    assert find_part_name("Violin").abbreviation is False


@pytest.mark.parametrize(
    ("text", "instrument_id"),
    [
        # Printed as full part names in headers (gershwin: "Eb Alto Sax. 1,
        # 2"), so not abbreviations, or the header would skip them.
        ("Eb Alto Sax. 1, 2", "alto_saxophone"),
        ("Eb Alto Cl.", "alto_clarinet"),
        ("Bb Bass Cl.", "bass_clarinet"),
        ("Bass Cl", "bass_clarinet"),
        ("Bari. Sax.", "baritone_saxophone"),
        ("Tub. Bells", "chimes"),
        ("Bass Gtr.", "bass_guitar"),
        ("El. Bass", "bass_guitar"),
    ],
)
def test_short_written_full_names_are_not_abbreviations(text: str, instrument_id: str) -> None:
    part = find_part_name(text)
    assert (part.instrument_id, part.abbreviation) == (instrument_id, False)


@pytest.mark.parametrize(
    ("text", "instrument_id", "key"),
    [
        ("Cowbell", "cowbell", None),
        ("COWBELL", "cowbell", None),
        ("Kuhglocke", "cowbell", None),
        ("Koebel", "cowbell", None),
        ("1st Bb FLUGEL HORN", "flugelhorn", "Bb"),
        ("Flügel Horn", "flugelhorn", "Bb"),
        ("Strijkbas", "double_bass", None),
        ("Contrebasse à cordes", "double_bass", None),
        # The alto horn is its own instrument, usually in Eb.
        ("Alto Horn", "alto_horn", "Eb"),
        ("Althorn in F", "alto_horn", "F"),
        ("Althoorn", "alto_horn", "Eb"),
        ("Flicorno contralto", "alto_horn", "Eb"),
        ("Tenor Horn", "tenor_horn", "Bb"),
    ],
)
def test_names_added_for_the_new_examples(text: str, instrument_id: str, key: str | None) -> None:
    # key: the printed key, else the instrument's default key.
    part = find_part_name(text)
    assert part.instrument_id == instrument_id
    assert (part.key or default_key(instrument_id)) == key


@pytest.mark.parametrize(
    "text",
    [
        "I. Aruba",  # the suite's movement title, not "truba" (trumpet)
        "true",
        "tube",
        "McDowell",  # not "cowbell"
        "Cromwell",
    ],
)
def test_look_alikes_of_short_names_are_not_instruments(text: str) -> None:
    assert find_part_name(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Conductor", True),
        ("Conductor Score", True),
        ("Directie C", True),  # Dutch (Antilliaanse Suite)
        ("Partition du conducteur", True),  # French
        ("Full Score", True),
        ("Concert Band - Full Score", True),  # "score" is never a person
        # Sedona's dedication on every part's first page: a person's job title
        ("Paul Sharle - Founder/Conductor", False),
        ("John Smith, conductor", False),
        ("Music Director and Conductor", False),
    ],
)
def test_score_keyword_but_not_a_persons_role(text: str, expected: bool) -> None:
    assert contains_score_keyword(text) is expected


# ---------------------------------------------------------------------------
# Keys (see `keys:` in data/instruments.yaml)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("instrument_id", "key"),
    [
        ("flute", "Eb"),  # a key the flute isn't listed in
        ("violin", "Bb"),  # any key on an instrument without keys
        ("bass", "A"),  # a band bass is written in C, Bb, Eb or F
    ],
)
def test_an_unusual_key_gets_a_note(instrument_id: str, key: str) -> None:
    assert "unusual key" in unusual_key_note(instrument_id, key)


def test_a_listed_key_or_no_key_needs_no_note() -> None:
    assert unusual_key_note("clarinet", "Eb") is None
    assert unusual_key_note("tuba", "F") is None
    # A band bass is written in the same keys as the tuba.
    for key in ("C", "Bb", "Eb", "F"):
        assert unusual_key_note("bass", key) is None
    assert unusual_key_note("clarinet", None) is None
    assert unusual_key_note("violin", None) is None
    assert unusual_key_note(None, "Bb") is None


def test_a_typo_in_the_keys_list_stops_the_program() -> None:
    # pytest.raises checks that the code inside the "with" block raises
    # this error; match= is a regular expression the message must contain.
    with pytest.raises(ValueError, match="'x'"):
        _parse_instrument({"id": "x", "name": "X", "keys": ["Hb"]})


def test_the_same_id_twice_stops_the_program() -> None:
    entry = {"id": "x", "name": "X"}
    with pytest.raises(ValueError, match="used twice"):
        _build_dictionary({"instruments": [entry, entry]})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Piccolo in Db", "Piccolo in Db"),
        ("Db Piccolo", "Piccolo in Db"),
        ("Piccolo Db", "Piccolo in Db"),
        ("Piccolo in Des", "Piccolo in Db"),
    ],
)
def test_d_flat_is_read_as_a_key(text: str, expected: str) -> None:
    assert find_part_name(text).display() == expected


@pytest.mark.parametrize(
    ("text", "instrument_id", "display"),
    [
        # "Tenor tuba" is another name for the euphonium (Deep Purple).
        ("Tenor Tuba", "euphonium", "Euphonium"),
        # Two names of the same instrument: the key after the second one counts.
        ("Tenor Tuba/Euphonium in Bb", "euphonium", "Euphonium in Bb"),
        # "Bass" stays the bass, with or without a key: the instrument is
        # what the PDF says (Deep Purple, Gardijska print "Bass in Bb").
        ("Bass in Bb", "bass", "Bass in Bb"),
        ("Bass 4 in Eb", "bass", "Bass 4 in Eb"),
        ("Bb Bass", "bass", "Bass in Bb"),
        ("Bass in F", "bass", "Bass in F"),
        ("Bass", "bass", "Bass"),
        # "Bass Tuba" is the tuba.
        ("Bass Tuba in Eb", "tuba", "Tuba in Eb"),
        ("Bass Tuba", "tuba", "Tuba"),
        # The bass VOICE only when the name says so.
        ("Bass Voice", "bass_voice", "Bass"),
        # Two DIFFERENT instruments still keep the publisher's wording.
        ("Trumpet/Cornet 1", "trumpet", "Trumpet/Cornet 1"),
        # A generic first name and a specific second one: the second says
        # which instrument it is (Funk Attack). The label keeps both names.
        ("Bass/Electric Bass", "bass_guitar", "Bass/Electric Bass"),
        ("Saxophone/Alto Saxophone 1", "alto_saxophone", "Saxophone/Alto Saxophone 1"),
    ],
)
def test_names_that_mean_another_instrument(text: str, instrument_id: str, display: str) -> None:
    part = find_part_name(text)
    assert (part.instrument_id, part.display()) == (instrument_id, display)


# ---------------------------------------------------------------------------
# Generic names ("Saxophone", "Bass", ...)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "instrument_id"),
    [
        ("Saxophone 1", "saxophone"),
        ("Bass", "bass"),
        ("Voice", "voice"),
        ("Percussion 2", "percussion"),
        ("Mallets 2", "mallets"),
    ],
)
def test_a_generic_name_is_read_as_the_generic_instrument(text: str, instrument_id: str) -> None:
    part = find_part_name(text)
    assert part.instrument_id == instrument_id


@pytest.mark.parametrize(
    ("text", "instrument_id"),
    [
        ("Alto Saxophone 1", "alto_saxophone"),
        # Longer names that start with "Bass" keep their own instrument.
        ("Bass Clarinet", "bass_clarinet"),
        ("Bass Trombone", "bass_trombone"),
        ("Bass Drum", "bass_drum"),
        ("Bass Guitar", "bass_guitar"),
        ("Electric Bass", "bass_guitar"),
        ("Double Bass", "double_bass"),
        # A generic first name and a specific second one (the slash rule).
        ("Mallets/Xylophone", "xylophone"),
    ],
)
def test_specific_names_keep_their_own_instrument(text: str, instrument_id: str) -> None:
    assert find_part_name(text).instrument_id == instrument_id


def test_unlisted_key_and_default_key() -> None:
    assert is_unlisted_key("bass_clarinet", "Eb")  # a bass clarinet has no Eb
    assert not is_unlisted_key("bass_clarinet", "Bb")
    assert not is_unlisted_key("violin", "Bb")  # no `keys` list: nothing to compare
    assert not is_unlisted_key("clarinet", None)
    assert default_key("bass_clarinet") == "Bb"
    assert default_key("violin") is None
    # Generic: a plain "Bass" may be a bass guitar, so no key is guessed.
    assert default_key("bass") is None


def test_the_clef_gives_the_key_of_a_euphonium_or_baritone() -> None:
    # Treble clef euphonium/baritone/trombone parts are written in Bb, bass
    # clef ones at concert pitch (C).
    assert key_from_clef("euphonium", "treble") == "Bb"
    assert key_from_clef("euphonium", "bass") == "C"
    assert key_from_clef("baritone", "treble") == "Bb"
    assert key_from_clef("trombone", "treble") == "Bb"
    assert key_from_clef("euphonium", None) is None  # clef unknown
    # No `treble_clef_key`: the clef says nothing about the key.
    assert key_from_clef("flute", "treble") is None
    assert key_from_clef("bass_trombone", "bass") is None


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("Bb BARITONE, Bb EUPHONIUM", "Bb"),  # vangelis.pdf: both say Bb
        ("Bb Baritone, Euphonium", None),  # only one says it: don't guess
        ("Bb Baritone, C Euphonium", None),  # they disagree
        ("Shaker, Tambourine, Wind Chimes", None),  # no key at all
    ],
)
def test_a_combined_part_keeps_a_key_all_its_instruments_share(text: str, key: str | None) -> None:
    part = find_part_name(text)
    assert part.label is not None  # still the combined part
    assert part.key == key


def test_a_key_from_the_clef_gets_a_note_only_when_it_is_not_the_default() -> None:
    # "Baritone (T.C.)": treble clef, no key printed -> Bb, which isn't the
    # default (C), so a person sees why.
    assert clef_key_note("baritone", None, "treble") == "key not printed: Bb, from the treble clef"
    assert clef_key_note("baritone", None, "bass") is None  # C is the default anyway
    assert clef_key_note("euphonium", "Eb", "treble") is None  # a printed key wins
    assert clef_key_note("flute", None, "treble") is None  # the clef says nothing


def test_a_treble_clef_key_must_be_one_of_the_keys() -> None:
    with pytest.raises(ValueError, match="treble_clef_key"):
        _parse_instrument({"id": "x", "name": "X", "keys": ["C"], "treble_clef_key": "Bb"})
    # C must be there too: it is the key of the bass clef version.
    with pytest.raises(ValueError, match="treble_clef_key"):
        _parse_instrument({"id": "x", "name": "X", "keys": ["Bb"], "treble_clef_key": "Bb"})
    ok = _parse_instrument({"id": "x", "name": "X", "keys": ["C", "Bb"], "treble_clef_key": "Bb"})
    assert ok.treble_clef_key == "Bb"


def test_an_instrument_written_in_one_clef_has_that_clef() -> None:
    # The clef a part gets when none was printed or detected (round 9.5).
    assert instrument_clef("trumpet") == "treble"
    assert instrument_clef("clarinet") == "treble"
    assert instrument_clef("alto_horn") == "treble"
    assert instrument_clef("bassoon") == "bass"
    assert instrument_clef("timpani") == "bass"
    # Written in either clef: an "Eb Bass" (italiano) is in treble clef, a
    # trombone or euphonium part comes in both. No guess.
    for instrument_id in ("trombone", "euphonium", "baritone", "tuba", "bass"):
        assert instrument_clef(instrument_id) is None
    # No single pitched clef: keyboards, unpitched percussion.
    assert instrument_clef("piano") is None
    assert instrument_clef("snare_drum") is None
    assert instrument_clef(None) is None
    assert instrument_clef("no_such_instrument") is None


def test_a_clef_in_the_yaml_must_be_treble_or_bass_and_not_with_a_treble_clef_key() -> None:
    with pytest.raises(ValueError, match="unknown clef 'alto'"):
        _parse_instrument({"id": "x", "name": "X", "clef": "alto"})
    with pytest.raises(ValueError, match="both a clef and a treble_clef_key"):
        _parse_instrument(
            {"id": "x", "name": "X", "keys": ["C", "Bb"], "treble_clef_key": "Bb", "clef": "treble"}
        )
    assert _parse_instrument({"id": "x", "name": "X", "clef": "bass"}).clef == "bass"
    assert _parse_instrument({"id": "x", "name": "X"}).clef is None


@pytest.mark.parametrize(
    "text, instrument_id, display",
    [
        ("Soprano Voice", "soprano_voice", "Soprano"),
        ("Alto Vocals", "alto_voice", "Alto"),
        ("Tenor Voice 2", "tenor_voice", "Tenor 2"),
        ("Bass Voice", "bass_voice", "Bass"),
        ("Contralto", "alto_voice", "Alto"),  # only ever a voice
    ],
)
def test_a_voice_only_when_the_name_says_so(text: str, instrument_id: str, display: str) -> None:
    # The ids end in "_voice" so "alto_voice" can't be mixed up with
    # "alto_saxophone"; the display name stays short.
    part = find_part_name(text)
    assert part.instrument_id == instrument_id
    assert part.display() == display


@pytest.mark.parametrize(
    "text, instrument_id, display",
    [
        # In band parts a plain range word is an instrument, not a singer.
        ("Tenor", "tenor_horn", "Tenor Horn"),
        ("2. tenor", "tenor_horn", "Tenor Horn 2"),
        ("Tenor Horn in Eb", "tenor_horn", "Tenor Horn in Eb"),
        ("Tenorhorn in C", "tenor_horn", "Tenor Horn in C"),
        ("Alto 1", "alto_saxophone", "Alto Saxophone 1"),
        ("1. alt", "alto_saxophone", "Alto Saxophone 1"),
        ("Soprano", "soprano_saxophone", "Soprano Saxophone"),
        # Longer names still win over the plain range word.
        ("Tenor Saxophone", "tenor_saxophone", "Tenor Saxophone"),
        ("Tenor Tuba", "euphonium", "Euphonium"),
        ("Tenor Trombone", "trombone", "Trombone"),
        ("Tenor Drum", "tenor_drum", "Tenor Drum"),
        ("Alto Flute", "alto_flute", "Alto Flute"),
        ("Alto Clarinet", "alto_clarinet", "Alto Clarinet"),
        ("Soprano Cornet in Eb", "cornet", "Cornet in Eb"),
    ],
)
def test_plain_range_words_are_band_instruments(text: str, instrument_id: str, display: str) -> None:
    part = find_part_name(text)
    assert part.instrument_id == instrument_id
    assert part.display() == display


def test_generic_flag_is_read_from_the_yaml() -> None:
    assert _parse_instrument({"id": "x", "name": "X", "generic": True}).generic is True
    assert _parse_instrument({"id": "x", "name": "X"}).generic is False


# --- The printed name (PartName.wording) --------------------------------------


@pytest.mark.parametrize(
    ("text", "wording"),
    [
        # As printed: word order, language and numbering are kept.
        ("Clarinet in Bb 1", "Clarinet in Bb 1"),
        ("2. klarinet in Es", "2. klarinet in Es"),
        ("1/2 Flute", "1/2 Flute"),
        ("Tuba Bb", "Tuba Bb"),
        ("Cassa", "Cassa"),
        ("String Bass", "String Bass"),
        ("Bb Tenor Tuba/Euphonium", "Bb Tenor Tuba/Euphonium"),
        # Title, credits and remarks around the name are trimmed off.
        ("Eb Alto Saxophone 1-2 DEEP PURPLE MEDLEY arr. T. Sahashi", "Eb Alto Saxophone 1-2"),
        ("English Horn (optional)", "English Horn"),
        ("Percussion 2: Drums", "Percussion 2"),
        ("BARITONE in C (Wind Band)", "Baritone in C"),
        ("FLUTE 1, 2 (PICCOLO)", "Flute 1&2"),
        # The clef has its own column: left out of the name.
        ("Baritone (T.C.)", "Baritone"),
        ("Euphonium (B.C.)", "Euphonium"),
        # OCR noise fixed: a misread flat sign, "rn" read as "m", glued words.
        ("B: Trumpet/Cornet 1", "Bb Trumpet/Cornet 1"),
        ("B, TENOR SAXOPHONE 1, 2", "Bb Tenor Saxophone 1&2"),
        ("E> Alto Saxophone 1 (ha)", "Eb Alto Saxophone 1"),
        ("Clarinet in B»", "Clarinet in Bb"),
        ("Hom 2in F", "Horn 2 in F"),
        ("Bass Tubain C", "Bass Tuba in C"),
        ("BASS I ELECTRIC BASS", "Bass/Electric Bass"),
        ("piccolo", "Piccolo"),
        # Capitals calmed; Roman numerals and plurals kept.
        ("TUBA in Eb &", "Tuba in Eb"),
        ("PERCUSSION II", "Percussion II"),
        ("KEYBOARDS", "Keyboards"),
        ("FLUTES & OBOES", "Flutes & Oboes"),
        # A list of instruments is its own printed name.
        ("Shaker, Susp. Cymbal,Wind Chimes", "Shaker, Susp. Cymbal, Wind Chimes"),
    ],
)
def test_printed_name(text: str, wording: str) -> None:
    part = find_part_name(text)
    assert part is not None
    assert part.wording == wording


@pytest.mark.parametrize("text", ["mTRUMPETl", "PERCOSSJON 1", "FHORN 1,2"])
def test_a_garbled_name_has_no_printed_name(text: str) -> None:
    # Recognised by a fuzzy match, but the words are OCR noise: not shown.
    part = find_part_name(text)
    assert part is not None
    assert part.wording is None


# ---------------------------------------------------------------------------
# Round 9.3: reading the header of scanned parts (mission, suite, italiano,
# gershwin). The texts are what OCR really read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "number", "wording"),
    [
        # A Roman numeral after the name, its strokes misread (mission).
        ("Bb Clarinet |", "1", "Bb Clarinet I"),
        ("Eb Horn l", "1", "Eb Horn I"),
        ("Eb Alto Saxophone Il", "2", "Eb Alto Saxophone II"),
        ("Eb Horn ll", "2", "Eb Horn II"),
        ("Horn 11", "2", "Horn II"),
        ("Bb Clarinet Ill", "3", "Bb Clarinet III"),
        ("Eb Horn lIl", "3", "Eb Horn III"),
        ("Eb Horn ITI", "3", "Eb Horn III"),
        ("Bb Trumpet/Cornet Il", "2", "Bb Trumpet/Cornet II"),
        ("FAGOT I+11", "1&2", "Fagot I+II"),
        # The Dutch ordinal "1e" in front, its "1" misread (suite).
        ("le FLUIT C", "1", "1. Fluit C"),
        ("Ie KLARINET Bb", "1", "1. Klarinet Bb"),
        ("{e TROMPET Bb", "1", "1. Trompet Bb"),
        ("fe TROMBONE C", "1", "1. Trombone C"),
        ("|e BUGEL Bb", "1", "1. Bugel Bb"),
        # "Sax. 1, 2" is one part for two players, like "Sax 1, 2" (gershwin).
        ("Eb Alto Sax. 1, 2", "1&2", "Eb Alto Sax. 1&2"),
        # The English "1st" in front, its "1" misread (american).
        ("Ist Bb Trumpet", "1", "1. Bb Trumpet"),
        ("lst Bb Clarinet", "1", "1. Bb Clarinet"),
        # Two players in front: both numbers, not only the second (american).
        ("Ist & 2nd Flute", "1&2", "1. & 2. Flute"),
        ("Ist/2nd Bb Tenor Sax", "1&2", "1./2. Bb Tenor Sax"),
        ("3rd & 4th F Horn", "3&4", "3. & 4. F Horn"),
    ],
)
def test_numbers_misread_by_ocr(text: str, number: str, wording: str) -> None:
    part = find_part_name(text)
    assert part is not None
    assert (part.number, part.wording) == (number, wording)


@pytest.mark.parametrize(
    ("text", "number"),
    [
        # The French "le" before a word in small letters is not "1e".
        ("le tacet Bassoon", None),
        # A real number at the end stays a number.
        ("Clarinet in Bb 1", "1"),
        # Plain digits that are not "11" are left alone: "1 1" is still 1.
        ("Flute 1 1", "1"),
        # A stroke that is not at the end is noise, not a numeral.
        ("Bb | Clarinet", None),
        # A pair in front must go up ("1. & 2."): "2. & 1." keeps only the 1.
        ("2. & 1. Flute", "1"),
    ],
)
def test_strokes_elsewhere_are_not_numbers(text: str, number: str | None) -> None:
    part = find_part_name(text)
    assert part is not None
    assert part.number == number


def test_normalise_text_reads_misread_numbers() -> None:
    assert normalise_text("Eb Horn Il") == "Eb Horn II"
    assert normalise_text("le FLUIT C") == "1. FLUIT C"
    assert normalise_text("Ist & 2nd Flute") == "1. & 2. Flute"
    # Only "Ist" + a space, "/" or "&" at the start: a word stays a word.
    assert normalise_text("Isthmus") == "Isthmus"
    assert normalise_text("Eb Alto Sax. 1, 2") == "Eb Alto Sax. 1&2"


def test_a_garbled_saxophone_after_a_range_word() -> None:
    # OCR read "Eb Baritone Saxophone" as "Eb Baritone Saigphonja_r" (mission):
    # it is the baritone saxophone, not the brass baritone.
    part = find_part_name("Eb Baritone Saigphonja_r")
    assert part is not None
    assert (part.instrument_id, part.key, part.wording) == (
        "baritone_saxophone",
        "Eb",
        "Eb Baritone Saxophone",
    )


@pytest.mark.parametrize(
    ("text", "instrument_id"),
    [
        # Not similar enough to "saxophone": the baritone stays.
        ("Eb Baritone Sample", "baritone"),
        # The Dutch "ALT" + a garbled "SAXOFOON" keeps the alto saxophone that
        # "ALT" gives ("Alt Saxophone" is not a name, the generic "Saxophone"
        # would win).
        ("ALT SAXOFOONI+ 1O Eb", "alto_saxophone"),
    ],
)
def test_other_sa_words_are_not_a_saxophone(text: str, instrument_id: str) -> None:
    part = find_part_name(text)
    assert part is not None
    assert part.instrument_id == instrument_id


@pytest.mark.parametrize(
    ("text", "instrument_id", "number", "wording"),
    [
        # The number in the brackets wins over the ordinal in front (suite).
        ("1e COR Eb (HORN III)", "horn", "3", "1. Cor Eb (Horn III)"),
        ("2e COR Eb (HORN IV)", "horn", "4", "2. Cor Eb (Horn IV)"),
        # "Alt" with a horn in brackets is the alto horn, not the saxophone.
        ("1e ALT Eb (HORN I)", "alto_horn", "1", "1. Alt Eb (Horn I)"),
        # ...also when OCR lost the closing bracket and the numeral.
        ("2e ALT Eb (HORN", "alto_horn", "2", "2. Alt Eb (Horn)"),
    ],
)
def test_a_name_in_brackets(text: str, instrument_id: str, number: str, wording: str) -> None:
    part = find_part_name(text)
    assert part is not None
    assert (part.instrument_id, part.number, part.key) == (instrument_id, number, "Eb")
    assert part.wording == wording
    # A full name in brackets makes "COR" a real part name, not an abbreviation.
    assert not part.abbreviation


def test_a_doubling_in_brackets_is_still_left_out() -> None:
    # A DIFFERENT instrument in brackets is one the player doubles on.
    part = find_part_name("FLUTE 1, 2 (PICCOLO 2)")
    assert part is not None
    assert (part.instrument_id, part.number) == ("flute", "1&2")


def test_a_number_in_front_of_a_slash_name_is_in_its_label() -> None:
    # The label is the canonical name, so the 1st and 2nd Trumpet/Cornet
    # must differ (italiano: otherwise all four were compared as copies).
    first = find_part_name("1st Bb TRUMPET / CORNET")
    second = find_part_name("2nd Bb TRUMPET / CORNET")
    assert first is not None and second is not None
    assert first.display() == "Trumpet/Cornet 1 in Bb"
    assert second.display() == "Trumpet/Cornet 2 in Bb"
    assert first.wording == "1. Bb Trumpet/Cornet"


@pytest.mark.parametrize(
    ("first_line", "second_line", "instrument_id", "number", "key", "wording"),
    [
        ("Bb BASS", "CLARINET", "bass_clarinet", None, "Bb", "Bb Bass Clarinet"),
        ("1st Bb TENOR", "SAXOPHONE", "tenor_saxophone", "1", "Bb", "1. Bb Tenor Saxophone"),
        ("Eb BARITONE", "SAXOPHONE", "baritone_saxophone", None, "Eb", "Eb Baritone Saxophone"),
        ("1st Eb ALTO", "SAXOPHONE", "alto_saxophone", "1", "Eb", "1. Eb Alto Saxophone"),
        ("Eb TRUMPET /", "CORNET", "trumpet", None, "Eb", "Eb Trumpet/Cornet"),
    ],
)
def test_a_name_over_two_lines(
    first_line: str,
    second_line: str,
    instrument_id: str,
    number: str | None,
    key: str,
    wording: str,
) -> None:
    part = find_two_line_name(first_line, second_line)
    assert part is not None
    assert (part.instrument_id, part.number, part.key, part.wording) == (
        instrument_id,
        number,
        key,
        wording,
    )


@pytest.mark.parametrize(
    ("first_line", "second_line"),
    [
        # Two names, one per line: a list, not one name.
        ("Mallets", "Xylophone"),
        ("Percussion 2", "Timpani"),
        # The "I" is the number 1, not a slash misread (suite).
        ("Percussion I", "KLEINE TROM"),
    ],
)
def test_two_names_on_two_lines_are_not_joined(first_line: str, second_line: str) -> None:
    assert find_two_line_name(first_line, second_line) is None


@pytest.mark.parametrize(
    ("text", "number", "key"),
    [
        # "C" in front is the key C as the first word or after a number with
        # a dot (american: "1st C Trombone", "C Baritone/Euphonium").
        ("1st C Trombone", "1", "C"),
        ("C Baritone/Euphonium", None, "C"),
        # Anywhere else a lone "C" is more often a rehearsal letter.
        ("Rehearsal C Trumpet", None, None),
    ],
)
def test_a_c_in_front_is_a_key_only_at_the_start_or_after_a_number(
    text: str, number: str | None, key: str | None
) -> None:
    part = find_part_name(text)
    assert part is not None
    assert (part.number, part.key) == (number, key)


@pytest.mark.parametrize(
    ("text", "instrument_id", "number", "wording"),
    [
        # OCR read the "e" as "c" (cornfield, american). In a short word the
        # fuzzy match misses it ("oboc" is 75% like "oboe"), so each "c" is
        # also read as "e", which must then BE a name.
        ("Oboc |", "oboe", "1", "Oboe I"),
        ("Bb Baritonc", "baritone", None, "Bb Baritone"),
        ("2nd & C Trombonc", "trombone", None, "Trombone"),
        ("Clarinct 2", "clarinet", "2", "Clarinet 2"),
        # "Flügel" alone is the flugelhorn; OCR read the ü as "a", "i" or
        # "ii" (UMLAUT_MISREADINGS). Shown as the real name.
        ("Flagel 1", "flugelhorn", "1", "Flügel 1"),
        ("Fligel Il", "flugelhorn", "2", "Flügel II"),
        ("FLIIGEL", "flugelhorn", None, "Flügel"),
    ],
)
def test_names_with_a_letter_misread_by_ocr(
    text: str, instrument_id: str, number: str | None, wording: str | None
) -> None:
    part = find_part_name(text)
    assert part is not None
    assert (part.instrument_id, part.number, part.wording) == (instrument_id, number, wording)


@pytest.mark.parametrize("text", ["McDowell", "Cromwell", "Taco", "Cocoa", "Acre", "Fuel", "luge"])
def test_a_misread_letter_must_give_a_name_exactly(text: str) -> None:
    # Only an EXACT name after the "c" -> "e" or "ü" fix counts: no fuzzy
    # match on top of a guessed letter (see the cowbell learning:
    # McDowell). "flugel" alone is exact-only: it is 80% like "fuel".
    assert find_part_name(text) is None


@pytest.mark.parametrize(
    ("text", "number", "wording"),
    [
        # Two Roman numerals with a dash: one part for two players.
        ("Horn III - IV", "3&4", "Horn III&IV"),
        ("Horn 111 - 1V", "3&4", "Horn III&IV"),  # strokes read as digits
        # The first must be the smaller one (else only it is read); plain
        # digits are left alone.
        ("Horn IV - III", "4", "Horn IV"),
        ("Flute 1 - 2", "1", "Flute 1"),
    ],
)
def test_two_roman_numerals_with_a_dash(text: str, number: str | None, wording: str) -> None:
    part = find_part_name(text)
    assert part is not None
    assert (part.number, part.wording) == (number, wording)


def test_a_key_in_front_joined_by_a_dash() -> None:
    # cornfield page 45: "C - Trombone II".
    part = find_part_name("C - Trombone 11")
    assert (part.display(), part.wording) == ("Trombone 2 in C", "C Trombone II")


@pytest.mark.parametrize(
    ("text", "display", "wording"),
    [
        # The solo player and the 1st together (superstar).
        ("SOLO + 1e KLARINET Bb", "Clarinet Solo&1 in Bb", "Solo + 1. Klarinet Bb"),
        # The bracket's "1" is part of "Solo&1": the number outside says more.
        ("SOLO +1e BUGEL Bb (FLUGELHORN I)", "Flugelhorn Solo&1 in Bb", None),
        # A capital C as the LAST word after the name is the key C.
        ("FLUIT C", "Flute in C", "Fluit C"),
        ("2e TROMBONE C (POSAUNE II)", "Trombone 2 in C", "2. Trombone C (Posaune II)"),
        # The same instrument in brackets, in another language, is kept.
        ("FAGOT C (BASSON)", "Bassoon in C", "Fagot C (Basson)"),
        # The small Eb flugelhorn of Dutch fanfares.
        ("Eb BUGEL", "Flugelhorn in Eb", "Eb Bugel"),
    ],
)
def test_dutch_names_and_keys(text: str, display: str, wording: str | None) -> None:
    part = find_part_name(text)
    assert part.display() == display
    if wording is not None:
        assert part.wording == wording
    assert not is_unlisted_key(part.instrument_id, part.key)


def test_a_lone_c_elsewhere_after_the_name_is_no_key() -> None:
    # Not the last word: more often a rehearsal letter.
    assert find_part_name("Trumpet C 2").key is None


def test_two_names_joined_by_a_dash_the_first_wins() -> None:
    # The user's rule: when a part prints two names, the first one is the
    # instrument. The dash stays in the printed name.
    part = find_part_name("BARITON - EUPHONIUM C")
    assert (part.instrument_id, part.key, part.wording) == (
        "baritone",
        "C",
        "Bariton - Euphonium C",
    )


def test_a_bare_number_before_percussion_counts_the_players() -> None:
    # "2 SLAGWERK" is plain Percussion (the user's decision); a bare number
    # in front of another name still numbers the part (Sedona).
    assert find_part_name("2 SLAGWERK").number is None
    assert find_part_name("1 Clarinet in Bb").number == "1"


@pytest.mark.parametrize(
    ("text", "instrument_id", "key"),
    [
        # Tullamore p111/113: "E♭ Bass" with the flat read as a dot. "e. bass"
        # is also a Bass Guitar abbreviation, but a Bass can be in Eb.
        ("E. Bass", "bass", "Eb"),
        ("E: Bass", "bass", "Eb"),
        ("E. Bass 2", "bass", "Eb"),
        # One word: no room for a flat sign, so the abbreviation stays.
        ("E.Bass", "bass_guitar", None),
        # A fully abbreviated name stays what it abbreviates: English horn,
        # electric guitar (never read as "Eb Horn" / "Eb Guitar").
        ("E. Hn.", "english_horn", None),
        ("E.H.", "english_horn", None),
        ("E. Gtr.", "electric_guitar", None),
        # "e. horn" is no abbreviation: an Eb horn (Tullamore p95-98).
        ("E. Horn", "horn", "Eb"),
    ],
)
def test_e_with_a_dot_is_e_flat_only_before_a_full_name(
    text: str, instrument_id: str, key: str | None
) -> None:
    part = find_part_name(text)
    assert part is not None
    assert (part.instrument_id, part.key) == (instrument_id, key)
