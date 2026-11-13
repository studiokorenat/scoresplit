"""
Tests for part names in other languages: English, German, Italian, French,
Dutch, Spanish and Croatian/Serbian (Latin script only).

Two kinds of tests:
  * a hand-written table of real-world names per language, with the
    instrument, number and key each one must give;
  * every name from the instrument research (tests/data/instrument_research.json), which must
    read as the right instrument, except the few listed below as "not taken".

Run with:  uv run pytest tests/test_languages.py
"""

import json
import re
from pathlib import Path

import pytest

from scoresplit.instruments import _parse_instrument, find_part_name, parse_key, parse_number

# ---------------------------------------------------------------------------
# Real-world names per language: (printed name, instrument id, number, key)
# ---------------------------------------------------------------------------


# Each line is one test. `None` = no number / no key.
@pytest.mark.parametrize(
    ("text", "instrument_id", "number", "key"),
    [
        # --- German: key after "in" or glued with a hyphen, B = Bb, Es = Eb
        ("1. Klarinette in B", "clarinet", "1", "Bb"),
        ("Es-Klarinette", "clarinet", None, "Eb"),
        ("B-Trompete 2", "trumpet", "2", "Bb"),
        ("F-Horn 3", "horn", "3", "F"),
        ("2. Horn in Es", "horn", "2", "Eb"),
        ("Altsaxophon 1", "alto_saxophone", "1", None),
        ("Tenorsaxophon", "tenor_saxophone", None, None),
        ("Bassposaune", "bass_trombone", None, None),
        ("Kleine Klarinette in Es", "clarinet", None, "Eb"),
        ("Große Trommel", "bass_drum", None, None),
        ("Querflöte 2", "flute", "2", None),
        ("Röhrenglocken", "chimes", None, None),
        # --- Dutch: compound words, "1e"/"2de" ordinals, "Bes"
        ("2e KLARINET Bb", "clarinet", "2", "Bb"),
        ("Bes-klarinet 1", "clarinet", "1", "Bb"),
        ("2de Trompet in Bes", "trumpet", "2", "Bb"),
        ("Dwarsfluit", "flute", None, None),
        ("Altsaxofoon", "alto_saxophone", None, None),
        ("Tenor Saxofoon", "tenor_saxophone", None, None),
        ("Bastuba", "tuba", None, None),
        ("Klokkenspel", "glockenspiel", None, None),
        ("Grote Trom", "bass_drum", None, None),
        ("Kleine Trom", "snare_drum", None, None),
        ("Woudhoorn in F", "horn", None, "F"),
        # --- Italian: solfège keys, ordinal words, type AFTER the instrument
        ("Clarinetto in Sib", "clarinet", None, "Bb"),
        ("Clarinetto piccolo in Mib", "clarinet", None, "Eb"),
        ("Corno in Fa", "horn", None, "F"),
        ("Tromba in Do", "trumpet", None, "C"),
        ("Primo Clarinetto in Sib", "clarinet", "1", "Bb"),
        ("Corno secondo in Fa", "horn", "2", "F"),
        ("1º Flauto", "flute", "1", None),
        ("Sassofono contralto", "alto_saxophone", None, None),
        ("Flicorno soprano in Sib", "flugelhorn", None, "Bb"),
        ("Flicorno baritono", "euphonium", None, None),
        ("Trombone basso", "bass_trombone", None, None),
        ("Clarinetto in Si bemolle", "clarinet", None, "Bb"),
        # --- French: "en" for the key, "1er/2e" ordinals, accents
        ("Clarinette en Sib", "clarinet", None, "Bb"),
        ("1er Cornet en Sib", "cornet", "1", "Bb"),
        ("2e Cor en Fa", "horn", "2", "F"),
        ("Trompette en Ut", "trumpet", None, "C"),
        ("Flûte", "flute", None, None),
        ("Saxophone ténor", "tenor_saxophone", None, None),
        ("Clarinette basse", "bass_clarinet", None, None),
        ("Petite clarinette en Mi♭", "clarinet", None, "Eb"),
        ("Deuxième Trompette", "trumpet", "2", None),
        ("Hautbois", "oboe", None, None),
        ("Grosse caisse", "bass_drum", None, None),
        ("Caisse claire", "snare_drum", None, None),
        # --- Spanish: "en", "Si bemol", "1ª", type after the instrument
        ("Clarinete en Si bemol", "clarinet", None, "Bb"),
        ("Trompeta 1ª en Sib", "trumpet", "1", "Bb"),
        ("Saxofón alto", "alto_saxophone", None, None),
        ("Trompa en Fa", "horn", None, "F"),
        ("Trompa alto", "alto_horn", None, None),
        ("Fliscorno", "flugelhorn", None, None),
        ("Bombardino", "euphonium", None, None),
        ("Segundo Trombón", "trombone", "2", None),
        ("Platillos", "cymbals", None, None),
        ("Requinto", "clarinet", None, "Eb"),
        # --- Croatian / Serbian (Latin script)
        ("Es-klarinet", "clarinet", None, "Eb"),
        ("B-klarinet 2", "clarinet", "2", "Bb"),
        ("Be-klarinet", "clarinet", None, "Bb"),
        ("A-klarinet", "clarinet", None, "A"),
        ("Mali klarinet", "clarinet", None, "Eb"),
        ("Mali klarinet in B", "clarinet", None, "Bb"),  # a printed key wins
        ("Pozauna 2", "trombone", "2", None),
        ("Flighorna in B", "flugelhorn", None, "Bb"),
        ("Kotlovi", "timpani", None, None),
        ("Čelesta", "celesta", None, None),
        ("Bas-gitara", "bass_guitar", None, None),
        ("Druga truba in B", "trumpet", "2", "Bb"),
        ("Mecosopran", "mezzo_voice", None, None),
        # --- English
        ("Second Trumpet in Bb", "trumpet", "2", "Bb"),
        ("Contrabass Clarinet", "contrabass_clarinet", None, None),
        ("EEb Contra-alto Clarinet", "contra_alto_clarinet", None, None),
        ("Bass Saxophone", "bass_saxophone", None, None),
        ("Crotales", "crotales", None, None),
        ("Celesta", "celesta", None, None),
    ],
)
def test_names_in_other_languages(
    text: str, instrument_id: str, number: str | None, key: str | None
) -> None:
    part = find_part_name(text)
    assert part is not None, f"no part found in {text!r}"
    assert (part.instrument_id, part.number, part.key) == (instrument_id, number, key)


# ---------------------------------------------------------------------------
# The two code fixes from the research
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "instrument_id"),
    [
        # Spaces after the dots of an abbreviation don't matter.
        ("C. A.", "english_horn"),
        ("C.A.", "english_horn"),
        ("D. B.", "double_bass"),
        ("D.B.", "double_bass"),
        ("B. Cl.", "bass_clarinet"),
        ("B.Cl.", "bass_clarinet"),
        ("C. A. Cl.", "contra_alto_clarinet"),
    ],
)
def test_abbreviations_match_with_or_without_spaces(text: str, instrument_id: str) -> None:
    assert find_part_name(text).instrument_id == instrument_id


@pytest.mark.parametrize(
    ("text", "wording"),
    [
        # A key glued to the name with a hyphen is printed that way...
        ("Es-klarinet", "Es-klarinet"),
        ("2. B-Trompete", "2. B-Trompete"),
        # ...but "B- " with a space is a flat sign OCR misread: shown as Bb.
        ("B- Clarinet 1", "Bb Clarinet 1"),
    ],
)
def test_a_key_with_a_hyphen(text: str, wording: str) -> None:
    assert find_part_name(text).wording == wording


def test_b_flat_written_out_is_not_a_hyphen_key() -> None:
    part = find_part_name("Clarinet in B-flat")
    assert part.key == "Bb"
    assert part.wording == "Clarinet in B-flat"


# ---------------------------------------------------------------------------
# Accents
# ---------------------------------------------------------------------------


def test_accents_dont_matter_and_printed_accents_are_kept() -> None:
    # One alias covers the name with and without its accents...
    assert find_part_name("Flûte alto").instrument_id == "alto_flute"
    assert find_part_name("Flute alto").instrument_id == "alto_flute"
    # ...and the printed name keeps the accents as printed.
    assert find_part_name("Flûte alto").wording == "Flûte alto"
    assert find_part_name("Große Trommel").wording == "Große Trommel"


def test_an_accent_ocr_invented_is_not_shown() -> None:
    # Croatian OCR reads "Clarinet" as "Člarinet": the clarinet is found,
    # but the misread name isn't shown as the printed name.
    part = find_part_name("Člarinet in Bb")
    assert part.display() == "Clarinet in Bb"
    assert part.wording is None


# ---------------------------------------------------------------------------
# Words that look like a short foreign name are NOT part names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Poco pitt mosso",  # OCR of "Poco più mosso", not "piatti" (cymbals)
        "TRANGUILLO",  # OCR of "Tranquillo", not "triángulo"
        "Nuovo Cinema Paradiso",  # not "činela"
        "Orchestre d'Harmonie",  # not "cor d'harmonie"
        "Giacomo Puccini",  # not "alcorno" (alto horn)
        "Celeste",  # not "celesta"
        "Based on a theme by",  # not "basse"
        "Equinox",  # not "requinto"
    ],
)
def test_ordinary_words_are_not_foreign_names(text: str) -> None:
    assert find_part_name(text) is None


# ---------------------------------------------------------------------------
# Keys and numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "key"),
    [
        ("Sib", "Bb"), ("Mib", "Eb"), ("Lab", "Ab"), ("Fa", "F"), ("Do", "C"),
        ("Ut", "C"), ("Sol", "G"), ("La", "A"), ("Re", "D"), ("Be", "Bb"),
    ],
)
def test_solfege_and_other_key_spellings(token: str, key: str) -> None:
    assert parse_key(token) == key


@pytest.mark.parametrize(
    ("token", "number"),
    [
        ("primo", "1"), ("Seconda", "2"), ("terzo", "3"), ("premier", "1"),
        ("deuxième", "2"), ("deuxieme", "2"), ("segundo", "2"), ("eerste", "1"),
        ("tweede", "2"), ("prvi", "1"), ("druga", "2"), ("treći", "3"), ("second", "2"),
    ],
)
def test_ordinal_words(token: str, number: str) -> None:
    assert parse_number(token) == number


def test_an_implied_key_must_be_one_of_the_keys() -> None:
    entry = {"id": "x", "name": "X", "keys": ["Bb"], "aliases": ["small x"]}
    with pytest.raises(ValueError, match="not in its keys"):
        _parse_instrument({**entry, "implied_keys": {"Eb": ["small x"]}})


def test_an_implied_key_name_must_be_an_alias() -> None:
    entry = {"id": "x", "name": "X", "keys": ["Eb"], "aliases": ["small x"]}
    with pytest.raises(ValueError, match="not one of its aliases"):
        _parse_instrument({**entry, "implied_keys": {"Eb": ["tiny x"]}})


# ---------------------------------------------------------------------------
# Every name from the research in tests/data/instrument_research.json
# ---------------------------------------------------------------------------

RESEARCH_FILE = Path(__file__).parent / "data" / "instrument_research.json"

# Research entry -> the instrument its names must give.
RESEARCH_ENTRY_IDS = {
    "Piccolo": "piccolo", "Flute": "flute", "Alto Flute": "alto_flute",
    "Bass Flute": "bass_flute", "Oboe": "oboe", "English Horn": "english_horn",
    "Bassoon": "bassoon", "Contrabassoon": "contrabassoon", "E♭ Clarinet": "clarinet",
    "B♭ Clarinet": "clarinet", "A Clarinet": "clarinet", "Alto Clarinet": "alto_clarinet",
    "Bass Clarinet": "bass_clarinet", "Contra-alto Clarinet": "contra_alto_clarinet",
    "Contrabass Clarinet": "contrabass_clarinet", "Soprano Saxophone": "soprano_saxophone",
    "Alto Saxophone": "alto_saxophone", "Tenor Saxophone": "tenor_saxophone",
    "Baritone Saxophone": "baritone_saxophone", "Bass Saxophone": "bass_saxophone",
    "Trumpet / Cornet": "trumpet", "Flugelhorn": "flugelhorn", "French Horn": "horn",
    "Tenor Horn / Alto Horn": "alto_horn", "Tenor Trombone": "trombone",
    "Bass Trombone": "bass_trombone", "Euphonium / Baritone": "euphonium",
    "Tuba / Sousaphone / Basses": "tuba", "Timpani": "timpani",
    "Glockenspiel": "glockenspiel", "Xylophone": "xylophone", "Marimba": "marimba",
    "Vibraphone": "vibraphone", "Tubular Bells (Chimes)": "chimes", "Crotales": "crotales",
    "Celesta": "celesta", "Timbales (Latin)": "timbales",
    "Unpitched Percussion": "percussion", "Double Bass": "double_bass",
    "Electric Bass": "bass_guitar", "Electric Guitar": "electric_guitar",
    "Synthesizer / Keyboard / Piano": "keyboard", "Soprano": "soprano_voice",
    "Mezzo-Soprano / Alto": "mezzo_voice", "Tenor": "tenor_voice",
    "Baritone / Bass": "bass_voice",
}

# Names (lowercase) that give another instrument than their research entry:
# the entries above often hold several instruments ("Trumpet / Cornet").
RESEARCH_NAME_IDS = {
    "kornet": "cornet", "cornet": "cornet", "cornetta": "cornet", "kornett": "cornet",
    "cornet à pistons": "cornet", "corneta": "cornet",
    "bariton": "baritone", "baryton": "baritone", "barítono": "baritone",
    # The research entry "Tenor Horn / Alto Horn" (in Eb) holds both: the
    # alto horn names give alto_horn, the "Tenor..." names the tenor horn
    # (German "Tenorhorn" is the Bb one, see instruments.yaml).
    "tenorhorn": "tenor_horn", "tenorhorna": "tenor_horn", "tenorhoorn": "tenor_horn",
    "suzafon": "sousaphone", "sousafoon": "sousaphone", "soubassophone": "sousaphone",
    "sousafón": "sousaphone",
    # A plain "Bass" is the generic bass: the instrument is what the PDF says.
    "bas": "bass", "bajo": "bass",
    "piano": "piano", "klavir": "piano", "pianoforte": "piano", "klavier": "piano",
    "contralto": "alto_voice", "voix d'alto (contralto)": "alto_voice",
    # German "Schlagzeug" and French "Batterie" are read as the drum kit,
    # which is what they mean in band parts.
    "schlagzeug (unbestimmte tonhöhe)": "drum_set", "batterie": "drum_set",
    # The Italian name of the contra-alto clarinet says "contrabbasso": the
    # instrument is what the PDF says (a contrabass clarinet in Eb).
    "clarinetto contrabbasso in mi♭": "contrabass_clarinet",
}

# Not taken from the research, on purpose:
# - a plain range word is a band instrument, not a singer: "Soprano" is the
#   soprano saxophone, "Alt" the alto saxophone, "Tenor" the tenor horn,
#   "Bariton" the baritone, "Bas" the generic bass;
# - French/Spanish "Timbales" are read as the Latin timbales, not timpani.
RESEARCH_NOT_TAKEN = {
    ("Soprano", "*"), ("Tenor", "*"),
    ("Mezzo-Soprano / Alto", "alt"),
    ("Baritone / Bass", "*"),
    ("Timpani", "timbales"), ("Timpani", "timbales (clásicos)"),
}

# Abbreviations not taken: single letters and very short forms match far
# too much other text ("Tr." is also the trill sign), and "Sop."/"Ten." are
# the saxophone / tenor horn in band parts, not voices.
RESEARCH_ABBREVS_NOT_TAKEN = {"S.", "A.", "T.", "B.", "Bs.", "M. S.", "Tr.", "Ch.", "Sop.", "Ten."}


def _research_names() -> list[tuple[str, str, str]]:
    """(research entry, language, one name) for every Latin-script name."""
    data = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    names = []
    for group in data.values():
        for entry in group:
            for language, text in entry["languages"].items():
                if language == "serbian_cyrillic":
                    continue  # Latin script only, by the user's choice
                # "Pikolo / Mala flauta" = two names for the same instrument.
                for name in re.split(r"\s+/\s+", text):
                    names.append((entry["name"], language, name))
    return names


def _research_abbreviations() -> list[tuple[str, str]]:
    data = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    return [
        (entry["name"], abbrev)
        for group in data.values()
        for entry in group
        for abbrev in entry["abbreviations"]
    ]


@pytest.mark.parametrize(("entry", "language", "name"), _research_names())
def test_every_research_name_reads_right(entry: str, language: str, name: str) -> None:
    low = name.lower()
    if (entry, "*") in RESEARCH_NOT_TAKEN or (entry, low) in RESEARCH_NOT_TAKEN:
        pytest.skip("not taken (see RESEARCH_NOT_TAKEN)")
    expected = RESEARCH_NAME_IDS.get(low, RESEARCH_ENTRY_IDS[entry])
    part = find_part_name(name)
    assert part is not None, f"{language} {name!r} is not found"
    assert part.instrument_id == expected, f"{language} {name!r}"


@pytest.mark.parametrize(("entry", "abbrev"), _research_abbreviations())
def test_every_research_abbreviation_reads_right(entry: str, abbrev: str) -> None:
    if abbrev in RESEARCH_ABBREVS_NOT_TAKEN:
        pytest.skip("not taken (see RESEARCH_ABBREVS_NOT_TAKEN)")
    expected = RESEARCH_NAME_IDS.get(abbrev.lower(), RESEARCH_ENTRY_IDS[entry])
    # The keyboard entry lists the piano's abbreviations too; "Cnt." is the
    # cornet, "Bar." the baritone, "Alt." the alto saxophone, "T. Hn." the
    # tenor horn (its entry also holds the alto horn).
    expected = {
        "Pno.": "piano", "Pf.": "piano", "Cnt.": "cornet", "Crn.": "cornet",
        "Sous.": "sousaphone", "Bar.": "baritone", "Alt.": "alto_saxophone",
        "B♭ Bass": "bass", "E♭ Bass": "bass", "Mez.": "mezzo_voice", "T. Hn.": "tenor_horn",
    }.get(abbrev, expected)
    part = find_part_name(abbrev)
    assert part is not None, f"{abbrev!r} is not found"
    assert part.instrument_id == expected
