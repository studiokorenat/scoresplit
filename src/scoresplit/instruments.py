"""
instruments.py: recognise instrument part names in a piece of text.

Pipeline position:  pdf_reader/ocr -> [instruments] <- classify

``classify.py`` hands every piece of header or margin text to
``find_part_name()``. This module answers: "Is there a part name in this text,
and if so which instrument, which number, which key?"

The knowledge (which words mean which instrument) lives in
``data/instruments.yaml``, so adding a spelling never needs a code change.

Names are read in English, German, Italian, French, Dutch, Spanish and
Croatian/Serbian (Latin script).

Matching happens in three steps:
  1. normalise_text(): fix common OCR / layout glitches ("Horn3inF" -> "Horn 3 in F")
  2. find the best instrument word(s) in the text: exact match for
     abbreviations, FUZZY match for full names (so "Buphonium" still matches
     "euphonium"). Fuzzy matching uses the rapidfuzz library: it gives a
     similarity from 0 to 100 between two strings. Names are compared
     without accents (see lookup_form), so "Flûte" and "Flute" are the same.
  3. read the part number ("1", "2", "1&2", "Solo", "II", "secondo") and the
     key ("in Bb", "in F", "en Sib", "Es-klarinet") around the instrument word.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

import yaml
from rapidfuzz import fuzz, process

from .models import PartName

# ---------------------------------------------------------------------------
# Tuning constants: change these to make matching stricter or looser.
# ---------------------------------------------------------------------------

# Minimum fuzzy similarity (0..100) to accept a full instrument name.
# Lower = more forgiving of OCR errors, but more false matches.
# 80 accepts "hom"->"horn" (86) and "buphonium"->"euphonium" (89), but
# rejects "flauta"-like look-alikes of other words.
FUZZY_MIN_SCORE = 80

# Aliases shorter than this are only matched exactly (fuzzy matching very short
# words gives too many false hits: "alt" would match "all", "art", ...).
MIN_FUZZY_ALIAS_LENGTH = 4

# A part name is at most this many words long ("bass clarinet" = 2 words).
MAX_NAME_WORDS = 3

# How many words after the instrument name we look at for number and key.
# "Clarinet in Bb 2" needs 3 ("in", "Bb", "2").
MAX_WORDS_AFTER_NAME = 5

# Part numbers above this are almost certainly bar numbers ("Flute 1 86"
# has bar 86), so they're ignored.
MAX_PART_NUMBER = 8

# How OCR reads the "ü" of a name: "Flügel" came out as "Fligel", "Flagel"
# and "Fliigel" (cornfield). Each name in instruments.yaml with a "ü" is
# also found in these spellings, exactly (see _umlaut_misreadings). A fuzzy
# match can't do it safely: "flugel" is 80% like "fuel" and "luge".
UMLAUT_MISREADINGS = ("i", "a", "ii")

# How the many spellings of a key (transposition) are written, mapped to
# the one canonical spelling. OCR often reads "Bb" as "By", "Bp" or "B>".
# "B" alone means B-flat in German/Croatian convention ("Klarinet in B"),
# and in practice no band instrument is pitched in B natural. Serbian
# writes "Be" ("Be-klarinet"), Dutch "Bes".
# The second half is solfège, used in Italian, French and Spanish parts
# ("Clarinetto in Sib", "Cor en Fa", "Trompette en Ut"): do/ut = C, re = D,
# mi = E, fa = F, sol = G, la = A, si = B, and a "b" after it is the flat.
# They are only read after "in" / "en" (see _read_number_and_key), never on
# their own, so the words "la", "do" or "sol" elsewhere are not keys.
KEY_SPELLINGS: dict[str, str] = {
    "bb": "Bb", "b♭": "Bb", "bes": "Bb", "be": "Bb", "b-flat": "Bb", "b": "Bb",
    "by": "Bb", "bp": "Bb", "b>": "Bb", "b»": "Bb", "bh": "Bb",
    "eb": "Eb", "e♭": "Eb", "es": "Eb", "e-flat": "Eb", "ey": "Eb", "ep": "Eb",
    "ab": "Ab", "as": "Ab", "a♭": "Ab",
    # D-flat: the piccolo in Db of older band music. No "d" + OCR-flat
    # spellings here: they'd be too easy to confuse with ordinary words.
    "db": "Db", "d♭": "Db", "des": "Db",
    "f": "F", "c": "C", "a": "A", "d": "D", "g": "G",
    # solfège
    "do": "C", "ut": "C", "re": "D", "ré": "D", "reb": "Db", "réb": "Db",
    "mib": "Eb", "fa": "F", "sol": "G", "lab": "Ab", "la": "A", "sib": "Bb",
}

# The words that introduce a key after the name: "Clarinet in Bb" (English,
# German, Dutch, Croatian, Italian) and "Clarinette en Sib" (French, Spanish).
KEY_WORDS = {"in", "en"}

# A key glued to the name with a hyphen, as German, Dutch and Croatian parts
# write it: "Es-klarinet", "B-Trompete", "F-Horn", "Bes-klarinet". Only the
# spellings that are really printed that way, with a capital first letter,
# and never "B-flat" / "E-flat" (a key written out after "in").
# (?<![\w-]) = not inside a longer word; (?=[^\W\d_]) = a letter follows.
HYPHEN_KEY_PATTERN = re.compile(
    r"(?<![\w-])(?:Bb|Bes|Be|B|Es|Eb|As|Ab|A|Des|Db|D|C|F|G)-(?!flat)(?=[^\W\d_])"
)

# Keys that may also be written BEFORE the instrument ("Bb Trumpet",
# "F Horn"). B and E come with a flat sign; OCR often reads the flat as ":",
# "." or ">", so "B:" or "B." also mean B-flat here. F must be a CAPITAL
# letter: a lowercase "f" in front of a word is a dynamic marking (forte).
# ("E+ Clarinet": the flat read as "+", Ross Roy with eng+hrv OCR.)
# D-flat ("Db Piccolo") only with a capital D and a real flat sign, so a
# stray "D" (a rehearsal letter) is never taken for a key.
# Groups: 1 = B or E (+ flat), 2 = F, 3 = D (+ flat).
LEADING_KEY_PATTERN = re.compile(r"^(?:([BbEe])(?:b|♭|:|\.|>|p|y|\+)?|(F)|(D)(?:b|♭))$")

# A key written right AFTER the name without "in": "Tuba Bb", "Trumpet Bb 2".
# Only the clear flat spellings with a capital letter. Looser ones are real
# words or noise there: "by", "as", "es", a lone "f" (forte), "C" (a section
# letter in the music).
BARE_KEY_AFTER_NAME = re.compile(r"^[ABDE](?:b|♭)$")

ROMAN_NUMERALS = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}

# A clef written as part of the name: "Baritone (T.C.)" = treble clef,
# "Baritone (B.C.)" = bass clef. Some band instruments (baritone, euphonium,
# trombone) come as two parts with the same notes in different clefs, and
# each one needs its own output PDF. (Longer forms such as "Treble Clef" are
# turned into these short ones by normalise_text.)
CLEF_SPELLINGS: dict[str, str] = {
    "t.c": "treble", "tc": "treble",
    "b.c": "bass", "bc": "bass",
}


# ---------------------------------------------------------------------------
# Loading the dictionary
# ---------------------------------------------------------------------------


@dataclass
class Instrument:
    """One entry from instruments.yaml."""

    id: str
    name: str
    family: str
    aliases: list[str]
    abbrevs: list[str]
    # The keys this instrument is written in, the default key first
    # (["Bb", "A", "Eb", "C"] for the clarinet). Empty for instruments that
    # have no key (percussion, strings, ...). The first key is the one a part
    # gets when no key is printed (see default_key()), except for a generic
    # instrument.
    keys: list[str] = field(default_factory=list)
    # True for a name that doesn't say WHICH instrument it is ("Saxophone",
    # "Percussion", "Bass", "Voice"). It changes how a name with a slash is
    # read: "Saxophone/Alto Saxophone" is the alto saxophone, because the
    # second, specific name says which one is meant. See find_part_name().
    # A generic instrument also never gets a default key (see default_key()).
    generic: bool = False
    # The key of a part printed in TREBLE clef when no key is printed, or
    # None. A euphonium or baritone part in treble clef is written in Bb,
    # the same instrument in bass clef in C (concert pitch). See
    # key_from_clef() and segment._keys_from_clef.
    treble_clef_key: str | None = None
    # The ONE clef this instrument is written in ("treble" for the trumpet,
    # "bass" for the bassoon), or None when it may be either (trombone,
    # tuba), or none at all (drums, piano). A part whose clef wasn't printed
    # or detected gets this clef, marked as assumed. See instrument_clef().
    clef: str | None = None
    # Names that SAY the key without printing it: {"mali klarinet": "Eb"}
    # ("small clarinet" = the clarinet in Eb). Each name is one of the
    # instrument's aliases or abbrevs; the key is used only when no key is
    # printed. Written in lookup form (see lookup_form()).
    implied_keys: dict[str, str] = field(default_factory=dict)
    # The aliases that are never fuzzy-matched (`exact:` in the yaml): short
    # foreign words that an OCR slip would otherwise find in ordinary text
    # ("basse" in "based", "piatti" in "Poco più" read as "Poco pitt").
    exact: set[str] = field(default_factory=set)
    # The names that are written WITH accents in the yaml, accents kept:
    # {"flûte alto", "činele", ...} (see _accented_form).
    accented_names: set[str] = field(default_factory=set)


@dataclass
class InstrumentDictionary:
    """
    The whole instruments.yaml, prepared for fast lookups.

    ``fuzzy_aliases`` and ``fuzzy_owner`` are two parallel lists:
    ``fuzzy_aliases[7]`` is an alias string and ``fuzzy_owner[7]`` is the
    Instrument it belongs to. rapidfuzz searches a plain list of strings very
    quickly (in C code), which is why the data is laid out this way.
    """

    instruments: list[Instrument]
    by_id: dict[str, Instrument]  # "clarinet" -> the clarinet entry
    fuzzy_aliases: list[str]
    fuzzy_owner: list[Instrument]
    exact_lookup: dict[str, Instrument]  # short aliases and abbreviations
    abbreviations: set[str]  # the keys of exact_lookup that are abbreviations
    score_keywords: list[str]
    number_words: set[str]
    # Ordinal words and the number they stand for: {"secondo": "2", ...}.
    # In lookup form (no accents: "deuxieme").
    ordinal_words: dict[str, str] = field(default_factory=dict)
    # Every instrument's accented_names together.
    accented_names: set[str] = field(default_factory=set)
    # Where each fuzzy alias is in fuzzy_aliases: {"oboe": 41, ...}, for a
    # fast "is this exactly a name?" check (an alias listed twice keeps its
    # first place).
    fuzzy_index: dict[str, int] = field(default_factory=dict)
    # A name with a "ü" as OCR misreads it, and the name it really is:
    # {"fligel": (flugelhorn, "flügel"), ...}. See UMLAUT_MISREADINGS.
    umlaut_misreadings: dict[str, tuple[Instrument, str]] = field(default_factory=dict)


# The canonical key spellings ("Bb", "Eb", "F", ...): the values of
# KEY_SPELLINGS. The `keys:` lists in instruments.yaml must use these.
CANONICAL_KEYS = set(KEY_SPELLINGS.values())


def lookup_form(text: str) -> str:
    """
    The form in which names are compared with the dictionary: lowercase,
    without accents, an apostrophe as a space, and no space after a dot.

        "Flûte"          -> "flute"
        "Čelesta"        -> "celesta"
        "Große Trommel"  -> "grosse trommel"
        "Cor d'harmonie" -> "cor d harmonie"   (the tokeniser splits at "'")
        "C. A."          -> "c.a."             (= "c.a." written without spaces)

    So one alias in instruments.yaml covers a name with or without its
    accents (OCR often loses them, or adds them), and an abbreviation
    matches however the spaces after its dots are printed.
    """
    # Letters that Unicode doesn't see as "a letter + an accent".
    text = text.lower().replace("ß", "ss").replace("œ", "oe").replace("đ", "d")
    # NFKD splits a letter with an accent into the plain letter followed by
    # the accent on its own ("û" -> "u" + "̂"); the accents ("combining"
    # characters) are then left out.
    decomposed = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in decomposed if not unicodedata.combining(char))
    text = re.sub(r"['’]", " ", text)
    text = re.sub(r"\.\s+", ".", text)  # "c. a." -> "c.a."
    return re.sub(r"\s+", " ", text).strip()


def _accented_form(text: str) -> str:
    """
    Like lookup_form(), but KEEPING the accents: "Flûte Alto" -> "flûte alto".
    Used to tell an accent that is really printed from one OCR invented.
    """
    text = re.sub(r"['’]", " ", text.lower())
    text = re.sub(r"\.\s+", ".", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_instrument(entry: dict) -> Instrument:
    """
    Turn one entry of instruments.yaml (a dict) into an Instrument.

    A key in `keys:` that isn't a canonical spelling ("Hb", "bes") raises a
    ValueError naming the instrument, so a typo in the yaml stops the program
    at once instead of quietly never matching a printed key.
    """
    instrument_id = entry["id"]
    keys = list(entry.get("keys", []))  # .get() = default if the key is missing
    for key in keys:
        if key not in CANONICAL_KEYS:
            allowed = ", ".join(sorted(CANONICAL_KEYS))
            raise ValueError(
                f"instruments.yaml: unknown key {key!r} for instrument "
                f"{instrument_id!r} (use one of: {allowed})"
            )

    # The treble-clef key, and C for the bass clef (see key_from_clef), must
    # be among the instrument's own keys, or the key read from the clef would
    # be one the instrument is never written in.
    treble_clef_key = entry.get("treble_clef_key")
    if treble_clef_key is not None and (treble_clef_key not in keys or "C" not in keys):
        raise ValueError(
            f"instruments.yaml: treble_clef_key {treble_clef_key!r} of instrument "
            f"{instrument_id!r} needs both it and C in its keys {keys}"
        )

    # The instrument's one clef. Only treble and bass: those are the clefs
    # the image detection finds too. An instrument with a treble_clef_key is
    # written in two clefs, so it can't also have just one.
    clef = entry.get("clef")
    if clef not in (None, "treble", "bass"):
        raise ValueError(
            f"instruments.yaml: unknown clef {clef!r} for instrument "
            f"{instrument_id!r} (use treble or bass)"
        )
    if clef is not None and treble_clef_key is not None:
        raise ValueError(
            f"instruments.yaml: instrument {instrument_id!r} has both a clef and a "
            "treble_clef_key: an instrument with one clef has no treble-clef key"
        )

    # `exact` names are full names like the aliases, but matched only exactly
    # (see instruments.yaml). They are added to the aliases here, and also
    # kept apart so _build_dictionary knows not to fuzzy-match them.
    exact = [lookup_form(name) for name in entry.get("exact", [])]
    aliases = [lookup_form(alias) for alias in entry.get("aliases", [])] + exact
    abbrevs = [lookup_form(abbrev) for abbrev in entry.get("abbrevs", [])]
    # The names as written in the yaml, with their accents ("flûte alto"),
    # for the ones that have any. See _match_candidate.
    written = entry.get("aliases", []) + entry.get("exact", []) + entry.get("abbrevs", [])
    accented_names = {_accented_form(name) for name in written if not name.isascii()}

    # implied_keys in the yaml: {key: [names]}, e.g. {Eb: [mali klarinet]}.
    # Here it is turned around into {name: key}. Each key must be one of the
    # instrument's keys, and each name one of its aliases or abbrevs, so a
    # typo can't make a name that silently never matches.
    implied_keys: dict[str, str] = {}
    for key, names in entry.get("implied_keys", {}).items():
        if key not in keys:
            raise ValueError(
                f"instruments.yaml: implied key {key!r} of instrument "
                f"{instrument_id!r} is not in its keys {keys}"
            )
        for name in names:
            name = lookup_form(name)
            if name not in aliases and name not in abbrevs:
                raise ValueError(
                    f"instruments.yaml: implied-key name {name!r} of instrument "
                    f"{instrument_id!r} is not one of its aliases or abbrevs"
                )
            implied_keys[name] = key

    return Instrument(
        id=instrument_id,
        name=entry["name"],
        family=entry.get("family", "other"),
        aliases=aliases,
        abbrevs=abbrevs,
        keys=keys,
        # bool() turns a missing value (None) into False.
        generic=bool(entry.get("generic", False)),
        treble_clef_key=treble_clef_key,
        clef=clef,
        implied_keys=implied_keys,
        exact=set(exact),
        accented_names=accented_names,
    )


# @lru_cache(maxsize=1) remembers the return value, so the YAML file is read
# only once per program run, however often this function is called. Similar to
# Laravel's Cache::rememberForever(), but only in memory.
@lru_cache(maxsize=1)
def load_dictionary() -> InstrumentDictionary:
    """Read data/instruments.yaml (shipped inside the package) and index it."""
    # importlib.resources finds files that live inside the installed package,
    # wherever it was installed (uv tool install, a venv, ...).
    yaml_text = resources.files("scoresplit").joinpath("data/instruments.yaml").read_text(
        encoding="utf-8"
    )
    data = yaml.safe_load(yaml_text)  # safe_load = never executes code from YAML
    return _build_dictionary(data)


def _build_dictionary(data: dict) -> InstrumentDictionary:
    """
    Check and index the parsed instruments.yaml (`data` is the whole file as
    a dict). Split out of load_dictionary() so tests can feed it a small
    hand-made dict and see that a broken file is refused.
    """
    instruments: list[Instrument] = []
    fuzzy_aliases: list[str] = []
    fuzzy_owner: list[Instrument] = []
    exact_lookup: dict[str, Instrument] = {}
    abbreviations: set[str] = set()

    for entry in data["instruments"]:
        instrument = _parse_instrument(entry)
        instruments.append(instrument)

        for alias in instrument.aliases:
            if len(alias) >= MIN_FUZZY_ALIAS_LENGTH and alias not in instrument.exact:
                fuzzy_aliases.append(alias)
                fuzzy_owner.append(instrument)
            # Every alias also gets an exact entry. setdefault() only stores
            # it if no earlier instrument claimed the same word.
            exact_lookup.setdefault(alias, instrument)

        for abbrev in instrument.abbrevs:
            exact_lookup.setdefault(abbrev, instrument)
            abbreviations.add(abbrev)
            # Abbreviations of 3+ letters may also be written without the dot
            # ("Picc" as well as "Picc."). Two-letter ones need the dot.
            without_dot = abbrev.rstrip(".")
            if len(without_dot.replace(" ", "")) >= 3:
                exact_lookup.setdefault(without_dot, instrument)
                abbreviations.add(without_dot)

    # Two entries with the same id would make by_id below keep only the last
    # one, so one of them could never be looked up. Stop at once instead.
    seen_ids: set[str] = set()
    for instrument in instruments:
        if instrument.id in seen_ids:
            raise ValueError(f"instruments.yaml: id {instrument.id!r} is used twice")
        seen_ids.add(instrument.id)

    # A word that is both a full alias and an abbreviation counts as an alias.
    all_aliases = {alias for instrument in instruments for alias in instrument.aliases}
    abbreviations -= all_aliases

    return InstrumentDictionary(
        instruments=instruments,
        # A "dict comprehension": build {id: instrument} in one line, like
        # PHP's array_column($instruments, null, 'id').
        by_id={instrument.id: instrument for instrument in instruments},
        fuzzy_aliases=fuzzy_aliases,
        fuzzy_owner=fuzzy_owner,
        exact_lookup=exact_lookup,
        abbreviations=abbreviations,
        score_keywords=[k.lower() for k in data.get("score_keywords", [])],
        number_words={w.lower() for w in data.get("number_words", [])},
        # str(number): the yaml number 2 becomes the text "2", like any
        # other part number.
        ordinal_words={
            lookup_form(word): str(number)
            for number, words in data.get("ordinal_words", {}).items()
            for word in words
        },
        # set().union(*lists) joins all the sets into one.
        accented_names=set().union(*(instrument.accented_names for instrument in instruments)),
        # reversed(): the FIRST place wins, because later ones overwrite earlier.
        fuzzy_index={
            alias: index for index, alias in reversed(list(enumerate(fuzzy_aliases)))
        },
        umlaut_misreadings=_umlaut_misreadings(instruments, exact_lookup),
    )


def _umlaut_misreadings(
    instruments: list[Instrument], exact_lookup: dict[str, Instrument]
) -> dict[str, tuple[Instrument, str]]:
    """
    Every name with a "ü", in the spellings OCR makes of it (with the "ü"
    read as one of UMLAUT_MISREADINGS), and what it really is:

        {"fligel": (flugelhorn, "flügel"), "flagel": (...), "fliigel": (...), ...}

    A misreading that is itself a name in the dictionary is left out.
    """
    misreadings: dict[str, tuple[Instrument, str]] = {}
    for instrument in instruments:
        for name in instrument.accented_names:
            for position, letter in enumerate(name):
                if letter != "ü":
                    continue
                for reading in UMLAUT_MISREADINGS:
                    misread = lookup_form(name[:position] + reading + name[position + 1 :])
                    if misread not in exact_lookup:
                        misreadings.setdefault(misread, (instrument, name))
    return misreadings


# ---------------------------------------------------------------------------
# Keys: which key a part is in when none is printed, and unusual keys
# ---------------------------------------------------------------------------


def unusual_key_note(instrument_id: str | None, key: str | None) -> str | None:
    """
    A "please check" note when a part's key isn't one its instrument is
    written in ("Flute in Eb", "Violin in Bb"), else None. The key is kept,
    but probably the name was misread.
    """
    if not instrument_id or not key:
        return None
    instrument = load_dictionary().by_id.get(instrument_id)
    if instrument is not None and key in instrument.keys:
        return None
    name = instrument.name if instrument is not None else instrument_id
    return f"unusual key for {name} ({key}): please check"


def clef_key_note(instrument_id: str | None, key: str | None, clef: str | None) -> str | None:
    """
    The note for a part whose key came from its CLEF (see key_from_clef),
    or None:

        ("euphonium", None, "treble") -> "key not printed: Bb, from the treble clef"
        ("euphonium", None, "bass")   -> None  (C is the default key anyway)
        ("euphonium", "Eb", "treble") -> None  (a printed key always wins)

    The note is only there when the clef gives a key OTHER than the default,
    so a person sees why the key isn't the usual one.
    """
    if key:
        return None
    clef_key = key_from_clef(instrument_id, clef)
    if clef_key is None or clef_key == default_key(instrument_id):
        return None
    return f"key not printed: {clef_key}, from the {clef} clef"


def is_unlisted_key(instrument_id: str | None, key: str | None) -> bool:
    """
    True when `key` is a key this instrument is never written in: it has a
    `keys` list and the key isn't on it ("bass_clarinet", "Eb"). Instruments
    without a `keys` list, and parts without a key, give False.
    """
    if not instrument_id or not key:
        return False
    instrument = load_dictionary().by_id.get(instrument_id)
    return instrument is not None and bool(instrument.keys) and key not in instrument.keys


def default_key(instrument_id: str | None) -> str | None:
    """
    The key a part is assumed to be in when no key is printed:

        "clarinet" -> "Bb"   (the first of its keys)
        "violin"   -> None   (no keys at all)
        "bass"     -> None   (generic: a plain "Bass" may be a band bass in
                              Bb or Eb, but also a bass guitar, which has no
                              key, so no key is guessed; segment.py asks a
                              person to check instead)
    """
    instrument = load_dictionary().by_id.get(instrument_id or "")
    if instrument is None or not instrument.keys or instrument.generic:
        return None
    return instrument.keys[0]


def key_from_clef(instrument_id: str | None, clef: str | None) -> str | None:
    """
    The key a part is written in when only its clef is known, or None when
    the clef doesn't tell:

        ("euphonium", "treble") -> "Bb"   (treble clef euphonium is in Bb)
        ("euphonium", "bass")   -> "C"    (bass clef = concert pitch)
        ("flute", "treble")     -> None   (no `treble_clef_key`: the clef
                                           doesn't change the key)
    """
    instrument = load_dictionary().by_id.get(instrument_id or "")
    if instrument is None or instrument.treble_clef_key is None:
        return None
    if clef == "treble":
        return instrument.treble_clef_key
    if clef == "bass":
        return "C"
    return None


def instrument_clef(instrument_id: str | None) -> str | None:
    """
    The clef a part is assumed to be in when no clef was printed or
    detected (the `clef:` in instruments.yaml):

        "trumpet"   -> "treble"
        "bassoon"   -> "bass"
        "trombone"  -> None   (written in treble or bass clef: no guess)
        "snare_drum"-> None   (no pitched clef at all)
    """
    instrument = load_dictionary().by_id.get(instrument_id or "")
    if instrument is None:
        return None
    return instrument.clef


# ---------------------------------------------------------------------------
# Step 1: normalising text
# ---------------------------------------------------------------------------


# How OCR reads the strokes of a Roman numeral (I, II, III) in a scan: as
# "I", a small "l", "|" or the digit "1", in any mix ("Il", "lIl", "11"),
# and III sometimes as "ITI". Used only at the end of a text, after the
# name (see normalise_text), because a lone "|" or "l" elsewhere is noise.
OCR_STROKES = r"(?:[Il|1]{1,3}|ITI)"


def _strokes_as_roman(found: re.Match) -> str:
    """
    For normalise_text: the misread Roman numeral at the end of a name,
    written as a real one. `found.group(1)` is the misread numeral:

        "|" / "l" -> "I"     "Il" / "ll" / "11" -> "II"
        "Ill" / "lIl" / "ITI" -> "III"     "I+11" -> "I+II"

    Plain digits ("1", "1+1") are left alone: they are already numbers.
    "11" is the exception, as no part has number 11 (MAX_PART_NUMBER):
    it is II misread.
    """
    numeral = found.group(1)
    if re.fullmatch(r"[1+]+", numeral) and "11" not in numeral:
        return found.group()  # unchanged
    # "I+11" is two numerals joined by "+": fix each one. A numeral of N
    # strokes is N times "I" (and "ITI", with its middle stroke read as a
    # "T", is III).
    pieces = ["III" if piece == "ITI" else "I" * len(piece) for piece in numeral.split("+")]
    return " " + "+".join(pieces)


# The Roman numeral IV as OCR reads it: "IV", "1V", "lV", "|V".
OCR_FOUR = r"[Il|1]V"


def _roman_pair(found: re.Match) -> str:
    """
    For normalise_text: two Roman numerals joined by a dash at the end of a
    name, written as a pair: "III - IV" -> " III&IV", "111 - 1V" -> " III&IV"
    (strokes misread as digits). Unchanged when the first numeral isn't the
    smaller one, or when both are plain digits ("1 - 2": maybe bar numbers).
    """
    first, second = found.group(1), found.group(2)
    if first.isdigit() and second.isdigit() and "11" not in first + second:
        return found.group()  # plain digits: leave them alone
    numerals = []
    for numeral in (first, second):
        if numeral.endswith("V"):
            numerals.append("IV")  # "1V" / "lV" -> "IV"
        else:
            numerals.append("III" if numeral == "ITI" else "I" * len(numeral))
    # The values: "III" -> 3, "IV" -> 4 (ROMAN_NUMERALS has them in lowercase).
    values = [int(ROMAN_NUMERALS[numeral.lower()]) for numeral in numerals]
    if values[0] >= values[1]:
        return found.group()
    return " " + "&".join(numerals)


def normalise_text(text: str) -> str:
    """
    Repair common glitches so that the words can be matched.

    Examples:
        "Horn3inF"         -> "Horn 3 in F"
        "Hom 1inF"         -> "Hom 1 in F"
        "Bass Tubain C"    -> "Bass Tuba in C"
        "2nd Trumpet"      -> "2. Trumpet"
        '2" Bugle'         -> "2. Bugle"
        "Clarinet in B♭"   -> "Clarinet in Bb"
        "Baritone (Bass Clef)" -> "Baritone (B.C.)"
        "le FLUIT C"       -> "1. FLUIT C"    (Dutch "1e", misread by OCR)
        "Ist Bb Trumpet"   -> "1. Bb Trumpet" ("1st", misread by OCR)
        "Bb Clarinet Il"   -> "Bb Clarinet II"  (Roman numeral, misread)
        "Alto Sax. 1, 2"   -> "Alto Sax. 1&2"
    """
    # OCR sometimes reads a space as "_" ("2nd_Eb Alto Saxophone").
    text = text.replace("♭", "b").replace("’", "'").replace("_", " ")

    # The Dutch ordinal "1e" (1st) at the START of a name, with its "1" misread
    # by OCR: "le FLUIT C", "Ie KLARINET", "{e TROMPET Bb", "fe", "|e" -> "1e",
    # which the ordinal rule below turns into "1.". Only when a word in
    # capitals follows: the French "le" in "le tacet Bassoon" stays.
    # ^ = the start of the text; (?=\s+[A-Z]) = followed by spaces and a capital.
    text = re.sub(r"^[lIf{|]e(?=\s+[A-Z])", "1e", text)

    # The same for the English "1st": "Ist & 2nd Flute", "Ist Bb Trumpet",
    # "lst/2nd Bb Tenor Sax" (american.pdf) -> "1st ...". Only at the START,
    # and only before a space, "/" or "&", so a word like "Isthmus" stays.
    text = re.sub(r"^[lI|]st(?=[\s/&])", "1st", text)

    # A key in front joined to the name by a dash: "C - Trombone II"
    # (cornfield) -> "C Trombone II", so the C is read as the key in front.
    # Only at the START, before a word with a capital letter.
    text = re.sub(r"^([A-G]b?)\s+[-–]\s+(?=[A-Z])", r"\1 ", text)

    # "SOLO +1e BUGEL" (superstar): the solo player and the 1st together.
    # A space on both sides of the "+", so that "Solo", "+" and "1e" are
    # words of their own (see _first_of_a_leading_pair).
    text = re.sub(r"\b(solo)\s*\+\s*(?=\d)", r"\1 + ", text, flags=re.IGNORECASE)

    # Ordinals: "1st", "2nd", "3rd", "4th" -> "1.", "2." ...; also French
    # ("1er", "1re", "2e", "2ème") and Dutch ("1e", "2de", "1ste").
    # Regex: \b = word boundary, (\d) = one digit (captured as group 1),
    # (?:st|nd|...) = one of these endings (not captured). The longer
    # endings come first, so "2ème" isn't cut after its "e".
    text = re.sub(r"\b(\d)(?:st|nd|rd|th|ère|ème|eme|er|re|ste|de|e)\b", r"\1.", text)

    # Ordinals printed as small raised letters ("1ˢᵗ", Italian/Spanish "1º",
    # "1ª") are often read by OCR as a symbol: "1* Bugle", '2" Bugle',
    # "1% Bugle", "1° Flute". Only directly before a word, so a lone digit
    # + symbol inside the music isn't touched.
    text = re.sub(r"\b([1-8])[*%\"”“°ºª](?=\s+[^\W\d_])", r"\1.", text)

    # A flat written out after a solfège key: "Si bemolle" (Italian), "Si
    # bémol" (French), "Si bemol" / "Si b" (Spanish) -> "Sib", the spelling
    # parse_key knows. (The flat sign itself, "Si♭", was turned into "Sib"
    # above.)
    text = re.sub(
        r"\b(Si|Mi|La|Re|Ré|Sol)[\s-]?(?:bemolle|bémol|bemol|b)\b",
        r"\1b",
        text,
        flags=re.IGNORECASE,
    )

    # A clef written out in full -> the short form that _read_number_and_key
    # understands. "violinski/bas ključ" is Croatian.
    text = re.sub(
        r"\b(?:treble|violin|violinski)\s+(?:clef|klju[čc])", "T.C.", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(?:bass|bas|basov|basovski)\s+(?:clef|klju[čc])", "B.C.", text, flags=re.IGNORECASE
    )

    # A small "l" or "|" glued to the end of a CAPITALS word is the digit 1
    # misread: "TRUMPETl" -> "TRUMPET 1". (Never a capital "I": "TIMPANI".)
    text = re.sub(r"(?<=[A-Z]{3})[l|](?=$|[\s,.])", " 1", text)

    # Two Roman numerals joined by a dash at the END of the text: one part
    # for two players, "Horn III - IV" -> "Horn III&IV" (= 3&4). OCR reads
    # the strokes as digits: "Horn 111 - 1V" (cornfield). See _roman_pair.
    text = re.sub(
        rf"(?<=[^\W\d_]{{2}}) ({OCR_STROKES}|{OCR_FOUR}) ?- ?({OCR_STROKES}|{OCR_FOUR})\s*$",
        _roman_pair,
        text,
    )

    # A Roman numeral at the END of the text, after a word, with its strokes
    # misread by OCR: "Bb Clarinet |" -> "Bb Clarinet I", "Eb Horn Il" ->
    # "Eb Horn II", "Clarinet Ill" -> "Clarinet III", "FAGOT I+11" ->
    # "FAGOT I+II" (see _strokes_as_roman).
    # (?<=[^\W\d_]{2}) = right after two letters (the end of a word) and a space.
    text = re.sub(
        rf"(?<=[^\W\d_]{{2}}) ({OCR_STROKES}(?:\+{OCR_STROKES})?)\s*$", _strokes_as_roman, text
    )

    # "Horn 1, 2" / "ALTO SAXOPHONE 1,2" / "Alto Sax. 1, 2": ONE part for the
    # 1st and 2nd player, written like "1&2". Only right after a name word
    # (which may end in a dot, "Sax."), so a list of bar numbers isn't touched.
    # (?:...)+ repeats, so "1, 2, 3" -> "1&2&3"; the lambda gets each match.
    # A lookbehind (?<=...) must have a fixed length, hence two of them: one
    # for "Horn 1, 2" and one for "Sax. 1, 2".
    text = re.sub(
        r"(?:(?<=[A-Za-z] )|(?<=[A-Za-z]\. ))[1-8](?:\s*,\s*[1-8]\b)+",
        lambda numbers: re.sub(r"\s*,\s*", "&", numbers.group()),
        text,
    )

    # A letter directly followed by a digit: "Horn3" -> "Horn 3".
    # (?<=[A-Za-z]) = "preceded by a letter" (the letter itself is not replaced).
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)

    # A digit directly followed by a letter: "1inF" -> "1 inF".
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)

    # "inF" / "inBb" -> "in F" / "in Bb". (?=[A-G]) = "followed by a capital
    # A..G", so normal words like "into" are left alone.
    text = re.sub(r"\bin(?=[A-G])", "in ", text)

    # "HominF" -> "Hom in F": OCR glued "in" and the key to the name.
    # (?<=[a-z]) = after a lowercase letter; the key must be a capital A..G.
    text = re.sub(r"(?<=[a-z])in(?=[A-G](?:b|#)?\b)", " in ", text)

    # "Tubain C" -> "Tuba in C": OCR swallowed the space before "in".
    # Only when the next word looks like a key, and never for words that
    # really end in "in" (violin, mandolin) - hence the (?<!ol) check.
    text = re.sub(r"(?<=[a-z]{3})(?<!ol)in(?=\s+[A-G][b#yp>]?\b)", " in", text)

    # Collapse repeated spaces into one.
    return re.sub(r"\s+", " ", text).strip()


def _tokenise_with_positions(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into words, and say where each word is in the text:
    a list of (word, start, end), where text[start:end] is the word.

    Brackets, quotes and similar punctuation count as separators; dots stay
    attached ("Picc." stays one token). A slash is a token of its own, so
    "Trumpet/Cornet" -> "Trumpet", "/", "Cornet". A colon stays attached to
    a single capital letter, because OCR often reads the flat sign that way
    ("B:" = B-flat). A key glued to the name with a hyphen is a word of its
    own: "Es-klarinet" -> "Es-", "klarinet" (see HYPHEN_KEY_PATTERN).

    The positions let find_part_name() cut the printed name out of the text
    with the punctuation between the words as printed (see _printed_wording).
    """
    words = []
    # The "|" in a regex means "or": the first alternative that fits at a
    # position wins, so the hyphen key is tried before an ordinary word.
    word_pattern = rf"/|{HYPHEN_KEY_PATTERN.pattern}|[^\s,;:()\[\]{{}}\"'|/]+:?|:"
    # re.finditer gives one "match" per word; .start()/.end() are its position.
    for found in re.finditer(word_pattern, text):
        word, start, end = found.group(), found.start(), found.end()
        # A colon is only kept on a one-letter token ("B:"); otherwise drop it.
        if word.endswith(":") and len(word) != 2:
            stripped = word.rstrip(":")
            end -= len(word) - len(stripped)  # the word got shorter
            word = stripped
        if word:
            words.append((word, start, end))
    return words


def _tokenise(text: str) -> list[str]:
    """Split text into words (see _tokenise_with_positions), without the positions."""
    return [word for word, _start, _end in _tokenise_with_positions(text)]


# ---------------------------------------------------------------------------
# Step 3 helpers: number and key
# ---------------------------------------------------------------------------


def parse_key(token: str) -> str | None:
    """'Bb' / 'B>' / 'Es' / 'F,' -> canonical key ('Bb', 'Eb', 'F'), else None."""
    cleaned = token.lower().strip(".,;:")
    return KEY_SPELLINGS.get(cleaned)


def _leading_key(token: str) -> str | None:
    """
    The key written IN FRONT of a name, or None:

        "Bb" / "B:" / "E+"  -> "Bb" / "Bb" / "Eb"   ("Bb Trumpet", flat misread)
        "F"                 -> "F"                  ("F Horn"; never "f" = forte)
        "Es-" / "B-"        -> "Eb" / "Bb"          ("Es-klarinet", "B-Trompete")
    """
    # A hyphen key ("Es-") is a word of its own (see _tokenise_with_positions).
    if token.endswith("-") and token[:1].isupper():
        return parse_key(token[:-1])
    found = LEADING_KEY_PATTERN.match(token)
    if not found:
        return None
    if found.group(2):  # "F Horn"
        return "F"
    if found.group(3):  # "Db Piccolo"
        return "Db"
    return found.group(1).upper() + "b"  # "B" + flat -> "Bb", "E" + flat -> "Eb"


def parse_clef(token: str) -> str | None:
    """'T.C.' / 'tc' -> 'treble', 'B.C.' -> 'bass', anything else -> None."""
    return CLEF_SPELLINGS.get(token.lower().strip(".,;:"))


def parse_number(token: str) -> str | None:
    """
    Recognise a part number token.

    Examples:
        "1" -> "1"      "2." -> "2"       "1&2" -> "1&2"
        "II" -> "2"     "Solo" -> "Solo"  "Solo&1" -> "Solo&1"
        "1-2" -> "1-2"  "2-4" -> "2-4"    (one part for the 2nd to 4th player)
        "secondo" -> "2"  "Premier" -> "1"  (ordinal words, see instruments.yaml)
        "89" -> None (bar number, too big)  "Flute" -> None
    """
    dictionary = load_dictionary()
    cleaned = token.strip(".,;:")
    if not cleaned:
        return None

    # A range of part numbers: "Eb Alto Saxophone 1-2", "F Horn 2-4". Both ends
    # must be part numbers and the first the smaller one, so a date or a
    # bar range ("12-16") isn't read as one. The dash is kept as printed,
    # so "Horn 1-3" and "Horn 2-4" stay two different names.
    range_match = re.fullmatch(r"([1-8])[-–]([1-8])", cleaned)
    if range_match:
        first, last = range_match.groups()
        return f"{first}-{last}" if first < last else None

    # A number can be several pieces joined by & or + ("1&2", "Solo&1").
    pieces = re.split(r"[&+]", cleaned)
    result: list[str] = []
    for piece in pieces:
        low = piece.lower()
        if piece.isdigit() and 1 <= int(piece) <= MAX_PART_NUMBER:
            result.append(piece)
        elif low in ROMAN_NUMERALS and piece.isupper():
            # Roman numerals must be capitals ("II"), so the word "i" isn't one.
            result.append(ROMAN_NUMERALS[low])
        elif low in dictionary.number_words:
            result.append(piece.capitalize())
        elif lookup_form(piece) in dictionary.ordinal_words:
            result.append(dictionary.ordinal_words[lookup_form(piece)])  # "primo" -> "1"
        else:
            return None  # one bad piece makes the whole token "not a number"
    return "&".join(result)


# ---------------------------------------------------------------------------
# Step 2: finding the instrument words
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    """
    Internal: the instrument that one candidate string ("bass clarinet")
    matched, found by _match_candidate.

    `trusted` and `shown_as` say how the matched words may appear in the
    printed name (PartName.wording):

        "Clarinet"  exact match    trusted, shown as printed
        "Flutes"    a plural       trusted, shown as printed
        "Hom"       OCR "rn" -> "m": trusted, shown as "Horn"
        "mTRUMPET"  only LOOKS like "trumpet" (a fuzzy match): not trusted,
                    so the part gets its recognised name instead
    """

    instrument: Instrument
    score: float  # similarity 0..100
    alias: str  # the alias that matched (longer = more specific)
    trusted: bool = True
    shown_as: str | None = None  # None = show the words as printed


@dataclass
class _Match:
    """Internal: the best instrument found in a list of tokens."""

    instrument: Instrument
    score: float  # similarity 0..100
    start: int  # index of the first matched token
    end: int  # index AFTER the last matched token (Python slice style)
    alias: str  # the alias that matched (longer = more specific)
    trusted: bool = True  # see _Candidate
    shown_as: str | None = None  # see _Candidate


def _in_case_of(printed: str, fixed: str) -> str:
    """
    Write `fixed` (lowercase) in the same capitals as `printed`:
    ("HOM", "horn") -> "HORN", ("Hom", "horn") -> "Horn", ("hom", "horn") -> "horn".
    """
    if printed.isupper():
        return fixed.upper()
    if printed[:1].isupper():
        return " ".join(word.capitalize() for word in fixed.split())
    return fixed


def _is_plural(word: str, alias: str) -> bool:
    """True for "flutes" / "basses" when the alias is "flute" / "bass"."""
    return word in (alias + "s", alias + "es")


def _match_candidate(candidate: str) -> _Candidate | None:
    """
    Compare one candidate string ("bass clarinet") with the dictionary.
    Returns what it matched (see _Candidate), or None.
    """
    dictionary = load_dictionary()
    # Without accents and spaces after dots, like the aliases (lookup_form).
    low = lookup_form(candidate)

    # 1) exact match (abbreviations and short aliases), always score 100.
    #    Its words are shown as printed, unless they carry an accent the name
    #    doesn't have: OCR read "Clarinet" as "Člarinet", which is not how it
    #    was printed. (A LOST accent is fine: "Flute" for "Flûte".)
    if low in dictionary.exact_lookup:
        printed = _accented_form(candidate)
        trusted = printed.isascii() or printed in dictionary.accented_names
        return _Candidate(dictionary.exact_lookup[low], 100.0, low, trusted)

    # A name with a "ü" that OCR misread: "Fligel" -> "Flügel" (see
    # UMLAUT_MISREADINGS). Shown as the real name, in the printed capitals.
    if low in dictionary.umlaut_misreadings:
        instrument, name = dictionary.umlaut_misreadings[low]
        return _Candidate(instrument, 100.0, lookup_form(name), True, _in_case_of(candidate, name))

    # 2) fuzzy match, only for reasonably long candidates made of letters
    letters_only = re.sub(r"[^a-z ]", "", low).strip()
    if len(letters_only) < 3 or letters_only != low.strip("."):
        return None  # contains digits/symbols -> not a plain name

    # OCR often reads a small "e" as "c": "Oboc", "Baritonc", "Trombonc".
    # In a long word fuzzy matching forgives that, but not in a short one
    # ("oboc" is only 75% like "oboe"), and in "Bb Baritonc" over "Bb
    # Euphonium" the near miss lost to the exact second line. So each "c" is
    # also read as an "e", but that reading must BE a name, exactly. A
    # fuzzy match on top of a guessed letter would find instruments in
    # ordinary words (as "cowbell" did in McDowell and Cromwell).
    exact_e_reading = _c_read_as_e(letters_only, dictionary)
    if exact_e_reading is not None:
        alias, index = exact_e_reading
        shown_as = _in_case_of(candidate, alias)
        return _Candidate(dictionary.fuzzy_owner[index], 100.0, alias, True, shown_as)

    # Try the text as it is, and then "OCR variants" of it. OCR very often
    # reads the letters "rn" as one "m" ("Horn" -> "Hom"), so we also try
    # with every "m" turned back into "rn".
    variants = [letters_only]
    if "m" in letters_only:
        variants.append(letters_only.replace("m", "rn"))

    for variant in variants:
        # process.extractOne compares `variant` with every alias and returns
        # the best (alias, score, index), or None if nothing reaches
        # score_cutoff.
        best = process.extractOne(
            variant, dictionary.fuzzy_aliases, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN_SCORE
        )
        if best is not None:
            alias, score, index = best
            # The words can be shown when they ARE the alias (after the
            # "rn" fix) or its plural; anything else only looks like it.
            trusted = variant == alias or _is_plural(variant, alias)
            shown_as = None
            if variant != letters_only:  # the "rn" fix was needed
                shown_as = _in_case_of(candidate, variant)
            return _Candidate(dictionary.fuzzy_owner[index], float(score), alias, trusted, shown_as)
    return None


def _c_read_as_e(word: str, dictionary: InstrumentDictionary) -> tuple[str, int] | None:
    """
    For _match_candidate: `word` (lowercase letters) with ONE "c" read as
    "e", when that gives a name exactly: "oboc" -> "oboe", "baritonc" ->
    "baritone". Returns (the name, its index in dictionary.fuzzy_aliases),
    or None. One "c" at a time, so "clarinct" still finds "clarinet"
    (turning every "c" into "e" would give "elarinet").
    """
    if word in dictionary.fuzzy_index:
        return None  # already a name as it is: no letter needs fixing
    for position, letter in enumerate(word):
        if letter != "c":
            continue
        reading = word[:position] + "e" + word[position + 1 :]
        if reading in dictionary.fuzzy_index:
            return reading, dictionary.fuzzy_index[reading]
    return None


def _has_letters(word: str) -> bool:
    """True when the word has at least one letter: "Bass", "B:" - not ".", "/", "2"."""
    # any() is True when at least one item is True.
    return any(char.isalpha() for char in word)


def _best_instrument_match(tokens: list[str]) -> _Match | None:
    """
    Slide a window of 1..MAX_NAME_WORDS tokens over the text and keep the best
    instrument match. "le tacet Bassoon 2" -> Bassoon at tokens[2:3].
    """
    best: _Match | None = None
    for size in range(MAX_NAME_WORDS, 0, -1):  # 3, 2, 1
        for start in range(0, len(tokens) - size + 1):
            window = tokens[start : start + size]
            # A name starts and ends with a word, never with a stray "." or
            # "/" ("B:. Bass Clarinet" is "Bass Clarinet", not ". Bass Clarinet").
            if not _has_letters(window[0]) or not _has_letters(window[-1]):
                continue
            candidate = " ".join(window)
            found = _match_candidate(candidate)
            if found is None:
                continue
            match = _Match(
                found.instrument,
                found.score,
                start,
                start + size,
                found.alias,
                found.trusted,
                found.shown_as,
            )
            # Better = higher score; on a tie, the longer (more specific) alias
            # wins, so "Bass Clarinet" beats "Clarinet". Python compares
            # tuples element by element, left to right.
            if best is None or (match.score, len(match.alias)) > (best.score, len(best.alias)):
                best = match
    return best


# Words that name only a RANGE (high or low) of an instrument family. Before
# "saxophone" they say which saxophone it is ("Baritone Saxophone"). Only the
# English ones: "<word> saxophone" must be a name in instruments.yaml. With
# the Dutch "Alt", the repaired "Alt Saxophone" is no name, and the exact
# generic "Saxophone" would win over the alto saxophone that "ALT" alone
# gives ("ALT SAXOFOONI+ 1O Eb" in suite).
RANGE_WORDS = {"soprano", "alto", "tenor", "baritone", "bass"}

# After a range word, a word that starts with "sa" and is at least this
# similar (0..100) to "saxophone" is a garbled "saxophone". Much looser than
# FUZZY_MIN_SCORE, because the word before already says what to expect: OCR
# read "Eb Baritone Saxophone" as "Eb Baritone Saigphonja" (63).
GARBLED_SAXOPHONE_MIN_SCORE = 60

# ...and is at least this long: short "sa..." words ("Sax", "Salsa") are not
# a garbled "saxophone".
GARBLED_SAXOPHONE_MIN_LENGTH = 6


def _e_flat_read_as_dot(tokens: list[str], match: _Match) -> _Match:
    """
    "E. Bass" is "Eb Bass": OCR read the flat sign as a dot.

    "e. bass" is also an abbreviation of Bass Guitar, and an abbreviation
    found as a whole wins over a key in front of a name. So when the match
    is an abbreviation that starts with such an "E." and the words after it
    are a FULL instrument name that can be in Eb, the "E." is the key and
    the match is only the name after it (find_part_name then reads "E."
    as Eb, see _leading_key):

        "E. Bass"  -> Bass, and "E." in front = Eb     (Tullamore)
        "E.Bass"   -> Bass Guitar   (one word: no flat sign could be there)
        "E. Hn."   -> English Horn  ("Hn." is itself abbreviated: the whole
                                     name is an abbreviation, not a key + name)
        "E. Gtr."  -> Electric Guitar (the same; a guitar is never in Eb)

    ("E. Horn" never reaches this: "e. horn" is no abbreviation, so it is
    read as Eb Horn, as in Tullamore. Rossroy prints "E. Horn" only on a
    page that continues its "English Horn" part, which keeps that name.)
    """
    dictionary = load_dictionary()
    matched_words = tokens[match.start : match.end]
    if len(matched_words) < 2 or _leading_key(matched_words[0]) != "Eb":
        return match
    if lookup_form(" ".join(matched_words)) not in dictionary.abbreviations:
        return match
    rest = _best_instrument_match(matched_words[1:])
    if (
        rest is None
        or rest.start != 0
        or rest.end != len(matched_words) - 1  # the name must be ALL the other words
        or rest.score < 100
        or lookup_form(" ".join(matched_words[1:])) in dictionary.abbreviations
        or "Eb" not in rest.instrument.keys
    ):
        return match
    # Count the words from the whole text again: one word later than before.
    return _Match(
        rest.instrument,
        rest.score,
        match.start + 1,
        match.end,
        rest.alias,
        rest.trusted,
        rest.shown_as,
    )


def _garbled_saxophone(tokens: list[str]) -> list[str]:
    """
    Repair a "saxophone" that OCR garbled beyond normal fuzzy matching, after
    a range word: ["Eb", "Baritone", "Saigphonja"] -> ["Eb", "Baritone",
    "Saxophone"]. Without this, "Baritone" alone wins, which is the brass
    baritone. Returns a new list; the number of words stays the same.
    """
    fixed = list(tokens)  # a copy, so the caller's list is not changed
    for index in range(1, len(tokens)):
        word = tokens[index]
        if (
            lookup_form(tokens[index - 1]) not in RANGE_WORDS
            or not word.lower().startswith("sa")
            or len(word) < GARBLED_SAXOPHONE_MIN_LENGTH
        ):
            continue
        # A word that is a name already ("Saxofoon", "Saxhorn") stays.
        if _match_candidate(word) is not None:
            continue
        if fuzz.ratio(lookup_form(word), "saxophone") >= GARBLED_SAXOPHONE_MIN_SCORE:
            # Shown in the capitals of the printed word ("SAIGPHONJA" -> "SAXOPHONE").
            fixed[index] = _in_case_of(word, "saxophone")
    return fixed


# A bracket and what is inside it (group 1). The closing bracket may be
# missing at the end of the text: OCR cut it off ("2e ALT Eb (HORN").
# [^()]* = any text without brackets; (?:\)|$) = a ")" or the end of the text.
BRACKET_PATTERN = re.compile(r"\(([^()]*)(?:\)|$)")

# The instrument that, in brackets after a plain range word, makes the part
# an alto horn: "1e ALT Eb (HORN I)" in a Dutch fanfare is the Eb alto horn
# (decided by the user), not the alto saxophone that "Alt" alone means.
ALTO_HORN_WORDS = {"alt", "alto"}


def _name_with_bracket(text: str) -> PartName | None:
    """
    A part name followed by a second name of the SAME part in brackets,
    which says more than the name outside (decided by the user):

        "1. COR Eb (HORN III)" -> Horn 3 in Eb       the bracket's number wins
        "1. ALT Eb (HORN I)"   -> Alto Horn 1 in Eb  a range word + a horn
        "FAGOT C (BASSON)"     -> Bassoon in C       the same name, translated

    The printed name keeps the brackets: "1. Cor Eb (Horn III)".
    Returns None for any other bracket. A different instrument in brackets
    is a doubling ("FLUTE 1, 2 (PICCOLO)"), which find_part_name handles.
    `text` has been through normalise_text already.
    """
    found = BRACKET_PATTERN.search(text)
    if found is None:
        return None
    inner = find_part_name(found.group(1))
    # The text without the bracket: before it + after it.
    outside_text = text[: found.start()] + " " + text[found.end() :]
    outside = find_part_name(outside_text)
    if inner is None or outside is None:
        return None

    dictionary = load_dictionary()
    # Which words outside the bracket named the instrument ("ALT", "COR").
    tokens = _tokenise(normalise_text(outside_text))
    match = _best_instrument_match(tokens)
    outside_words = lookup_form(" ".join(tokens[match.start : match.end])) if match else ""
    if inner.instrument_id == "horn" and outside_words in ALTO_HORN_WORDS:
        instrument = dictionary.by_id["alto_horn"]
    elif inner.instrument_id == outside.instrument_id:
        # The same instrument, with a number ("COR Eb (HORN III)") or just
        # in another language ("FAGOT C (BASSON)", superstar): the bracket
        # belongs to the printed name.
        instrument = dictionary.by_id[outside.instrument_id]
    else:
        return None

    # The bracket's number wins ("1e COR Eb (HORN III)" is Horn 3), unless
    # the number outside already holds it and says more: "SOLO + 1e BUGEL
    # Bb (FLUGELHORN I)" is Solo&1, not 1.
    number = inner.number or outside.number
    if inner.number and outside.number and inner.number in outside.number.split("&"):
        number = outside.number

    wording = None
    if outside.wording and inner.wording:
        wording = f"{outside.wording} ({inner.wording})"
    return PartName(
        instrument_id=instrument.id,
        instrument_name=instrument.name,
        family=instrument.family,
        number=number,
        # "or": the first one that is not None.
        key=outside.key or inner.key,
        clef=outside.clef or inner.clef,
        label=outside.label,
        score=min(outside.score, inner.score),
        # A full name in the brackets makes an abbreviation outside them
        # ("COR") a real part name.
        abbreviation=outside.abbreviation and inner.abbreviation,
        wording=wording,
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def find_part_name(text: str) -> PartName | None:
    """
    Find a part name in a piece of text, or return None.

    Examples:
        "Clarinet in Bb 2"          -> Clarinet 2 in Bb
        "Hom 2in F"                 -> Horn 2 in F
        "le tacet Bassoon 1"        -> Bassoon 1
        "2. truba in B"             -> Trumpet 2 in Bb
        "Shaker, Susp. Cymbal, ..." -> label "Shaker, Susp. Cymbal, ..."
        "Lentamente molto rubato"   -> None
    """
    text = normalise_text(text)
    if not text:
        return None

    # A second name of the same part in brackets, which says more than the
    # name outside: "1e COR Eb (HORN III)" is Horn 3.
    bracketed = _name_with_bracket(text)
    if bracketed is not None:
        return bracketed

    # An instrument in brackets after the name is one the player doubles on:
    # "FLUTE 1, 2 (PICCOLO)" is the Flute part. Drop such brackets when the
    # text outside them names an instrument. (Brackets without an instrument,
    # like the clef in "Baritone (T.C.)", stay.)
    without_doubling = BRACKET_PATTERN.sub(
        lambda group: " " if _best_instrument_match(_tokenise(group.group(1))) else group.group(),
        text,
    )
    if without_doubling != text and _best_instrument_match(_tokenise(without_doubling)):
        text = re.sub(r"\s+", " ", without_doubling).strip()

    # A comma-separated list of SEVERAL instruments is a combined part
    # (common in percussion: "Shaker, Tambourine, Wind Chimes"). Keep the
    # original wording as the label.
    chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if len(chunks) >= 2:
        chunk_tokens = [_tokenise(chunk) for chunk in chunks]
        chunk_matches = [_best_instrument_match(tokens) for tokens in chunk_tokens]
        if all(m is not None for m in chunk_matches):
            first = chunk_matches[0]
            # Keep each instrument's own wording, without the noise around it:
            # "Glockenspiel &" -> "Glockenspiel", "Bb Euphonium (bra)" -> "Bb Euphonium".
            names = [
                _name_wording(tokens, match)
                for tokens, match in zip(chunk_tokens, chunk_matches, strict=True)
            ]
            # The key, when every instrument in the list prints the same one:
            # "Bb Baritone, Bb Euphonium" -> Bb. (Each chunk has no comma, so
            # find_part_name reads it as a single name.) A set keeps each
            # value once: {"Bb"} means all agree, {None} means no key at all.
            chunk_keys = {part.key if part else None for part in map(find_part_name, chunks)}
            key = chunk_keys.pop() if len(chunk_keys) == 1 else None
            label = _calm_capitals(", ".join(names))
            return PartName(
                instrument_id=first.instrument.id,
                instrument_name=first.instrument.name,
                family=first.instrument.family,
                key=key,
                label=label,
                score=min(m.score for m in chunk_matches),
                # The list is printed as it is: it is its own printed name,
                # unless one of the names only looked like an instrument.
                wording=label if all(m.trusted for m in chunk_matches) else None,
            )

    # The words, where each one is in `text`, and the words alone. Only
    # _misread_slash ("I" -> "/") and _garbled_saxophone change a word, never
    # the number of words, so word i is still at positions[i].
    with_positions = _tokenise_with_positions(text)
    positions = [(start, end) for _word, start, end in with_positions]
    printed_tokens = [word for word, _start, _end in with_positions]
    tokens = _misread_slash(printed_tokens)
    tokens = _garbled_saxophone(tokens)
    match = _best_instrument_match(tokens)
    if match is None:
        return None
    match = _e_flat_read_as_dot(tokens, match)
    # How the instrument word(s) should appear in the printed name. The
    # "/"-joined second name below adds its own.
    shown = _shown_instrument_words(match)
    trusted = match.trusted

    # --- Two instruments joined by "/" form ONE part that either instrument
    #     can play: "Trumpet/Cornet 1", "Trombone 3/Bass Trombone". The first
    #     instrument is the main one, and the label keeps both names.
    #     Two names of the SAME instrument ("Tenor Tuba/Euphonium in Bb") are
    #     read the same way, so the key after the second name isn't lost, but
    #     get no label: it is simply "Euphonium in Bb".
    #     A GENERIC first name with a specific second one ("Bass/Electric
    #     Bass", "Saxophone/Alto Saxophone") is the specific instrument: the
    #     second name says which one is meant. The label still keeps both.
    slash_end: int | None = None  # where the second instrument name ends
    same_instrument_twice = False
    if "/" in tokens:
        slash = tokens.index("/")
        left = _best_instrument_match(tokens[:slash])
        right = _best_instrument_match(tokens[slash + 1 :])
        if left and right:
            match = left
            if left.instrument.generic and not right.instrument.generic:
                # Keep where the name STARTS (left.start), so the label and
                # the number/key reading below see the whole "Bass/Electric
                # Bass"; only the instrument comes from the right-hand name.
                match = _Match(right.instrument, right.score, left.start, left.end, right.alias)
            slash_end = slash + 1 + right.end
            same_instrument_twice = left.instrument.id == right.instrument.id
            # Both names appear in the printed name, so both must be readable.
            # (`right` counted its words from just after the slash.)
            shown = _shown_instrument_words(left) | _shown_instrument_words(right, slash + 1)
            trusted = left.trusted and right.trusted
            # A dash read as a "/" (see NAME_DASHES) is printed as a dash.
            if printed_tokens[slash] in NAME_DASHES:
                shown[slash] = printed_tokens[slash]

    number, key, clef, name_end = _read_number_and_key(tokens, match.end, slash_end)
    # True when the key is printed AFTER the name ("Tuba Bb", "Clarinet in
    # Bb"), so it is inside the label below.
    key_after_name = key is not None
    # True when the number is printed AFTER the name ("Trumpet/Cornet 1"),
    # so it is inside the label below; a number in front is added to it.
    number_after_name = number is not None

    # Where the printed name starts: at the instrument word, or earlier when a
    # key or number is written in front of it (see below).
    name_start = match.start

    # --- look BEFORE the name: "2. Trumpet", "Bb Trumpet", "II Horn",
    #     and both together: "1. Bb Clarinet" (from "1st Bb Clarinet")
    if match.start > 0:
        before = tokens[match.start - 1]
        leading = _leading_key(before)
        if leading is None and before == "C" and _c_is_a_key(tokens, match.start - 1):
            leading = "C"  # "1. C Trombone", "C Baritone/Euphonium"
        if leading:
            key = key or leading
            if key == leading:
                # The key in front belongs to the printed name: "Bb Trumpet".
                name_start = match.start - 1
                # A key glued to the name with a hyphen is shown as printed
                # ("Es-klarinet"). Any other key in front is shown as the
                # canonical key: it is the English way of writing ("Bb
                # Clarinet"), so a "B", "B:", "B." or "B- " (with a space)
                # there is a B-flat whose flat sign OCR misread.
                glued = positions[match.start - 1][1] == positions[match.start][0]
                if not (before.endswith("-") and glued):
                    shown[match.start - 1] = leading
        # With a key in front, the number (if any) is one word further left.
        number_index = match.start - 1
        if leading and number_index > 0:
            number_index -= 1
        number_word = tokens[number_index]
        # A leading number counts only with a dot ("2."), as a Roman numeral
        # or as an ordinal word ("Secondo Corno"): a bare "2" in front is
        # usually the PAGE number. A Roman numeral of ONE letter needs the
        # dot too ("V. Horn"): a lone "I" or "V" is too often OCR noise
        # ("}V( Ccolo" is not "Piccolo 5").
        roman_ok = len(number_word.strip(".")) >= 2 and number_word.strip(".").isupper()
        ordinal_ok = lookup_form(number_word) in load_dictionary().ordinal_words
        if number is None and (number_word.endswith(".") or roman_ok or ordinal_ok):
            number = parse_number(number_word)
            if number is not None:
                name_start = number_index
                # Two players in front: "1. & 2. Flute", "1./2. Bb Tenor Sax"
                # (from "1st & 2nd", "1st/2nd") -> "1&2", not just "2".
                first = _first_of_a_leading_pair(tokens, number_index)
                if first is not None:
                    number = f"{first}&{number}"
                    name_start = number_index - 2
        if number is None:
            # clean_line: the text is ONLY the number, the name and what
            # _read_number_and_key read after it (nothing left over), and the
            # name matched exactly.
            clean_line = name_end == len(tokens) and match.score >= 100
            # "2 SLAGWERK" (superstar) is plain Percussion, no number (the
            # user's decision): a bare digit before the generic percussion
            # name counts the players, it doesn't number the part.
            if match.instrument.id == "percussion":
                clean_line = False
            number = _bare_leading_number(tokens, number_index, clean_line)
            if number is not None:
                # "1/2 Flute" starts two words earlier than "1 Clarinet".
                name_start = number_index - 2 if "/" in number else number_index

    # A name that says the key without printing it: "Mali klarinet" (the
    # small clarinet) is in Eb. A printed key always wins.
    key = key or match.instrument.implied_keys.get(match.alias)

    label = None
    if slash_end is not None and not same_instrument_twice:
        # The publisher's wording, e.g. "Trumpet/Cornet 1" (+ " in Bb" when the
        # key was written in front: "B: Trumpet/Cornet 1").
        # The joining word as printed: a "/" (also when OCR read it as "I"),
        # or a dash ("Bariton - Euphonium C").
        label_words = [
            tokens[index] if tokens[index] != "/" else shown.get(index, "/")
            for index in range(match.start, name_end)
        ]
        label = _calm_capitals(" ".join(label_words).replace(" / ", "/"))
        # A number written in front ("1st Bb Trumpet/Cornet") goes after the
        # names, like in the canonical name: "Trumpet/Cornet 1 in Bb". The
        # label is the part's canonical name (PartName.display), so without
        # it the 1st and 2nd Trumpet/Cornet would have the same name.
        if number and not number_after_name:
            label += f" {number}"
        # The key, unless it is printed in the label already ("... C").
        if key and not key_after_name and f"in {key}" not in label:
            label += f" in {key}"

    # The printed name may go on with more instruments: "Flutes & Oboes".
    # They don't change which part this is, only how its name is shown.
    wording_end = name_end
    while wording_end + 1 < len(tokens) and tokens[wording_end].lower() in NAME_JOINERS:
        # Only words that come RIGHT after the "&" count.
        next_start = wording_end + 1
        following = _best_instrument_match(tokens[next_start : next_start + MAX_NAME_WORDS])
        if following is None or following.start != 0:
            break
        shown.update(_shown_instrument_words(following, next_start))
        trusted = trusted and following.trusted
        # Its number and key belong to the name too: "Flute 1 & Oboe 2".
        wording_end = _read_number_and_key(tokens, next_start + following.end, None)[3]

    wording = None
    if trusted:
        shown.update(_shown_after_name(tokens, name_start, wording_end))
        wording = _printed_wording(text, positions, tokens, name_start, wording_end, shown)

    instrument = match.instrument
    matched_text = " ".join(tokens[match.start : match.end]).lower()
    return PartName(
        instrument_id=instrument.id,
        instrument_name=instrument.name,
        family=instrument.family,
        number=number,
        key=key,
        clef=clef,
        label=label,
        score=match.score,
        abbreviation=lookup_form(matched_text) in load_dictionary().abbreviations,
        wording=wording or None,  # an empty wording ("" after the cleanup) is None
    )


def find_two_line_name(first_line: str, second_line: str) -> PartName | None:
    """
    A part name printed over two lines, read as one, or None:

        "Bb BASS" / "CLARINET"          -> Bass Clarinet in Bb
        "1st Bb TENOR" / "SAXOPHONE"    -> Tenor Saxophone 1 in Bb
        "Eb TRUMPET /" / "CORNET"       -> Trumpet/Cornet in Eb

    Only when the instrument name RUNS OVER the line break, i.e. the best
    instrument match uses words of both lines. Two lines that each name an
    instrument ("Mallets" / "Xylophone") are a list, not one name: None.
    """
    # Each line is tokenised on its own, so we know exactly how many words
    # the first line has: the line break is between tokens[first_count - 1]
    # and tokens[first_count].
    first_tokens = _tokenise(normalise_text(first_line))
    first_count = len(first_tokens)
    # No _misread_slash here: only a PRINTED "/" joins two lines. The "I" in
    # "Percussion I" / "KLEINE TROM" (suite) is the number 1, not a slash
    # between two names.
    tokens = _garbled_saxophone(first_tokens + _tokenise(normalise_text(second_line)))
    match = _best_instrument_match(tokens)
    if match is None:
        return None
    # "BASS" + "CLARINET": the matched words start on the first line and end
    # on the second.
    runs_over = match.start < first_count < match.end
    # "TRUMPET /" + "CORNET": two names joined by a slash at the line break.
    if "/" in tokens:
        slash = tokens.index("/")
        at_break = slash in (first_count - 1, first_count)
        both_names = _best_instrument_match(tokens[:slash]) and _best_instrument_match(
            tokens[slash + 1 :]
        )
        if at_break and both_names:
            runs_over = True
    if not runs_over:
        return None
    return find_part_name(f"{first_line} {second_line}")


# ---------------------------------------------------------------------------
# The printed name (PartName.wording)
# ---------------------------------------------------------------------------

# Key spellings that are really printed that way, in some language: they are
# shown as printed ("klarinet in B", "Horn in Es"). Every other spelling in
# KEY_SPELLINGS is OCR noise for a flat sign ("By", "B>", "B:") and is shown
# as the canonical key ("Bb").
PRINTED_KEY_SPELLINGS = {
    "bb", "bes", "be", "b-flat", "b", "eb", "es", "e-flat", "ab", "as", "db", "des",
    "f", "c", "a", "d", "g",
    # solfège ("in Sib", "en Fa")
    "do", "ut", "re", "ré", "reb", "réb", "mib", "fa", "sol", "lab", "la", "sib",
}

# A name counts as printed in CAPITALS when at least this share of its words
# of 3 or more letters are in capitals. Shorter words don't count: a key
# ("Bb") or "in" may be printed in small letters in a capitals name
# ("TUBA in Eb", "Bb CLARINET 1").
CAPITALS_SHARE = 0.8

# Small words that are written in lowercase when capitals are calmed:
# "HORN IN F" -> "Horn in F", not "Horn In F".
SMALL_WORDS = {"in", "and"}

# Words that join two instrument names into one printed name: "Flutes & Oboes".
NAME_JOINERS = {"&", "+", "and"}


def _shown_key(printed: str, key: str) -> str:
    """
    How a key after "in" appears in the printed name: as printed when that
    is a real spelling ("in Bb", "in Es", "in B"), else the canonical key
    ("in B>" -> "in Bb").
    """
    return printed if printed.lower() in PRINTED_KEY_SPELLINGS else key


def _shown_instrument_words(match: _Match, offset: int = 0) -> dict[int, str]:
    """
    The instrument words to show differently from how they were printed:
    {token index: text}. Only an OCR fix ("Hom" -> "Horn") changes them;
    the fixed text goes on the first word, and any other matched words are
    left out (""). `offset` is added to the indexes (for a name that was
    matched in a slice of the tokens).
    """
    if match.shown_as is None:
        return {}
    shown = {offset + match.start: match.shown_as}
    for index in range(match.start + 1, match.end):
        shown[offset + index] = ""
    return shown


def _shown_after_name(tokens: list[str], start: int, end: int) -> dict[int, str]:
    """
    The words between `start` and `end` (the printed name) to show
    differently: a key after "in" in its shown spelling ("in B>" -> "in Bb"),
    and a clef ("T.C.") left out: the clef has its own column in the table.
    """
    shown: dict[int, str] = {}
    for index in range(start, end):
        word = tokens[index]
        if word.lower() in KEY_WORDS and index + 1 < end and parse_key(tokens[index + 1]):
            key_word = tokens[index + 1]
            shown[index + 1] = _shown_key(key_word, parse_key(key_word))
        elif parse_clef(word):
            shown[index] = ""
    return shown


def _printed_wording(
    text: str,
    positions: list[tuple[int, int]],
    tokens: list[str],
    start: int,
    end: int,
    shown: dict[int, str],
) -> str:
    """
    Cut the printed name (tokens start..end) out of `text`, keeping the
    brackets between the words as printed ("Baritone (Wind Band)"), with the
    words in `shown` replaced, then tidy it up (_tidy_wording).
    """
    pieces: list[str] = []
    for index in range(start, end):
        if index > start:
            # The separator: what is printed between this word and the one
            # before. Brackets are kept; other punctuation there (a comma,
            # colon, quote or "|") is OCR noise or a flat sign, and becomes
            # a space: "B, TENOR SAXOPHONE" -> "Bb Tenor Saxophone".
            separator = text[positions[index - 1][1] : positions[index][0]]
            pieces.append(re.sub(r"[^()\s]", " ", separator))
        # dict.get(key, default): the shown text, or the word itself.
        pieces.append(shown.get(index, tokens[index]))
    return _tidy_wording("".join(pieces))


def _tidy_wording(wording: str) -> str:
    """
    The last cleanup of a printed name:

        "Baritone ("        -> "Baritone"       (the clef "T.C." was left out)
        "Trumpet / Cornet"  -> "Trumpet/Cornet"
        "FLUTE 1"           -> "Flute 1"
        "piccolo"           -> "Piccolo"
    """
    wording = re.sub(r"\s+", " ", wording)
    wording = re.sub(r"\s*/\s*", "/", wording)
    wording = re.sub(r"\(\s*\)", "", wording)  # empty brackets
    # One bracket without its partner: drop an opening one at the end, close
    # any other ("Baritone (Wind" -> "Baritone (Wind)").
    wording = wording.strip()
    if wording.count("(") > wording.count(")"):
        wording = wording.rstrip("( ")
        if wording.count("(") > wording.count(")"):
            wording += ")"
    elif wording.count(")") > wording.count("("):
        wording = wording.replace(")", "", 1)
    wording = wording.strip(" ,;:&-")
    wording = _calm_capitals(wording)
    # A name starts with a capital: OCR read "piccolo" where "Piccolo" was printed.
    if wording[:1].islower():
        wording = wording[0].upper() + wording[1:]
    return wording


def _c_is_a_key(tokens: list[str], index: int) -> bool:
    """
    True when the "C" at tokens[index], in front of a name, is the key C.

    A lone capital C is often something else (a rehearsal letter in the
    music), so it only counts as a key in two places: as the first word
    ("C Baritone/Euphonium") or right after a number with a dot ("1. C
    Trombone", from "1st C Trombone", american.pdf).
    """
    if index == 0:
        return True
    word_before = tokens[index - 1]
    return word_before.endswith(".") and parse_number(word_before) is not None


def _first_of_a_leading_pair(tokens: list[str], index: int) -> str | None:
    """
    The FIRST number of a pair written in front of a name with dots:
    "1. & 2. Flute" or "1./2. Tenor Sax" -> "1", when tokens[index] is the
    "2.". None when there is no such pair.

    The first number must be smaller ("1. & 2.", never "3. & 2."), and both
    need a dot, so a page number in front isn't taken for a player.

    The solo player with a number works too: "Solo + 1. Klarinet"
    (superstar, from "SOLO + 1e KLARINET") -> "Solo".
    """
    if index < 2 or tokens[index - 1] not in ("&", "/", "+"):
        return None
    first_word, second_word = tokens[index - 2], tokens[index]
    if lookup_form(first_word) == "solo" and parse_number(second_word) is not None:
        return "Solo"
    if not first_word.endswith("."):
        return None
    first, second = parse_number(first_word), parse_number(second_word)
    if first is None or second is None or not (first.isdigit() and second.isdigit()):
        return None
    if int(first) >= int(second):
        return None
    return first


def _bare_leading_number(tokens: list[str], index: int, clean_line: bool) -> str | None:
    """
    A part number written in front of the name WITHOUT a dot, as some US
    publishers do (Sedona): "1 Clarinet in Bb", "2/3 Trumpet in Bb",
    "1/2 Flute". `index` is the position of the word just before the name
    (or before its key).

    Two forms are accepted:
      * "N/M" (tokenised as "1", "/", "2"): the Nth and Mth player. A page number
        is never written like that, so it may stand anywhere in the line.
      * a bare digit, but only as the FIRST word of a CLEAN line
        (`clean_line`: nothing but the number and the name, e.g. "1 Clarinet
        in Bb", and an exact name match). A page number printed next to the
        name is normally far enough away to be split into its own span
        (pdf_reader / ocr split at big gaps), so a digit that opens a clean
        line belongs to the name. A messy line is OCR noise: on Deep Purple
        page 64 the handwritten "1º-3º" beside "F Horn" came out as
        "2 Homf{— 39", which is NOT Horn 2.
    """
    word = tokens[index]
    if not word.isdigit():
        return None
    second = parse_number(word)
    if second is None:
        return None
    # tokens[index - 2 : index] = the two words before: "1", "/".
    if index >= 2 and tokens[index - 1] == "/":
        first = parse_number(tokens[index - 2])
        if first is not None and tokens[index - 2].isdigit() and int(first) < int(second):
            return f"{first}/{second}"
    if index == 0 and clean_line:
        return second
    return None


# A dash between two instrument names joins them like a slash: "BARITON -
# EUPHONIUM C" (superstar) is one part, and the first name is the
# instrument (the user's rule: when a part prints two names, the first wins).
NAME_DASHES = ("-", "–")


def _misread_slash(tokens: list[str]) -> list[str]:
    """
    OCR often reads the "/" between two instrument names as "I", "l" or "|":
    "BASS I ELECTRIC BASS" is really "BASS / ELECTRIC BASS" (one part that
    either instrument plays). Turn such a token back into "/", but only when
    it sits directly between two different instrument names. A dash there
    (NAME_DASHES) is read as a "/" too; find_part_name still prints it as a
    dash.
    """
    if "/" in tokens:
        return tokens
    for i, token in enumerate(tokens):
        if token not in ("I", "l", "|", *NAME_DASHES) or i == 0 or i == len(tokens) - 1:
            continue
        left = _best_instrument_match(tokens[:i])
        right = _best_instrument_match(tokens[i + 1 :])
        if (
            left is not None
            and right is not None
            and left.end == i  # the name ends right before the "I"...
            and right.start == 0  # ...and the other starts right after it
            and left.instrument.id != right.instrument.id
        ):
            # tokens[:i] + ["/"] + tokens[i+1:] = a copy with one token replaced.
            return tokens[:i] + ["/"] + tokens[i + 1 :]
    return tokens


def _calm_capitals(label: str) -> str:
    """
    A name printed in CAPITALS, written normally; other names are left alone:

        "BASS/ELECTRIC BASS" -> "Bass/Electric Bass"
        "Bb CLARINET 1"      -> "Bb Clarinet 1"   (mostly capitals)
        "HORN IN F"          -> "Horn in F"       (small words in lowercase)
        "PERCUSSION II"      -> "Percussion II"   (Roman numerals stay)
        "Bass Guitar"        -> "Bass Guitar"
    """
    # [^\W\d_]+ = a run of letters (a word without digits or punctuation).
    long_words = [word for word in re.findall(r"[^\W\d_]+", label) if len(word) >= 3]
    capitals = sum(1 for word in long_words if word.isupper())
    if not long_words or capitals < CAPITALS_SHARE * len(long_words):
        return label

    def calm(found: re.Match) -> str:
        word = found.group()
        if word.lower() in SMALL_WORDS:
            return word.lower()
        if word.isupper() and word.lower() in ROMAN_NUMERALS:
            return word  # "II" stays "II", not "Ii"
        return word.capitalize()

    # [^\W\d_]+ = a run of letters. re.sub calls calm() for each one.
    return re.sub(r"[^\W\d_]+", calm, label)


def _name_wording(tokens: list[str], match: _Match) -> str:
    """
    The original words of one instrument name: the instrument word(s), plus a
    key/number written in front ("Bb", "2.") and the number/key after it.
    """
    start = match.start
    # Up to two words in front: a key ("Bb") and/or a number ("2.").
    for _ in range(2):
        if start > 0 and (_leading_key(tokens[start - 1]) or parse_number(tokens[start - 1])):
            start -= 1
    _, _, _, end = _read_number_and_key(tokens, match.end, None)
    return " ".join(tokens[start:end])


def _read_number_and_key(
    tokens: list[str], name_end: int, slash_end: int | None
) -> tuple[str | None, str | None, str | None, int]:
    """
    Read the part number, key and clef written after an instrument name:
    "Clarinet in Bb 2", "Horn 1 in F", "Flute 1&2", "Trombone 3/Bass Trombone",
    "Baritone (T.C.)".

    `slash_end`, when given, is where a second "/"-joined instrument name
    ends; the words up to there are skipped over (but numbers in between,
    like the "3" in "Trombone 3/Bass Trombone", are still read).

    Returns (number, key, clef, index after the last word that belonged to
    the name).
    """
    number: str | None = None
    key: str | None = None
    clef: str | None = None
    i = name_end
    limit = max(name_end, slash_end or 0) + MAX_WORDS_AFTER_NAME
    while i < min(len(tokens), limit):
        word = tokens[i]
        if slash_end is not None and i < slash_end and (word == "/" or _match_candidate(word)):
            i += 1  # part of the "/"-joined second instrument name
            continue
        if word.lower() in KEY_WORDS and i + 1 < len(tokens) and parse_key(tokens[i + 1]):
            key = key or parse_key(tokens[i + 1])
            i += 2  # skip "in" / "en" and the key
            continue
        if key is None and BARE_KEY_AFTER_NAME.match(word):
            key = parse_key(word)  # "Tuba Bb", "Trumpet Bb 2"
            i += 1
            continue
        if key is None and word == "C" and i == len(tokens) - 1:
            # "FLUIT C", "HOBO C", "FAGOT C (BASSON)" (superstar): a capital
            # C as the LAST word after the name is the key C. Anywhere else
            # a lone C is more often a rehearsal letter (see BARE_KEY_AFTER_NAME).
            # (A bracket after it is a doubling, already left out here.)
            key = "C"
            i += 1
            continue
        parsed_number = parse_number(word)
        if parsed_number and number is None:
            number = parsed_number
            i += 1
            # "1/2" (tokenised as "1", "/", "2") is one number: the 1st and 2nd player.
            if i + 1 < len(tokens) and tokens[i] == "/" and parse_number(tokens[i + 1]):
                number += "/" + parse_number(tokens[i + 1])
                i += 2
            continue
        if parse_clef(word) and clef is None:
            clef = parse_clef(word)  # "(T.C.)" -> treble
            i += 1
            continue
        if slash_end is not None and i < slash_end:
            i += 1  # e.g. "Bass" of "Bass Trombone" on its own
            continue
        break  # anything else ends the part name
    return number, key, clef, max(i, slash_end or 0)


# Score keywords that can also be the job title of a PERSON. In a dedication
# like "Paul Sharle - Founder/Conductor" (printed on every first page of
# Sedona) or "John Smith, conductor", the word describes that person, not
# the page. ("Score" or "Partitura" are never a person, so they aren't here.)
PERSON_ROLE_KEYWORDS = {"conductor", "dirigent"}

# How such a role is written: joined to another title with "/" ("Founder/
# Conductor", "Conductor/Director"), or after a name and a separator
# ("Paul Sharle - Conductor", "John Smith, conductor", "Founder & Conductor",
# "music director and conductor"). {kw} is filled in with the keyword.
PERSON_ROLE_TEMPLATE = r"(?:/\s*{kw}\b|\b{kw}\s*/|(?:[,&–—-]|\band)\s*{kw}\b)"


def contains_score_keyword(text: str) -> bool:
    """
    True if the text contains a word like "Score", "Conductor", "Partitura".

    A role of a person doesn't count: "Paul Sharle - Founder/Conductor" ->
    False, "Conductor" -> True, "Conductor Score" -> True.
    """
    low = normalise_text(text).lower()
    for keyword in load_dictionary().score_keywords:
        # re.escape makes dots etc. literal; \b...\b = whole words only.
        if not re.search(rf"\b{re.escape(keyword)}\b", low):
            continue
        if keyword in PERSON_ROLE_KEYWORDS:
            role_pattern = PERSON_ROLE_TEMPLATE.format(kw=re.escape(keyword))
            if re.search(role_pattern, low):
                continue  # the job title of a person, look at the next keyword
        return True
    return False
