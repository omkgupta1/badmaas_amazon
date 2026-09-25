"""Dependency-free transliteration of the nine major Indic scripts to phonetic Latin.

The Unicode blocks of Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada
and Malayalam share one layout (ISCII heritage): the same offset inside each 128-code-point
block is the same phonetic unit. One offset table therefore covers all nine scripts, plus a
few per-script overrides. Consonants carry an inherent "a" that a vowel sign replaces and a
virama removes; for Indo-Aryan scripts the word-final inherent "a" is dropped (schwa
deletion: रेड -> "red", लिमिटेड -> "limited"), for Dravidian scripts it is kept
(ಕರ್ನಾಟಕ -> "karnaataka").

Frequent words are better served by the lexicon mined from training matches
(``lexicon_mining.mine_indic_lexicon``); this module is the fallback for everything else.
"""
from __future__ import annotations

import re

INDIC_CHAR_RE = re.compile(r"[ऀ-ൿ]")
INDIC_WORD_RE = re.compile(r"[ऀ-ൿ]+")

# (script id, name, block base, indo-aryan?)
SCRIPTS = [
    (1, "devanagari", 0x0900, True),
    (2, "bengali", 0x0980, True),
    (3, "gurmukhi", 0x0A00, True),
    (4, "gujarati", 0x0A80, True),
    (5, "oriya", 0x0B00, True),
    (6, "tamil", 0x0B80, False),
    (7, "telugu", 0x0C00, False),
    (8, "kannada", 0x0C80, False),
    (9, "malayalam", 0x0D00, False),
]

_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "rii", 0x61: "lii",
}
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_MATRAS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "rii",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x62: "li", 0x63: "lii",
}
_SIGNS = {
    0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h", 0x3D: "", 0x50: "om",
    0x51: "", 0x52: "", 0x53: "", 0x54: "", 0x55: "", 0x56: "", 0x57: "",
    0x64: " ", 0x65: " ", 0x70: "", 0x71: "",
}
# per-script differences: (type, latin); types: C consonant (inherent a), F final consonant
# (no inherent vowel), S sign, M vowel sign, V vowel, X ignore
_OVERRIDES = {
    "gurmukhi": {0x70: ("S", "n"), 0x71: ("X", ""), 0x72: ("X", ""), 0x73: ("X", ""),
                 0x74: ("S", "ek onkar"), 0x75: ("X", "")},
    "bengali": {0x4E: ("F", "t"), 0x70: ("C", "r"), 0x71: ("C", "v"), 0x57: ("X", "")},
    "oriya": {0x71: ("C", "v"), 0x57: ("X", "")},
    "telugu": {0x58: ("C", "ts"), 0x59: ("C", "dz"), 0x5A: ("C", "r")},
    "kannada": {0x5E: ("C", "l")},
    "malayalam": {0x4E: ("F", "r"), 0x54: ("F", "m"), 0x55: ("F", "y"), 0x56: ("F", "l"),
                  0x7A: ("F", "n"), 0x7B: ("F", "n"), 0x7C: ("F", "r"), 0x7D: ("F", "l"),
                  0x7E: ("F", "l"), 0x7F: ("F", "k")},
}
_NUKTA = {"k": "q", "g": "g", "j": "z", "d": "r", "dh": "rh", "ph": "f", "kh": "kh", "y": "y"}


def _build_table() -> dict[str, tuple[str, str, bool, int]]:
    table: dict[str, tuple[str, str, bool, int]] = {}
    for sid, name, base, indo in SCRIPTS:
        overrides = _OVERRIDES.get(name, {})
        for off in range(0x80):
            if off in overrides:
                typ, lat = overrides[off]
            elif off in _CONSONANTS:
                typ, lat = "C", _CONSONANTS[off]
            elif off in _MATRAS:
                typ, lat = "M", _MATRAS[off]
            elif off == 0x4D:
                typ, lat = "H", ""  # virama / halant
            elif off == 0x3C:
                typ, lat = "N", ""  # nukta
            elif off in _VOWELS:
                typ, lat = "V", _VOWELS[off]
            elif 0x66 <= off <= 0x6F:
                typ, lat = "D", str(off - 0x66)
            elif off in _SIGNS:
                typ, lat = "S", _SIGNS[off]
            else:
                typ, lat = "X", ""
            table[chr(base + off)] = (typ, lat, indo, sid)
    return table


_TABLE = _build_table()


def has_indic(text: str) -> bool:
    return bool(text) and INDIC_CHAR_RE.search(text) is not None


def script_of(text: str) -> int:
    """Dominant Indic script id of ``text`` (0 when there is no Indic character)."""
    if not text or not has_indic(text):
        return 0
    counts: dict[int, int] = {}
    for ch in text:
        info = _TABLE.get(ch)
        if info is not None:
            counts[info[3]] = counts.get(info[3], 0) + 1
    return max(counts, key=counts.get) if counts else 0


def translit_indic(text: str) -> str:
    """Transliterate a string that may mix Indic and other characters."""
    out: list[str] = []
    pending = False          # an inherent "a" is waiting after the last consonant
    sylls = 0                # aksharas in the current word (for final-schwa deletion)
    indo = True

    def boundary() -> None:
        nonlocal pending, sylls
        if pending and not (indo and sylls > 1):
            out.append("a")
        pending = False
        sylls = 0

    for ch in text:
        info = _TABLE.get(ch)
        if info is None:
            boundary()
            out.append(ch)
            continue
        typ, lat, indo, _ = info
        if typ == "C":
            if pending:
                out.append("a")
            out.append(lat)
            pending = True
            sylls += 1
        elif typ == "F":
            if pending:
                out.append("a")
            out.append(lat)
            pending = False
            sylls += 1
        elif typ == "M":
            out.append(lat)
            pending = False
        elif typ == "H":
            pending = False
        elif typ == "N":
            if out:
                out[-1] = _NUKTA.get(out[-1], out[-1])
        elif typ == "V":
            if pending:
                out.append("a")
                pending = False
            out.append(lat)
            sylls += 1
        elif typ == "D":
            boundary()
            out.append(lat)
        elif typ == "S":
            if pending:
                out.append("a")
                pending = False
            out.append(lat)
    boundary()
    return "".join(out)


def transliterate_text(text: str, lexicon: dict[str, str] | None = None) -> str:
    """Replace every Indic word by its lexicon entry, else by rule-based transliteration."""
    lex = lexicon or {}

    def repl(m: re.Match) -> str:
        word = m.group(0)
        return " " + (lex.get(word) or translit_indic(word)) + " "

    return INDIC_WORD_RE.sub(repl, text)


_REPEAT_RE = re.compile(r"(.)\1+")


def phonetic_simplify(s: str) -> str:
    """Collapse spelling variation typical of romanised Indian (and generic) names.

    raibareilly / raebareli -> "raibareli"; sharma / sharmaa -> "sarma"; builders / bildars
    -> "bilders" / "bildars". Used for fuzzy features and for lexicon alignment.
    """
    if not s:
        return ""
    s = s.lower()
    s = s.replace("chh", "\x01").replace("ch", "\x01").replace("sh", "\x02").replace("ph", "f")
    s = s.replace("ck", "k").replace("c", "k").replace("q", "k").replace("x", "ks")
    s = s.replace("z", "j").replace("w", "v")
    s = s.replace("\x01", "c").replace("\x02", "s")
    s = s.replace("th", "t").replace("dh", "d").replace("kh", "k").replace("gh", "g")
    s = s.replace("bh", "b").replace("jh", "j")
    s = s.replace("ee", "i").replace("oo", "u").replace("ou", "u").replace("y", "i")
    s = s.replace("ae", "ai").replace("ei", "e")  # bareilly / bareli, sheikh / shekh
    return _REPEAT_RE.sub(r"\1", s)
