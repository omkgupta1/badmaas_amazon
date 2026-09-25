"""Unicode cleanup and business-name normalisation.

``normalize_name`` turns a raw business name into the fields the rest of the pipeline uses:
alias / DBA splitting, domain & handle collapse, leetspeak repair, legal-form canonicalisation,
core name, phonetic keys. Everything is country-agnostic.
"""
from __future__ import annotations

import re
import unicodedata

import jellyfish

from .seeds import LEGAL_BIT, LEGAL_SET, NAME_CANON, NAME_JUNK, NAME_STOP
from .translit import has_indic, phonetic_simplify, script_of, transliterate_text

_SPECIAL = str.maketrans({
    "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o", "ł": "l",
    "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "þ": "th", "ð": "d",
    "’": "'", "‘": "'", "´": "'", "`": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "‐": "-", "‑": "-", "·": " ", "•": " ",
})
_ZW_RE = re.compile("[­​-‏‪-‮⁠-⁤﻿]")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")


def fold_accents(s: str) -> str:
    if s.isascii():
        return s
    nk = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nk if not unicodedata.combining(ch))


def clean_text(s: str, indic_lexicon: dict | None = None) -> str:
    """NFKC, drop zero-width/control chars, transliterate Indic words, lowercase, fold accents."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _ZW_RE.sub("", s)
    s = _CTRL_RE.sub(" ", s)
    if has_indic(s):
        s = transliterate_text(s, indic_lexicon)
    s = fold_accents(s.translate(_SPECIAL).lower())
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------------------------
# name variants: "X doing business as Y" / "X formerly Y" / "X | www.x.com"
# ---------------------------------------------------------------------------------------------
_ALIAS_RE = re.compile(
    r"\s*\b(?:doing\s+business\s+as|d\s*/\s*b\s*/\s*a|d\.\s*b\.\s*a|dba|formerly\s+known\s+as"
    r"|formerly|f\s*/\s*k\s*/\s*a|f\.\s*k\.\s*a|fka|also\s+known\s+as|a\s*/\s*k\s*/\s*a"
    r"|a\.\s*k\.\s*a|aka|trading\s+as|t\s*/\s*a|previously)\b\.?\s*"
)
_DOMAIN_RE = re.compile(
    r"^(?:https?\s*:\s*//)?\s*(?:www\s*\.\s*)?([a-z0-9][a-z0-9\-]*?)\s*\.\s*"
    r"(?:com|net|org|co\.in|org\.in|net\.in|in|co\.uk|uk|co|fr|io|biz|us|info)\s*/?$"
)
_HANDLE_RE = re.compile(r"^[@#]\s*([a-z0-9_.\-]+)$")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def split_name_variants(s: str) -> tuple[str, str, bool]:
    """Return ``(main, other, has_alias)`` for a cleaned lower-case name.

    For "X doing business as Y" the operating name Y is ``main``; for "X | www.x.com" the text
    before the pipe is ``main``.
    """
    m = _ALIAS_RE.search(s)
    if m and m.start() > 0 and m.end() < len(s):
        return s[m.end():].strip(), s[:m.start()].strip(), True
    if "|" in s:
        before, _, after = s.partition("|")
        before, after = before.strip(), after.strip()
        if before:
            return before, after, True
        return after, "", True
    return s, "", False


def domain_core(s: str) -> str | None:
    """"bastmartipa.com" -> "bastmartipa"; "@kadamba" -> "kadamba"; "pediatricspecialistscom"."""
    t = s.strip()
    if not t:
        return None
    m = _DOMAIN_RE.match(t)
    if m:
        return _NON_ALNUM_RE.sub("", m.group(1)) or None
    m = _HANDLE_RE.match(t)
    if m:
        return _NON_ALNUM_RE.sub("", m.group(1)) or None
    if " " not in t and len(t) >= 10 and t.isalnum():
        if t.startswith("www"):
            t = t[3:]
        if t.endswith("com"):
            return t[:-3]
    return None


# ---------------------------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------------------------
_PVT_P_RE = re.compile(r"\(\s*p\s*\)")
# "p.a." / "l.l.c." / "s.a.s" / "p. c." -> "pa" / "llc" / "sas" / "pc"
_DOTTED_RE = re.compile(r"(?<![a-z])(?:[a-z]\.\s?){2,}[a-z]?(?![a-z])")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def fix_leet(tok: str) -> str:
    """"ec0nomic" -> "economic", but keep "1st", "b2b", "4608", "k2"."""
    if tok.isalpha() or tok.isdigit() or _ORDINAL_RE.match(tok):
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    n_digit = len(tok) - n_alpha
    if n_alpha >= 3 and 0 < n_digit <= 2 and n_alpha >= 2 * n_digit and all(
            c in "013457" for c in tok if c.isdigit()):
        return tok.translate(_LEET)
    return tok


def tokenize_name(s: str) -> list[str]:
    if not s:
        return []
    s = _PVT_P_RE.sub(" pvt ", s)
    s = _DOTTED_RE.sub(lambda m: m.group(0).replace(".", "").replace(" ", "") + " ", s)
    s = s.replace("'", "").replace("&", " and ").replace("+", " and ").replace("@", "a")
    s = _NON_ALNUM_RE.sub(" ", s)
    toks: list[str] = []
    prev = None
    for t in s.split():
        t = fix_leet(t)
        t = NAME_CANON.get(t, t)
        if t in NAME_JUNK or t == prev:
            continue
        toks.append(t)
        prev = t
    return toks


def legal_mask(tokens: list[str]) -> int:
    mask = 0
    for t in tokens:
        mask |= LEGAL_BIT.get(t, 0)
    return mask


def core_tokens(tokens: list[str]) -> list[str]:
    core = [t for t in tokens if t not in LEGAL_SET and t not in NAME_STOP]
    return core or [t for t in tokens if t not in NAME_STOP] or tokens


def _metaphone(tok: str) -> str:
    try:
        return jellyfish.metaphone(tok)
    except Exception:  # pragma: no cover - defensive, jellyfish handles ASCII
        return tok


NAME_FIELDS = [
    "name_norm", "name_core", "name_concat", "name_sorted", "name_other", "name_nums",
    "name_phon", "name_simpl", "legal_mask", "is_domain", "has_alias", "name_script",
    "n_name_tok",
]


def normalize_name(raw: str, indic_lexicon: dict | None = None) -> tuple:
    """Parse one business name into the ``NAME_FIELDS`` tuple."""
    script = script_of(raw)
    s = clean_text(raw, indic_lexicon)
    main, other, has_alias = split_name_variants(s)

    dom = domain_core(main)
    is_domain = dom is not None
    main_toks = [dom] if dom else tokenize_name(main)
    if other:
        dom_o = domain_core(other)
        other_toks = [dom_o] if dom_o else tokenize_name(other)
    else:
        other_toks = []
    if not main_toks and other_toks:
        main_toks, other_toks = other_toks, []

    core = core_tokens(main_toks)
    nums = sorted({t.lstrip("0") or "0" for t in core if t.isdigit()})
    phon = sorted({_metaphone(t) for t in core if t.isalpha()} - {""})
    return (
        " ".join(main_toks),
        " ".join(core),
        "".join(main_toks),
        " ".join(sorted(set(core))),
        " ".join(other_toks),
        " ".join(nums),
        " ".join(phon),
        phonetic_simplify(" ".join(core)),
        legal_mask(main_toks + other_toks),
        int(is_domain),
        int(has_alias),
        script,
        len(main_toks),
    )
