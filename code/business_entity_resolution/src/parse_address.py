"""Address parsing into canonical tokens, house number(s), unit, street, city, state, landmarks.

Components are reordered freely across sources ("OH, Columbus, 5559 Orville Avenue"), so the
parser works per comma-separated component and never relies on component position except for
the state (first/last component). House-number handling is the core of decoy detection:

* labelled numbers win ("H.No.16-11-23/37/A", "Door No D/1", "Plot No 1", "No. 5");
* then a component that starts with a number followed by street words ("1795 Westchester Dr");
* then a bare number component ("2505"), then the first standalone number;
* "##34" / "(41)" / "0010413" / "274-278" / "19 1/2" / "5 bis" are normalised.

Mapping-derived fields (state) are empty when the value cannot be mapped — the feature layer
turns that into NaN, never into a mismatch.
"""
from __future__ import annotations

import re

from .seeds import (ADDR_JUNK, CAREOF_WORDS, DIRECTIONS, FRENCH_ARTICLES, HOUSE_SUFFIXES,
                    LANDMARK_WORDS, OTHER_ADDR, PO_WORDS, STREET_TYPE_SET, STREET_TYPES,
                    UNIT_WORDS)
from .text_norm import clean_text
from .translit import script_of

_CAREOF_RE = re.compile(r"\b([csdw])\s*/\s*o\b")
_CAREOF_MAP = {"c": "careof", "s": "sonof", "d": "daughterof", "w": "wifeof"}
_PO_BOX_RE = re.compile(r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s+office\s+box)\b")
_FRACTION_RE = re.compile(r"(?<=\d)\s+[13]\s*/\s*[24]\b")
_BRACKETS_RE = re.compile(r"[()\[\]{}<>\"*|]")
_SPLIT_RE = re.compile(r"[,;]")
# continuation of a house id: "-a", "-4", "/37", "/a", "-e1" — never a whole word ("2260- housecreek")
_ID_CONT = r"(?:\s*[\-/]\s*(?:\d+[a-z]?|[a-z]{1,2}\d*)(?![a-z]))*"
# digits that are not an ordinal ("12th") and cannot be shortened by backtracking
_ID_DIGITS = r"\d+(?![0-9])(?!(?:st|nd|rd|th)(?![0-9a-z]))"
_LABEL_RE = re.compile(
    r"(?<![0-9a-z])(?:h\s*no|house\s*no|hno|hn|door\s*no|d\s*no|dno|plot\s*no|plot|flat\s*no|"
    r"shop\s*no|office\s*no|sy\s*no|survey\s*no|s\s*no|kh\s*no|khasra\s*no|ward\s*no|gali\s*no|"
    r"no|number)\s*[:\-]?\s*([a-z]{0,2}(?:\s*[\-/]\s*)?" + _ID_DIGITS + r"[a-z]?" + _ID_CONT + r")"
)
_LEAD_RE = re.compile(
    r"^([a-z]{0,2}(?:\s*[\-/]\s*)?" + _ID_DIGITS + r"[a-z]?(?![a-z])" + _ID_CONT + r")"
    r"(?:\s+(bis|ter|quater|[a-d])(?![0-9a-z]))?"
)
_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_NUM_RE = re.compile(r"(?<![0-9a-z])\d+(?![0-9a-z])")
_DIGITS_RE = re.compile(r"\d+")
_TOKEN_RE = re.compile(r"[0-9a-z]+")
_ALNUM_RE = re.compile(r"[^0-9a-z]+")

LABEL_WORDS = {"h", "house", "hno", "hn", "door", "d", "dno", "plot", "flat", "shop", "office",
               "sy", "survey", "s", "kh", "khasra", "ward", "gali", "no", "number"}
_CONTEXT_TOKENS = {"st", "ste", "dr", "r", "all", "ch", "ter"}
_STRUCTURAL = (set(STREET_TYPES) | set(DIRECTIONS) | UNIT_WORDS | {"no", "pobox"}
               | STREET_TYPE_SET)
_SUFFIX_CANON = {"b": "bis", "t": "ter"}

ADDR_FIELDS = [
    "addr_norm", "addr_block", "hn", "hn_int", "hn_hi", "hid", "nums", "unit", "has_po",
    "street", "street_type", "city", "state", "loc", "careof", "landmark", "addr_empty",
    "n_comps", "addr_script", "comps",
]
EMPTY_ADDR = ("", "", "", -1, -1, "", "", "", 0, "", "", "", "", "", "", "", 1, 0, 0, "")


def _strip_zeros(d: str) -> str:
    return d.lstrip("0") or "0"


def canon_tokens(toks: list[str], abbr: dict | None = None) -> list[str]:
    """Canonicalise address tokens with a little context (st = street | saint, r = rue ...)."""
    out: list[str] = []
    n = len(toks)
    if n == 1:
        # a lone token is a city / state ("CT", "FL", "NE", "MT"), never "court" / "floor" ...
        t = toks[0]
        if abbr and t in abbr and len(t) > 2:
            t = abbr[t]
        t = _strip_zeros(t) if t.isdigit() else t
        return [] if t in ADDR_JUNK else [t]
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else ""
        prv = toks[i - 1] if i > 0 else ""
        if abbr and t not in _CONTEXT_TOKENS and t in abbr:
            t = abbr[t]
        followed_by_name = nxt.isalpha() and nxt not in _STRUCTURAL
        if t == "st":
            t = "saint" if followed_by_name else "street"
        elif t == "ste":
            t = "sainte" if followed_by_name else "suite"
        elif t == "dr":
            t = "doctor" if followed_by_name else "drive"
        elif t == "r":
            t = "rue" if (nxt in FRENCH_ARTICLES or (prv.isdigit() and nxt.isalpha())) else "r"
        elif t == "all":
            t = "allee" if nxt in FRENCH_ARTICLES else "all"
        elif t == "ch":
            t = "chemin" if nxt in FRENCH_ARTICLES else "ch"
        elif t == "ter":
            t = "ter" if prv.isdigit() else "terrace"
        elif t in STREET_TYPES:
            t = STREET_TYPES[t]
        elif t in DIRECTIONS:
            t = DIRECTIONS[t]
        elif t in OTHER_ADDR:
            t = OTHER_ADDR[t]
        elif t.isdigit():
            t = _strip_zeros(t)
        if t in ADDR_JUNK or (out and out[-1] == t):
            continue
        out.append(t)
    return out


def _is_unit_comp(toks: list[str]) -> bool:
    return bool(toks) and toks[0] in UNIT_WORDS and (len(toks) > 1 or toks[0] in PO_WORDS)


def _house_candidate(comp: str) -> tuple[int, str, str] | None:
    """Best house-number candidate inside one component: (priority, id_string, suffix)."""
    m = _LABEL_RE.search(comp)
    if m:
        return 0, m.group(1).strip(" -/"), ""
    m = _LEAD_RE.match(comp)
    if m:
        rest = comp[m.end():]
        has_words = any(len(t) >= 2 and t.isalpha() for t in _TOKEN_RE.findall(rest))
        suffix = m.group(2) or ""
        return (1 if has_words else 2), m.group(1).strip(" -/"), suffix
    m = _NUM_RE.search(comp)
    if m:
        return 3, m.group(0), ""
    return None


def find_state(comp_strs: list[str], state_map: dict | None) -> tuple[str, int]:
    """Map the last / first / any other digit-free component to a canonical state key."""
    if not state_map or not comp_strs:
        return "", -1
    n = len(comp_strs)
    order = [n - 1] + ([0] if n > 1 else []) + list(range(n - 2, 0, -1))
    for i in order:
        c = comp_strs[i]
        if c and c in state_map:
            return state_map[c], i
    return "", -1


def parse_address(raw: str, indic_lexicon: dict | None = None, state_map: dict | None = None,
                  abbr_map: dict | None = None) -> tuple:
    """Parse one address into the ``ADDR_FIELDS`` tuple."""
    s = clean_text(raw, indic_lexicon)
    if not s:
        return EMPTY_ADDR
    script = script_of(raw)
    s = _CAREOF_RE.sub(lambda m: f" {_CAREOF_MAP[m.group(1)]} ", s)
    s = _PO_BOX_RE.sub(" pobox ", s)
    s = _FRACTION_RE.sub(" ", s)
    s = s.replace("#", " ").replace(".", " ").replace("'", "")
    s = _BRACKETS_RE.sub(" ", s)
    comps_raw = [" ".join(c.split()) for c in _SPLIT_RE.split(s)]
    comps_raw = [c for c in comps_raw if c and c not in ADDR_JUNK]
    if not comps_raw:
        return EMPTY_ADDR

    comp_toks = [canon_tokens(_TOKEN_RE.findall(c), abbr_map) for c in comps_raw]
    comp_strs = [" ".join(t) for t in comp_toks]
    unit_flags = [_is_unit_comp(t) for t in comp_toks]

    # ---- house number ------------------------------------------------------------------
    best = None  # (priority, comp_index, id_string, suffix)
    for i, comp in enumerate(comps_raw):
        if unit_flags[i]:
            continue
        cand = _house_candidate(comp)
        if cand is not None and (best is None or cand[0] < best[0]):
            best = (cand[0], i, cand[1], cand[2])
    hn, hn_int, hn_hi, hid, street_idx = "", -1, -1, "", -1
    id_tokens: set[str] = set()
    if best is not None:
        _, street_idx, id_str, suffix = best
        digits = _DIGITS_RE.findall(id_str)
        if digits:
            hn = _strip_zeros(digits[0])
            hn_int = int(hn) if len(hn) <= 12 else -1
            rm = _RANGE_RE.match(id_str.replace(" ", ""))
            if rm:
                lo, hi = int(rm.group(1)), int(rm.group(2))
                if 0 < hi - lo <= 100:
                    hn_hi = hi
            suffix = _SUFFIX_CANON.get(suffix, suffix)
            hid = _ALNUM_RE.sub("", id_str) + suffix
            hid = re.sub(r"(?<![0-9])0+(?=\d)", "", hid)
            id_tokens = {(_strip_zeros(t) if t.isdigit() else t) for t in _TOKEN_RE.findall(id_str)}
            if suffix:
                id_tokens.add(suffix)

    # ---- street -------------------------------------------------------------------------
    def street_tokens(toks: list[str], drop: set[str]) -> tuple[list[str], str]:
        st_type = ""
        kept: list[str] = []
        for t in toks:
            if t in UNIT_WORDS and t not in STREET_TYPE_SET:
                break  # "... drive apt 4" -> stop at the unit
            if t in drop or t in LABEL_WORDS or t.isdigit() or t in HOUSE_SUFFIXES:
                continue
            if t in STREET_TYPE_SET:
                st_type = st_type or t
                continue
            if t in DIRECTIONS.values():
                continue
            kept.append(t)
        return kept, st_type

    street, street_type = "", ""
    if street_idx >= 0:
        kept, street_type = street_tokens(comp_toks[street_idx], id_tokens)
        if not kept and not street_type:
            street_idx = -1
        else:
            street = " ".join(kept)
    if street_idx < 0:
        for i, toks in enumerate(comp_toks):
            if not unit_flags[i] and any(t in STREET_TYPE_SET for t in toks):
                kept, street_type = street_tokens(toks, id_tokens)
                street, street_idx = " ".join(kept), i
                break

    # ---- state / city / locality -------------------------------------------------------
    state, state_idx = find_state(comp_strs, state_map)
    city = ""
    cands = [i for i, c in enumerate(comp_strs)
             if c and i != street_idx and i != state_idx and not unit_flags[i]
             and not any(ch.isdigit() for ch in c)]
    if cands:
        if state_idx >= 0:
            before = [i for i in cands if i < state_idx]
            city = comp_strs[max(before)] if before else comp_strs[min(cands)]
        else:
            city = comp_strs[cands[-1]]
    loc_toks: list[str] = []
    for i, toks in enumerate(comp_toks):
        if i in (street_idx, state_idx) or unit_flags[i]:
            continue
        loc_toks.extend(t for t in toks if not t.isdigit() and t not in LABEL_WORDS)

    # ---- unit / PO / landmarks / care-of ------------------------------------------------
    unit_parts: list[str] = []
    has_po = 0
    careof: list[str] = []
    landmark: list[str] = []
    for i, toks in enumerate(comp_toks):
        if unit_flags[i]:
            if toks[0] in PO_WORDS:
                has_po = 1
            else:
                unit_parts.append("".join(t for t in toks if t not in UNIT_WORDS))
        mode = None
        for t in toks:
            if t in CAREOF_WORDS:
                mode = "c"
                continue
            if t in LANDMARK_WORDS:
                mode = "l"
                continue
            if mode == "c":
                careof.append(t)
            elif mode == "l":
                landmark.append(t)

    all_toks = [t for toks in comp_toks for t in toks]
    nums = sorted({_strip_zeros(d) for d in _NUM_RE.findall(" ".join(comps_raw))},
                  key=lambda x: (len(x), x))
    block = [t for i, toks in enumerate(comp_toks) if not unit_flags[i] for t in toks
             if t not in LABEL_WORDS]
    if hn:
        block.append("hn" + hn)
    if street:
        block.append("st_" + street.replace(" ", "_"))

    return (
        " ".join(all_toks),
        " ".join(block),
        hn,
        hn_int,
        hn_hi,
        hid,
        " ".join(nums),
        " ".join(p for p in unit_parts if p),
        has_po,
        street,
        street_type,
        city,
        state,
        " ".join(loc_toks),
        " ".join(careof),
        " ".join(landmark),
        0,
        len(comps_raw),
        script,
        "|".join(comp_strs),
    )
