"""Data-driven lexicons (only the provided data is used).

1. Indic lexicon  — native-script word -> Latin word, aligned on training matches:
   "प्राइवेट" -> "private", "लिमिटेड" -> "limited", "महाराष्ट्र" -> "maharashtra".
2. State lexicon  — per country, any address component -> canonical state key. Built from the
   components that end S1 addresses, a small seed table (chosen per country by overlap, so no
   country name is hard-coded) and synonyms mined from aligned pairs ("texas" -> "tx",
   "mh" -> "maharashtra", "nord" -> "hauts de france").
3. Abbreviations  — per country, token -> expansion mined from aligned pairs ("av" -> "avenue",
   "r" -> "rue"), where the short form is a subsequence of the long one.

Aligned pairs are ground-truth matches (train only) and, for every split, exact-key
pseudo-pairs: S1 and S2/S3 records sharing (country, sorted core name, house number) with a
unique S1 — which is what lets the unseen test country adapt without labels.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler

from .seeds import INDIA_STATES, US_STATES
from .text_norm import fold_accents
from .translit import INDIC_WORD_RE, phonetic_simplify, translit_indic
from .utils import LOG

_ZW_RE = re.compile("[­​-‏‪-‮⁠-⁤﻿]")
_LATIN_TOK_RE = re.compile(r"[a-z0-9]+")
_CONTEXT_TOKENS = {"st", "ste", "dr", "r", "all", "ch", "ter"}


def _native_clean(s: str) -> str:
    return _ZW_RE.sub("", unicodedata.normalize("NFKC", s or ""))


@lru_cache(maxsize=500_000)
def _native_key(word: str) -> str:
    return phonetic_simplify(translit_indic(word))


def mine_indic_lexicon(b_texts: list[str], a_texts: list[str], min_count: int = 3,
                       min_share: float = 0.5, min_sim: float = 0.55) -> dict[str, str]:
    """Align native-script words of ``b_texts`` with Latin words of the matching ``a_texts``."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for b, a in zip(b_texts, a_texts):
        b = _native_clean(b)
        words = INDIC_WORD_RE.findall(b)
        if not words:
            continue
        a_toks = [t for t in _LATIN_TOK_RE.findall(fold_accents(a.lower())) if len(t) >= 2
                  and not t.isdigit()]
        if not a_toks:
            continue
        a_keys = [phonetic_simplify(t) for t in a_toks]
        for w in set(words):
            key = _native_key(w)
            if not key:
                continue
            best, best_sim = None, 0.0
            for tok, tk in zip(a_toks, a_keys):
                sim = JaroWinkler.normalized_similarity(key, tk)
                if sim > best_sim:
                    best, best_sim = tok, sim
            if best is not None and best_sim >= min_sim:
                counts[w][best] += 1
    lexicon: dict[str, str] = {}
    for w, ctr in counts.items():
        tok, c = ctr.most_common(1)[0]
        if c >= min_count and c / sum(ctr.values()) >= min_share:
            lexicon[w] = tok
    LOG.info("indic lexicon: %d native words mapped (from %d candidates)", len(lexicon), len(counts))
    return lexicon


# ---------------------------------------------------------------------------------------------
# states and abbreviations
# ---------------------------------------------------------------------------------------------

def _has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def state_vocab(s1_lite: pl.DataFrame, min_count: int) -> dict[str, dict[str, str]]:
    """Per country: components that typically END S1 addresses -> canonical state key.

    A component qualifies when it ends >= ``min_count`` S1 addresses and at least half of its
    occurrences are at the end (so cities that sometimes end an address are excluded).
    A seed table is attached to a country only when most of its end-components are seed keys.
    """
    out: dict[str, dict[str, str]] = {}
    for country, grp in s1_lite.group_by("country"):
        country = country[0] if isinstance(country, tuple) else country
        n_last: Counter = Counter()
        n_any: Counter = Counter()
        for comps in grp["comps"].to_list():
            if not comps:
                continue
            parts = comps.split("|")
            for c in set(parts):
                n_any[c] += 1
            if parts[-1] and not _has_digit(parts[-1]):
                n_last[parts[-1]] += 1
        vocab = {c for c, n in n_last.items() if n >= min_count and n / n_any[c] >= 0.5}
        mapping = {c: c for c in vocab}
        for seed in (US_STATES, INDIA_STATES):
            seed_keys = set(seed) | set(seed.values())
            hits = sum(n_last[c] for c in vocab if c in seed_keys)
            total = sum(n_last[c] for c in vocab) or 1
            if hits / total >= 0.6:
                canon = {c: seed.get(c, c) for c in vocab if c in seed_keys}
                mapping.update(canon)
                for k, v in seed.items():
                    mapping.setdefault(k, v)
                    mapping.setdefault(v, v)
        out[country] = mapping
        LOG.info("state vocab [%s]: %d end-components -> %d canonical states", country,
                 len(vocab), len(set(mapping.values())))
    return out


def _is_abbrev(short: str, long: str) -> bool:
    if len(short) >= len(long) or len(short) > 5 or short[0] != long[0]:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def mine_address_lexicon(a_comps: list[str], b_comps: list[str], countries: list[str],
                         states: dict[str, dict[str, str]], s1_comp_counts: dict[str, Counter],
                         cfg: dict) -> dict[str, dict[str, dict[str, str]]]:
    """Mine state synonyms and abbreviations from aligned (S1, S2/S3) component strings."""
    p = cfg["prep"]["address_lexicon"]
    st_cnt: dict[tuple, Counter] = defaultdict(Counter)
    ab_cnt: dict[tuple, Counter] = defaultdict(Counter)
    ab_tot: Counter = Counter()
    for ac, bc, country in zip(a_comps, b_comps, countries):
        if not ac or not bc:
            continue
        a_parts = ac.split("|")
        b_parts = bc.split("|")
        smap = states.get(country, {})
        # ---- state synonyms
        a_state = None
        if a_parts[-1] in smap:
            a_state = smap[a_parts[-1]]
        elif a_parts[0] in smap:
            a_state = smap[a_parts[0]]
        if a_state is not None and not any(smap.get(x) == a_state for x in b_parts):
            a_set = set(a_parts)
            free = [x for x in b_parts if x and not _has_digit(x)]
            ends = {free[0], free[-1]} if free else set()
            for c in ends:
                if c not in a_set and c not in smap:
                    st_cnt[(country, c)][a_state] += 1
        # ---- abbreviations
        a_toks = {t for x in a_parts for t in x.split() if t.isalpha()}
        b_toks = {t for x in b_parts for t in x.split() if t.isalpha()}
        a_only = a_toks - b_toks
        for bt in b_toks - a_toks:
            if bt in _CONTEXT_TOKENS:
                continue
            ab_tot[(country, bt)] += 1
            for at in a_only:
                if _is_abbrev(bt, at):
                    ab_cnt[(country, bt)][at] += 1

    lex: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: {"state": {}, "abbr": {}})
    for (country, c), ctr in st_cnt.items():
        state, n = ctr.most_common(1)[0]
        tot = sum(ctr.values())
        if (n >= p["state_min_count"] and n / tot >= p["state_min_share"]
                and s1_comp_counts.get(country, Counter()).get(c, 0) < 5):
            lex[country]["state"][c] = state
    for (country, bt), ctr in ab_cnt.items():
        at, n = ctr.most_common(1)[0]
        if n >= p["abbr_min_count"] and n / ab_tot[(country, bt)] >= p["abbr_min_share"]:
            lex[country]["abbr"][bt] = at
    for country, d in lex.items():
        LOG.info("address lexicon [%s]: %d state synonyms, %d abbreviations", country,
                 len(d["state"]), len(d["abbr"]))
    return dict(lex)


def s1_component_counts(s1_lite: pl.DataFrame) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    for country, grp in s1_lite.group_by("country"):
        country = country[0] if isinstance(country, tuple) else country
        ctr: Counter = Counter()
        for comps in grp["comps"].to_list():
            if comps:
                ctr.update(set(comps.split("|")))
        out[country] = ctr
    return out


def pseudo_pairs(s1_lite: pl.DataFrame, q_lite: pl.DataFrame, max_pairs: int,
                 seed: int = 0) -> pl.DataFrame:
    """Exact-key pseudo-matches: same (country, sorted core name, house number), unique S1."""
    keys = ["country", "name_sorted", "hn"]
    s1k = s1_lite.filter((pl.col("name_sorted") != "") & (pl.col("hn") != "")).select(
        ["idx"] + keys)
    uniq = s1k.group_by(keys).agg(pl.len().alias("n")).filter(pl.col("n") == 1).drop("n")
    s1k = s1k.join(uniq, on=keys, how="semi")
    qk = q_lite.filter((pl.col("name_sorted") != "") & (pl.col("hn") != "")).select(
        ["idx"] + keys)
    pairs = qk.join(s1k, on=keys, how="inner", suffix="_s").select(
        pl.col("idx_s").alias("s"), pl.col("idx").alias("q"))
    if pairs.height > max_pairs:
        pairs = pairs.sample(n=max_pairs, seed=seed)
    LOG.info("pseudo-pairs: %d", pairs.height)
    return pairs


def merge_address_lexicons(primary: dict, secondary: dict) -> dict:
    """Union per country; ``primary`` wins on conflicts."""
    out: dict = {}
    for country in set(primary) | set(secondary):
        a = primary.get(country, {"state": {}, "abbr": {}})
        b = secondary.get(country, {"state": {}, "abbr": {}})
        out[country] = {"state": {**b.get("state", {}), **a.get("state", {})},
                        "abbr": {**b.get("abbr", {}), **a.get("abbr", {})}}
    return out


def sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))
