"""Stage 3 — pairwise features for every candidate pair (vectorised, one cand part at a time).

a = the S1 record, b = the S2/S3 record. Groups:
  name     fuzzy scores on canonical / core / space-free / phonetic / alias forms, IDF-weighted
           token overlap, extra & missing tokens, legal forms, numbers inside the name
  hn       house-number suite: equality, digit edit distances, |delta|, prefix/suffix, first/last
           digit, single-digit substitution, truncation, range containment, number sets
  address  fuzzy + IDF overlap, street, city, state, unit, locality, landmarks, care-of names
  ambiguity name-family size, shared-address counts (how many S1 compete for this key)
Values that cannot be computed (empty field on either side) are NaN, never 0.
Output: work/<split>/feat/part-*.parquet, row-aligned with work/<split>/cand/part-*.parquet.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Hamming, JaroWinkler, Levenshtein, OSA, Postfix, Prefix
from sklearn.feature_extraction.text import CountVectorizer

from .config import models_dir, report_dir, split_dir
from .records import load_raw, load_records
from .utils import LOG, is_done, list_parts, mark_done, part_path, reset_dir, stage, write_df

NAME_COLS = ["name_norm", "name_core", "name_concat", "name_other", "name_simpl", "name_nums",
             "legal_mask", "is_domain", "has_alias", "name_script", "n_name_tok"]
ADDR_COLS = ["addr_norm", "hn", "hn_int", "hn_hi", "hid", "nums", "unit", "has_po", "street",
             "street_type", "city", "state", "loc", "careof", "landmark", "addr_empty",
             "n_comps", "addr_script"]
REC_COLS = ["country"] + NAME_COLS + ADDR_COLS


# ---------------------------------------------------------------------------------------------
# token spaces: IDF-weighted binary bags for set-overlap features
# ---------------------------------------------------------------------------------------------

IDF_MAX_RATE = 1e-5   # tokens rarer than this share of a country's docs all get the same IDF


class TokenSpace:
    """Binary bag-of-tokens over S1 + S2/S3 docs with sqrt(IDF) weights (dot = sum of IDF).

    IDF is computed per country (pairs never cross countries) and capped for very rare tokens,
    so a token's weight depends on its share of the country's documents only, and common French
    words are not treated as rare because they are diluted by US / India documents.
    ``idf_ref`` (country -> DataFrame[token, idf], saved from train) replaces the IDF of countries
    seen in train, so test features use exactly the train weights (unseen tokens: the cap);
    ``keep_idf`` stores the fitted tables in ``self.idf_tables`` for saving.
    """

    def __init__(self, docs_s1: list[str], docs_q: list[str], country_s1: np.ndarray | None = None,
                 country_q: np.ndarray | None = None, idf_ref: dict | None = None,
                 keep_idf: bool = False):
        cv = CountVectorizer(analyzer=str.split, binary=True, dtype=np.float32)
        P = cv.fit_transform(docs_s1 + docs_q).tocsr()
        n_s1 = len(docs_s1)
        self.vocab_size = P.shape[1]
        self.idf_tables: dict[str, pl.DataFrame] = {}
        vocab = cv.get_feature_names_out() if (idf_ref or keep_idf) else None
        country = (np.zeros(P.shape[0], dtype=object) if country_s1 is None
                   else np.concatenate([country_s1, country_q]).astype(object))
        cap = np.log(1.0 / IDF_MAX_RATE) + 1.0
        blocks, order = [], []
        for c in sorted(set(country.tolist())):
            rows = np.flatnonzero(country == c)
            Pc = P[rows]
            dfreq = np.asarray(Pc.sum(axis=0)).ravel()
            idf = np.minimum(np.log((1.0 + len(rows)) / (1.0 + dfreq)) + 1.0, cap)
            if idf_ref and c in idf_ref:
                j = (pl.DataFrame({"token": vocab}).with_row_index("j")
                     .join(idf_ref[c], on="token", how="inner"))
                idf = np.full(len(vocab), cap)
                idf[j["j"].to_numpy()] = j["idf"].to_numpy()
            elif keep_idf:
                seen = dfreq > 0
                self.idf_tables[str(c)] = pl.DataFrame({"token": vocab[seen],
                                                        "idf": idf[seen].astype(np.float64)})
            blocks.append((Pc @ sp.diags(np.sqrt(idf).astype(np.float32))).tocsr())
            order.append(rows)
        inv = np.empty(P.shape[0], dtype=np.int64)
        inv[np.concatenate(order)] = np.arange(P.shape[0])
        W = sp.vstack(blocks, format="csr")[inv].astype(np.float32)
        cnt = np.diff(P.indptr).astype(np.float32)
        del P, blocks
        self.WS, self.WQ = W[:n_s1], W[n_s1:]
        sq = np.asarray(W.multiply(W).sum(axis=1)).ravel().astype(np.float32)
        self.norm_s, self.norm_q = sq[:n_s1], sq[n_s1:]
        self.cnt_s, self.cnt_q = cnt[:n_s1], cnt[n_s1:]

    @staticmethod
    def _binary(M: sp.csr_matrix) -> sp.csr_matrix:
        B = M.copy()
        B.data[:] = 1.0
        return B

    def pair(self, s: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
        A, B = self.WS[s], self.WQ[q]
        PA, PB = self._binary(A), self._binary(B)
        inter = _rowsum(A.multiply(B))
        na, nb = self.norm_s[s], self.norm_q[q]
        ca, cb = self.cnt_s[s], self.cnt_q[q]
        inter_c = _rowsum(PA.multiply(PB))
        valid = (ca > 0) & (cb > 0)
        extra_b = (B - B.multiply(PA)).tocsr()
        extra_b.eliminate_zeros()
        miss_a = (A - A.multiply(PB)).tocsr()
        miss_a.eliminate_zeros()
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {
                "idf_jacc": np.where(valid, inter / (na + nb - inter), np.nan),
                "tok_jacc": np.where(valid, inter_c / (ca + cb - inter_c), np.nan),
                "inter_idf": np.where(valid, inter, np.nan),
                "extra_b_maxidf": np.where(valid, _rowmax(extra_b) ** 2, np.nan),
                "extra_b_cnt": np.where(valid, np.diff(extra_b.indptr), np.nan),
                "miss_a_maxidf": np.where(valid, _rowmax(miss_a) ** 2, np.nan),
                "miss_a_cnt": np.where(valid, np.diff(miss_a.indptr), np.nan),
            }
        return {k: v.astype(np.float32) for k, v in out.items()}


def _rowsum(M: sp.spmatrix) -> np.ndarray:
    return np.asarray(M.sum(axis=1)).ravel().astype(np.float32)


def _rowmax(M: sp.csr_matrix) -> np.ndarray:
    r = M.max(axis=1)
    return np.asarray(r.todense() if sp.issparse(r) else r).ravel().astype(np.float32)


def build_spaces(cfg: dict, split: str, s1: pl.DataFrame) -> dict[str, TokenSpace]:
    """Token spaces for name core tokens, address words, address numbers and name phonetics."""
    spaces: dict[str, TokenSpace] = {}
    specs = {
        "name": pl.col("name_core"),
        "addr": pl.col("addr_norm").str.replace_all(r"\b\d+\b", " "),
        "nums": pl.col("nums"),
        "phon": pl.col("name_phon"),
    }
    q_country = load_records(cfg, split, "q", ["country"])["country"].to_numpy()
    s_country = load_records(cfg, split, "s1", ["country"])["country"].to_numpy()
    for key, expr in specs.items():
        col = expr.meta.root_names()[0]
        q_docs = load_records(cfg, split, "q", [col]).select(expr.alias("d"))["d"].to_list()
        s_docs = s1.select(expr.alias("d"))["d"].to_list() if col in s1.columns else \
            load_records(cfg, split, "s1", [col]).select(expr.alias("d"))["d"].to_list()
        path = models_dir(cfg) / f"idf_{key}.parquet"
        ref = None
        if split != "train" and path.exists():  # train weights for countries seen in train
            t = pl.read_parquet(path)
            ref = {c: t.filter(pl.col("country") == c).select(["token", "idf"])
                   for c in t["country"].unique().to_list()}
        spaces[key] = TokenSpace(s_docs, q_docs, s_country, q_country, idf_ref=ref,
                                 keep_idf=split == "train")
        if split == "train":
            write_df(pl.concat([t.with_columns(pl.lit(c).alias("country"))
                                for c, t in spaces[key].idf_tables.items()], how="vertical"), path)
            spaces[key].idf_tables = {}
        del q_docs, s_docs
        LOG.info("  token space %s: vocab %d", key, spaces[key].vocab_size)
    return spaces


# ---------------------------------------------------------------------------------------------
# ambiguity counts
# ---------------------------------------------------------------------------------------------

def ambiguity_arrays(cfg: dict, split: str, s1: pl.DataFrame) -> dict[str, np.ndarray]:
    """How many S1 records share a record's name key / name+city / house+street key."""
    q = load_records(cfg, split, "q", ["country", "name_sorted", "hn", "street"])
    s1k = load_records(cfg, split, "s1", ["country", "name_sorted", "hn", "street", "city"])
    addr_key = pl.when((pl.col("hn") != "") & (pl.col("street") != "")).then(
        pl.col("hn") + "|" + pl.col("street")).otherwise(pl.lit(None, pl.Utf8))
    s1k = s1k.with_columns(addr_key.alias("akey"))
    q = q.with_columns(addr_key.alias("akey"))
    fam = s1k.group_by(["country", "name_sorted"]).agg(pl.len().alias("fam"))
    famc = s1k.group_by(["country", "name_sorted", "city"]).agg(pl.len().alias("famc"))
    adr = s1k.filter(pl.col("akey").is_not_null()).group_by(["country", "akey"]).agg(
        pl.len().alias("adr"))
    s = (s1k.join(fam, on=["country", "name_sorted"], how="left")
            .join(famc, on=["country", "name_sorted", "city"], how="left")
            .join(adr, on=["country", "akey"], how="left").sort("idx"))
    qq = (q.join(fam, on=["country", "name_sorted"], how="left")
           .join(adr, on=["country", "akey"], how="left").sort("idx"))
    return {
        "a_name_family": s["fam"].fill_null(0).to_numpy().astype(np.float32),
        "a_name_city_family": s["famc"].fill_null(0).to_numpy().astype(np.float32),
        "a_addr_share": s["adr"].cast(pl.Float32).to_numpy(),
        "b_name_amb": qq["fam"].fill_null(0).to_numpy().astype(np.float32),
        "b_addr_amb": qq["adr"].fill_null(0).to_numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------------------------
# raw-text formatting of the S2/S3 record
# ---------------------------------------------------------------------------------------------

_FMT_NAME = {
    "fmt_n_accent": r"[À-ÖØ-öø-ÿ]",
    "fmt_n_leet": r"[A-Za-z][0-9][A-Za-z]",
    "fmt_n_junk": r">>|<<|--|\*\*\*|\bNULL\b|\bnull\b|N/A",
    "fmt_n_idnum": r"#\d|\(ID:|\bID\b|\||\d{5,}",
    "fmt_n_bracket": r"[\[\]()]",
    "fmt_n_dbl": r"  ",
}
_FMT_ADDR = {
    "fmt_a_dbl": r"  ",
    "fmt_a_hashhash": r"##",
    "fmt_a_zeropad": r"(?:^|[\s,#])0\d",
    "fmt_a_null": r"\bNULL\b",
}
FMT_NOISE = ["fmt_n_accent", "fmt_n_leet", "fmt_n_junk", "fmt_n_idnum", "fmt_n_dbl"]


def _upper_share(col: str) -> pl.Expr:
    letters = pl.col(col).str.count_matches(r"[A-Za-z]")
    return (pl.when(letters > 0).then(pl.col(col).str.count_matches(r"[A-Z]") / letters)
            .otherwise(None).cast(pl.Float32))


def format_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Formatting flags of raw ``business_name`` / ``business_address`` strings (one row each)."""
    exprs = [_upper_share("business_name").alias("fmt_n_upper"),
             _upper_share("business_address").alias("fmt_a_upper")]
    exprs += [pl.col("business_name").str.contains(rx).alias(k) for k, rx in _FMT_NAME.items()]
    exprs += [pl.col("business_address").str.contains(rx).alias(k)
              for k, rx in _FMT_ADDR.items()]
    return raw.select(exprs).with_columns(
        pl.sum_horizontal([pl.col(c).cast(pl.Int32) for c in FMT_NOISE]).alias("fmt_n_noise"))


def format_arrays(cfg: dict, split: str) -> dict[str, np.ndarray]:
    """Record-level priors measured on the raw S2/S3 strings (indexed by query idx).

    Character noise (accents, leetspeak, junk tokens, id suffixes, case changes) is applied to
    true records more often than to decoys (train match rate ~0.82 vs ~0.74 overall), and the
    normalisation erases it, so it is read here before any cleaning.
    """
    raw = load_raw(cfg, split, "q", ["idx", "business_name", "business_address"]).sort("idx")
    f = format_frame(raw)
    return {c: f[c].cast(pl.Float32).to_numpy() for c in f.columns}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _empty(s: pl.Series) -> np.ndarray:
    return (s.str.len_chars() == 0).to_numpy()


class _Lists:
    """Caches ``.to_list()`` of gathered string columns within one part."""

    def __init__(self, df: pl.DataFrame, idx: np.ndarray):
        self.df, self.idx, self.cache = df, idx, {}

    def series(self, col: str) -> pl.Series:
        key = ("s", col)
        if key not in self.cache:
            self.cache[key] = self.df[col].gather(self.idx)
        return self.cache[key]

    def list(self, col: str) -> list[str]:
        key = ("l", col)
        if key not in self.cache:
            self.cache[key] = self.series(col).to_list()
        return self.cache[key]

    def num(self, col: str) -> np.ndarray:
        return self.series(col).cast(pl.Float64).to_numpy()


def _fz(A: _Lists, B: _Lists, ca: str, cb: str, scorer, nt: int, distance=False) -> np.ndarray:
    dtype = np.int32 if distance else np.float32
    v = process.cpdist(A.list(ca), B.list(cb), scorer=scorer, workers=nt, dtype=dtype)
    v = v.astype(np.float32)
    v[_empty(A.series(ca)) | _empty(B.series(cb))] = np.nan
    return v


def _eq(a: pl.Series, b: pl.Series) -> np.ndarray:
    both = ~(_empty(a) | _empty(b))
    return np.where(both, (a == b).to_numpy(), np.nan).astype(np.float32)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------------------------
# feature groups
# ---------------------------------------------------------------------------------------------

def name_features(A: _Lists, B: _Lists, s: np.ndarray, q: np.ndarray,
                  spaces: dict[str, TokenSpace], nt: int) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    f["n_ratio"] = _fz(A, B, "name_norm", "name_norm", fuzz.ratio, nt)
    f["n_partial"] = _fz(A, B, "name_norm", "name_norm", fuzz.partial_ratio, nt)
    f["n_tsort"] = _fz(A, B, "name_norm", "name_norm", fuzz.token_sort_ratio, nt)
    f["n_tset"] = _fz(A, B, "name_norm", "name_norm", fuzz.token_set_ratio, nt)
    f["n_wratio"] = _fz(A, B, "name_norm", "name_norm", fuzz.WRatio, nt)
    f["n_core_ratio"] = _fz(A, B, "name_core", "name_core", fuzz.ratio, nt)
    f["n_core_tset"] = _fz(A, B, "name_core", "name_core", fuzz.token_set_ratio, nt)
    f["n_core_jw"] = _fz(A, B, "name_core", "name_core", JaroWinkler.normalized_similarity, nt)
    f["n_core_lev"] = _fz(A, B, "name_core", "name_core", Levenshtein.normalized_similarity, nt)
    f["n_cat_ratio"] = _fz(A, B, "name_concat", "name_concat", fuzz.ratio, nt)
    f["n_cat_partial"] = _fz(A, B, "name_concat", "name_concat", fuzz.partial_ratio, nt)
    pre = _fz(A, B, "name_concat", "name_concat", Prefix.similarity, nt, distance=True)
    la = A.series("name_concat").str.len_chars().to_numpy().astype(np.float32)
    lb = B.series("name_concat").str.len_chars().to_numpy().astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["n_cat_prefix"] = pre / np.maximum(np.minimum(la, lb), 1)
        f["n_len_ratio"] = np.where((la > 0) & (lb > 0), np.minimum(la, lb) / np.maximum(la, lb),
                                    np.nan).astype(np.float32)
    f["n_simpl_ratio"] = _fz(A, B, "name_simpl", "name_simpl", fuzz.ratio, nt)
    f["n_other_tset"] = _fz(A, B, "name_norm", "name_other", fuzz.token_set_ratio, nt)
    B.cache[("s", "_other_cat")] = B.series("name_other").str.replace_all(" ", "")
    f["n_other_cat_partial"] = _fz(A, B, "name_concat", "_other_cat", fuzz.partial_ratio, nt)
    f["n_best"] = np.fmax(np.fmax(f["n_tset"], f["n_other_tset"]), f["n_cat_partial"])
    for k, v in spaces["name"].pair(s, q).items():
        f[f"n_{k}"] = v
    f["n_phon_jacc"] = spaces["phon"].pair(s, q)["tok_jacc"]
    first_a = A.series("name_core").str.split(" ").list.first()
    first_b = B.series("name_core").str.split(" ").list.first()
    f["n_first_eq"] = _eq(first_a, first_b)
    f["n_nums_match"] = _eq(A.series("name_nums"), B.series("name_nums"))
    la_m = A.series("legal_mask").to_numpy().astype(np.int64)
    lb_m = B.series("legal_mask").to_numpy().astype(np.int64)
    f["legal_same"] = ((la_m == lb_m) & (la_m > 0)).astype(np.float32)
    f["legal_conflict"] = ((la_m > 0) & (lb_m > 0) & ((la_m & lb_m) == 0)).astype(np.float32)
    f["legal_a_none"] = (la_m == 0).astype(np.float32)
    f["legal_b_none"] = (lb_m == 0).astype(np.float32)
    f["legal_overlap"] = _popcount(la_m & lb_m).astype(np.float32)
    na, nb = A.num("n_name_tok"), B.num("n_name_tok")
    f["a_ntok"], f["b_ntok"] = na.astype(np.float32), nb.astype(np.float32)
    f["ntok_diff"] = np.abs(na - nb).astype(np.float32)
    f["b_is_domain"] = B.num("is_domain").astype(np.float32)
    f["b_has_alias"] = B.num("has_alias").astype(np.float32)
    f["a_has_alias"] = A.num("has_alias").astype(np.float32)
    f["b_name_script"] = B.num("name_script").astype(np.float32)
    return f


def hn_features(A: _Lists, B: _Lists, nt: int) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    ah, bh = A.list("hn"), B.list("hn")
    has_a = ~_empty(A.series("hn"))
    has_b = ~_empty(B.series("hn"))
    both = has_a & has_b
    nan = np.float32(np.nan)
    f["hn_a_missing"] = (~has_a).astype(np.float32)
    f["hn_b_missing"] = (~has_b).astype(np.float32)
    f["hn_eq"] = np.where(both, (A.series("hn") == B.series("hn")).to_numpy(), nan)
    f["hn_lev"] = _fz(A, B, "hn", "hn", Levenshtein.distance, nt, distance=True)
    f["hn_osa"] = _fz(A, B, "hn", "hn", OSA.distance, nt, distance=True)
    f["hn_hamming_raw"] = _fz(A, B, "hn", "hn", Hamming.distance, nt, distance=True)
    f["hn_prefix"] = _fz(A, B, "hn", "hn", Prefix.similarity, nt, distance=True)
    f["hn_suffix"] = _fz(A, B, "hn", "hn", Postfix.similarity, nt, distance=True)
    la = A.series("hn").str.len_chars().to_numpy().astype(np.float32)
    lb = B.series("hn").str.len_chars().to_numpy().astype(np.float32)
    f["hn_len_a"], f["hn_len_b"] = la, lb
    len_eq = la == lb
    f["hn_len_eq"] = np.where(both, len_eq, nan)
    f["hn_len_diff"] = np.where(both, np.abs(la - lb), nan)
    f["hn_hamming"] = np.where(both & len_eq, f.pop("hn_hamming_raw"), nan)
    f["hn_single_sub"] = np.where(both, len_eq & (f["hn_hamming"] == 1), nan)
    ai, bi = A.num("hn_int"), B.num("hn_int")
    ok = both & (ai >= 0) & (bi >= 0)
    diff = np.where(ok, np.abs(ai - bi), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["hn_logdiff"] = np.log1p(diff).astype(np.float32)
        f["hn_reldiff"] = (diff / np.maximum(np.maximum(ai, bi), 1)).astype(np.float32)
        k = np.floor(np.log10(np.where(diff > 0, diff, 1)))
        p10 = np.power(10.0, k)
        f["hn_pow10"] = np.where(ok, (diff > 0) & (np.mod(diff, p10) == 0), nan)
    f["hn_first_eq"] = np.where(both, np.array([x[:1] == y[:1] for x, y in zip(ah, bh)]), nan)
    f["hn_last_eq"] = np.where(both, np.array([x[-1:] == y[-1:] for x, y in zip(ah, bh)]), nan)
    f["hn_substr"] = np.where(both, np.array([x != y and (x in y or y in x)
                                              for x, y in zip(ah, bh)]), nan)
    ahi, bhi = A.num("hn_hi"), B.num("hn_hi")
    rng = ((bhi >= 0) & (bi <= ai) & (ai <= bhi)) | ((ahi >= 0) & (ai <= bi) & (bi <= ahi))
    f["hn_range"] = np.where(ok, rng, nan)
    an, bn = A.list("nums"), B.list("nums")
    f["hn_a_in_bnums"] = np.where(has_a & ~_empty(B.series("nums")),
                                  np.array([f" {x} " in f" {y} " for x, y in zip(ah, bn)]), nan)
    f["hn_b_in_anums"] = np.where(has_b & ~_empty(A.series("nums")),
                                  np.array([f" {y} " in f" {x} " for x, y in zip(an, bh)]), nan)
    f["hid_ratio"] = _fz(A, B, "hid", "hid", fuzz.ratio, nt)
    f["hid_eq"] = _eq(A.series("hid"), B.series("hid"))
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def address_features(A: _Lists, B: _Lists, s: np.ndarray, q: np.ndarray,
                     spaces: dict[str, TokenSpace], nt: int) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    f["b_addr_empty"] = B.num("addr_empty").astype(np.float32)
    f["ad_ratio"] = _fz(A, B, "addr_norm", "addr_norm", fuzz.ratio, nt)
    f["ad_tset"] = _fz(A, B, "addr_norm", "addr_norm", fuzz.token_set_ratio, nt)
    f["ad_tsort"] = _fz(A, B, "addr_norm", "addr_norm", fuzz.token_sort_ratio, nt)
    f["ad_partial"] = _fz(A, B, "addr_norm", "addr_norm", fuzz.partial_ratio, nt)
    for k, v in spaces["addr"].pair(s, q).items():
        f[f"ad_{k}"] = v
    nums = spaces["nums"].pair(s, q)
    f["nums_jacc"] = nums["tok_jacc"]
    f["nums_a_only"] = nums["miss_a_cnt"]
    f["nums_b_only"] = nums["extra_b_cnt"]
    f["st_ratio"] = _fz(A, B, "street", "street", fuzz.ratio, nt)
    f["st_tset"] = _fz(A, B, "street", "street", fuzz.token_set_ratio, nt)
    f["st_in_b"] = _fz(A, B, "street", "addr_norm", fuzz.partial_ratio, nt)
    f["st_type_eq"] = _eq(A.series("street_type"), B.series("street_type"))
    f["city_eq"] = _eq(A.series("city"), B.series("city"))
    f["city_jw"] = _fz(A, B, "city", "city", JaroWinkler.normalized_similarity, nt)
    f["city_in_b"] = _fz(A, B, "city", "addr_norm", fuzz.partial_ratio, nt)
    f["state_eq"] = _eq(A.series("state"), B.series("state"))
    f["unit_eq"] = _eq(A.series("unit"), B.series("unit"))
    f["a_has_unit"] = (~_empty(A.series("unit"))).astype(np.float32)
    f["b_has_unit"] = (~_empty(B.series("unit"))).astype(np.float32)
    f["b_has_po"] = B.num("has_po").astype(np.float32)
    f["loc_tset"] = _fz(A, B, "loc", "loc", fuzz.token_set_ratio, nt)
    f["careof_tset"] = _fz(A, B, "careof", "careof", fuzz.token_set_ratio, nt)
    f["landmark_tset"] = _fz(A, B, "landmark", "landmark", fuzz.token_set_ratio, nt)
    f["a_ncomps"] = A.num("n_comps").astype(np.float32)
    f["b_ncomps"] = B.num("n_comps").astype(np.float32)
    f["b_addr_script"] = B.num("addr_script").astype(np.float32)
    return f


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, qrec: pl.DataFrame, qloc: np.ndarray,
                  spaces: dict[str, TokenSpace], amb: dict[str, np.ndarray], src: np.ndarray,
                  fmt: dict[str, np.ndarray], nt: int) -> pl.DataFrame:
    s = cand["s"].to_numpy().astype(np.int64)
    q = cand["q"].to_numpy().astype(np.int64)
    A = _Lists(s1, s)
    B = _Lists(qrec, qloc)
    feats: dict[str, np.ndarray] = {"pair_id": cand["pair_id"].to_numpy()}
    feats.update(name_features(A, B, s, q, spaces, nt))
    feats.update(hn_features(A, B, nt))
    feats.update(address_features(A, B, s, q, spaces, nt))
    for k in ("a_name_family", "a_name_city_family", "a_addr_share"):
        feats[k] = np.log1p(amb[k][s]).astype(np.float32)
    for k in ("b_name_amb", "b_addr_amb"):
        feats[k] = np.log1p(amb[k][q]).astype(np.float32)
    feats["b_src"] = src[q].astype(np.float32)
    for k, v in fmt.items():
        feats[k] = v[q]
    return pl.DataFrame(feats)


def _id_runs(ids: np.ndarray, gap: int = 1_000_000) -> list[tuple[int, int]]:
    """(lo, hi) ranges covering ``ids``, split wherever consecutive sorted ids jump by > gap."""
    u = np.unique(ids)
    cut = np.flatnonzero(np.diff(u) > gap)
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(u) - 1]
    return [(int(u[a]), int(u[b])) for a, b in zip(starts, ends)]


def run_features(cfg: dict, split: str) -> None:
    out = split_dir(cfg, split)
    feat_dir = out / "feat"
    if is_done(feat_dir):
        LOG.info("features[%s]: cached", split)
        return
    rep = report_dir(cfg)
    nt = int(cfg["runtime"]["n_threads"])
    with stage(f"features[{split}] setup", rep):
        s1 = load_records(cfg, split, "s1", REC_COLS)
        spaces = build_spaces(cfg, split, s1)
        amb = ambiguity_arrays(cfg, split, s1)
        src = load_records(cfg, split, "q", ["src"])["src"].to_numpy()
        fmt = format_arrays(cfg, split)
    reset_dir(feat_dir)
    q_glob = str(out / "records_q" / "*.parquet")
    parts = list_parts(out / "cand")
    with stage(f"features[{split}] pairs", rep):
        for i, p in enumerate(parts):
            cand = pl.read_parquet(p, columns=["pair_id", "q", "s"])
            if cand.height == 0:
                raise ValueError(f"{p} is empty — blocking never writes empty parts")
            # one idx range per run of query ids (augmented train parts have a second run of
            # synthetic ids at the end of the id space)
            rng_filter = pl.lit(False)
            for lo, hi in _id_runs(cand["q"].to_numpy()):
                rng_filter = rng_filter | pl.col("idx").is_between(lo, hi)
            qrec = (pl.scan_parquet(q_glob).filter(rng_filter)
                    .select(["idx"] + REC_COLS).collect().sort("idx"))
            qloc = np.searchsorted(qrec["idx"].to_numpy(), cand["q"].to_numpy())
            feats = pair_features(cand, s1, qrec, qloc, spaces, amb, src, fmt, nt)
            write_df(feats, part_path(feat_dir, i))
            LOG.info("  features part %d/%d: %d pairs, %d columns", i + 1, len(parts),
                     feats.height, feats.width)
    mark_done(feat_dir)
