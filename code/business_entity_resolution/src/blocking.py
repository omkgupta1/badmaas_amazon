"""Stage 2 — candidate generation (blocking), per country, never across countries.

For every S2/S3 record ("query") we retrieve S1 candidates of the same country from:
  N   character 3-gram TF-IDF of the space-free name   (typos, reordering, domain names)
  A   word TF-IDF of the canonical address + "hn<num>" + "st_<street>" tokens
  NA  the L2-normalised concatenation of N and A        (name families, shared addresses)
  keys exact sorted-core-name key and exact house-number+street key (small S1 groups only)
  REV reverse queries: each S1 keeps its top-k S2/S3 records on NA (name-family crowding)
The union is scored by the stage-0 ranker (``prerank``); the top-M per query become the final
candidate set — exactly the pairs the matching models score, written to candidate_pairs.tsv.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import models_dir, report_dir, split_dir
from .records import load_records, load_truth
from .utils import (LOG, chunks, is_done, list_parts, load_json, mark_done, part_path,
                    reset_dir, save_json, stage, write_df)

FLAG_N, FLAG_A, FLAG_NA, FLAG_KN, FLAG_KA, FLAG_REV, FLAG_RN = 1, 2, 4, 8, 16, 32, 64
UNION_FEATURES = [
    "cos_n", "cos_a", "cos_na", "r_n", "r_a", "r_na", "in_n", "in_a", "in_na", "key_name",
    "key_addr", "rev", "hn_eq0", "name_ratio0", "addr_tset0", "n_union", "b_addr_empty0",
    "b_is_domain0", "in_rn",
]
_REC_COLS = ["country", "name_concat", "addr_block", "name_sorted", "hn", "street", "name_norm",
             "addr_norm", "addr_empty", "is_domain", "state", "name_core"]
STATE_GROUPS_FILE = "state_groups.json"


# ---------------------------------------------------------------------------------------------
# sparse top-k
# ---------------------------------------------------------------------------------------------

def _topn(A: sp.csr_matrix, B: sp.csr_matrix, k: int, n_threads: int) -> sp.coo_matrix:
    """Top-``k`` entries per row of ``A @ B`` (B has shape n_features x n_cols)."""
    A = sp.csr_matrix(A, dtype=np.float32)
    B = sp.csr_matrix(B, dtype=np.float32)
    A.sort_indices()
    B.sort_indices()
    try:
        from sparse_dot_topn import sp_matmul_topn

        return sp_matmul_topn(A, B, top_n=k, n_threads=n_threads).tocoo()
    except ImportError:
        pass
    try:  # sparse_dot_topn < 1.0
        from sparse_dot_topn import awesome_cossim_topn

        return awesome_cossim_topn(A, B, k, 0.0, use_threads=n_threads > 1,
                                   n_jobs=n_threads).tocoo()
    except ImportError:
        LOG.warning("sparse_dot_topn missing — using the slow scipy fallback")
        return _scipy_topn(A, B, k)


def _scipy_topn(A: sp.csr_matrix, B: sp.csr_matrix, k: int, batch: int = 2000) -> sp.coo_matrix:
    rows, cols, vals = [], [], []
    for a in range(0, A.shape[0], batch):
        C = (A[a:a + batch] @ B).tocsr()
        for i in range(C.shape[0]):
            st, en = C.indptr[i], C.indptr[i + 1]
            if en == st:
                continue
            d, ix = C.data[st:en], C.indices[st:en]
            if en - st > k:
                sel = np.argpartition(-d, k)[:k]
                d, ix = d[sel], ix[sel]
            rows.append(np.full(len(d), a + i, dtype=np.int64))
            cols.append(ix)
            vals.append(d)
    if not rows:
        return sp.coo_matrix((A.shape[0], B.shape[1]), dtype=np.float32)
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(A.shape[0], B.shape[1]))


def rowdot(A: sp.csr_matrix, ia: np.ndarray, B: sp.csr_matrix, ib: np.ndarray,
           batch: int = 1_000_000) -> np.ndarray:
    """Dot products of row ``ia[i]`` of A with row ``ib[i]`` of B."""
    out = np.empty(len(ia), dtype=np.float32)
    for a, b in chunks(len(ia), batch):
        prod = A[ia[a:b]].multiply(B[ib[a:b]])
        out[a:b] = np.asarray(prod.sum(axis=1)).ravel()
    return out


# ---------------------------------------------------------------------------------------------
# per-country views
# ---------------------------------------------------------------------------------------------

class CountryViews:
    """TF-IDF vectorisers and S1 matrices of one country."""

    def __init__(self, cfg: dict, s_name: list[str], s_addr: list[str], q_name_fit: list[str],
                 q_addr_fit: list[str], s_words: list[str] | None = None,
                 q_words_fit: list[str] | None = None):
        b = cfg["blocking"]
        # country-wide view over RARE name words (cheap: short posting lists); it keeps
        # cross-state matches with a distinctive name when retrieval is split by state
        rv = b.get("rare_name_view", {})
        self.rare_vec, self.S_R_T = None, None
        if s_words is not None and int(rv.get("k", 0)) > 0:
            self.rare_vec = TfidfVectorizer(analyzer=str.split, lowercase=False, min_df=2,
                                            max_df=float(rv["max_df"]), sublinear_tf=True,
                                            dtype=np.float32)
            self.rare_vec.fit(s_words + (q_words_fit or []))
            self.rare_vec.stop_words_ = None
            self.S_R_T = self.rare_vec.transform(s_words).T.tocsr()
        nv, av = b["name_view"], b["addr_view"]
        self.w = float(b["joint_view"]["name_weight"])
        n_fit = len(s_name) + len(q_name_fit)
        self.name_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(nv["ngram"], nv["ngram"]), lowercase=False,
            min_df=min(nv["min_df"], max(1, n_fit // 1000)), max_df=nv["max_df"],
            sublinear_tf=True, dtype=np.float32)
        self.addr_vec = TfidfVectorizer(
            analyzer=str.split, lowercase=False, min_df=min(av["min_df"], max(1, n_fit // 1000)),
            max_df=av["max_df"], sublinear_tf=True, dtype=np.float32)
        self.name_vec.fit(_name_docs(s_name) + _name_docs(q_name_fit))
        self.addr_vec.fit(s_addr + q_addr_fit)
        # sklearn keeps every pruned term in stop_words_ (millions of rare address tokens);
        # it is for introspection only and transform() does not use it
        self.name_vec.stop_words_ = None
        self.addr_vec.stop_words_ = None
        self.S_N, self.S_A, self.S_NA = self.transform(s_name, s_addr)
        self.S_N_T = self.S_N.T.tocsr()
        self.S_A_T = self.S_A.T.tocsr()
        self.S_NA_T = self.S_NA.T.tocsr()

    def transform(self, names: list[str], addrs: list[str]):
        N = self.name_vec.transform(_name_docs(names)).tocsr()
        A = self.addr_vec.transform(addrs).tocsr()
        NA = normalize(sp.hstack([N * np.sqrt(self.w), A * np.sqrt(1.0 - self.w)],
                                 format="csr")).astype(np.float32)
        return N, A, NA


def _name_docs(names: list[str]) -> list[str]:
    return [f" {n} " if n else "" for n in names]


class Partition:
    """S1 columns of one country split by state group. A query whose state belongs to a group
    searches only that group's S1 records; a query without a (known) state searches the whole
    country. With ~25-45 groups this cuts the sparse top-k cost ~10x and removes the crowding of
    true matches by same-name / same-street S1 records of other states."""

    def __init__(self, s_group: np.ndarray, views: CountryViews):
        self.cols = {int(g): np.flatnonzero(s_group == g) for g in np.unique(s_group) if g >= 0}
        mats = (views.S_N, views.S_A, views.S_NA)
        self.T = {g: tuple(M[c].T.tocsr() for M in mats) for g, c in self.cols.items()}
        self.S_NA = {g: views.S_NA[c] for g, c in self.cols.items()}
        self.full_T = (views.S_N_T, views.S_A_T, views.S_NA_T)

    def rows_by_group(self, q_group: np.ndarray) -> dict[int, np.ndarray]:
        """Chunk rows per group; key -1 = rows searched against the whole country."""
        known = np.isin(q_group, np.fromiter(self.cols, dtype=np.int64))
        out = {-1: np.flatnonzero(~known)}
        order = np.argsort(q_group, kind="stable")
        g_sorted = q_group[order]
        for g in self.cols:
            lo, hi = np.searchsorted(g_sorted, g, "left"), np.searchsorted(g_sorted, g, "right")
            if hi > lo:
                out[g] = np.sort(order[lo:hi])
        return out

    def topn(self, M: sp.csr_matrix, vi: int, by_group: dict[int, np.ndarray], k: int,
             nt: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows, cols, vals = [], [], []
        for g, r in by_group.items():
            if len(r) == 0:
                continue
            B = self.full_T[vi] if g < 0 else self.T[g][vi]
            C = _topn(M[r], B, k, nt)
            rows.append(r[C.row])
            cols.append(C.col if g < 0 else self.cols[g][C.col])
            vals.append(C.data)
        if not rows:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def reverse(self, Q_NA: sp.csr_matrix, by_group: dict[int, np.ndarray], k: int,
                nt: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each S1 keeps its k best queries of its own group (S1 column idx, chunk row, value)."""
        s_loc, q_loc, vals = [], [], []
        for g, r in by_group.items():
            if g < 0 or len(r) == 0:
                continue
            R = _topn(self.S_NA[g], Q_NA[r].T.tocsr(), k, nt)
            s_loc.append(self.cols[g][R.row])
            q_loc.append(r[R.col])
            vals.append(R.data)
        if not s_loc:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(s_loc), np.concatenate(q_loc), np.concatenate(vals)


def state_group_maps(cfg: dict, split: str, s1: pl.DataFrame, q: pl.DataFrame,
                     q_true: np.ndarray | None) -> dict[str, dict[str, int]]:
    """Per country: S1 state -> group id. Confusable states are merged: on train, two states are
    linked when >= min_merge_pairs true pairs (and >= min_merge_share of the query state's true
    pairs) connect them (e.g. Andhra Pradesh records of Telangana businesses); the merges are
    saved for test. Explicit merges can be added in config (blocking.partition.merge)."""
    pc = cfg["blocking"].get("partition", {})
    path = models_dir(cfg) / STATE_GROUPS_FILE
    s_state, s_c = s1["state"].to_numpy(), s1["country"].to_numpy()
    merges: dict[str, list[list[str]]] = {c: [list(m) for m in v]
                                          for c, v in (pc.get("merge") or {}).items()}
    if split == "train" and q_true is not None:
        pos = np.flatnonzero(q_true >= 0)
        d = pl.DataFrame({"c": s_c[q_true[pos]], "a": q["state"].to_numpy()[pos],
                          "b": s_state[q_true[pos]]}).filter(pl.col("a") != "")
        tot = d.group_by(["c", "a"]).agg(pl.len().alias("tot"))
        links = (d.filter((pl.col("b") != "") & (pl.col("a") != pl.col("b")))
                 .group_by(["c", "a", "b"]).agg(pl.len().alias("n")).join(tot, on=["c", "a"])
                 .filter((pl.col("n") >= int(pc.get("min_merge_pairs", 200)))
                         & (pl.col("n") / pl.col("tot") >= float(pc.get("min_merge_share", 0.01)))))
        for c, a, b, n, t in links.iter_rows():
            merges.setdefault(c, []).append([a, b])
            LOG.info("  state merge [%s]: %s <-> %s (%d true pairs, %.1f%%)", c, a, b, n, 100 * n / t)
        save_json(merges, path)
    elif path.exists():
        merges = load_json(path)
    out: dict[str, dict[str, int]] = {}
    for c in sorted(set(s_c.tolist())):
        states = sorted(set(s_state[s_c == c].tolist()) - {""})
        parent = {st: st for st in states}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for grp in merges.get(c, []):
            members = [m for m in grp if m in parent]
            for m in members[1:]:
                parent[find(m)] = find(members[0])
        roots = sorted({find(st) for st in states})
        rid = {r: i for i, r in enumerate(roots)}
        out[c] = {st: rid[find(st)] for st in states}
    return out


def _group_codes(states: np.ndarray, gmap: dict[str, int]) -> np.ndarray:
    return np.array([gmap.get(st, -1) for st in states.tolist()], dtype=np.int64)


def _key_pairs(s1c: pl.DataFrame, qc: pl.DataFrame, key: pl.Expr, cap: int) -> pl.DataFrame:
    s = s1c.select(pl.col("idx"), key.alias("key")).filter(pl.col("key") != "")
    small = s.group_by("key").agg(pl.len().alias("n")).filter(pl.col("n") <= cap)
    s = s.join(small.select("key"), on="key", how="semi")
    q = qc.select(pl.col("idx"), key.alias("key")).filter(pl.col("key") != "")
    return q.join(s, on="key", how="inner", suffix="_s").select(
        pl.col("idx").alias("q"), pl.col("idx_s").alias("s"))


def _group_rank(q_local: np.ndarray, score: np.ndarray) -> np.ndarray:
    """1-based descending rank of ``score`` within groups of ``q_local``."""
    return (pl.DataFrame({"q": q_local, "c": score})
            .select(pl.col("c").rank("ordinal", descending=True).over("q"))
            .to_series().to_numpy().astype(np.float32))


def _cpdist(a: pl.Series, b: pl.Series, scorer, nt: int, batch: int = 2_000_000) -> np.ndarray:
    """rapidfuzz cpdist in batches, so only ``batch`` Python strings per side exist at a time."""
    out = np.empty(len(a), dtype=np.float32)
    for i, j in chunks(len(a), batch):
        out[i:j] = process.cpdist(a[i:j].to_list(), b[i:j].to_list(), scorer=scorer, workers=nt,
                                  dtype=np.float32)
    return out


def _top_k_per_s(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Best ``k`` (s, q, v) rows per S1 (ties broken by q, so the result is deterministic)."""
    return (df.sort(["s", "v", "q"], descending=[False, True, False])
              .group_by("s", maintain_order=True).head(k))


def _nan_empty(vals: np.ndarray, a: pl.Series, b: pl.Series) -> np.ndarray:
    bad = ((a.str.len_chars() == 0) | (b.str.len_chars() == 0)).to_numpy()
    vals = vals.astype(np.float32, copy=False)
    vals[bad] = np.nan
    return vals


def union_features(qc_ids: np.ndarray, s_ids: np.ndarray, views: CountryViews, Q, extra,
                   s1: pl.DataFrame, q: pl.DataFrame, n_s1_total: int, cfg: dict,
                   part: Partition | None = None, by_group: dict | None = None) -> pl.DataFrame:
    """Union of all candidate sources for one query chunk + cheap stage-0 features."""
    b = cfg["blocking"]
    nt = int(cfg["runtime"]["n_threads"])
    Q_N, Q_A, Q_NA = Q
    qs, ss, fs = [extra[0]], [extra[1]], [extra[2]]
    full_T = (views.S_N_T, views.S_A_T, views.S_NA_T)
    for vi, (M, k, flag) in enumerate(((Q_N, b["name_view"]["k"], FLAG_N),
                                       (Q_A, b["addr_view"]["k"], FLAG_A),
                                       (Q_NA, b["joint_view"]["k"], FLAG_NA))):
        if k <= 0:
            continue
        if part is None:
            C = _topn(M, full_T[vi], int(k), nt)
            rows, cols, vals = C.row, C.col, C.data
        else:
            rows, cols, vals = part.topn(M, vi, by_group, int(k), nt)
        keep = vals > 0
        qs.append(qc_ids[rows[keep]])
        ss.append(s_ids[cols[keep]])
        fs.append(np.full(int(keep.sum()), flag, dtype=np.int16))
    if views.rare_vec is not None:  # country-wide rare-name-word view
        R = views.rare_vec.transform(q["name_core"].gather(qc_ids).to_list()).tocsr()
        C = _topn(R, views.S_R_T, int(b["rare_name_view"]["k"]), nt)
        keep = C.data > 0
        qs.append(qc_ids[C.row[keep]])
        ss.append(s_ids[C.col[keep]])
        fs.append(np.full(int(keep.sum()), FLAG_RN, dtype=np.int16))
    qa = np.concatenate(qs).astype(np.int64)
    sa = np.concatenate(ss).astype(np.int64)
    fa = np.concatenate(fs).astype(np.int16)
    key = qa * n_s1_total + sa
    uk, inv = np.unique(key, return_inverse=True)
    flags = np.zeros(len(uk), dtype=np.int16)
    np.bitwise_or.at(flags, inv, fa)
    q_u = (uk // n_s1_total).astype(np.int32)
    s_u = (uk % n_s1_total).astype(np.int32)
    rq = np.searchsorted(qc_ids, q_u)
    rs = np.searchsorted(s_ids, s_u)

    cos_n = rowdot(Q_N, rq, views.S_N, rs)
    cos_a = rowdot(Q_A, rq, views.S_A, rs)
    cos_na = rowdot(Q_NA, rq, views.S_NA, rs)
    n_union = np.bincount(rq, minlength=len(qc_ids)).astype(np.float32)[rq]

    q_hn, s_hn = q["hn"].gather(q_u), s1["hn"].gather(s_u)
    hn_eq = np.where((q_hn != "").to_numpy() & (s_hn != "").to_numpy(),
                     (q_hn == s_hn).to_numpy().astype(np.float32), np.nan).astype(np.float32)
    qn, sn = q["name_norm"].gather(q_u), s1["name_norm"].gather(s_u)
    name_ratio = _nan_empty(_cpdist(qn, sn, fuzz.ratio, nt), qn, sn)
    qa_, sa_ = q["addr_norm"].gather(q_u), s1["addr_norm"].gather(s_u)
    addr_tset = _nan_empty(_cpdist(qa_, sa_, fuzz.token_set_ratio, nt), qa_, sa_)
    del qn, sn, qa_, sa_
    return pl.DataFrame({
        "q": q_u, "s": s_u,
        "cos_n": cos_n, "cos_a": cos_a, "cos_na": cos_na,
        "r_n": _group_rank(rq, cos_n), "r_a": _group_rank(rq, cos_a),
        "r_na": _group_rank(rq, cos_na),
        "in_n": ((flags & FLAG_N) > 0).astype(np.float32),
        "in_a": ((flags & FLAG_A) > 0).astype(np.float32),
        "in_na": ((flags & FLAG_NA) > 0).astype(np.float32),
        "key_name": ((flags & FLAG_KN) > 0).astype(np.float32),
        "key_addr": ((flags & FLAG_KA) > 0).astype(np.float32),
        "rev": ((flags & FLAG_REV) > 0).astype(np.float32),
        "in_rn": ((flags & FLAG_RN) > 0).astype(np.float32),
        "hn_eq0": hn_eq, "name_ratio0": name_ratio, "addr_tset0": addr_tset,
        "n_union": n_union,
        "b_addr_empty0": q["addr_empty"].gather(q_u).cast(pl.Float32).to_numpy(),
        "b_is_domain0": q["is_domain"].gather(q_u).cast(pl.Float32).to_numpy(),
    })


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

def run_blocking(cfg: dict, split: str, resume: bool = False) -> None:
    """``resume``: keep the parts an interrupted run already wrote and compute only the missing
    query chunks (the interrupted country's views, keys and reverse queries are rebuilt first,
    with the same random draws, so the result equals an uninterrupted run)."""
    from .prerank import apply_prerank, load_prerank, train_prerank

    out = split_dir(cfg, split)
    cand_dir = out / "cand"
    if is_done(cand_dir):
        LOG.info("blocking[%s]: cached", split)
        return
    rep = report_dir(cfg)
    ranker = None
    if split != "train":
        ranker = load_prerank(cfg)
        if ranker is None:
            raise FileNotFoundError("stage-0 ranker missing — run `block --split train` first")
    target = cand_dir if ranker is not None else out / "union"
    if split == "train":
        from .augment import strip_synthetic

        strip_synthetic(cfg)  # synthetic twins are added after blocking, never blocked
    if not (resume and target.exists()):
        reset_dir(target)
    with stage(f"blocking[{split}] retrieve", rep):
        _retrieve(cfg, split, target, ranker, resume)
    if ranker is None:
        with stage("blocking[train] stage-0 ranker", rep):
            ranker = train_prerank(cfg, out / "union")
        with stage("blocking[train] cut to top-M", rep):
            apply_prerank(cfg, ranker, out / "union", reset_dir(cand_dir))
        if not cfg["blocking"].get("keep_union", False):
            shutil.rmtree(out / "union", ignore_errors=True)
    mark_done(cand_dir)
    with stage(f"blocking[{split}] report", rep):
        blocking_report(cfg, split)


def _existing_parts(target: Path, q_country: np.ndarray) -> tuple[dict[str, list[int]], int, int]:
    """Parts written by an interrupted run: first query of every part (by country), number of
    parts and rows. Only the last part can be truncated (killed while writing); it is removed."""
    parts = list_parts(target)
    if [p.name for p in parts] != [part_path(target, i).name for i in range(len(parts))]:
        raise RuntimeError(f"resume: part files in {target} are not numbered 0..n-1")
    first_q: dict[str, list[int]] = {}
    n_rows = 0
    for i, p in enumerate(parts):
        try:
            st = pl.scan_parquet(p).select(pl.col("q").min().alias("q0"),
                                           pl.len().alias("n")).collect()
        except Exception:  # noqa: BLE001 - truncated parquet file
            if i != len(parts) - 1:
                raise
            LOG.warning("resume: removing unreadable last part %s", p)
            p.unlink()
            parts = parts[:-1]
            break
        q0 = int(st["q0"][0])
        first_q.setdefault(str(q_country[q0]), []).append(q0)
        n_rows += int(st["n"][0])
    return first_q, len(parts), n_rows


def _retrieve(cfg: dict, split: str, target: Path, ranker, resume: bool = False,
              max_chunks: int | None = None) -> None:
    """``max_chunks``: only the first chunks of each country (``block-bench``)."""
    from .prerank import cut_top_m

    b = cfg["blocking"]
    nt = int(cfg["runtime"]["n_threads"])
    rng = np.random.default_rng(cfg["runtime"]["seed"])
    s1 = load_records(cfg, split, "s1", _REC_COLS)
    q = load_records(cfg, split, "q", _REC_COLS)
    n_s1_total = s1.height
    q_true = load_truth(cfg)["q_true"] if split == "train" else None
    s_country = s1["country"].to_numpy()
    q_country = q["country"].to_numpy()
    countries = sorted(set(s_country.tolist()) | set(q_country.tolist()))
    done_q0, part_i, offset = _existing_parts(target, q_country) if resume else ({}, 0, 0)
    if resume:
        LOG.info("resume: %d parts (%d rows) already written", part_i, offset)
    size = int(b["query_chunk"])
    tmp = split_dir(cfg, split) / "_tmp_block"
    use_part = bool(b.get("partition", {}).get("enabled", False))
    gmaps = state_group_maps(cfg, split, s1, q, q_true) if use_part else {}
    use_rare = int(b.get("rare_name_view", {}).get("k", 0)) > 0
    for country in countries:
        t0 = time.perf_counter()
        s_ids = np.flatnonzero(s_country == country).astype(np.int64)
        q_ids = np.flatnonzero(q_country == country).astype(np.int64)
        LOG.info("blocking[%s] %s: %d S1 x %d queries", split, country, len(s_ids), len(q_ids))
        if len(s_ids) == 0 or len(q_ids) == 0:
            continue
        # drawn for every country (also when resume skips it) so later countries get the same draw
        fit_ids = q_ids if len(q_ids) <= b["fit_sample_queries"] else np.sort(
            rng.choice(q_ids, size=int(b["fit_sample_queries"]), replace=False))
        q_chunks = list(chunks(len(q_ids), size))[:max_chunks]
        done_ci = {int(np.searchsorted(q_ids, x)) // size for x in done_q0.get(country, [])}
        if len(done_ci) == len(q_chunks):
            LOG.info("  resume: all %d chunks already written", len(q_chunks))
            continue
        if done_ci:
            LOG.info("  resume: %d of %d chunks already written", len(done_ci), len(q_chunks))
        views = CountryViews(cfg, s1["name_concat"].gather(s_ids).to_list(),
                             s1["addr_block"].gather(s_ids).to_list(),
                             q["name_concat"].gather(fit_ids).to_list(),
                             q["addr_block"].gather(fit_ids).to_list(),
                             s1["name_core"].gather(s_ids).to_list() if use_rare else None,
                             q["name_core"].gather(fit_ids).to_list() if use_rare else None)
        del fit_ids
        LOG.info("  vocab: name %d, addr %d", len(views.name_vec.vocabulary_),
                 len(views.addr_vec.vocabulary_))
        part, q_group = None, None
        if use_part:
            gmap = gmaps.get(country, {})
            part = Partition(_group_codes(s1["state"].gather(s_ids).to_numpy(), gmap), views)
            q_group = _group_codes(q["state"].gather(q_ids).to_numpy(), gmap)
            LOG.info("  partition: %d state groups, %.1f%% of queries search the whole country",
                     len(part.cols), 100 * float((~np.isin(q_group, list(part.cols))).mean()))

        # exact keys
        key_cols = ["idx", "name_sorted", "hn", "street"]
        s1c = s1.select(key_cols).select(pl.all().gather(s_ids))
        qc = q.select(key_cols).select(pl.all().gather(q_ids))
        kn = _key_pairs(s1c, qc, pl.col("name_sorted"), int(b["key_name_cap"]))
        ka = _key_pairs(s1c, qc, pl.when((pl.col("hn") != "") & (pl.col("street") != ""))
                        .then(pl.col("hn") + "|" + pl.col("street")).otherwise(pl.lit("")),
                        int(b["key_addr_cap"]))
        ex_q = [kn["q"].to_numpy(), ka["q"].to_numpy()]
        ex_s = [kn["s"].to_numpy(), ka["s"].to_numpy()]
        ex_f = [np.full(kn.height, FLAG_KN, np.int16), np.full(ka.height, FLAG_KA, np.int16)]
        LOG.info("  key pairs: name %d, addr %d", kn.height, ka.height)
        del s1c, qc, kn, ka

        k_rev = int(b.get("reverse_k", 0))
        cache: dict[int, tuple] = {}
        if k_rev > 0:
            # each S1 keeps its k_rev best queries over ALL chunks; the running top-k is merged
            # chunk by chunk, so memory stays at ~2 * k_rev rows per S1
            tmp.mkdir(parents=True, exist_ok=True)
            rev = None
            for ci, (a, e) in enumerate(q_chunks):
                qc_ids = q_ids[a:e]
                Q = views.transform(q["name_concat"].gather(qc_ids).to_list(),
                                    q["addr_block"].gather(qc_ids).to_list())
                if ci not in done_ci:
                    for vi, M in enumerate(Q):
                        sp.save_npz(tmp / f"{ci}_{vi}.npz", M, compressed=False)
                    cache[ci] = tuple(tmp / f"{ci}_{vi}.npz" for vi in range(3))
                if part is None:
                    R = _topn(views.S_NA, Q[2].T.tocsr(), k_rev, nt)
                    r_s, r_q, r_v = R.row, R.col, R.data
                else:
                    r_s, r_q, r_v = part.reverse(Q[2], part.rows_by_group(q_group[a:e]), k_rev, nt)
                keep = r_v > 0
                new = pl.DataFrame({"s": s_ids[r_s[keep]], "q": qc_ids[r_q[keep]],
                                    "v": r_v[keep]})
                rev = new if rev is None else _top_k_per_s(pl.concat([rev, new]), k_rev)
                del Q, keep, new, r_s, r_q, r_v
                LOG.info("  reverse chunk %d/%d: %d queries | %.1f min", ci + 1, len(q_chunks),
                         len(qc_ids), (time.perf_counter() - t0) / 60)
            rev = _top_k_per_s(rev, k_rev)
            ex_q.append(rev["q"].to_numpy())
            ex_s.append(rev["s"].to_numpy())
            ex_f.append(np.full(rev.height, FLAG_REV, np.int16))
            LOG.info("  reverse pairs: %d", rev.height)
            del rev
        ex_q_all = np.concatenate(ex_q).astype(np.int64)
        order = np.argsort(ex_q_all, kind="stable")
        ex_q_all = ex_q_all[order]
        ex_s_all = np.concatenate(ex_s).astype(np.int64)[order]
        ex_f_all = np.concatenate(ex_f)[order]
        del ex_q, ex_s, ex_f, order

        for ci, (a, e) in enumerate(q_chunks):
            if ci in done_ci:
                continue
            qc_ids = q_ids[a:e]
            if ci in cache:
                Q = tuple(sp.load_npz(p).tocsr() for p in cache[ci])
            else:
                Q = views.transform(q["name_concat"].gather(qc_ids).to_list(),
                                    q["addr_block"].gather(qc_ids).to_list())
            lo = np.searchsorted(ex_q_all, qc_ids[0], side="left")
            hi = np.searchsorted(ex_q_all, qc_ids[-1], side="right")
            extra = (ex_q_all[lo:hi], ex_s_all[lo:hi], ex_f_all[lo:hi])
            by_group = part.rows_by_group(q_group[a:e]) if part is not None else None
            df = union_features(qc_ids, s_ids, views, Q, extra, s1, q, n_s1_total, cfg,
                                part, by_group)
            if q_true is not None:
                df = df.with_columns(
                    pl.Series("label", (q_true[df["q"].to_numpy()] == df["s"].to_numpy())
                              .astype(np.int8)))
            if ranker is not None:
                df = cut_top_m(ranker, df, part=part_i)
                df = df.with_columns(pl.int_range(offset, offset + df.height, dtype=pl.Int64)
                                     .alias("pair_id"))
                offset += df.height
            if df.height == 0:
                continue
            write_df(df, part_path(target, part_i))
            LOG.info("  chunk %d/%d: %d queries -> %d pairs | %.1f min", ci + 1, len(q_chunks),
                     len(qc_ids), df.height, (time.perf_counter() - t0) / 60)
            part_i += 1
            del df, Q, extra
        shutil.rmtree(tmp, ignore_errors=True)
        del views, ex_q_all, ex_s_all, ex_f_all, cache, part


# ---------------------------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------------------------

def run_block_bench(cfg: dict) -> dict:
    """Retrieval with the current settings (no stage-0 cut) on the first ``bench_chunks`` query
    chunks of every train country: union recall, union size and time, next to the recall of the
    cached candidates for the same queries. Only writes work/train/_bench_union + a report."""
    n = int(cfg["blocking"].get("bench_chunks", 1))
    out = split_dir(cfg, "train")
    target = reset_dir(out / "_bench_union")
    t0 = time.perf_counter()
    _retrieve(cfg, "train", target, None, max_chunks=n)
    minutes = (time.perf_counter() - t0) / 60
    src_cols = ["in_n", "in_a", "in_na", "key_name", "key_addr", "rev"]
    u = pl.concat([pl.read_parquet(p, columns=["q", "s", "label", "in_rn"] + src_cols)
                   for p in list_parts(target)], how="vertical")
    q_true = load_truth(cfg)["q_true"]
    qs = u["q"].unique().to_numpy()
    qs = qs[q_true[qs] >= 0]
    found = np.isin(qs, u.filter(pl.col("label") == 1)["q"].to_numpy())
    only_rn = u.filter((pl.col("label") == 1) & (pl.col("in_rn") == 1)
                       & (pl.sum_horizontal(src_cols) == 0)).height
    old_found = np.zeros(len(qs), dtype=bool)
    if (out / "cand").exists():
        for p in list_parts(out / "cand"):
            c = pl.read_parquet(p, columns=["q", "label"]).filter(pl.col("label") == 1)
            old_found |= np.isin(qs, c["q"].to_numpy())
    qc = load_records(cfg, "train", "q", ["country"])["country"].to_numpy()
    rep = {"minutes_total": round(minutes, 2), "chunks_per_country": n,
           "union_pairs_per_query": float(u.height / max(u["q"].n_unique(), 1)),
           "true_pairs_only_via_rare_view": int(only_rn),
           "note": "reverse search limited to the benched chunks (slightly optimistic)"}
    for c in sorted(set(qc[qs].tolist())):
        m = qc[qs] == c
        rep[c] = {"queries_with_match": int(m.sum()),
                  "new_union_recall": float(found[m].mean()),
                  "cached_candidate_recall_M12": float(old_found[m].mean())}
    save_json(rep, report_dir(cfg) / "blocking_bench.json")
    LOG.info("block-bench: %s", rep)
    shutil.rmtree(target, ignore_errors=True)
    return rep


def blocking_report(cfg: dict, split: str) -> dict:
    out = split_dir(cfg, split)
    cand = pl.concat([pl.read_parquet(p, columns=["q", "s"] + (["label"] if split == "train"
                                                              else []))
                      for p in list_parts(out / "cand")], how="vertical")
    s1 = load_records(cfg, split, "s1", ["country"])
    q = load_records(cfg, split, "q", ["country", "src"])
    n_q, n_s1 = q.height, s1.height
    per_q = np.bincount(cand["q"].to_numpy(), minlength=n_q)
    stats = {
        "pairs": cand.height,
        "cand_per_query_mean": float(per_q.mean()),
        "queries_without_candidates": int((per_q == 0).sum()),
        "cand_per_s1_mean": float(cand.height / max(n_s1, 1)),
    }
    s_c = s1["country"].to_numpy()
    q_c = q["country"].to_numpy()
    full = sum(int((s_c == c).sum()) * int((q_c == c).sum()) for c in set(s_c.tolist()))
    stats["reduction_ratio"] = 1.0 - cand.height / max(full, 1)
    if split == "train":
        tr = load_truth(cfg)
        q_true, s_ntrue = tr["q_true"], tr["s_ntrue"]
        pos = cand.filter(pl.col("label") == 1)
        n_pos_total = int((q_true >= 0).sum())
        stats["pair_recall"] = pos.height / max(n_pos_total, 1)
        tp = np.bincount(pos["s"].to_numpy(), minlength=n_s1)
        f = np.where(s_ntrue == 0, 1.0,
                     np.where(tp > 0, 1.25 * tp / (0.25 * s_ntrue + tp + 1e-12), 0.0))
        stats["oracle_macro_f05"] = float(f.mean())
        by = {}
        q_src = q["src"].to_numpy()
        found = np.zeros(n_q, dtype=bool)
        found[pos["q"].to_numpy()] = True
        for c in sorted(set(q_c.tolist())):
            for src in (2, 3):
                m = (q_c == c) & (q_src == src) & (q_true >= 0)
                if m.any():
                    by[f"{c}/S{src}"] = float(found[m].mean())
        stats["pair_recall_by_country_source"] = by
    save_json(stats, report_dir(cfg) / f"blocking_{split}.json")
    LOG.info("blocking[%s]: %s", split, stats)
    return stats
