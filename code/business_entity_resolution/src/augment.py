"""Synthetic decoy groups for the train split.

Test decoy businesses come as groups of 2+ S2/S3 records that share their own house number, a small
shift of the S1's number; train decoys are almost always single records. After train blocking two
kinds of synthetic orphan records are added:

  decoy copies  a share of train decoys (orphans whose top stage-0 candidate has a similar name and
                a different house number) get a copy in the same or the other source;
  true twins    for a share of S1s, a group of records is made from the S1's TRUE records: the
                address of a true record with the house number shifted by a small delta (profile
                measured on test), each copy named like a different true record of the S1 (so the
                names are noised independently, as in test), formatted like its source.

Every synthetic record is parsed like a real one (prep._parse_chunk) and inherits the candidate S1
list of the record its address came from. All stage-0 columns (TF-IDF cosines and their ranks, fuzzy
scores, exact keys, house-number equality, s0 and s0_rank) are RECOMPUTED with blocking's own views,
so no column gives a synthetic record away. Rates per country follow the decoy excess measured on
test, which makes train OOF a test-like metric for tuning the decision rule.

Synthetic query ids are >= the real query count stored in records_q/_DONE; ``strip_synthetic``
removes everything this stage added (blocking calls it before re-blocking).
"""
from __future__ import annotations

import multiprocessing as mp
import re
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .config import lexicon_dir, split_dir
from .prep import _init_worker, _parse_chunk
from .records import load_records
from .utils import LOG, chunks, list_parts, load_json, part_path, save_json, write_df

MARKER = "_AUGMENTED"
_LEGAL = {
    "US": ["Inc", "LLC", "Co", "Corp", "Ltd", "Group"],
    "India": ["Pvt Ltd", "Private Limited", "LLP", "Ltd", "Limited"],
}
_LEGAL_DEFAULT = ["Inc", "Ltd", "Co", "Group"]
_LEGAL_WORDS = {w.lower().strip(".") for ws in list(_LEGAL.values()) + [_LEGAL_DEFAULT]
                for x in ws for w in x.split()} | {"l.l.c", "llc.", "inc.", "pvt", "private"}
_RESTAGED = ["cos_n", "cos_a", "cos_na", "r_n", "r_a", "r_na", "key_name", "key_addr", "hn_eq0",
             "name_ratio0", "addr_tset0", "b_addr_empty0", "b_is_domain0"]


def n_real_queries(cfg: dict) -> int:
    return int(load_json(split_dir(cfg, "train") / "records_q" / "_DONE")["rows"])


def strip_synthetic(cfg: dict) -> int:
    """Remove synthetic records, raw rows, truth entries and candidate rows added by augment."""
    out = split_dir(cfg, "train")
    if not (out / "records_q" / "_DONE").exists():
        return 0
    n_real = n_real_queries(cfg)
    removed = 0
    for p in list_parts(out / "records_q"):
        if pl.scan_parquet(p).select(pl.col("idx").min()).collect().item() >= n_real:
            p.unlink()
            removed += 1
    raw_path = out / "raw_q.parquet"
    if raw_path.exists():
        n_raw = pl.scan_parquet(raw_path).select(pl.len()).collect().item()
        if n_raw > n_real:
            write_df(pl.read_parquet(raw_path).filter(pl.col("idx") < n_real), raw_path)
    qt_path = out / "q_true.npy"
    if qt_path.exists():
        qt = np.load(qt_path)
        if len(qt) > n_real:
            np.save(qt_path, qt[:n_real])
    cand = out / "cand"
    for p in list_parts(cand) if cand.exists() else []:  # also cleans an augment that crashed
        if pl.scan_parquet(p).select(pl.col("q").max()).collect().item() >= n_real:
            write_df(pl.read_parquet(p).filter(pl.col("q") < n_real), p)
    if (cand / MARKER).exists():
        (cand / MARKER).unlink()
    if removed:
        LOG.info("augment: removed %d synthetic record parts", removed)
    return removed


# ---------------------------------------------------------------------------------------------
# record construction
# ---------------------------------------------------------------------------------------------

def _noisy_name(name: str, country: str, rng: np.random.Generator, p_noise: float) -> str:
    toks = name.split()
    if not toks or rng.random() >= p_noise:
        return name
    op = rng.integers(4)
    last = toks[-1].lower().strip(".,()")
    if op == 0 and last not in _LEGAL_WORDS:                      # add a legal form
        forms = _LEGAL.get(country, _LEGAL_DEFAULT)
        toks.append(forms[rng.integers(len(forms))])
    elif op == 1 and len(toks) >= 3:                              # drop the last token
        toks = toks[:-1]
    elif op == 2:                                                 # duplicate the last token
        toks.append(toks[-1])
    elif op == 3 and len(toks) >= 2:                              # swap two adjacent tokens
        i = int(rng.integers(len(toks) - 1))
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return " ".join(toks)


def _formatted(name: str, addr: str, src: int, country: str, rng: np.random.Generator,
               p_s2_upper: dict) -> tuple[str, str]:
    """Casing of the copy follows its source (S2 addresses are mostly upper case in the US)."""
    if src == 2:
        if rng.random() < float(p_s2_upper.get(country, p_s2_upper.get("default", 0.5))):
            addr = addr.upper()
        if rng.random() < 0.2:
            name = name.upper()
    else:
        if addr and addr == addr.upper() and rng.random() < 0.95:
            addr = addr.title()
        if name and name == name.upper() and rng.random() < 0.8:
            name = name.title()
    return name, addr


def replace_house_number(addr: str, hn_int: int, new: int) -> str | None:
    """``addr`` with the first standalone occurrence of number ``hn_int`` (zero padding allowed)
    replaced by ``new``; None when the number does not occur."""
    m = re.search(r"(?<![0-9])0*" + str(int(hn_int)) + r"(?![0-9])", addr)
    if m is None:
        return None
    return addr[:m.start()] + str(int(new)) + addr[m.end():]


def sample_shift(rng: np.random.Generator, buckets: list) -> int:
    """Signed house-number shift from ``[[lo, hi, weight], ...]`` buckets of |delta|."""
    b = np.asarray(buckets, dtype=np.float64)
    i = int(rng.choice(len(b), p=b[:, 2] / b[:, 2].sum()))
    d = int(rng.integers(int(b[i, 0]), int(b[i, 1]) + 1))
    return d if rng.random() < 0.5 else -d


def _parse(raw: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    lex = lexicon_dir(cfg)
    indic = load_json(lex / "indic.json")
    addr_lex = load_json(lex / "address_train.json")
    n_workers = max(1, int(cfg["runtime"]["n_workers"]))
    tasks = [(raw["business_name"].slice(a, b - a).to_list(),
              raw["business_address"].slice(a, b - a).to_list(),
              raw["country"].slice(a, b - a).to_list()) for a, b in chunks(raw.height, 25_000)]
    ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(indic, addr_lex, False)) as pool:
        parts = pool.map(_parse_chunk, tasks)
    return pl.concat(parts, how="vertical")


# ---------------------------------------------------------------------------------------------
# stage-0 columns of synthetic candidate rows
# ---------------------------------------------------------------------------------------------

def _restage(cfg: dict, rows: pl.DataFrame, syn: pl.DataFrame, n_real: int) -> pl.DataFrame:
    """Recompute the stage-0 columns of synthetic candidate ``rows`` (q = synthetic id, with a
    ``_part`` column) from the parsed synthetic records ``syn``, exactly as blocking computes them
    (same per-country TF-IDF views: same fit documents and random draws as blocking._retrieve)."""
    from .blocking import (_REC_COLS, CountryViews, _cpdist, _group_rank, _nan_empty, rowdot)
    from .prerank import load_prerank

    b = cfg["blocking"]
    nt = int(cfg["runtime"]["n_threads"])
    ranker = load_prerank(cfg)
    if ranker is None:
        raise FileNotFoundError("stage-0 ranker missing — run `block --split train` first")
    s1 = load_records(cfg, "train", "s1", _REC_COLS)
    qr = load_records(cfg, "train", "q", ["country", "name_concat", "addr_block"]).head(n_real)
    s_country = s1["country"].to_numpy()
    q_country = qr["country"].to_numpy()
    syn = syn.sort("idx")
    syn_idx = syn["idx"].to_numpy()
    rng = np.random.default_rng(cfg["runtime"]["seed"])  # the draws blocking._retrieve made
    row_country = s_country[rows["s"].to_numpy()]
    out = []
    for country in sorted(set(s_country.tolist()) | set(q_country.tolist())):
        s_ids = np.flatnonzero(s_country == country).astype(np.int64)
        q_ids = np.flatnonzero(q_country == country).astype(np.int64)
        if len(s_ids) == 0 or len(q_ids) == 0:
            continue
        fit_ids = q_ids if len(q_ids) <= b["fit_sample_queries"] else np.sort(
            rng.choice(q_ids, size=int(b["fit_sample_queries"]), replace=False))
        m = row_country == country
        if not m.any():
            continue
        views = CountryViews(cfg, s1["name_concat"].gather(s_ids).to_list(),
                             s1["addr_block"].gather(s_ids).to_list(),
                             qr["name_concat"].gather(fit_ids).to_list(),
                             qr["addr_block"].gather(fit_ids).to_list())
        r = rows.filter(pl.Series(m))
        qs = r["q"].to_numpy().astype(np.int64)
        ss = r["s"].to_numpy().astype(np.int64)
        uq = np.unique(qs)
        sub = syn.select(pl.all().gather(np.searchsorted(syn_idx, uq)))
        QN, QA, QNA = views.transform(sub["name_concat"].to_list(), sub["addr_block"].to_list())
        rq = np.searchsorted(uq, qs)
        rs = np.searchsorted(s_ids, ss)
        cos = {"cos_n": rowdot(QN, rq, views.S_N, rs), "cos_a": rowdot(QA, rq, views.S_A, rs),
               "cos_na": rowdot(QNA, rq, views.S_NA, rs)}
        del views, QN, QA, QNA
        # exact keys with blocking's group-size caps
        s1c = s1.select(pl.all().gather(s_ids))
        fam = dict(zip(*s1c.group_by("name_sorted").agg(pl.len().alias("n"))
                       .filter(pl.col("name_sorted") != "").select(["name_sorted", "n"])
                       .to_dict(as_series=False).values()))
        akey = pl.when((pl.col("hn") != "") & (pl.col("street") != "")).then(
            pl.col("hn") + "|" + pl.col("street")).otherwise(pl.lit(""))
        s_ak = s1.select(akey.alias("k"))["k"]
        grp = dict(zip(*s1c.select(akey.alias("k")).filter(pl.col("k") != "").group_by("k")
                       .agg(pl.len().alias("n")).to_dict(as_series=False).values()))
        q_ns = sub["name_sorted"].gather(rq)
        s_ns = s1["name_sorted"].gather(ss)
        q_ak = sub.select(akey.alias("k"))["k"].gather(rq)
        s_akg = s_ak.gather(ss)
        key_name = np.array([a != "" and a == c and fam.get(c, 0) <= int(b["key_name_cap"])
                             for a, c in zip(q_ns.to_list(), s_ns.to_list())], dtype=np.float32)
        key_addr = np.array([a != "" and a == c and grp.get(c, 0) <= int(b["key_addr_cap"])
                             for a, c in zip(q_ak.to_list(), s_akg.to_list())], dtype=np.float32)
        q_hn, s_hn = sub["hn"].gather(rq), s1["hn"].gather(ss)
        hn_eq = np.where((q_hn != "").to_numpy() & (s_hn != "").to_numpy(),
                         (q_hn == s_hn).to_numpy().astype(np.float32), np.nan).astype(np.float32)
        qn, sn = sub["name_norm"].gather(rq), s1["name_norm"].gather(ss)
        qa, sa = sub["addr_norm"].gather(rq), s1["addr_norm"].gather(ss)
        new = {**cos,
               "r_n": _group_rank(rq, cos["cos_n"]), "r_a": _group_rank(rq, cos["cos_a"]),
               "r_na": _group_rank(rq, cos["cos_na"]),
               "key_name": key_name, "key_addr": key_addr, "hn_eq0": hn_eq,
               "name_ratio0": _nan_empty(_cpdist(qn, sn, fuzz.ratio, nt), qn, sn),
               "addr_tset0": _nan_empty(_cpdist(qa, sa, fuzz.token_set_ratio, nt), qa, sa),
               "b_addr_empty0": sub["addr_empty"].gather(rq).cast(pl.Float32).to_numpy(),
               "b_is_domain0": sub["is_domain"].gather(rq).cast(pl.Float32).to_numpy()}
        r = r.with_columns([pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in new.items()])
        # s0 with the booster of the part the rows are appended to, then ranks within each query
        scored = []
        for p_idx in np.unique(r["_part"].to_numpy()):
            d = r.filter(pl.col("_part") == p_idx)
            scored.append(d.with_columns(pl.Series("s0", ranker.score(d, part=int(p_idx)))))
        r = pl.concat(scored, how="vertical")
        r = r.with_columns(pl.col("s0").rank("ordinal", descending=True).over("q")
                           .cast(pl.Int16).alias("s0_rank"))
        out.append(r)
        LOG.info("  augment restage %s: %d synthetic rows", country, r.height)
    return pl.concat(out, how="vertical")


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------

def run_augment(cfg: dict) -> dict:
    out = split_dir(cfg, "train")
    cand_dir = out / "cand"
    a = cfg.get("augment", {})
    if not a.get("enabled", False):
        LOG.info("augment: disabled")
        return {}
    if (cand_dir / MARKER).exists():
        prev = load_json(cand_dir / MARKER)
        if prev.get("config") == a:
            LOG.info("augment: cached")
            return prev
        LOG.info("augment: settings changed since the last augment — rebuilding")
    strip_synthetic(cfg)
    n_real = n_real_queries(cfg)
    rng = np.random.default_rng(int(cfg["runtime"]["seed"]) + 17)
    q_true = np.load(out / "q_true.npy")
    raw = pl.read_parquet(out / "raw_q.parquet")
    q_country = raw["country"].to_numpy()
    q_src = raw["src"].to_numpy().astype(np.int64)
    q_name = raw["business_name"].to_numpy()
    q_addr = raw["business_address"].to_numpy()
    s1 = load_records(cfg, "train", "s1", ["country", "hn", "hn_int"])
    s_country = s1["country"].to_numpy()
    s_hn_int = s1["hn_int"].to_numpy()
    s_has_hn = (s1["hn"].str.len_chars() > 0).to_numpy()
    dc, tt = a.get("decoy_copies", {}), a.get("true_twins", {})
    p_noise = float(a.get("name_noise", 0.6))
    p_upper = a.get("s2_address_upper", {"US": 0.9, "India": 0.25, "default": 0.5})

    # 1. scan candidates: eligible decoys (per part) and true pairs
    parts = list_parts(cand_dir)
    entries: list[tuple[int, int, int, str, str]] = []  # (part, address-source q, src, name, addr)
    true_rows = []
    for pi, p in enumerate(parts):
        t = pl.read_parquet(p, columns=["q", "s", "s0_rank", "name_ratio0", "hn_eq0", "label"])
        top = t.filter((pl.col("s0_rank") == 1)
                       & (pl.col("name_ratio0") >= float(dc.get("min_name_ratio", 80)))
                       & (pl.col("hn_eq0") == 0))
        dq = np.sort(top["q"].to_numpy().astype(np.int64))
        dq = dq[q_true[dq] < 0]
        rate = np.array([float(dc.get("rate", {}).get(c, dc.get("default_rate", 0.0)))
                         for c in q_country[dq]])
        for q in dq[rng.random(len(dq)) < rate]:
            src = int(q_src[q]) if rng.random() < float(a.get("same_source_frac", 0.5)) \
                else 5 - int(q_src[q])
            nm = _noisy_name(q_name[q], q_country[q], rng, p_noise)
            nm, ad = _formatted(nm, q_addr[q], src, q_country[q], rng, p_upper)
            entries.append((pi, int(q), src, nm, ad))
        tr = t.filter(pl.col("label") == 1).select(["q", "s", "hn_eq0"])
        true_rows.append(tr.with_columns(pl.lit(pi, dtype=pl.Int32).alias("_part")))
    n_copies = len(entries)

    # 2. true twins: per selected S1, one address-source record with the S1's own house number
    tp = pl.concat(true_rows, how="vertical")
    names_by_s = tp.group_by("s").agg(pl.col("q"))
    names_by_s = dict(zip(names_by_s["s"].to_list(), names_by_s["q"].to_list()))
    base = (tp.filter(pl.col("hn_eq0") == 1).sample(fraction=1.0, shuffle=True,
                                                    seed=int(cfg["runtime"]["seed"]))
            .unique(subset=["s"], keep="first", maintain_order=True).sort("s"))
    bs = base["s"].to_numpy().astype(np.int64)
    ok = s_has_hn[bs] & (s_hn_int[bs] >= 1)
    rate = np.array([float(tt.get("rate", {}).get(c, tt.get("default_rate", 0.0)))
                     for c in s_country[bs]])
    pick = ok & (rng.random(len(bs)) < rate)
    size = int(tt.get("group_size", 2))
    buckets = tt.get("shift_buckets", [[1, 1, 0.1], [2, 9, 0.6], [10, 99, 0.3]])
    n_groups = 0
    for s, q_addr_src, part in base.filter(pl.Series(pick)).select(["s", "q", "_part"]).iter_rows():
        hn = int(s_hn_int[s])
        d = sample_shift(rng, buckets)
        new_hn = hn + d if hn + d >= 1 else hn - d
        pool = names_by_s.get(s, [q_addr_src])
        made = []
        for _ in range(size):
            qn = int(pool[int(rng.integers(len(pool)))])
            src = int(q_src[qn]) if rng.random() >= float(tt.get("cross_source", 0.5)) \
                else 5 - int(q_src[qn])
            ad = replace_house_number(q_addr[q_addr_src], hn, new_hn)
            if ad is None:
                break
            nm = _noisy_name(q_name[qn], s_country[s], rng, p_noise)
            nm, ad = _formatted(nm, ad, src, s_country[s], rng, p_upper)
            made.append((int(part), int(q_addr_src), src, nm, ad))
        if len(made) == size:
            entries.extend(made)
            n_groups += 1
    if not entries:
        save_json({"n_syn": 0, "n_real": n_real, "config": a}, cand_dir / MARKER)
        return {"n_syn": 0}

    # 3. synthetic ids consecutive within each part; raw rows, parsed records, truth
    entries.sort(key=lambda e: e[0])
    n_syn = len(entries)
    new_ids = n_real + np.arange(n_syn, dtype=np.int64)
    e_part = np.array([e[0] for e in entries], dtype=np.int64)
    e_src_q = np.array([e[1] for e in entries], dtype=np.int64)
    e_src = np.array([e[2] for e in entries], dtype=np.int64)
    syn_raw = pl.DataFrame({
        "idx": new_ids, "entity_id": [f"SYN{int(sv)}-{int(i)}" for sv, i in zip(e_src, new_ids)],
        "src": e_src, "business_name": [e[3] for e in entries],
        "business_address": [e[4] for e in entries], "country": q_country[e_src_q],
    }).cast(dict(raw.schema))
    write_df(pl.concat([raw, syn_raw], how="vertical"), out / "raw_q.parquet")
    del raw
    rec_parts = list_parts(out / "records_q")
    schema = pl.read_parquet_schema(rec_parts[0])
    rec = pl.concat([syn_raw.select(["idx", "entity_id", "src", "country"]), _parse(syn_raw, cfg)],
                    how="horizontal").select(list(schema)).cast(dict(schema))
    write_df(rec, part_path(out / "records_q", len(rec_parts)))
    np.save(out / "q_true.npy", np.concatenate([q_true, np.full(n_syn, -1, q_true.dtype)]))

    # 4. candidate rows: copy the address-source record's rows, then recompute stage-0 columns
    mapping = pl.DataFrame({"q": e_src_q, "q_new": new_ids, "_part": e_part})
    syn_rows = []
    for pi, p in enumerate(parts):
        mp_ = mapping.filter(pl.col("_part") == pi)
        if mp_.height == 0:
            continue
        df = pl.read_parquet(p)
        m = mp_.select(["q", "q_new"]).cast({"q": df.schema["q"], "q_new": df.schema["q"]})
        syn_rows.append(df.join(m, on="q", how="inner").drop("q").rename({"q_new": "q"})
                        .with_columns(pl.lit(pi, dtype=pl.Int32).alias("_part")))
    rows = pl.concat(syn_rows, how="vertical")
    rows = _restage(cfg, rows, rec, n_real)
    max_pid = max(pl.scan_parquet(p).select(pl.col("pair_id").max()).collect().item()
                  for p in parts)
    next_pid = int(max_pid) + 1
    for pi, p in enumerate(parts):
        extra = rows.filter(pl.col("_part") == pi)
        if extra.height == 0:
            continue
        df = pl.read_parquet(p)
        extra = (extra.drop("_part").with_columns(pl.lit(0).alias("label"))
                 .select(df.columns).cast(dict(df.schema)).sort(["q", "s0_rank"]))
        extra = extra.with_columns(
            pl.int_range(next_pid, next_pid + extra.height, dtype=df.schema["pair_id"])
            .alias("pair_id"))
        next_pid += extra.height
        write_df(pl.concat([df, extra], how="vertical"), p)
    info = {"n_syn": n_syn, "n_real": n_real, "decoy_copies": n_copies,
            "true_twin_records": n_syn - n_copies, "true_twin_groups": n_groups,
            "per_country": {c: int((q_country[e_src_q] == c).sum())
                            for c in sorted(set(q_country[e_src_q].tolist()))},
            "candidate_rows": rows.height}
    save_json({**info, "config": a}, cand_dir / MARKER)
    LOG.info("augment: %s", info)
    return info
