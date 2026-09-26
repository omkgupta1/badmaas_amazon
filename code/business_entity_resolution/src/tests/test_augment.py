import numpy as np

from src.augment import _formatted, _noisy_name
from src.context_features import s_level_features
from src.features import _id_runs


def test_noisy_name_is_light():
    rng = np.random.default_rng(0)
    assert _noisy_name("Acme Holdings", "US", rng, 0.0) == "Acme Holdings"
    for _ in range(50):
        out = _noisy_name("Acme Blue Holdings", "US", rng, 1.0)
        assert set(out.split()) - {"Acme", "Blue", "Holdings", "Inc", "LLC", "Co", "Corp", "Ltd",
                                   "Group"} == set()


def test_formatted_follows_source():
    rng = np.random.default_rng(0)
    _, addr = _formatted("Acme", "12 Main St, Springfield, IL", 2, "US", rng, {"US": 1.0})
    assert addr == "12 MAIN ST, SPRINGFIELD, IL"
    _, addr = _formatted("Acme", "12 MAIN ST, SPRINGFIELD, IL", 3, "US", np.random.default_rng(1),
                         {"US": 1.0})
    assert addr != addr.upper()


def test_id_runs_split_at_gaps():
    ids = np.array([7, 5, 6, 2_000_001, 2_000_000, 6])
    assert _id_runs(ids) == [(5, 7), (2_000_000, 2_000_001)]
    assert _id_runs(np.array([3])) == [(3, 3)]


def test_twin_group_is_not_the_best_supported_number():
    # S1 0 at house number 150; its group: three records at 150 (true), two twins at 151
    s = np.zeros(5, dtype=np.int64)
    q = np.arange(5)
    q_hn = np.array([150, 150, 150, 151, 151])
    f = s_level_features(s, q, np.ones(5, np.float32), np.ones(5, bool), q_hn,
                         np.array([2, 3, 2, 2, 3]), np.array([150]), 1)
    assert f["g_n_hn"].tolist() == [2.0] * 5
    assert f["g_hn_b_is_max"].tolist() == [1, 1, 1, 0, 0]
    assert f["g_hn_a_is_max"].tolist() == [1] * 5
    assert f["g_hn_ba_diff"][3] < 0 < 1 + f["g_hn_ba_diff"][0]


def test_nan_augment_blanks_state_only_for_non_empty_addresses():
    from src.train_gbdt import nan_augment
    cols = ["state_eq", "b_addr_empty", "n_ratio"]
    X = np.zeros((2000, 3), dtype=np.float32)
    X[:, 0] = 1.0
    X[1000:, 1] = 1.0                      # second half: empty address
    cfg = {"model": {"nan_augment": {"cols": ["state_eq"], "frac": 0.35}}}
    nan_augment(X, cols, cfg, 0)
    blank = np.isnan(X[:, 0])
    assert 0.25 < blank[:1000].mean() < 0.45 and not blank[1000:].any()
    assert not np.isnan(X[:, 2]).any()


def test_token_space_idf_is_per_country():
    from src.features import TokenSpace
    ts = TokenSpace(["rue paris a", "rue lyon b", "main st x"], ["rue paris a", "main st y"],
                    np.array(["FR", "FR", "US"]), np.array(["FR", "US"]))
    shared = TokenSpace(["rue paris a", "rue lyon b", "main st x"], ["rue paris a", "main st y"])
    s, q = np.array([0]), np.array([0])
    # "rue" is in 3 of 3 French docs: weighted less than when mixed with US docs
    assert ts.pair(s, q)["inter_idf"][0] < shared.pair(s, q)["inter_idf"][0]


def test_replace_house_number():
    from src.augment import replace_house_number
    assert replace_house_number("1206 Lincoln Drive, Upper Dublin, PA", 1206, 1219) == \
        "1219 Lincoln Drive, Upper Dublin, PA"
    assert replace_house_number("095 TAMARISK LN, DEERFIELD, IL", 95, 97) == \
        "97 TAMARISK LN, DEERFIELD, IL"
    assert replace_house_number("B-13, F/F Kailash Colony, New Delhi", 13, 15) == \
        "B-15, F/F Kailash Colony, New Delhi"
    assert replace_house_number("Suite 1206B, Main St", 120, 121) is None   # not standalone
    assert replace_house_number("", 5, 6) is None


def test_sample_shift_profile():
    from src.augment import sample_shift
    rng = np.random.default_rng(0)
    d = np.array([sample_shift(rng, [[1, 1, 0.1], [2, 9, 0.6], [10, 99, 0.3]]) for _ in range(4000)])
    a = np.abs(d)
    assert (a >= 1).all() and (a <= 99).all()
    assert 0.07 < (a == 1).mean() < 0.13 and 0.55 < ((a >= 2) & (a <= 9)).mean() < 0.65
    assert 0.45 < (d > 0).mean() < 0.55


def test_token_space_reuses_train_idf():
    from src.features import TokenSpace
    c = np.array(["US", "US", "US"])
    train = TokenSpace(["main st a", "main st b", "oak rd c"], ["main st a"], c, np.array(["US"]),
                       keep_idf=True)
    ref = {"US": train.idf_tables["US"]}
    # same docs + the saved table: identical weights; a new token gets the capped (rare) IDF
    same = TokenSpace(["main st a", "main st b", "oak rd c"], ["main st a"], c, np.array(["US"]),
                      idf_ref=ref)
    assert np.allclose(same.pair(np.array([0]), np.array([0]))["inter_idf"],
                       train.pair(np.array([0]), np.array([0]))["inter_idf"])
    test = TokenSpace(["main st zzz"], ["main st zzz"], np.array(["US"]), np.array(["US"]),
                      idf_ref=ref)
    assert test.pair(np.array([0]), np.array([0]))["extra_b_maxidf"][0] == 0
    assert test.pair(np.array([0]), np.array([0]))["inter_idf"][0] > \
        train.pair(np.array([0]), np.array([0]))["inter_idf"][0]


def test_agreement_stats_on_a_small_frame():
    import polars as pl

    from src.diagnose import agreement_stats
    nan = float("nan")
    d = pl.DataFrame({"q": [1, 2, 10, 11], "name_ratio0": [90.0, nan, 80.0, 70.0],
                      "addr_tset0": [100.0, 50.0, 90.0, 90.0], "hn_eq0": [1.0, nan, 0.0, 1.0],
                      "n_ratio": [90.0, nan, 80.0, 60.0], "ad_tset": [100.0, 50.0, 90.0, 90.0],
                      "hn_eq": [1.0, nan, 0.0, 0.0]})
    t = agreement_stats(d, n_real=10).sort("synthetic")
    real, syn = t.row(0, named=True), t.row(1, named=True)
    assert real["n"] == 2 and real["d_name"] == 0 and real["hn_agree"] == 2
    assert syn["n"] == 2 and syn["d_name"] == 10 and syn["hn_agree"] == 1
