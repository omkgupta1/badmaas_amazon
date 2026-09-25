import numpy as np
import polars as pl

from src.decide import EFGroups, assign
from src.evaluate import f05_per_s1, macro_f05_pairs, macro_f05_sets
from src.features import _Lists, hn_features


def test_problem_statement_example():
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    truth = {"S1-00001": {"S2-00047", "S3-00812"}}
    assert abs(macro_f05_sets(pred, truth, ["S1-00001"]) - 0.7142857) < 1e-6


def test_singleton_rules():
    assert macro_f05_sets({}, {}, ["a"]) == 1.0
    assert macro_f05_sets({"a": {"x"}}, {}, ["a"]) == 0.0
    assert macro_f05_sets({}, {"a": {"x"}}, ["a"]) == 0.0


def test_count_formula_matches_sets():
    # S1 0: 2 true, predicts 3 with 2 correct; S1 1: singleton, empty; S1 2: 1 true, empty
    n_true = np.array([2, 0, 1])
    s_kept = np.array([0, 0, 0])
    lab = np.array([1, 1, 0])
    got = macro_f05_pairs(s_kept, lab, n_true)
    assert abs(got - (0.7142857 + 1.0 + 0.0) / 3) < 1e-6
    assert f05_per_s1(np.array([0]), np.array([0]), np.array([0]))[0] == 1.0


def test_expected_f_rule():
    s = np.array([0, 0, 0, 1])
    p = np.array([0.95, 0.9, 0.2, 0.3])
    g = EFGroups(s, p)
    keep = g.select(np.array([0.01, 0.9]), mhat=0.0)
    assert keep.tolist() == [True, True, False, False]
    keep = g.select(np.array([0.99, 0.0]), mhat=0.0)
    assert keep.tolist() == [False, False, False, True]


def test_assign_keeps_best_candidate_and_margin():
    df = pl.DataFrame({"q": [7, 7, 8], "s": [1, 2, 1], "p": [0.2, 0.9, 0.6]})
    a = assign(df).sort("q")
    assert a["s"].to_list() == [2, 1]
    assert abs(a["margin"][0] - 0.7) < 1e-6 and abs(a["margin"][1] - 0.6) < 1e-6


def test_house_number_features():
    A = pl.DataFrame({"hn": ["106", "1133", "19015"], "hn_int": [106, 1133, 19015],
                      "hn_hi": [-1, -1, -1], "nums": ["106", "1133", "19015"],
                      "hid": ["106", "1133", "19015"]})
    B = pl.DataFrame({"hn": ["109", "8133", "1901"], "hn_int": [109, 8133, 1901],
                      "hn_hi": [-1, -1, -1], "nums": ["109", "8133", "1901"],
                      "hid": ["109", "8133", "1901"]})
    idx = np.arange(3)
    f = hn_features(_Lists(A, idx), _Lists(B, idx), nt=1)
    assert f["hn_eq"].tolist() == [0, 0, 0]
    assert f["hn_single_sub"].tolist() == [1, 1, 0]
    assert f["hn_pow10"][:2].tolist() == [1, 1]
    assert f["hn_last_eq"].tolist() == [0, 1, 0]
    assert f["hn_first_eq"].tolist() == [1, 0, 1]
    assert f["hn_substr"].tolist() == [0, 0, 1]
