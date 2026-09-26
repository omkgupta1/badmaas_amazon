import numpy as np
import polars as pl
import scipy.sparse as sp

from src.blocking import Partition, _group_codes, state_group_maps


class _Views:
    """Minimal stand-in for CountryViews: 4 S1 rows over 3 features."""

    def __init__(self):
        S = sp.csr_matrix(np.array([[1.0, 0, 0], [0.9, 0.1, 0], [1.0, 0, 0], [0, 0, 1.0]],
                                   dtype=np.float32))
        self.S_N = self.S_A = self.S_NA = S
        self.S_N_T = self.S_A_T = self.S_NA_T = S.T.tocsr()


def test_partition_searches_own_group_and_falls_back_to_country():
    views = _Views()
    part = Partition(np.array([0, 0, 1, 1]), views)       # S1 0,1 in group 0; 2,3 in group 1
    Q = sp.csr_matrix(np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]], dtype=np.float32))
    by = part.rows_by_group(np.array([0, 1, -1]))         # q0 group 0, q1 group 1, q2 unknown
    rows, cols, vals = part.topn(Q, 0, by, 1, 1)
    best = {int(r): int(c) for r, c in zip(rows, cols)}
    assert best[0] in (0, 1)          # only group-0 S1 records
    assert best[1] == 2               # only group-1 S1 records
    assert best[2] in (0, 2)          # whole country
    s_loc, q_loc, v = part.reverse(Q, by, 1, 1)
    assert set(zip(s_loc.tolist(), q_loc.tolist())) <= {(0, 0), (1, 0), (2, 1), (3, 1)}


def test_state_groups_merge_confusable_states(tmp_path):
    cfg = {"paths": {"work_dir": str(tmp_path)},
           "blocking": {"partition": {"min_merge_pairs": 2, "min_merge_share": 0.2}}}
    (tmp_path / "models").mkdir()
    s1 = pl.DataFrame({"country": ["India"] * 3, "state": ["andhra pradesh", "telangana", "delhi"]})
    # 3 true pairs: two AP-labelled records of the Telangana S1, one Delhi pair
    q = pl.DataFrame({"state": ["andhra pradesh", "andhra pradesh", "delhi", ""]})
    q_true = np.array([1, 1, 2, -1])
    maps = state_group_maps(cfg, "train", s1, q, q_true)["India"]
    assert maps["andhra pradesh"] == maps["telangana"] != maps["delhi"]
    codes = _group_codes(np.array(["telangana", "", "bihar"]), maps)
    assert codes[0] == maps["telangana"] and codes[1] == -1 and codes[2] == -1
