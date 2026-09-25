import math

import polars as pl

from src.features import format_frame


def _flags(name: str, addr: str = "12 Main Street, Springfield, IL") -> dict:
    df = pl.DataFrame({"business_name": [name], "business_address": [addr]})
    return format_frame(df).row(0, named=True)


def test_case_share():
    assert _flags("ACME LTD")["fmt_n_upper"] == 1.0
    assert _flags("acme ltd")["fmt_n_upper"] == 0.0
    assert 0.0 < _flags("Acme Ltd")["fmt_n_upper"] < 1.0
    assert _flags("रेड बिल्डर्स")["fmt_n_upper"] is None  # no Latin letters


def test_noise_flags():
    assert _flags("Acmé Ltd")["fmt_n_accent"]
    assert _flags("Ec0nomic Ltd")["fmt_n_leet"]
    assert not _flags("3M Company")["fmt_n_leet"]
    assert _flags(">> Acme Ltd")["fmt_n_junk"]
    assert _flags("Faloify Inc #77339")["fmt_n_idnum"]
    assert _flags("Korgild (ID: 58498)")["fmt_n_idnum"]
    assert _flags("Acme  Ltd")["fmt_n_dbl"]
    assert _flags("Glyphiuma Payments [Partners]")["fmt_n_bracket"]
    clean = _flags("Acme Ltd")
    assert clean["fmt_n_noise"] == 0
    assert _flags("Acmé  Ltd #12345")["fmt_n_noise"] == 3


def test_address_flags():
    assert _flags("Acme", "C-##56, SOUTH DELHI, Delhi")["fmt_a_hashhash"]
    assert _flags("Acme", "095 TAMARISK LN, DEERFIELD, IL")["fmt_a_zeropad"]
    assert not _flags("Acme", "1095 TAMARISK LN, DEERFIELD, IL")["fmt_a_zeropad"]
    assert _flags("Acme", "2598 79, NULL, INDIAN MOUND, TN")["fmt_a_null"]
    assert _flags("Acme", "")["fmt_a_upper"] is None
    assert not math.isnan(_flags("Acme", "12 MAIN ST")["fmt_a_upper"])
