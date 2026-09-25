from src.text_norm import NAME_FIELDS, domain_core, fix_leet, normalize_name

F = {f: i for i, f in enumerate(NAME_FIELDS)}


def name(raw, lex=None):
    out = normalize_name(raw, lex)
    return {f: out[i] for f, i in F.items()}


def test_alias_split_keeps_operating_name():
    n = name("Veodovaio doing business as Big 67 Hypnosis Inc.")
    assert n["name_norm"] == "big 67 hypnosis inc"
    assert n["name_other"] == "veodovaio"
    assert n["has_alias"] == 1
    assert n["name_core"] == "big 67 hypnosis"
    assert n["name_nums"] == "67"


def test_formerly_and_pipe():
    assert name("Evoonyx Formerly Naqvi & Gao Goldman Associates")["name_core"] == \
        "naqvi gao goldman associates"
    n = name("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com")
    assert n["name_norm"] == "shivshakti vidyalaya overseas corp"
    assert n["name_other"] == "shivshakti"


def test_dotted_legal_forms():
    n = name("Bast, Marti, P.A., M.D., P.C.")
    assert n["name_norm"] == "bast marti pa md pc"
    assert n["name_core"] == "bast marti md"


def test_domains_and_handles():
    assert domain_core("bastmartipa.com") == "bastmartipa"
    assert domain_core("www.arcyuma.com") == "arcyuma"
    assert domain_core("@kadambagachiconstructions") == "kadambagachiconstructions"
    assert domain_core("pediatricspecialistscom") == "pediatricspecialists"
    n = name("PÉDIATRICSPECIALISTSCOM")
    assert n["is_domain"] == 1 and n["name_concat"] == "pediatricspecialists"


def test_leetspeak_and_ordinals():
    assert fix_leet("ec0nomic") == "economic"
    assert fix_leet("behavi0ral") == "behavioral"
    assert fix_leet("1st") == "1st"
    assert fix_leet("4608") == "4608"


def test_indian_legal_forms_and_duplicates():
    n = name("Kadambagachi Constructions Constructions Porveat Limited")
    assert n["name_norm"] == "kadambagachi constructions porveat ltd"
    n = name("Pvt. EFS Print Ventures Ltd.")
    assert n["name_core"] == "efs print ventures"
    assert name("Abc (P) Ltd")["name_norm"] == "abc pvt ltd"


def test_transliterated_name_matches_latin():
    lex = {"प्राइवेट": "private", "बिल्डर्स": "builders"}
    assert name("रेड बिल्डर्स प्राइवेट लिमिटेड", lex)["name_norm"] == "red builders pvt ltd"
    assert name("Red Builders Private Limited")["name_norm"] == "red builders pvt ltd"


def test_accents_folded():
    assert name("Big 67 Hypnosis Ínc.")["name_norm"] == "big 67 hypnosis inc"
