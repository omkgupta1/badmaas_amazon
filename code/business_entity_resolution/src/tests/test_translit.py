from src.translit import has_indic, phonetic_simplify, script_of, transliterate_text, translit_indic


def test_devanagari_schwa_deletion():
    assert translit_indic("रेड") == "red"
    assert translit_indic("लिमिटेड") == "limited"


def test_dravidian_keeps_final_vowel():
    assert translit_indic("ಕರ್ನಾಟಕ") == "karnaataka"


def test_native_digits():
    assert translit_indic("१२३") == "123"


def test_lexicon_first_then_rules():
    out = transliterate_text("रेड बिल्डर्स", {"बिल्डर्स": "builders"})
    assert out.split() == ["red", "builders"]


def test_script_detection():
    assert script_of("Kolhapur, महाराष्ट्र") == 1
    assert script_of("plain latin") == 0
    assert has_indic("ಕರ್ನಾಟಕ") and not has_indic("Karnataka")


def test_phonetic_simplify_merges_spellings():
    assert phonetic_simplify("raibareilly") == phonetic_simplify("raebareli")
