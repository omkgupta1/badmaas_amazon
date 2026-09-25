from src.parse_address import ADDR_FIELDS, parse_address

F = {f: i for i, f in enumerate(ADDR_FIELDS)}
US = {"nc": "nc", "oh": "oh", "vt": "vt", "vermont": "vt", "tx": "tx", "texas": "tx",
      "mn": "mn", "mo": "mo", "missouri": "mo"}


def addr(raw, states=None, abbr=None):
    out = parse_address(raw, None, states, abbr)
    return {f: out[i] for f, i in F.items()}


def test_us_basic():
    a = addr("1795 Westchester Drive, High Point, NC", US)
    assert (a["hn"], a["street"], a["street_type"]) == ("1795", "westchester", "drive")
    assert (a["city"], a["state"]) == ("high point", "nc")


def test_reordered_components():
    a = addr("OH, Columbus, 5559 Orville Avenue", US)
    assert (a["hn"], a["street"], a["state"], a["city"]) == ("5559", "orville", "oh", "columbus")


def test_state_synonyms():
    assert addr("Vermont, Essex Town, 3 Jacson Heights", US)["state"] == "vt"
    assert addr("3 JACKSON HEIGHTS, ESSEX TOWN, VT", US)["state"] == "vt"


def test_labelled_indian_house_numbers():
    a = addr("H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building")
    assert a["hn"] == "16" and a["hid"] == "16112337a"
    a = addr("DOOR NO D/1 JHULELAL MARG, SHIPRA PATH MANSAROVAR, JAIPUR, Rajasthan")
    assert a["hn"] == "1" and a["hid"] == "d1" and a["street"] == "jhulelal"
    assert addr("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi")["hn"] == "570"
    assert addr("NO 12 , SECTOR 25 PANCHKULA, PANCHKUAL, Haryana")["hn"] == "12"


def test_number_noise():
    assert addr("0010413 COBALT FALLS DRIVE, HOUTON, TX", US)["hn"] == "10413"
    a = addr("274-278 Central Avenue, Lockport, New York")
    assert (a["hn"], a["hn_hi"]) == ("274", 278)
    assert addr("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", US)["hn"] == "19"
    assert addr("2260- Housecreek Trail, Unit 407, Raleigh, North Carolina")["street"] == \
        "housecreek"


def test_units_and_hashes():
    a = addr("##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri", US)
    assert a["hn"] == "8" and a["unit"] == "0" and a["state"] == "mo"
    a = addr("Unit L-2, TN, 320 Welch Road, Nashville")
    assert a["hn"] == "320" and a["street"] == "welch"
    assert addr("#109 EDGEWOOD LN, PO BOX 623, LA PORTE, IN")["has_po"] == 1


def test_ordinals_are_not_house_numbers():
    a = addr("TWELFTH STREET, MOORHEAD, MN", US)
    assert a["hn"] == "" and a["street"] == "12th"
    assert addr("630 12th Street, MN, Moorhead", US)["street"] == "12th"


def test_french():
    a = addr("5 bis Rue Pierre Dignac, La Teste-de-Buch, Nouvelle-Aquitaine")
    assert a["hn"] == "5" and a["hid"] == "5bis"
    a = addr("63 R. DE DIEPPE, LILLE, Hauts-de-France")
    assert a["street_type"] == "rue" and a["street"] == "de dieppe"


def test_careof_and_empty():
    a = addr("C/O Anil Kumar, S/O Bishnath Rai, Saguna More, Nr Dsp Office, Patna, Bihar")
    assert "anil" in a["careof"] and "dsp" in a["landmark"]
    assert addr("")["addr_empty"] == 1
