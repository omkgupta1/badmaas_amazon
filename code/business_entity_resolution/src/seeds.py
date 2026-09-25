"""Small hand-written normalisation seed lists (generic linguistic knowledge, no entity data).

Everything here is extended at run time by equivalences mined from the provided data
(``lexicon_mining``); nothing is looked up externally.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# business names
# ---------------------------------------------------------------------------------------------
LEGAL_FORMS = [
    "pvt", "ltd", "llc", "llp", "lp", "lllp", "inc", "corp", "co", "plc", "pllc", "pc", "pa",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop", "selarl", "gie", "opc",
    "gmbh", "ag", "bv", "nv", "pte", "pty",
]
LEGAL_SET = set(LEGAL_FORMS)
LEGAL_BIT = {form: 1 << i for i, form in enumerate(LEGAL_FORMS)}

NAME_CANON = {
    "private": "pvt", "pvte": "pvt", "priv": "pvt", "prv": "pvt",
    "limited": "ltd", "ltda": "ltd", "limitada": "ltd",
    "corporation": "corp", "corpn": "corp", "corporate": "corp",
    "incorporated": "inc", "incorporation": "inc",
    "company": "co", "compagnie": "co", "cie": "co",
    "centre": "center", "ctr": "center",
    "intl": "international", "int'l": "international",
    "bros": "brothers", "bro": "brothers",
    "svcs": "services", "svc": "service", "srvcs": "services",
    "mfg": "manufacturing", "assoc": "associates", "assocs": "associates",
    "grp": "group", "hldgs": "holdings", "hldg": "holding", "mgmt": "management",
    "univ": "university", "dept": "department", "natl": "national",
    "st": "saint", "ste": "sainte",
    "et": "and",
    "societe": "societe", "ste.": "societe",
}
NAME_STOP = {"the", "of", "and", "de", "du", "des", "la", "le", "les", "d", "l"}
NAME_JUNK = {"null", "none", "nan", "dba", "aka", "fka", "formerly", "www", "com", "http", "https"}

# ---------------------------------------------------------------------------------------------
# addresses
# ---------------------------------------------------------------------------------------------
STREET_TYPES = {
    "road": "road", "rd": "road",
    "street": "street", "str": "street", "strt": "street",
    "avenue": "avenue", "ave": "avenue", "av": "avenue", "avn": "avenue", "aven": "avenue",
    "avenu": "avenue",
    "drive": "drive", "drv": "drive",
    "lane": "lane", "ln": "lane",
    "court": "court", "ct": "court", "crt": "court",
    "boulevard": "boulevard", "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard",
    "bld": "boulevard", "blv": "boulevard",
    "place": "place", "pl": "place",
    "circle": "circle", "cir": "circle", "circ": "circle",
    "highway": "highway", "hwy": "highway",
    "parkway": "parkway", "pkwy": "parkway", "pky": "parkway",
    "trail": "trail", "trl": "trail",
    "terrace": "terrace", "terr": "terrace",
    "square": "square", "sq": "square",
    "way": "way",
    "loop": "loop",
    "expressway": "expressway", "expy": "expressway",
    "freeway": "freeway", "fwy": "freeway",
    "heights": "heights", "hts": "heights",
    "crossing": "crossing", "xing": "crossing",
    "route": "route", "rte": "route",
    "rue": "rue",
    "allee": "allee",
    "impasse": "impasse", "imp": "impasse",
    "chemin": "chemin", "chem": "chemin", "che": "chemin",
    "quai": "quai",
    "cours": "cours",
    "faubourg": "faubourg", "fbg": "faubourg",
    "marg": "marg", "path": "path",
    "nagar": "nagar", "ngr": "nagar",
    "colony": "colony", "clny": "colony",
    "sector": "sector", "sec": "sector", "sect": "sector",
    "block": "block", "blk": "block",
    "layout": "layout",
    "cross": "cross",
}
STREET_TYPE_SET = set(STREET_TYPES.values())

DIRECTIONS = {
    "north": "north", "n": "north", "south": "south", "s": "south", "east": "east", "e": "east",
    "west": "west", "w": "west", "northeast": "northeast", "ne": "northeast",
    "northwest": "northwest", "nw": "northwest", "southeast": "southeast", "se": "southeast",
    "southwest": "southwest", "sw": "southwest",
}
DIRECTION_SET = set(DIRECTIONS.values())

OTHER_ADDR = {
    "apartment": "apartment", "apt": "apartment", "apts": "apartment",
    "suite": "suite", "floor": "floor", "fl": "floor", "flr": "floor",
    "room": "room", "rm": "room", "building": "building", "bldg": "building",
    "near": "near", "nr": "near", "nearby": "near",
    "opposite": "opposite", "opp": "opposite", "opposit": "opposite",
    "behind": "behind", "bhd": "behind", "beside": "beside", "besides": "beside",
    "adjacent": "adjacent", "adj": "adjacent",
    "village": "village", "vill": "village", "vil": "village", "vlg": "village",
    "district": "district", "dist": "district", "distt": "district",
    "tehsil": "tehsil", "teh": "tehsil", "taluk": "taluk", "taluka": "taluk", "tq": "taluk",
    "county": "county", "cnty": "county", "cty": "county",
    "township": "township", "twp": "township",
    "center": "center", "centre": "center", "ctr": "center",
    "mount": "mount", "mt": "mount", "fort": "fort", "ft": "fort",
    "point": "point", "pt": "point",
    "saint": "saint", "sainte": "sainte",
    "residence": "residence", "res": "residence",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th",
    "fifteenth": "15th", "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
    "nineteenth": "19th", "twentieth": "20th",
}

# words that open a unit / PO-box component ("Unit APT 42", "PMB 3480", "PO Box 623", "Fl. 0")
UNIT_WORDS = {"unit", "apt", "apartment", "apts", "suite", "ste", "fl", "floor", "flr", "room",
              "rm", "pmb", "box", "pobox", "lot", "spc", "space", "trlr", "bldg", "building"}
PO_WORDS = {"pmb", "pobox", "box"}
LANDMARK_WORDS = {"near", "opposite", "behind", "beside", "adjacent"}
CAREOF_WORDS = {"careof", "sonof", "daughterof", "wifeof"}
CITY_NOISE = {"city", "town", "village", "county", "township", "cdp", "of", "ownship"}
FRENCH_ARTICLES = {"de", "du", "des", "d", "la", "le", "les", "l"}
ADDR_JUNK = {"null", "none", "nan", "unknown"}
HOUSE_SUFFIXES = {"bis", "ter", "quater", "a", "b", "c", "d"}

# ---------------------------------------------------------------------------------------------
# states: canonical key = the form Source 1 uses (US: 2-letter code; India: lower-case name)
# ---------------------------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "guam": "gu",
}
INDIA_STATES = {
    "andhra pradesh": "andhra pradesh", "ap": "andhra pradesh",
    "arunachal pradesh": "arunachal pradesh", "ar": "arunachal pradesh",
    "assam": "assam", "as": "assam",
    "bihar": "bihar", "br": "bihar",
    "chhattisgarh": "chhattisgarh", "chattisgarh": "chhattisgarh", "cg": "chhattisgarh",
    "goa": "goa", "ga": "goa",
    "gujarat": "gujarat", "gj": "gujarat",
    "haryana": "haryana", "hr": "haryana",
    "himachal pradesh": "himachal pradesh", "hp": "himachal pradesh",
    "jharkhand": "jharkhand", "jh": "jharkhand",
    "karnataka": "karnataka", "ka": "karnataka",
    "kerala": "kerala", "kl": "kerala",
    "madhya pradesh": "madhya pradesh", "mp": "madhya pradesh",
    "maharashtra": "maharashtra", "mh": "maharashtra",
    "manipur": "manipur", "mn": "manipur",
    "meghalaya": "meghalaya", "ml": "meghalaya",
    "mizoram": "mizoram", "mz": "mizoram",
    "nagaland": "nagaland", "nl": "nagaland",
    "odisha": "odisha", "orissa": "odisha", "od": "odisha", "or": "odisha",
    "punjab": "punjab", "pb": "punjab",
    "rajasthan": "rajasthan", "rj": "rajasthan",
    "sikkim": "sikkim", "sk": "sikkim",
    "tamil nadu": "tamil nadu", "tamilnadu": "tamil nadu", "tn": "tamil nadu",
    "telangana": "telangana", "ts": "telangana", "tg": "telangana",
    "tripura": "tripura", "tr": "tripura",
    "uttar pradesh": "uttar pradesh", "up": "uttar pradesh",
    "uttarakhand": "uttarakhand", "uttaranchal": "uttarakhand", "uk": "uttarakhand",
    "west bengal": "west bengal", "wb": "west bengal",
    "delhi": "delhi", "new delhi": "delhi", "dl": "delhi",
    "chandigarh": "chandigarh", "ch": "chandigarh",
    "puducherry": "puducherry", "pondicherry": "puducherry", "py": "puducherry",
    "jammu and kashmir": "jammu and kashmir", "jammu kashmir": "jammu and kashmir",
    "jk": "jammu and kashmir",
    "ladakh": "ladakh", "la": "ladakh",
}
