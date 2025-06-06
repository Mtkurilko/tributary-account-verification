"""
data.py

hard-coded datasets and constants for the dataset generator
separated from main logic for better modularity and expandability

date: 06/06/2025
author: thomas bruce
"""

# generation constants
DEFAULT_DUPLICATE_LIKELIHOOD = 0.1
DEFAULT_SEQUENCE_STEPS = 5
EXACT_DUPLICATE_CHANCE = 0.02
FAMILY_REUSE_LIKELIHOOD = 0.4
BIRTH_YEAR_START = 1920
BIRTH_YEAR_END = 2020
DEATH_CHANCE_PRE_1960 = 0.15
BIRTH_CITY_CHANCE = 0.8
PHONE_NUMBER_CHANCE = 0.9
MIN_DEATH_AGE_YEARS = 20
MAX_DEATH_AGE_YEARS = 100
SEQUENCE_TIME_INCREMENT_DAYS = 30

# email TLDs
EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "proton.me",
    "aol.com",
    "outlook.com",
    "yandex.ru",
    "protonmail.com",
    "tutanota.com",
    "posteo.de",
    "startmail.com",
    "disroot.org",
    "mailbox.org",
]

# major us cities
US_CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
    "Austin",
    "Jacksonville",
    "Fort Worth",
    "Columbus",
    "Charlotte",
    "San Francisco",
    "Indianapolis",
    "Seattle",
    "Denver",
    "Washington",
    "Boston",
    "El Paso",
    "Nashville",
    "Detroit",
    "Oklahoma City",
    "Portland",
    "Las Vegas",
    "Memphis",
    "Louisville",
    "Baltimore",
    "Milwaukee",
    "Albuquerque",
    "Tucson",
    "Fresno",
    "Sacramento",
    "Kansas City",
    "Long Beach",
    "Mesa",
    "Atlanta",
    "Colorado Springs",
    "Virginia Beach",
    "Raleigh",
    "Omaha",
    "Miami",
    "Oakland",
    "Minneapolis",
    "Tulsa",
    "Wichita",
    "New Orleans",
]

# common name cleaning - titles, prefixes, and suffixes to remove
BLOCKED_NAME_TERMS = {
    "dr.",  # titles
    "dr",
    "prof.",
    "prof",
    "professor",
    "mr.",
    "mr",
    "mrs.",
    "mrs",
    "ms.",
    "ms",
    "miss",
    "sir",
    "lord",
    "lady",
    "hon.",
    "hon",
    "rev.",
    "rev",
    "reverend",
    "father",
    "fr.",
    "fr",
    "sister",
    "sr.",
    "brother",
    "br.",
    "captain",
    "capt.",
    "capt",
    "major",
    "col.",
    "colonel",
    "general",
    "gen.",
    "admiral",
    "sergeant",
    "sgt.",
    "lieutenant",
    "lt.",
    "md",  # professional suffixes
    "dvm",
    "phd",
    "dds",
    "do",
    "jd",
    "esq",
    "esq.",
    "cpa",
    "rn",
    "pharmd",
    "dpt",
    "edd",
    "psyd",
    "dnp",
    "crna",
    "pa",
    "np",
    "jr.",  # name suffixes
    "jr",
    "sr.",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th",
    # other
    "the",
    "von",
    "van",
    "de",
    "del",
    "da",
    "di",
    "du",
}

# qwerty keyboard adjacent key mappings for realistic typos
ADJACENT_KEYS = {
    "a": "s",
    "s": "a",
    "d": "s",
    "f": "d",
    "g": "f",
    "h": "g",
    "j": "h",
    "k": "j",
    "l": "k",
    "q": "w",
    "w": "q",
    "e": "w",
    "r": "e",
    "t": "r",
    "y": "t",
    "u": "y",
    "i": "u",
    "o": "i",
    "p": "o",
    "z": "x",
    "x": "z",
    "c": "x",
    "v": "c",
    "b": "v",
    "n": "b",
    "m": "n",
}

# common vowel substitutions for realistic typos
VOWEL_SWAPS = {"a": "e", "e": "a", "i": "y", "y": "i", "o": "u", "u": "o"}

# common nickname mappings
NICKNAME_MAPPINGS = {
    "William": "Bill",
    "Robert": "Bob",
    "Richard": "Dick",
    "Michael": "Mike",
    "Christopher": "Chris",
    "Matthew": "Matt",
    "Anthony": "Tony",
    "Elizabeth": "Liz",
    "Katherine": "Kate",
    "Jennifer": "Jen",
    "Rebecca": "Becky",
    "Jessica": "Jess",
}

# valid nanp area codes
NANP_AREA_CODES = [
    "555",  # reserved for fictional use
    "800",
    "888",
    "877",
    "866",
    "855",
    "844",
    "833",
    "822",  # toll-free
    "900",
    "901",
    "902",
    "903",
    "904",
    "905",
    "906",
    "907",
    "908",
    "909",  # premium rate
]
