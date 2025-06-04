"""
generate.py

dataset generator for testing comparison methods with realistic duplicate data
also generates comparison sequences for training transformer-based models off
of realistic data
follows the tributary-account-verification database schema

date: 06/04/2025
author: thomas bruce
last update: major refactor, remove duplicated datasets, make generator tweakable
"""

# TODO:
# - move hard-coded datasets to external files and expand
# - split up for modularity/readability
# - finalize timestamp creation system for simulated sybil attacks
# - improve consistency of name variation

import os
import sys
import json
import uuid
import random
import argparse

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.gdb import GraphDatabase

from faker import Faker

# constants
DEFAULT_DUPLICATE_LIKELIHOOD = 0.1
DEFAULT_SEQUENCE_STEPS = 5
EXACT_DUPLICATE_CHANCE = 0.02
FAMILY_REUSE_LIKELIHOOD = 0.4
BIRTH_YEAR_START = 1920
BIRTH_YEAR_END = 2020
DEATH_CHANCE_PRE_1960 = 0.15
BIRTH_CITY_CHANCE = 0.8
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
# TODO: move this to an external file and include international locations
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
    # titles/prefixes
    "dr.",
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
    # professional suffixes
    "md",
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
    # name suffixes
    "jr.",
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
# TODO: move to an external file to expand dataset
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


class Generator:
    """
    generate randomized person data with configurable duplicate likelihood
    """

    def __init__(self, duplicate_likelihood: float = DEFAULT_DUPLICATE_LIKELIHOOD):
        self.faker = Faker()
        self.duplicate_likelihood = duplicate_likelihood
        self.used_last_names: List[str] = []
        self.existing_people: List[Tuple[str, Dict[str, Any]]] = []

    def _clean_name_part(self, name_part: str) -> Optional[str]:
        """clean name parts of any titles, prefixes, or suffixes"""
        cleaned = name_part.lower().strip(".,")
        return None if cleaned in BLOCKED_NAME_TERMS else name_part

    def _generate_realistic_typo(self, name: str) -> str:
        """generate realistic typos based on common keyboard/spelling errors"""
        if len(name) < 3:
            return name

        chars = list(name)
        typo_type = random.choice(
            [
                "adjacent_key",
                "double_letter",
                "missing_letter",
                "vowel_swap",
                "transpose",
                "common_misspelling",
            ]
        )

        if typo_type == "adjacent_key" and chars:
            idx = random.randint(1, len(chars) - 2)
            if chars[idx].lower() in ADJACENT_KEYS:
                chars[idx] = ADJACENT_KEYS[chars[idx].lower()]

        elif typo_type == "double_letter":
            idx = random.randint(1, len(chars) - 1)
            chars.insert(idx, chars[idx])

        elif typo_type == "missing_letter" and len(chars) > 3:
            idx = random.randint(1, len(chars) - 2)
            chars.pop(idx)

        elif typo_type == "vowel_swap":
            for i, char in enumerate(chars):
                if char.lower() in VOWEL_SWAPS and random.random() < 0.3:
                    chars[i] = VOWEL_SWAPS[char.lower()]
                    break

        elif typo_type == "transpose" and len(chars) > 3:
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        elif typo_type == "common_misspelling":
            name_lower = name.lower()
            if "ph" in name_lower:
                return name.replace("ph", "f")
            elif "ck" in name_lower:
                return name.replace("ck", "k")
            elif name_lower.endswith("ie"):
                return name[:-2] + "y"

        return "".join(chars)

    def _generate_names(self) -> Tuple[str, str, Optional[str]]:
        """generate cleaned first, last, and optional middle names"""
        profile = self.faker.profile()
        name_parts = profile["name"].split()

        # clean and validate name parts
        cleaned_parts = [
            part for part in name_parts if self._clean_name_part(part) is not None
        ]

        # ensure we have at least first and last name
        if len(cleaned_parts) < 2:
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
        else:
            first_name = cleaned_parts[0]
            last_name = cleaned_parts[-1]

        # middle name, 70%
        # TODO: make globally tweakable!
        middle_name = self.faker.first_name() if random.random() < 0.7 else None

        # reuse last names for family connections
        if self.used_last_names and random.random() < FAMILY_REUSE_LIKELIHOOD:
            last_name = random.choice(self.used_last_names)
        else:
            self.used_last_names.append(last_name)

        return first_name, last_name, middle_name

    def _generate_dates(self) -> Tuple[datetime, Optional[datetime]]:
        """generate birth date and optional death date."""
        start_date = datetime(BIRTH_YEAR_START, 1, 1)
        end_date = datetime(BIRTH_YEAR_END, 12, 31)
        birth_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        # death date for pre-1960 births
        # TODO: make globally tweakable!
        death_date = None
        if birth_date.year < 1960 and random.random() < DEATH_CHANCE_PRE_1960:
            min_death = birth_date + timedelta(days=MIN_DEATH_AGE_YEARS * 365)
            max_death = birth_date + timedelta(days=MAX_DEATH_AGE_YEARS * 365)
            max_death = min(max_death, datetime.now())

            if min_death < max_death:
                death_date = min_death + timedelta(
                    days=random.randint(0, (max_death - min_death).days)
                )

        return birth_date, death_date

    def _generate_email(self, first_name: str, last_name: str) -> str:
        """generate realistic email address"""
        username = f"{first_name.lower()}.{last_name.lower()}"
        domain = random.choice(EMAIL_DOMAINS)
        return f"{username}@{domain}"

    def generate_person(self) -> Tuple[str, Dict[str, Any]]:
        """generate a single person with potential for duplicates."""
        # check for duplicate generation
        if self.existing_people and random.random() < self.duplicate_likelihood:
            if random.random() < EXACT_DUPLICATE_CHANCE:
                return self._generate_exact_duplicate()
            else:
                return self._generate_similar_person()

        # generate new person
        uid = str(uuid.uuid4())[:8]
        first_name, last_name, middle_name = self._generate_names()
        birth_date, death_date = self._generate_dates()
        email = self._generate_email(first_name, last_name)

        # optional birth city
        birth_city = (
            random.choice(US_CITIES) if random.random() < BIRTH_CITY_CHANCE else None
        )

        # build metadata
        metadata = {
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": birth_date.strftime("%Y-%m-%d"),
            "email": email,
            "base_uuid": uid,  # ground truth tracking
        }

        # add optional fields
        if middle_name:
            metadata["middle_name"] = middle_name
        if death_date:
            metadata["date_of_death"] = death_date.strftime("%Y-%m-%d")
        if birth_city:
            metadata["birth_city"] = birth_city

        # store for potential duplicate generation
        self.existing_people.append((uid, metadata))
        return uid, metadata

    def _generate_exact_duplicate(self) -> Tuple[str, Dict[str, Any]]:
        """generate an exact 100% duplicate of an existing person"""
        base_uid, base_metadata = random.choice(self.existing_people)
        uid = str(uuid.uuid4())[:8]

        # preserve original base_uuid for ground truth
        metadata = base_metadata.copy()
        return uid, metadata

    def _apply_name_variation(self, metadata: Dict[str, Any]) -> None:
        """apply various name transformations for similar person generation"""
        variation_type = random.choice(
            [
                "typo_first",
                "typo_last",
                "typo_first",
                "typo_last",  # weight typos
                "nickname",
                "middle_initial",
                "hyphenated_last",
                "maiden_name",
            ]
        )

        if variation_type == "typo_first":
            metadata["first_name"] = self._generate_realistic_typo(
                metadata["first_name"]
            )
        elif variation_type == "typo_last":
            metadata["last_name"] = self._generate_realistic_typo(metadata["last_name"])
        elif variation_type == "nickname":
            first_name = metadata["first_name"]
            if first_name in NICKNAME_MAPPINGS:
                metadata["first_name"] = NICKNAME_MAPPINGS[first_name]
        elif variation_type == "middle_initial":
            if "middle_name" in metadata:
                metadata["middle_name"] = metadata["middle_name"][0]
            else:
                metadata["middle_name"] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def _apply_date_variation(self, metadata: Dict[str, Any]) -> None:
        """apply date variations with realistic error patterns"""
        birth_date = datetime.strptime(metadata["date_of_birth"], "%Y-%m-%d")

        # weighted toward smaller errors (70% within 30 days)
        if random.random() < 0.7:
            days_variation = random.randint(-30, 30)
        elif random.random() < 0.9:
            days_variation = random.randint(-90, 90)
        else:
            days_variation = random.randint(-365, 365)

        new_birth_date = birth_date + timedelta(days=days_variation)
        metadata["date_of_birth"] = new_birth_date.strftime("%Y-%m-%d")

    def _apply_email_variation(self, metadata: Dict[str, Any]) -> None:
        """apply email variations with different formatting patterns"""
        first = metadata["first_name"].lower()
        last = metadata["last_name"].lower()

        email_formats = [
            f"{first}.{last}",
            f"{first}{last}",
            f"{first[0]}{last}",
            f"{first}.{last[0]}",
            f"{first}_{last}",
            f"{first}{random.randint(10, 99)}",
        ]

        username = random.choice(email_formats)
        domain = random.choice(EMAIL_DOMAINS)
        metadata["email"] = f"{username}@{domain}"

    def _generate_similar_person(self) -> Tuple[str, Dict[str, Any]]:
        """generate a similar person for duplicate testing."""
        base_uid, base_metadata = random.choice(self.existing_people)
        uid = str(uuid.uuid4())[:8]

        metadata = base_metadata.copy()
        # preserve original base_uuid for ground truth
        metadata["base_uuid"] = base_metadata["base_uuid"]

        # apply variations
        # TODO: make globally tweakable!
        if random.random() < 0.85:
            self._apply_name_variation(metadata)

        # date variations
        # TODO: make globally tweakable!
        if random.random() < 0.6:
            self._apply_date_variation(metadata)

        # email variations
        # TODO: make globally tweakable!
        if random.random() < 0.65:
            self._apply_email_variation(metadata)

        # birth city variations
        # TODO: make globally tweakable!
        if random.random() < 0.2 and "birth_city" in metadata:
            if random.random() < 0.5:
                del metadata["birth_city"]
            else:
                metadata["birth_city"] = random.choice(US_CITIES)

        return uid, metadata

    def generate_identity_sequence(
        self, steps: int = DEFAULT_SEQUENCE_STEPS
    ) -> List[Tuple[str, Dict[str, Any], int]]:
        """generate a sequence of identity evolution over time."""
        if steps < 1:
            raise ValueError("steps must be at least 1")

        # generate initial identity
        base_uid, base_metadata = self.generate_person()
        base_timestamp = datetime(2020, 1, 1)
        base_metadata["timestamp"] = base_timestamp.isoformat()

        sequence = [(base_uid, base_metadata, 0)]

        # generate evolution steps
        for step in range(1, steps):
            evolved_uid = str(uuid.uuid4())[:8]
            evolved_metadata = base_metadata.copy()
            evolved_metadata["base_uuid"] = base_metadata["base_uuid"]

            # update timestamp
            current_timestamp = base_timestamp + timedelta(
                days=SEQUENCE_TIME_INCREMENT_DAYS * step
            )
            evolved_metadata["timestamp"] = current_timestamp.isoformat()

            # apply evolutionary changes
            if random.random() < 0.4:
                self._apply_name_variation(evolved_metadata)
            if random.random() < 0.3:
                self._apply_date_variation(evolved_metadata)
            if random.random() < 0.5:
                self._apply_email_variation(evolved_metadata)

            sequence.append((evolved_uid, evolved_metadata, step))

        return sequence


class ConnectionGenerator:
    """generate edges to connect nodes with realistic relationship patterns"""

    def __init__(self, people: List[Tuple[str, Dict[str, Any]]]):
        self.people = people

    def generate_connections(self, num_edges: int) -> List[Dict[str, Any]]:
        """generate connections between people with family and random relationships"""
        connections = []

        # group people by last name for family connections
        by_last_name = {}
        for uid, metadata in self.people:
            last_name = metadata["last_name"]
            by_last_name.setdefault(last_name, []).append((uid, metadata))

        # connection types with distribution ratios
        connection_types = [
            ("family", self._generate_family_connections, by_last_name, 0.3),
            ("random", self._generate_random_connections, None, 0.7),
        ]

        # distribute edges across connection types
        remaining_edges = num_edges
        for i, (conn_type, generator, groups, ratio) in enumerate(connection_types):
            if i == len(connection_types) - 1:
                edges_count = remaining_edges
            else:
                edges_count = int(num_edges * ratio)
                remaining_edges -= edges_count

            new_connections = generator(groups, edges_count)
            connections.extend(new_connections)

        # remove duplicates and self-loops
        return self._deduplicate_connections(connections, num_edges)

    def _deduplicate_connections(
        self, connections: List[Dict[str, Any]], max_count: int
    ) -> List[Dict[str, Any]]:
        """remove duplicate connections and self-loops"""
        seen: Set[Tuple[str, str]] = set()
        unique_connections = []

        for conn in connections:
            key = (conn["source"], conn["target"])
            reverse_key = (conn["target"], conn["source"])

            if (
                key not in seen
                and reverse_key not in seen
                and conn["source"] != conn["target"]
            ):
                seen.add(key)
                unique_connections.append(conn)

        return unique_connections[:max_count]

    def _generate_family_connections(
        self, by_last_name: Dict[str, List], count: int
    ) -> List[Dict[str, Any]]:
        """generate family connections based on shared last names and age differences"""
        connections = []
        families = [family for family in by_last_name.values() if len(family) > 1]

        for _ in range(count):
            if not families:
                break

            family = random.choice(families)
            if len(family) < 2:
                continue

            person1, person2 = random.sample(family, 2)

            # determine relationship based on age difference
            # TODO: come up with a smarter way to do this
            age1 = datetime.now().year - int(person1[1]["date_of_birth"][:4])
            age2 = datetime.now().year - int(person2[1]["date_of_birth"][:4])
            age_diff = abs(age1 - age2)

            if age_diff > 20:
                relationship = "parent_child"
            elif age_diff < 5:
                relationship = "sibling"
            else:
                relationship = "relative"

            connections.append(
                {
                    "source": person1[0],
                    "target": person2[0],
                    "directed": False,
                    "metadata": {"type": "family", "relationship": relationship},
                }
            )

        return connections

    def _generate_random_connections(
        self, groups: None, count: int
    ) -> List[Dict[str, Any]]:
        """generate random connections between people"""
        connections = []

        for _ in range(count):
            if len(self.people) < 2:
                break

            person1, person2 = random.sample(self.people, 2)

            connections.append(
                {
                    "source": person1[0],
                    "target": person2[0],
                    "directed": random.choice([True, False]),
                    "metadata": {"type": "random", "relationship": "acquaintance"},
                }
            )

        return connections


def generate_dataset(
    num_people: int,
    num_edges: int,
    output_file: str = "dataset.json",
    duplicate_likelihood: float = DEFAULT_DUPLICATE_LIKELIHOOD,
) -> None:
    """generate a complete dataset with people and connections."""
    # initialize database
    db = GraphDatabase(output_file)
    db.clear()

    # generate people
    person_gen = Generator(duplicate_likelihood=duplicate_likelihood)
    people = []

    print("generating people")
    print("-----------------")
    for i in range(num_people):
        uid, metadata = person_gen.generate_person()
        db.add_node(uid, metadata)
        people.append((uid, metadata))

        if (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{num_people} people")

    # generate connections
    print("generating connections")
    print("----------------------")
    conn_gen = ConnectionGenerator(people)
    connections = conn_gen.generate_connections(num_edges)

    for i, conn in enumerate(connections):
        db.add_edge(
            conn["source"],
            conn["target"],
            directed=conn["directed"],
            metadata=conn["metadata"],
        )

        if (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{len(connections)} connections")

    db.save()

    # print statistics
    similar_count = int(num_people * duplicate_likelihood)
    print(f"\ndataset generation complete!")
    print(f"generated {num_people} people with ~{similar_count} potential duplicates")
    print(f"unique last names: {len(person_gen.used_last_names)}")
    print(f"generated {len(connections)} connections")


def generate_sequences(
    num_sequences: int,
    steps_per_sequence: int = DEFAULT_SEQUENCE_STEPS,
    output_file: Optional[str] = None,
) -> List[List[Tuple[str, Dict[str, Any], int]]]:
    """generate multiple identity sequences for testing purposes."""
    generator = Generator(duplicate_likelihood=0.0)  # no duplicates for clean sequences
    sequences = []

    print(
        f"generating {num_sequences} identity sequences with {steps_per_sequence} steps each"
    )

    for i in range(num_sequences):
        sequence = generator.generate_identity_sequence(steps_per_sequence)
        sequences.append(sequence)

        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{num_sequences} sequences")

    if output_file:
        # initialize database for proper schema (without path to avoid loading existing file)
        db = GraphDatabase()

        # add all sequence nodes to database with sequence metadata
        for sequence_idx, sequence in enumerate(sequences):
            for uid, metadata, step_index in sequence:
                # add sequence information to metadata
                enhanced_metadata = metadata.copy()
                enhanced_metadata["sequence_id"] = sequence_idx
                enhanced_metadata["sequence_step"] = step_index
                enhanced_metadata["total_sequences"] = num_sequences
                enhanced_metadata["steps_per_sequence"] = steps_per_sequence

                db.add_node(uid, enhanced_metadata)

        # save using database schema (nodes and edges format)
        db.save(output_file)
        print(f"sequences saved to {output_file}")

    return sequences


def main() -> None:
    """entrypoint + argparse"""
    parser = argparse.ArgumentParser(description="generate a test dataset")
    parser.add_argument(
        "num_people",
        type=int,
        nargs="?",
        default=None,
        help="number of people to generate",
    )
    parser.add_argument(
        "num_edges",
        type=int,
        nargs="?",
        default=None,
        help="number of edges to generate",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dataset.json",
        help="output file path (default: dataset.json)",
    )
    parser.add_argument(
        "-d",
        "--duplicate-likelihood",
        type=float,
        default=DEFAULT_DUPLICATE_LIKELIHOOD,
        help=f"likelihood of generating similar/duplicate people (0.0-1.0, default: {DEFAULT_DUPLICATE_LIKELIHOOD})",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        type=int,
        help="generate n identity sequences instead of regular dataset",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_SEQUENCE_STEPS,
        help=f"number of steps per sequence (default: {DEFAULT_SEQUENCE_STEPS})",
    )

    args = parser.parse_args()

    # handle sequence generation mode
    if args.sequence is not None:
        if args.sequence <= 0:
            print("error: sequence count must be positive")
            sys.exit(1)

        output_file = args.output if args.output != "dataset.json" else "sequences.json"
        sequences = generate_sequences(args.sequence, args.steps, output_file)

        # show example sequence
        if sequences:
            print(f"\nexample sequence (first of {len(sequences)}):")
            for uuid_val, metadata, step in sequences[0]:
                print(
                    f"  step {step}: {uuid_val} -> {metadata['first_name']} "
                    f"{metadata['last_name']} ({metadata['timestamp']})"
                )

        return

    # regular dataset generation mode
    if args.num_people is None or args.num_edges is None:
        print(
            "error: num_people and num_edges are required for regular dataset generation"
        )
        print("use --sequence n to generate identity sequences instead")
        sys.exit(1)

    # validation
    max_edges = args.num_people * (args.num_people - 1)
    if args.num_edges > max_edges:
        print(f"warn: num_edges ({args.num_edges}) is very high relative to num_people")
        print(f"maximum possible edges: {max_edges}")

    if not 0.0 <= args.duplicate_likelihood <= 1.0:
        print("error: duplicate-likelihood must be between 0.0 and 1.0")
        sys.exit(1)

    generate_dataset(
        args.num_people, args.num_edges, args.output, args.duplicate_likelihood
    )


if __name__ == "__main__":
    main()
