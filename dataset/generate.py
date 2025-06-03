import os
import sys
import random
import uuid

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.gdb import GraphDatabase

from faker import Faker


class Generator:
    """
    generate randomized person data
    """

    def __init__(self, duplicate_likelihood: float = 0.1):
        self.faker = Faker()
        self.duplicate_likelihood = duplicate_likelihood
        self.used_last_names = []  # track used last names for reuse
        self.existing_people = []  # track people for potential duplicates

        # list of major usa cities
        # prompt (gpt-4o): "generate a python list of 50 major usa cities"
        self.cities = [
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

    # prompt (gpt-4o): "generate a python function that strips prefixes, titles, and suffices from generated names"
    def _clean_name_part(self, name_part: str) -> str:
        """aggressively clean name parts of any titles, prefixes, or suffixes"""
        # remove common titles, prefixes, and suffixes (case insensitive)
        blocked_terms = {
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

        cleaned = name_part.lower().strip(".,")
        if cleaned in blocked_terms:
            return None  # signal to skip this part
        return name_part

    def generate_person(self) -> Dict[str, Any]:
        """generate a single person"""

        # check if should generate a similar person for duplicate testing
        if self.existing_people and random.random() < self.duplicate_likelihood:
            # very slim chance (2%) of exact 100% duplicate
            if random.random() < 0.02:
                return self._generate_exact_duplicate()
            else:
                return self._generate_similar_person()

        uid = str(uuid.uuid4())[:8]

        # generate a profile of consistent data
        profile = self.faker.profile()
        name_parts = profile["name"].split()

        # clean name parts
        cleaned_parts = []
        for part in name_parts:
            cleaned_part = self._clean_name_part(part)
            if cleaned_part is not None:
                cleaned_parts.append(cleaned_part)

        # ensure we have at least first and last name
        if len(cleaned_parts) < 2:
            # fall back to generating fresh names if cleaning removed too much
            first_name = self.faker.first_name()
            last_name = self.faker.last_name()
        else:
            first_name = cleaned_parts[0]
            last_name = cleaned_parts[-1]

        middle_name = self.faker.first_name() if random.random() < 0.7 else None

        # bias toward reusing last names, 40%
        if self.used_last_names and random.random() < 0.4:
            last_name = random.choice(self.used_last_names)
        else:
            self.used_last_names.append(last_name)

        # birth date (1920-2020)
        start_date = datetime(1920, 1, 1)
        end_date = datetime(2020, 12, 31)
        birth_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )

        # death date, 15% chance, pre-1960
        death_date = None
        if birth_date.year < 1960 and random.random() < 0.15:
            # between ages 20 and 100
            min_death = birth_date + timedelta(days=20 * 365)
            max_death = birth_date + timedelta(days=100 * 365)
            max_death = min(max_death, datetime.now())
            if min_death < max_death:
                death_date = min_death + timedelta(
                    days=random.randint(0, (max_death - min_death).days)
                )

        # birth city, 80%
        birth_city = random.choice(self.cities) if random.random() < 0.8 else None

        # emails
        email_username = f"{first_name.lower()}.{last_name.lower()}"
        email_domain = random.choice(
            [
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
        )
        email = f"{email_username}@{email_domain}"

        metadata = {
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": birth_date.strftime("%Y-%m-%d"),
            "email": email,
        }

        if middle_name:
            metadata["middle_name"] = middle_name
        if death_date:
            metadata["date_of_death"] = death_date.strftime("%Y-%m-%d")
        if birth_city:
            metadata["birth_city"] = birth_city

        # store this person for potential duplicate generation
        self.existing_people.append((uid, metadata))

        return uid, metadata

    def _generate_exact_duplicate(self) -> Dict[str, Any]:
        """generate an exact 100% duplicate of an existing person"""
        # choose a random existing person as the base
        base_uid, base_metadata = random.choice(self.existing_people)

        # generate new UID but keep all metadata exactly the same
        uid = str(uuid.uuid4())[:8]

        # return exact copy of metadata
        return uid, base_metadata.copy()

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

        # prompt (gpt-4o): "generate a mapping of common adjacent key errors on the qwerty keyboard as a python dict"
        if typo_type == "adjacent_key":
            # common adjacent key errors on QWERTY keyboard
            adjacent_keys = {
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
            idx = random.randint(1, len(chars) - 2)
            if chars[idx].lower() in adjacent_keys:
                chars[idx] = adjacent_keys[chars[idx].lower()]

        elif typo_type == "double_letter":
            # accidentally double a letter
            idx = random.randint(1, len(chars) - 1)
            chars.insert(idx, chars[idx])

        elif typo_type == "missing_letter":
            # skip a letter
            if len(chars) > 3:
                idx = random.randint(1, len(chars) - 2)
                chars.pop(idx)

        elif typo_type == "vowel_swap":
            # common vowel confusions
            vowel_swaps = {"a": "e", "e": "a", "i": "y", "y": "i", "o": "u", "u": "o"}
            for i, char in enumerate(chars):
                if char.lower() in vowel_swaps and random.random() < 0.3:
                    chars[i] = vowel_swaps[char.lower()]
                    break

        elif typo_type == "transpose":
            # swap adjacent letters
            if len(chars) > 3:
                idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        elif typo_type == "common_misspelling":
            # common letter pattern mistakes
            name_lower = name.lower()
            if "ph" in name_lower:
                return name.replace("ph", "f")
            elif "ck" in name_lower:
                return name.replace("ck", "k")
            elif name_lower.endswith("ie"):
                return name[:-2] + "y"

        return "".join(chars)

    def _generate_similar_person(self) -> Dict[str, Any]:
        """generate a similar person for duplicate testing"""
        # chose a random existing uid as the base
        base_uid, base_metadata = random.choice(self.existing_people)

        uid = str(uuid.uuid4())[:8]

        metadata = base_metadata.copy()
        variations = []

        # name variations, 85% chance of some variation
        if random.random() < 0.85:
            variation_type = random.choice(
                [
                    "typo_first",
                    "typo_last",
                    "typo_first",  # weight typos more heavily
                    "typo_last",  # weight typos more heavily
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
                metadata["last_name"] = self._generate_realistic_typo(
                    metadata["last_name"]
                )

            elif variation_type == "nickname":
                # common nickname mappings
                # prompt (gpt-4o): "generate a list of some common nicknames as a python dict"
                nicknames = {
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
                first_name = metadata["first_name"]
                if first_name in nicknames:
                    metadata["first_name"] = nicknames[first_name]

            elif variation_type == "middle_initial":
                if "middle_name" in metadata:
                    metadata["middle_name"] = metadata["middle_name"][0]  # Just initial
                else:
                    metadata["middle_name"] = random.choice(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    )

        # variations on date, 60% chance with bias toward small errors
        if random.random() < 0.6:
            birth_date = datetime.strptime(metadata["date_of_birth"], "%Y-%m-%d")

            # weighted toward smaller date errors (70% are within 30 days)
            if random.random() < 0.7:
                # small error: +-1-30 days (data entry errors, transcription)
                days_variation = random.randint(-30, 30)
            elif random.random() < 0.9:
                # medium error: +-31-90 days (month confusion)
                days_variation = random.randint(-90, 90)
            else:
                # large error: +-91-365 days (year confusion)
                days_variation = random.randint(-365, 365)

            new_birth_date = birth_date + timedelta(days=days_variation)
            metadata["date_of_birth"] = new_birth_date.strftime("%Y-%m-%d")

        # email variations, 65% chance
        if random.random() < 0.65:
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

            email_username = random.choice(email_formats)
            email_domain = random.choice(
                [
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
            )
            metadata["email"] = f"{email_username}@{email_domain}"

        # birth cities, 20%
        if random.random() < 0.2 and "birth_city" in metadata:
            if random.random() < 0.5:
                del metadata["birth_city"]
            else:
                metadata["birth_city"] = random.choice(self.cities)

        return uid, metadata


class ConnectionGenerator:
    """
    generates edges to connect nodes
    """

    def __init__(self, people: List[tuple]):
        self.people = people

    def generate_connections(self, num_edges: int) -> List[Dict[str, Any]]:
        connections = []

        # create indices
        by_last_name = {}

        for uid, metadata in self.people:
            last_name = metadata["last_name"]
            if last_name not in by_last_name:
                by_last_name[last_name] = []
            by_last_name[last_name].append((uid, metadata))

        connection_types = [
            ("family", self._generate_family_connections, by_last_name, 0.3),
            ("random", self._generate_random_connections, None, 0.2),
        ]

        edges_per_type = []
        remaining_edges = num_edges

        for i, (conn_type, generator, groups, ratio) in enumerate(connection_types):
            if i == len(connection_types) - 1:
                edges_count = remaining_edges
            else:
                edges_count = int(num_edges * ratio)
                remaining_edges -= edges_count
            edges_per_type.append((conn_type, generator, groups, edges_count))

        for conn_type, generator, groups, count in edges_per_type:
            new_connections = generator(groups, count)
            connections.extend(new_connections)

        # remove duplicates and loops
        seen = set()
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

        return unique_connections[:num_edges]

    def _generate_family_connections(
        self, by_last_name: Dict, count: int
    ) -> List[Dict]:
        connections = []
        families = [family for family in by_last_name.values() if len(family) > 1]

        for _ in range(count):
            if not families:
                break

            family = random.choice(families)
            if len(family) < 2:
                continue

            person1, person2 = random.sample(family, 2)

            # determine relationship type based on age diff
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

    def _generate_random_connections(self, groups: None, count: int) -> List[Dict]:
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


"""
actually generating a dataset
"""


def generate_dataset(
    num_people: int,
    num_edges: int,
    output_file: str = "dataset.json",
    duplicate_likelihood: float = 0.1,
):
    db = GraphDatabase(output_file)
    db.clear()

    person_gen = Generator(duplicate_likelihood=duplicate_likelihood)
    people = []

    print("generating people...")
    for i in range(num_people):
        uid, metadata = person_gen.generate_person()
        db.add_node(uid, metadata)
        people.append((uid, metadata))

        if (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{num_people} people")

    print("generating connections...")
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

    # Print statistics
    similar_count = int(num_people * duplicate_likelihood)
    print(f"\ndataset generation complete")
    print(f"generated {num_people} people with ~{similar_count} potential duplicates")
    print(f"unique last names: {len(person_gen.used_last_names)}")
    print(f"generated {len(connections)} connections")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="generate a test dataset")
    parser.add_argument("num_people", type=int, help="number of people to gen")
    parser.add_argument("num_edges", type=int, help="number of edges to gen")
    parser.add_argument(
        "-o", "--output", default="dataset.json", help="output file path"
    )
    parser.add_argument(
        "-d",
        "--duplicate-likelihood",
        type=float,
        default=0.1,
        help="likelihood of generating similar/duplicate people (0.0-1.0, default: 0.1)",
    )

    args = parser.parse_args()

    if args.num_edges > args.num_people * (args.num_people - 1):
        print("warn: num_edges is very high relative to num_people")
        print(f"max possible edges: {args.num_people * (args.num_people - 1)}")

    if not (0.0 <= args.duplicate_likelihood <= 1.0):
        print("error: duplicate-likelihood must be between 0.0 and 1.0")
        sys.exit(1)

    generate_dataset(
        args.num_people, args.num_edges, args.output, args.duplicate_likelihood
    )
