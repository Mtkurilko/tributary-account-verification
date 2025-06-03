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

    def __init__(self):
        self.faker = Faker()

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

    def generate_person(self) -> Dict[str, Any]:
        """generate a single person"""
        uid = str(uuid.uuid4())[:8]

        # generate a profile of consistent data
        profile = self.faker.profile()
        name_parts = profile["name"].split()

        # name extraction
        first_name = name_parts[0]
        middle_name = self.faker.first_name() if random.random() < 0.7 else None
        last_name = name_parts[-1]

        # birth date (1900-2020)
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
        email_domain = random.choice([
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
        ])
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
    num_people: int, num_edges: int, output_file: str = "dataset.json"
):
    db = GraphDatabase(output_file)
    db.clear()

    person_gen = Generator()
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="generate a test dataset")
    parser.add_argument("num_people", type=int, help="number of people to gen")
    parser.add_argument("num_edges", type=int, help="number of edges to gen")
    parser.add_argument(
        "-o", "--output", default="dataset.json", help="output file path"
    )

    args = parser.parse_args()

    if args.num_edges > args.num_people * (args.num_people - 1):
        print("warn: num_edges is very high relative to num_people")
        print(f"max possible edges: {args.num_people * (args.num_people - 1)}")

    generate_dataset(args.num_people, args.num_edges, args.output)
