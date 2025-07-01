"""
generate.py

dataset generator for testing comparison methods with realistic duplicate data
also generates comparison sequences for training transformer-based models off
of realistic data
follows the tributary-account-verification database schema

date: 06/06/2025
author: thomas bruce
last update: NANP phone numbers, moved data to another file
"""

# TODO:
# - finalize timestamp creation system for simulated sybil attacks
# - improve consistency of name variation

import os
import sys
import json
import uuid
import random
import argparse

from data import *

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.gdb import GraphDatabase

from faker import Faker


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

    def _generate_phone_number(self) -> str:
        """generate nanp phone numbers from reserved area codes"""
        # reserved and fictional area codes from dataset
        area_code = random.choice(NANP_AREA_CODES)

        # exchange code
        exchange = f"{random.randint(2, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"

        # subscriber number
        subscriber = f"{random.randint(0, 9999):04d}"

        return f"{area_code}{exchange}{subscriber}"

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

        # optional phone number, 90%
        phone_number = (
            self._generate_phone_number()
            if random.random() < PHONE_NUMBER_CHANCE
            else None
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
        if phone_number:
            metadata["phone_number"] = phone_number

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

    def _apply_phone_variation(self, metadata: Dict[str, Any]) -> None:
        """apply phone number variations for similar person generation"""
        if "phone_number" in metadata:
            # 30% chance to remove phone number entirely
            if random.random() < 0.3:
                del metadata["phone_number"]
            else:
                # generate a new phone number (representing different phone/carrier)
                metadata["phone_number"] = self._generate_phone_number()
        else:
            # 20% chance to add a phone number
            if random.random() < 0.2:
                metadata["phone_number"] = self._generate_phone_number()

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

        # phone number variations
        # TODO: make globally tweakable!
        if random.random() < 0.4:
            self._apply_phone_variation(metadata)

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


class FamilyGenerator:
    """generate realistic family structures with proper genealogical relationships"""

    def __init__(self, people: List[Tuple[str, Dict[str, Any]]]):
        self.people = people
        self.families = {}  # family_id -> list of people
        self.family_trees = {}  # family_id -> tree structure

    def _get_birth_year(self, person_data: Dict[str, Any]) -> int:
        """extract birth year from person data"""
        return int(person_data["date_of_birth"][:4])

    def _create_family_connection(
        self,
        uid1: str,
        uid2: str,
        relationship: str,
        family_id: str,
        directed: bool = False,
    ) -> Dict[str, Any]:
        """create a family connection with consistent structure"""
        return {
            "source": uid1,
            "target": uid2,
            "directed": directed,
            "metadata": {
                "type": "family",
                "relationship": relationship,
                "family_id": family_id,
            },
        }

    def create_family_structures(self) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """organize people into realistic family units"""
        # group by last name as starting point
        by_last_name = {}
        for uid, metadata in self.people:
            last_name = metadata["last_name"]
            by_last_name.setdefault(last_name, []).append((uid, metadata))

        # create family units, considering ages and names
        family_id = 0
        for last_name, people_with_name in by_last_name.items():
            if len(people_with_name) < 2:
                continue

            # sort by birth date to establish generations
            people_with_name.sort(key=lambda x: x[1]["date_of_birth"])

            # group into generations
            generations = self._group_into_generations(people_with_name)

            # create family tree structure
            if len(generations) >= 2:
                family_key = f"family_{family_id}"
                self.families[family_key] = people_with_name
                self.family_trees[family_key] = generations
                family_id += 1

        return self.families

    def _group_into_generations(
        self, people: List[Tuple[str, Dict[str, Any]]]
    ) -> List[List]:
        """group people into generations based on age gaps"""
        generations = []
        current_gen = [people[0]]

        for person in people[1:]:
            birth_year = self._get_birth_year(person[1])
            last_birth_year = self._get_birth_year(current_gen[-1][1])

            if birth_year - last_birth_year > GENERATION_AGE_GAP:
                generations.append(current_gen)
                current_gen = [person]
            else:
                current_gen.append(person)

        if current_gen:
            generations.append(current_gen)

        return generations

    def generate_family_connections(self, target_count: int) -> List[Dict[str, Any]]:
        """generate realistic family relationships"""
        connections = []

        for family_id, generations in self.family_trees.items():
            # parent-child relationships
            connections.extend(
                self._generate_parent_child_connections(family_id, generations)
            )

            # sibling relationships
            connections.extend(
                self._generate_sibling_connections(family_id, generations)
            )

            # spouse relationships
            connections.extend(self._generate_spouse_connections(family_id))

        return connections[:target_count]

    def _generate_parent_child_connections(
        self, family_id: str, generations: List[List]
    ) -> List[Dict[str, Any]]:
        """generate parent-child relationships between consecutive generations"""
        connections = []

        for i in range(len(generations) - 1):
            parents = generations[i]
            children = generations[i + 1]

            for parent_uid, parent_data in parents:
                for child_uid, child_data in children:
                    parent_birth = self._get_birth_year(parent_data)
                    child_birth = self._get_birth_year(child_data)
                    age_diff = child_birth - parent_birth

                    if (
                        PARENT_CHILD_MIN_AGE_DIFF
                        <= age_diff
                        <= PARENT_CHILD_MAX_AGE_DIFF
                    ):
                        connections.append(
                            self._create_family_connection(
                                parent_uid,
                                child_uid,
                                "parent_child",
                                family_id,
                                directed=True,
                            )
                        )

        return connections

    def _generate_sibling_connections(
        self, family_id: str, generations: List[List]
    ) -> List[Dict[str, Any]]:
        """generate sibling relationships within same generation"""
        connections = []

        for generation in generations:
            if len(generation) > 1:
                for i, (uid1, data1) in enumerate(generation):
                    for uid2, data2 in generation[i + 1 :]:
                        birth1 = self._get_birth_year(data1)
                        birth2 = self._get_birth_year(data2)

                        if abs(birth1 - birth2) <= SIBLING_MAX_AGE_DIFF:
                            connections.append(
                                self._create_family_connection(
                                    uid1, uid2, "sibling", family_id
                                )
                            )

        return connections

    def _generate_spouse_connections(self, family_id: str) -> List[Dict[str, Any]]:
        """generate spouse relationships (different last names, similar ages)"""
        connections = []

        # this is a simplified version - could be more sophisticated
        for person_uid, person_data in self.people:
            if person_data["last_name"] == family_id.split("_")[1]:
                continue

            # find potential spouses in this family
            for family_person_uid, family_person_data in self.families.get(
                family_id, []
            ):
                person_birth = self._get_birth_year(person_data)
                family_birth = self._get_birth_year(family_person_data)

                # spouses typically within age difference limit
                if (
                    abs(person_birth - family_birth) <= SPOUSE_MAX_AGE_DIFF
                    and random.random() < SPOUSE_CONNECTION_CHANCE
                ):
                    connections.append(
                        self._create_family_connection(
                            person_uid, family_person_uid, "spouse", family_id
                        )
                    )

        return connections


class ConnectionGenerator:
    """generate edges to connect nodes with realistic relationship patterns"""

    def __init__(self, people: List[Tuple[str, Dict[str, Any]]]):
        self.people = people
        self.family_gen = FamilyGenerator(people)

        # organize people by demographic clusters for realistic connections
        self.city_clusters = self._create_city_clusters()
        self.age_clusters = self._create_age_clusters()
        self.email_clusters = self._create_email_clusters()

        # degree tracking for preferential attachment
        self.node_degrees = {uid: 0 for uid, _ in people}

    def _create_city_clusters(self) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """group people by birth city for geographic clustering"""
        clusters = {}
        for uid, metadata in self.people:
            city = metadata.get("birth_city", "unknown")
            clusters.setdefault(city, []).append((uid, metadata))
        return clusters

    def _create_age_clusters(self) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """group people by birth decade for age-based clustering"""
        clusters = {}
        for uid, metadata in self.people:
            birth_year = int(metadata["date_of_birth"][:4])
            decade = f"{birth_year // 10 * 10}s"
            clusters.setdefault(decade, []).append((uid, metadata))
        return clusters

    def _create_email_clusters(self) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """group people by email domain for workplace/school clustering"""
        clusters = {}
        for uid, metadata in self.people:
            domain = metadata["email"].split("@")[1]
            clusters.setdefault(domain, []).append((uid, metadata))
        return clusters

    def _weighted_random_choice(
        self, items_with_weights: Dict[str, float]
    ) -> Optional[str]:
        """select item based on weights, return None if no items"""
        if not items_with_weights:
            return None

        total_weight = sum(items_with_weights.values())
        if total_weight <= 0:
            return random.choice(list(items_with_weights.keys()))

        r = random.uniform(0, total_weight)
        cumulative = 0

        for item, weight in items_with_weights.items():
            cumulative += weight
            if r <= cumulative:
                return item

        return list(items_with_weights.keys())[-1]  # fallback

    def _create_connection(
        self,
        person1_uid: str,
        person2_uid: str,
        conn_type: str,
        relationship: str,
        **extra_metadata,
    ) -> Dict[str, Any]:
        """create a connection dict with consistent structure"""
        metadata = {"type": conn_type, "relationship": relationship}
        metadata.update(extra_metadata)

        return {
            "source": person1_uid,
            "target": person2_uid,
            "directed": False,
            "metadata": metadata,
        }

    def _generate_cluster_connections(
        self,
        count: int,
        clusters: Dict[str, List],
        conn_type: str,
        relationship: str,
        weight_func=None,
        **extra_metadata_func,
    ) -> List[Dict[str, Any]]:
        """generic method to generate connections within clusters"""
        connections = []

        for _ in range(count):
            # filter clusters with at least 2 people
            valid_clusters = {
                key: people for key, people in clusters.items() if len(people) >= 2
            }
            if not valid_clusters:
                break

            # calculate weights for cluster selection
            if weight_func:
                cluster_weights = {
                    key: weight_func(key, people)
                    for key, people in valid_clusters.items()
                }
            else:
                cluster_weights = {
                    key: len(people) for key, people in valid_clusters.items()
                }

            # select cluster
            selected_cluster = self._weighted_random_choice(cluster_weights)
            if not selected_cluster:
                continue

            # select two people from cluster
            people_in_cluster = valid_clusters[selected_cluster]
            person1 = random.choice(people_in_cluster)
            person2 = self._select_with_preferential_attachment(
                people_in_cluster, person1[0]
            )

            if person2 and person1[0] != person2[0]:
                # build extra metadata
                extra_metadata = {}
                if extra_metadata_func:
                    for key, func in extra_metadata_func.items():
                        extra_metadata[key] = func(selected_cluster, person1, person2)

                connections.append(
                    self._create_connection(
                        person1[0],
                        person2[0],
                        conn_type,
                        relationship,
                        **extra_metadata,
                    )
                )

        return connections

    def _select_with_preferential_attachment(
        self, candidates: List[Tuple[str, Dict[str, Any]]], exclude_uid: str = None
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """select a person with preferential attachment (popular nodes more likely)"""
        if not candidates:
            return None

        # filter out excluded person
        if exclude_uid:
            candidates = [(uid, data) for uid, data in candidates if uid != exclude_uid]

        if not candidates:
            return None

        # simple weighted selection based on degree
        weights = {uid: self.node_degrees.get(uid, 0) + 1 for uid, _ in candidates}
        selected_uid = self._weighted_random_choice(weights)

        # return the full tuple for the selected uid
        for uid, data in candidates:
            if uid == selected_uid:
                return (uid, data)

        return candidates[0]  # fallback

    def _generate_geographic_connections(self, count: int) -> List[Dict[str, Any]]:
        """generate connections between people in same cities"""
        return self._generate_cluster_connections(
            count=count,
            clusters=self.city_clusters,
            conn_type="geographic",
            relationship="neighbor",
            city=lambda cluster_key, p1, p2: cluster_key,
        )

    def _generate_age_cohort_connections(self, count: int) -> List[Dict[str, Any]]:
        """generate connections between people of similar ages"""
        return self._generate_cluster_connections(
            count=count,
            clusters=self.age_clusters,
            conn_type="age_cohort",
            relationship="peer",
            decade=lambda cluster_key, p1, p2: cluster_key,
        )

    def _generate_email_domain_connections(self, count: int) -> List[Dict[str, Any]]:
        """generate connections between people with same email domain (workplace/school)"""
        connections = []

        for _ in range(count):
            # filter domains with at least 2 people
            valid_domains = {
                domain: people
                for domain, people in self.email_clusters.items()
                if len(people) >= 2
            }
            if not valid_domains:
                break

            # calculate domain weights
            domain_weights = {}
            for domain, people in valid_domains.items():
                base_weight = len(people)
                domain_weights[domain] = base_weight * (
                    CONSUMER_EMAIL_WEIGHT
                    if domain in CONSUMER_EMAIL_DOMAINS
                    else WORKPLACE_EMAIL_WEIGHT
                )

            # select domain
            selected_domain = self._weighted_random_choice(domain_weights)
            if not selected_domain:
                continue

            # select two people from domain
            people_in_domain = valid_domains[selected_domain]
            person1 = random.choice(people_in_domain)
            person2 = self._select_with_preferential_attachment(
                people_in_domain, person1[0]
            )

            if person2 and person1[0] != person2[0]:
                relationship = (
                    "colleague"
                    if selected_domain not in CONSUMER_EMAIL_DOMAINS
                    else "acquaintance"
                )
                connections.append(
                    self._create_connection(
                        person1[0],
                        person2[0],
                        "workplace",
                        relationship,
                        domain=selected_domain,
                    )
                )

        return connections

    def _generate_small_world_connections(self, count: int) -> List[Dict[str, Any]]:
        """generate random long-distance connections for small-world property"""
        connections = []

        for _ in range(count):
            if len(self.people) < 2:
                break

            person1, person2 = random.sample(self.people, 2)
            connections.append(
                self._create_connection(
                    person1[0], person2[0], "small_world", "distant_acquaintance"
                )
            )
            # small world connections can be directed
            connections[-1]["directed"] = random.choice([True, False])

        return connections

    def generate_connections(self, num_edges: int) -> List[Dict[str, Any]]:
        """generate connections with realistic social network patterns"""
        connections = []

        # create family structures first
        print("  organizing family structures")
        families = self.family_gen.create_family_structures()
        print(f"  created {len(families)} family units")

        # connection types with realistic distribution
        connection_types = [
            (
                "family",
                self.family_gen.generate_family_connections,
                CONNECTION_TYPE_RATIOS["family"],
            ),
            (
                "geographic",
                self._generate_geographic_connections,
                CONNECTION_TYPE_RATIOS["geographic"],
            ),
            (
                "age_cohort",
                self._generate_age_cohort_connections,
                CONNECTION_TYPE_RATIOS["age_cohort"],
            ),
            (
                "workplace",
                self._generate_email_domain_connections,
                CONNECTION_TYPE_RATIOS["workplace"],
            ),
            (
                "small_world",
                self._generate_small_world_connections,
                CONNECTION_TYPE_RATIOS["small_world"],
            ),
        ]

        # distribute edges across connection types
        remaining_edges = num_edges
        for i, (conn_type, generator, ratio) in enumerate(connection_types):
            if i == len(connection_types) - 1:
                edges_count = remaining_edges
            else:
                edges_count = int(num_edges * ratio)
                remaining_edges -= edges_count

            if edges_count > 0:
                new_connections = generator(edges_count)
                connections.extend(new_connections)
                print(f"  generated {len(new_connections)} {conn_type} connections")

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

                # update degree counts for preferential attachment
                self.node_degrees[conn["source"]] += 1
                self.node_degrees[conn["target"]] += 1

        return unique_connections[:max_count]


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
    family_connections = len(
        [c for c in connections if c["metadata"].get("type") == "family"]
    )

    print(f"\ndataset generation complete!")
    print(f"generated {num_people} people with ~{similar_count} potential duplicates")
    print(f"unique last names: {len(person_gen.used_last_names)}")
    print(f"generated {len(connections)} connections:")
    print(f"  - {family_connections} family relationships")
    print(f"  - {len(connections) - family_connections} random connections")


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


# Michael: This function allows running this script outside the terminal
# and is useful for programmatically generating datasets or
# sequences using an imported function.
def generate_from_args(
    num_people: int = None,
    num_edges: int = None,
    output: str = "dataset.json",
    duplicate_likelihood: float = DEFAULT_DUPLICATE_LIKELIHOOD,
    sequence: int = None,
    steps: int = DEFAULT_SEQUENCE_STEPS,
):
    """
    Generate a dataset or sequences, just like the CLI, but callable from Python.
    """
    if sequence is not None:
        if sequence <= 0:
            raise ValueError("sequence count must be positive")
        output_file = output if output != "dataset.json" else "sequences.json"
        sequences = generate_sequences(sequence, steps, output_file)
        return sequences
    else:
        if num_people is None or num_edges is None:
            raise ValueError(
                "num_people and num_edges are required for regular dataset generation"
            )
        max_edges = num_people * (num_people - 1)
        if num_edges > max_edges:
            print(f"warn: num_edges ({num_edges}) is very high relative to num_people")
            print(f"maximum possible edges: {max_edges}")
        if not 0.0 <= duplicate_likelihood <= 1.0:
            raise ValueError("duplicate-likelihood must be between 0.0 and 1.0")
        generate_dataset(num_people, num_edges, output, duplicate_likelihood)
        return output


if __name__ == "__main__":
    main()
