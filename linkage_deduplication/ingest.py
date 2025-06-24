'''
Author: Michael Kurilko
Date: 6/4/2025
Description: This script runs the ingestion process that grabs data from the dataset.json file and prepares it for the comaprison model.
'''

from .Subject import Subject
import json
import multiprocessing
from itertools import combinations
from functools import partial


def read_json(file_path):
    """
    Reads a JSON file and returns the data.
    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file).get('nodes')


def _pair_from_indices(i, j, subjects):
    return (subjects[i], subjects[j])

def _create_subject(item):
    uuid = item.get('node_id')
    item = item.get('metadata', item)
    return Subject(
        first_name=item.get('first_name'),
        middle_name=item.get('middle_name', None),
        last_name=item.get('last_name'),
        dob=item.get('date_of_birth', None),
        dod=item.get('date_of_death', None),
        email=item.get('email'),
        phone_number=item.get('phone_number', None),
        birth_city=item.get('birth_city', None),
        attributes={'base_id': item.get('base_uuid', None), 'uuid': uuid}
    )


def _pair_indices(n):
    """Efficient generator for i, j index pairs where i < j"""
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, j)


def _label_pair(pair):
    subj1, subj2 = pair
    base1 = subj1.attributes.get('base_id')
    base2 = subj2.attributes.get('base_id')
    return 1 if base1 is not None and base1 == base2 else 0


def obtain_subject_labels(subject_pairs):
    """
    Parallelized label generation.
    :param subject_pairs: List of subject pairs.
    :return: List of labels (1 for match, 0 for non-match).
    """
    with multiprocessing.Pool() as pool:
        return list(pool.map(_label_pair, subject_pairs))


def ingest_data(file_path):
    """
    Generates subjects and all unique pairs (i < j), parallel-safe.
    :param file_path: Path to the JSON file.
    :return: Tuple (subjects, subject_pairs)
    """
    if not file_path.endswith('.json'):
        raise ValueError("File must be a JSON file.")

    data = read_json(file_path)

    # Use parallel map to create Subject instances
    with multiprocessing.Pool() as pool:
        subjects = pool.map(_create_subject, data)

    # Efficiently generate all subject pairs (i < j)
    indices = list(_pair_indices(len(subjects)))
    with multiprocessing.Pool() as pool:
        make_pair = partial(_pair_from_indices, subjects=subjects)
        subject_pairs = pool.starmap(make_pair, indices)

    return subjects, subject_pairs

