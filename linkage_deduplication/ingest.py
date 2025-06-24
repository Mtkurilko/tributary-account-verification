"""
Authors: Michael Kurilko, Thomas Bruce
Date: 6/24/2025
Description: This script runs the ingestion process that grabs data from the
             dataset.json file and prepares it for the comparison model.
"""

import json
import multiprocessing as mp
from functools import partial

from .Subject import Subject


def read_json(file_path):
    """
    Reads a JSON file and returns the data.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file).get("nodes")


def _create_subject(item):
    """Create a Subject instance from JSON data."""
    uuid = item.get("node_id")
    item = item.get("metadata", item)
    return Subject(
        first_name=item.get("first_name"),
        middle_name=item.get("middle_name", None),
        last_name=item.get("last_name"),
        dob=item.get("date_of_birth", None),
        dod=item.get("date_of_death", None),
        email=item.get("email"),
        phone_number=item.get("phone_number", None),
        birth_city=item.get("birth_city", None),
        attributes={"base_id": item.get("base_uuid", None), "uuid": uuid},
    )


def _pair_indices(n):
    """generator for i, j index pairs where i < j"""
    for i in range(n):
        for j in range(i + 1, n):
            yield (i, j)


def _pair_from_indices(i, j, subjects):
    """Create a pair from indices."""
    return (subjects[i], subjects[j])


def _label_pair(pair):
    """Create label for a pair of subjects."""
    subj1, subj2 = pair
    base1 = subj1.attributes.get("base_id")
    base2 = subj2.attributes.get("base_id")
    return 1 if base1 is not None and base1 == base2 else 0


def ingest_data(file_path, use_multiprocessing=True, num_workers=None):
    """
    Ingest data from a JSON file and create Subject instances.

    :param file_path: Path to the JSON file containing the dataset.
    :param use_multiprocessing: Whether to use multiprocessing for data loading
    :param num_workers: Number of worker processes (None for auto-detection)
    :return: List of Subject instances and subject pairs.
    """
    if not file_path.endswith(".json"):
        raise ValueError("File must be a JSON file.")

    data = read_json(file_path)
    n_subjects = len(data)

    # determine if multiprocessing should be used
    if not use_multiprocessing or n_subjects < 1000:
        print(f"using sequential processing for {n_subjects} subjects")
        return _ingest_data_sequential(data)

    # auto-detect number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 20)

    print(f"using multiprocessing with {num_workers} workers for {n_subjects} subjects")

    try:
        return _ingest_data_parallel(data, num_workers)
    except Exception as e:
        print(f"multiprocessing failed ({e}), falling back to sequential processing")
        return _ingest_data_sequential(data)


def _ingest_data_sequential(data):
    """
    Sequential version of data ingestion.
    """
    n_subjects = len(data)
    subjects = [None] * n_subjects  # pre-allocate with exact size

    # Create subjects
    for idx, item in enumerate(data):
        subjects[idx] = _create_subject(item)

    # Create pairs
    subject_pairs = []
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            subject_pairs.append((subjects[i], subjects[j]))

    return subjects, subject_pairs


def _ingest_data_parallel(data, num_workers):
    """
    Parallel ingestion
    """
    # create subjects in parallel
    with mp.Pool(num_workers) as pool:
        subjects = pool.map(_create_subject, data)

    # create pairs in parallel
    indices = list(_pair_indices(len(subjects)))
    with mp.Pool(num_workers) as pool:
        make_pair = partial(_pair_from_indices, subjects=subjects)
        subject_pairs = pool.starmap(make_pair, indices)

    return subjects, subject_pairs


def obtain_subject_labels(subject_pairs, use_multiprocessing=True, num_workers=None):
    """
    Obtain labels for subject pairs.

    :param subject_pairs: List of tuples containing subject pairs.
    :param use_multiprocessing: Whether to use multiprocessing for label creation
    :param num_workers: Number of worker processes (None for auto-detection)
    :return: List of labels (1 for match, 0 for non-match).
    """
    n_pairs = len(subject_pairs)

    # Use multiprocessing for large datasets
    if use_multiprocessing and n_pairs > 10000:
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 20)

        print(f"creating labels in parallel with {num_workers} workers...")

        try:
            with mp.Pool(num_workers) as pool:
                return pool.map(_label_pair, subject_pairs)
        except Exception as e:
            print(f"parallel label creation failed ({e}), falling back to sequential")

    return [_label_pair(pair) for pair in subject_pairs]
