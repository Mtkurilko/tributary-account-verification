"""
Author: Michael Kurilko
Date: 6/4/2025
Description: This script runs the ingestion process that grabs data from the dataset.json file and prepares it for the comaprison model.
"""

from .Subject import Subject
import json
import multiprocessing as mp
from functools import partial
import numpy as np


def create_subject_chunk(data_chunk, start_idx):
    """
    Create Subject instances for a chunk of data.
    
    :param data_chunk: Chunk of JSON data
    :param start_idx: Starting index for this chunk
    :return: List of Subject instances
    """
    subjects = []
    for idx, item in enumerate(data_chunk):
        uuid = item.get("node_id")
        item = item.get("metadata", item)

        subject = Subject(
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
        subjects.append((start_idx + idx, subject))
    return subjects


def create_pair_chunk(subjects, chunk_range):
    """
    Create subject pairs for a specific range of indices.
    
    :param subjects: List of all subjects
    :param chunk_range: Tuple of (start_i, end_i) for the outer loop
    :return: List of (pair_index, subject_pair) tuples
    """
    pairs = []
    n_subjects = len(subjects)
    start_i, end_i = chunk_range
    
    for i in range(start_i, min(end_i, n_subjects)):
        for j in range(i + 1, n_subjects):
            # Calculate the pair index as it would be in the sequential version
            pair_idx = sum(n_subjects - k - 1 for k in range(i)) + (j - i - 1)
            pairs.append((pair_idx, (subjects[i], subjects[j])))
    
    return pairs


def ingest_data(file_path, use_multiprocessing=True, num_workers=12):
    """
    Ingest data from a JSON file and create Subject instances.

    :param file_path: Path to the JSON file containing the dataset.
    :param use_multiprocessing: Whether to use multiprocessing for data loading
    :param num_workers: Number of worker processes (None for auto-detection)
    :return: List of Subject instances and subject pairs.
    """
    if not file_path.endswith(".json"):
        raise ValueError("File must be a JSON file.")

    data = read_json(file_path)  # Read the JSON data from the file
    n_subjects = len(data)
    
    # Determine if multiprocessing should be used
    if not use_multiprocessing or n_subjects < 1000:
        print(f"Using sequential processing for {n_subjects} subjects")
        return _ingest_data_sequential(data)
    
    # Auto-detect number of workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), max(2, n_subjects // 1000))
    
    print(f"Using multiprocessing with {num_workers} workers for {n_subjects} subjects")
    
    try:
        return _ingest_data_parallel(data, num_workers)
    except Exception as e:
        print(f"Multiprocessing failed ({e}), falling back to sequential processing")
        return _ingest_data_sequential(data)


def _ingest_data_sequential(data):
    """
    Sequential version of data ingestion (original optimized version).
    """
    n_subjects = len(data)
    subjects = [None] * n_subjects  # pre-allocate with exact size
    n_pairs = n_subjects * (n_subjects - 1) // 2
    subject_pairs = [None] * n_pairs  # pre-allocate pairs list

    # Create subjects
    for idx, item in enumerate(data):
        uuid = item.get("node_id")
        item = item.get("metadata", item)

        subjects[idx] = Subject(
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

    # Create pairs
    pair_idx = 0
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            subject_pairs[pair_idx] = (subjects[i], subjects[j])
            pair_idx += 1

    return subjects, subject_pairs


def _ingest_data_parallel(data, num_workers):
    """
    Parallel version of data ingestion using multiprocessing.
    """
    n_subjects = len(data)
    n_pairs = n_subjects * (n_subjects - 1) // 2
    
    print("Creating subjects in parallel...")
    
    # Step 1: Create subjects in parallel
    chunk_size = max(1, n_subjects // num_workers)
    subject_chunks = []
    
    for i in range(0, n_subjects, chunk_size):
        chunk = data[i:i + chunk_size]
        subject_chunks.append((chunk, i))
    
    with mp.Pool(num_workers) as pool:
        subject_results = pool.starmap(create_subject_chunk, subject_chunks)
    
    # Flatten and sort results to maintain order
    indexed_subjects = []
    for chunk_result in subject_results:
        indexed_subjects.extend(chunk_result)
    
    indexed_subjects.sort(key=lambda x: x[0])  # Sort by index
    subjects = [subject for _, subject in indexed_subjects]
    
    print("Creating subject pairs in parallel...")
    
    # Step 2: Create pairs in parallel
    # Divide the work by splitting the outer loop iterations
    pairs_per_worker = max(1, n_subjects // num_workers)
    pair_chunks = []
    
    for i in range(0, n_subjects, pairs_per_worker):
        end_i = min(i + pairs_per_worker, n_subjects)
        pair_chunks.append((subjects, (i, end_i)))
    
    with mp.Pool(num_workers) as pool:
        pair_results = pool.starmap(create_pair_chunk, pair_chunks)
    
    # Flatten and sort results to maintain order
    indexed_pairs = []
    for chunk_result in pair_results:
        indexed_pairs.extend(chunk_result)
    
    indexed_pairs.sort(key=lambda x: x[0])  # Sort by pair index
    subject_pairs = [pair for _, pair in indexed_pairs]
    
    return subjects, subject_pairs


def obtain_subject_labels(subject_pairs, use_multiprocessing=True, num_workers=None):
    """
    Obtain labels for subject pairs based on user input.

    :param subject_pairs: List of tuples containing subject pairs.
    :param use_multiprocessing: Whether to use multiprocessing for label creation
    :param num_workers: Number of worker processes (None for auto-detection)
    :return: List of labels (1 for match, 0 for non-match).
    """
    n_pairs = len(subject_pairs)
    
    # Use multiprocessing for large datasets
    if use_multiprocessing and n_pairs > 10000:
        if num_workers is None:
            num_workers = min(mp.cpu_count(), max(2, n_pairs // 5000))
            
        print(f"Creating labels in parallel with {num_workers} workers...")
        
        try:
            chunk_size = max(1, n_pairs // num_workers)
            chunks = [subject_pairs[i:i + chunk_size] for i in range(0, n_pairs, chunk_size)]
            
            with mp.Pool(num_workers) as pool:
                label_chunks = pool.map(_obtain_labels_chunk, chunks)
            
            # Flatten results
            labels = []
            for chunk in label_chunks:
                labels.extend(chunk)
                
            return labels
            
        except Exception as e:
            print(f"Parallel label creation failed ({e}), falling back to sequential")
    
    # Sequential version (original optimized)
    return [
        (
            1
            if (
                subj1.attributes.get("base_id") == subj2.attributes.get("base_id")
                and subj1.attributes.get("base_id") is not None
            )
            else 0
        )
        for subj1, subj2 in subject_pairs
    ]


def _obtain_labels_chunk(subject_pairs_chunk):
    """
    Process a chunk of subject pairs to create labels.
    """
    return [
        (
            1
            if (
                subj1.attributes.get("base_id") == subj2.attributes.get("base_id")
                and subj1.attributes.get("base_id") is not None
            )
            else 0
        )
        for subj1, subj2 in subject_pairs_chunk
    ]


def read_json(file_path):
    """
    Reads a JSON file and returns the data.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    with open(file_path, "r") as file:
        return json.load(file).get("nodes")
