'''
Author: Michael Kurilko
Date: 6/4/2025
Description: This script runs the ingestion process that grabs data from the dataset.json file and prepares it for the comaprison model.
'''

from Subject import Subject
import json

def ingest_data(file_path):
    """
    Ingest data from a JSON file and create Subject instances.

    :param file_path: Path to the JSON file containing the dataset.
    :return: List of Subject instances.
    """
    subjects = [] # Create a list of Subject instances
    subject_pairs = [] # Create a list of subject pairs for training/evaluation

    if not file_path.endswith('.json'):
        raise ValueError("File must be a JSON file.")


    data = read_json(file_path) # Read the JSON data from the file

    for item in data:
        uuid = item.get('node_id') # Extract the unique identifier for the subject
        item = item.get('metadata', item) # Grab the metadata from the dataset

        subject = Subject(
            first_name=item.get('first_name'),
            middle_name=item.get('middle_name', None),
            last_name=item.get('last_name'),
            dob=item.get('date_of_birth', None),
            dod=item.get('date_of_death', None),
            email=item.get('email'),
            birth_city=item.get('birth_city', None),
            attributes={'base_id': item.get('base_uuid', None), 'uuid': uuid} # Additional attributes if any
        )
        subjects.append(subject)
    
    # Create pairs of subjects for training/evaluation
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            subject_pairs.append((subjects[i], subjects[j]))
    
    return subjects, subject_pairs


def obtain_subject_labels(subject_pairs):
    """
    Obtain labels for subject pairs based on user input.
    
    :param subject_pairs: List of tuples containing subject pairs.
    :return: List of labels (1 for match, 0 for non-match).
    """
    labels = [] # Create a list to store labels for each subject pair (1 for match, 0 for non-match)

    for subj1, subj2 in subject_pairs:
        # Check if the base_ids (the unqiue identifier variants are based on) are the same
        if (subj1.attributes.get('base_id') == subj2.attributes.get('base_id')) and subj1.attributes.get('base_id') is not None:
            label = 1
        else:
            label = 0

        #Append the label to the list
        labels.append(int(label))

    return labels


def read_json(file_path):
    """
    Reads a JSON file and returns the data.
    
    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file).get('nodes')