'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This module contains the Fellegi-Sunter model implementation for linkage and deduplication.
This model is used to calculate the probability of two records being a match based on their attributes.
'''

import numpy as np
import string

def fellegi_sunter_probability(subject1, subject2, m_probs, u_probs):
    """
    Calculate the Fellegi-Sunter match score using m and u probabilities.

    :param subject1: The first subject.
    :param subject2: The second subject.
    :param m_probs: Dict of m-probabilities for each attribute.
    :param u_probs: Dict of u-probabilities for each attribute.
    :return: The probability of the two subjects being a match.
    """

    score = 0.0000001  # Initialize score to a small value to avoid log(0)
    # List of features to compare
    features = ['first_name', 'middle_name', 'last_name', 'dob', 'dod', 'email', 'phone_number', 'birth_city']

    for feature in features:
        attr1 = normalize_string(getattr(subject1, feature, None)) if isinstance(getattr(subject1, feature, None), str) else getattr(subject1, feature, None) # Uniform case handling for string attributes
        attr2 = normalize_string(getattr(subject2, feature, None)) if isinstance(getattr(subject2, feature, None), str) else getattr(subject2, feature, None) # Uniform case handling for string attributes
        m = m_probs.get(feature, 0.9)  # Default values can be adjusted
        u = u_probs.get(feature, 0.1)
        if attr1 is None or attr2 is None:
            continue  # Skip missing data
        if attr1 == attr2:
            score += np.log(m / u)
        else:
            score += np.log((1 - m) / (1 - u))

    # Convert log-odds score to probability
    odds = np.exp(score)
    probability = odds / (1 + odds)
    return probability


def normalize_string(s):
    """
    Normalize a string by removing punctuation and converting to lowercase.

    :param s: The string to normalize.
    :return: The normalized string.
    """
    if not isinstance(s, str):
        return s  # Return as is if not a string
    s = s.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return s.lower()  # Convert to lowercase