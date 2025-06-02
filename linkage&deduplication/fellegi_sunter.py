'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This module contains the Fellegi-Sunter model implementation for linkage and deduplication.
This model is used to calculate the probability of two records being a match based on their attributes.
'''

import numpy as np

def fellegi_sunter_probability(subject1, subject2, m_probs, u_probs):
    """
    Calculate the Fellegi-Sunter match score using m and u probabilities.

    :param subject1: The first subject.
    :param subject2: The second subject.
    :param m_probs: Dict of m-probabilities for each attribute.
    :param u_probs: Dict of u-probabilities for each attribute.
    :return: The probability of the two subjects being a match.
    """

    score = 0.0
    features = ['first_name', 'middle_name', 'last_name', 'dob', 'email', 'birth_city']

    for feature in features:
        attr1 = getattr(subject1, feature, None)
        attr2 = getattr(subject2, feature, None)
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