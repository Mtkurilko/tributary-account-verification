'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This script is the main entry point for the linkage and deduplication process.
'''

from fellegi_sunter import fellegi_sunter_probability as fs_prob
from Subject import Subject

def main():
    # Example subjects
    subject1 = Subject(
        first_name="John",
        middle_name="A.",
        last_name="Doe",
        dob="1990-01-01",
        email="JohnDoe@gmail.com",
        birth_city="New York"
    )
    subject2 = Subject(
        first_name="John",
        middle_name="A.",
        last_name="Doe",
        dob="1990-01-01",
        email="JohnDoe@gmail.com",
        birth_city="New York"
    )

    # Example m and u probabilities (6/3/2025 - MORE REFINED match and unique probabilities)
    m_probs = {
        'first_name': 0.9,
        'middle_name': 0.8,
        'last_name': 0.95,
        'dob': 0.98,
        'email': 0.99,
        'birth_city': 0.9
    }
    u_probs = {
        'first_name': 0.1,
        'middle_name': 0.2,
        'last_name': 0.005,
        'dob': 0.005,
        'email': 0.3,
        'birth_city': 0.05
    }

    # Calculate the probability of a match
    probability = fs_prob(subject1, subject2, m_probs, u_probs)
    print(f"Probability of match between {subject1.first_name} and {subject2.first_name}: {probability:.4f}")
    
if __name__ == "__main__":
    main()
