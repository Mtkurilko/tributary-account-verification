'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This script is the main entry point for the linkage and deduplication process.
'''

from evaluation_models.fellegi_sunter import fellegi_sunter_probability as fs_prob
from evaluation_models.ComparisonModel import ComparisonModel
from Subject import Subject
import ingest

def main():
    '''
    # Example subjects
    subject1 = Subject(
        first_name="Richard",
        middle_name="L.",
        last_name="Jenkins",
        dob="1973-07-04",
        dod="1999-12-31",
        email="RLJ98@aol.com",
        birth_city="Ohio",
    )
    subject2 = Subject(
        first_name="Richard",
        middle_name="L.",
        last_name="Jankins",
        dob="1973-07-05",
        dod=None,
        email="RLJ98@aol.com",
        birth_city="Ohio",
    )
    '''

    subjects, subject_pairs  = ingest.ingest_data("dataset.json")  # Load subjects and pairs from dataset.json
    
    # Ask the user if they want to use the ComparisonModel
    model_requested = int(input("Do you want to use 1) ComparisonModel or 2) Fellegi-Sunter model? (Enter 1 or 2): "))

    # Run the requested model
    if model_requested == 1:
        # Initialize the ComparisonModel
        model = ComparisonModel()

        # Prompt for training data or loading a model
        load_model = input("Do you want to load a pre-trained model? (Y/N): ").strip().lower()
        if load_model == 'y':
            path = input("Enter the path to the pre-trained model: ").strip()
            model.load(path)
            print("Loading pre-trained model...")

        train_model = input("Do you want to train the model? (Y/N): ").strip().lower()
        if train_model == 'y':
            # Subject pair arew tuples of (subject1, subject2) and already created
            labels = ingest.obtain_subject_labels(subject_pairs)  # Obtain labels for the subject pairs

            #Train the model with the subject pairs and labels
            model.train_transformer(subject_pairs, labels, epochs=20, lr=1e-3)

            # Save the trained model (WHEN READY)
            save_model = input("Do you want to save the trained model? (Y/N): ").strip().lower()
            if save_model == 'y':
                path = input("Enter the path to save the model: ").strip()
                model.save(path)
                print(f"Model saved to {path}")

        # Run through each pair of subjects in the dataset
        for subject1, subject2 in subject_pairs:
            # Calculate the gradient-boosted score
            gb_score = model.gradient_boosted_score(subject1, subject2)
            print(f"Gradient Boosted Score between {subject1.first_name} and {subject2.first_name}: {gb_score:.4f}")
            
            # Calculate the transformer similarity score
            transformer_score = model.transformer_similarity(subject1, subject2)
            print(f"Transformer Similarity Score between {subject1.first_name} and {subject2.first_name}: {transformer_score:.4f}")

    elif model_requested == 2:
        # Example m and u probabilities (6/3/2025 - MORE REFINED match and unique probabilities)
        m_probs = {
            'first_name': 0.9,
            'middle_name': 0.8,
            'last_name': 0.95,
            'dob': 0.98,
            'dod': 0.98,
            'email': 0.99,
            'birth_city': 0.9
        }
        u_probs = {
            'first_name': 0.1,
            'middle_name': 0.2,
            'last_name': 0.005,
            'dob': 0.005,
            'dod': 0.005,
            'email': 0.3,
            'birth_city': 0.05
        }

        # Run through each pair of subjects in the dataset
        for subject1, subject2 in subject_pairs:
            # Calculate the probability of a match
            probability = fs_prob(subject1, subject2, m_probs, u_probs)
            print(f"Felligi-Sunter Similarity Score between {subject1.first_name} and {subject2.first_name}: {probability:.4f}")
    
if __name__ == "__main__":
    main()
