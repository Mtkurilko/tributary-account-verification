'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This script is the main entry point for the linkage and deduplication process.
'''

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkage_deduplication.evaluation_models.fellegi_sunter import fellegi_sunter_probability as fs_prob
from linkage_deduplication.evaluation_models.ComparisonModel import ComparisonModel
from linkage_deduplication.Subject import Subject
from linkage_deduplication import ingest

def main(modelRequested=None, jsonPath=None, doLoadModel=None, loadPath=None, doTrainModel=None, doSaveModel=None, savePath=None):
    '''
    # Example subjects
    subject1 = Subject(
        first_name="Michael",
        middle_name="L.",
        last_name="Jenkins",
        dob="1973-07-04",
        dod="1999-12-31",
        email="RLJ98@hotmail.com",
        birth_city="Ohio",
    )
    subject2 = Subject(
        first_name="Micheal",
        middle_name="L.",
        last_name="Jankins",
        dob="1973-07-05",
        dod=None,
        email="RLJ98@aol.com",
        birth_city="Ohio",
    )
    '''

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

    # Prompt the user for the path to the dataset
    path = jsonPath if jsonPath is not None else input("Enter the path to the dataset (dataset.json): ").strip()

    subjects, subject_pairs  = ingest.ingest_data(path)  # Load subjects and pairs from dataset

    labels = ingest.obtain_subject_labels(subject_pairs)  # Obtain labels for the subject pairs
    
    # Ask the user if they want to use the ComparisonModel
    model_requested = modelRequested if modelRequested is not None else int(input("Do you want to use 1) ComparisonModel, 2) Fellegi-Sunter model, or 3) Both? (Enter 1, 2, or 3): "))

    # Run the requested model
    if model_requested == 1:
        # Initialize the ComparisonModel
        model = ComparisonModel()

        # Prompt for training data or loading a model
        load_model = ('y' if doLoadModel == True else 'n') if doLoadModel is not None else input("Do you want to load a pre-trained model? (Y/N): ").strip().lower()
        if load_model == 'y':
            path = loadPath if loadPath is not None else input("Enter the path to the pre-trained model: ").strip()
            model.load(path)
            print("Loading pre-trained model...")

        train_model = ('y' if doTrainModel == True else 'n') if doTrainModel is not None else input("Do you want to train the model? (Y/N): ").strip().lower()
        if train_model == 'y':
            # Subject pair arew tuples of (subject1, subject2) and already created

            #Train the model with the subject pairs and labels
            model.train_transformer(subject_pairs, labels, epochs=20, lr=1e-3)

            # Save the trained model (WHEN READY)
            save_model = ('y' if doSaveModel == True else 'n') if doSaveModel is not None else input("Do you want to save the trained model? (Y/N): ").strip().lower()
            if save_model == 'y':
                path = savePath if savePath is not None else input("Enter the path to save the model: ").strip()
                model.save(path)
                print(f"Model saved to {path}")

        # Run through each pair of subjects in the dataset
        for i, (subject1, subject2) in enumerate(subject_pairs):
            # Create a matched f-string for output
            matched = f"Yes Base_ID:{subject2.attributes.get('base_id')}"

            # Calculate the gradient-boosted score
            gb_score = model.gradient_boosted_score(subject1, subject2)
            
            print(f"Gradient Boosted Score between {subject1.attributes.get('uuid')} and {subject2.attributes.get('uuid')}: {gb_score:.4f} | (Match: {matched if (labels[i] == 1) else 'No'})")
            
            # Calculate the transformer similarity score
            transformer_score = model.transformer_similarity(subject1, subject2)
            print(f"Transformer Similarity Score between {subject1.attributes.get('uuid')} and {subject2.attributes.get('uuid')}: {transformer_score:.4f} | (Match: {matched if (labels[i] == 1) else 'No'})")

            # Print a separator for clarity
            print("-" * 80)

    elif model_requested == 2:
        # Run through each pair of subjects in the dataset
        for i, (subject1, subject2) in enumerate(subject_pairs):
            # Create a matched f-string for output
            matched = f"Yes Base_ID:{subject2.attributes.get('base_id')}"

            # Calculate the probability of a match
            probability = fs_prob(subject1, subject2, m_probs, u_probs)
            print(f"Felligi-Sunter Similarity Score between {subject1.attributes.get('uuid')} and {subject2.attributes.get('uuid')}: {probability:.4f}  | (Match: {matched if (labels[i] == 1) else 'No'})")
    
    elif model_requested == 3:
        # Initialize the ComparisonModel
        model = ComparisonModel()

        # Prompt for training data or loading a model
        load_model = ('y' if doLoadModel == True else 'n') if doLoadModel is not None else input("Do you want to load a pre-trained model? (Y/N): ").strip().lower()
        if load_model == 'y':
            path = loadPath if loadPath is not None else input("Enter the path to the pre-trained model: ").strip()
            model.load(path)
            print("Loading pre-trained model...")

        train_model = ('y' if doTrainModel == True else 'n') if doTrainModel is not None else input("Do you want to train the model? (Y/N): ").strip().lower()
        if train_model == 'y':
            # Subject pair arew tuples of (subject1, subject2) and already created

            #Train the model with the subject pairs and labels
            model.train_transformer(subject_pairs, labels, epochs=20, lr=1e-3)

            # Save the trained model (WHEN READY)
            save_model = ('y' if doSaveModel == True else 'n') if doSaveModel is not None else input("Do you want to save the trained model? (Y/N): ").strip().lower()
            if save_model == 'y':
                path = savePath if savePath is not None else input("Enter the path to save the model: ").strip()
                model.save(path)
                print(f"Model saved to {path}")

        # Run through each pair of subjects in the dataset
        for i, (subject1, subject2) in enumerate(subject_pairs):
            # Create a matched f-string for output
            matched = f"Yes Base_ID:{subject2.attributes.get('base_id')}"

            # Calculate the gradient-boosted score
            gb_score = model.gradient_boosted_score(subject1, subject2)
            print(f"Gradient Boosted Score between {subject1.attributes.get('uuid')} and {subject2.attributes.get('uuid')}: {gb_score:.4f} | (Match: {matched if (labels[i] == 1) else 'No'})")
            
            # Calculate the transformer similarity score
            transformer_score = model.transformer_similarity(subject1, subject2)
            print(f"Transformer Similarity Score between {subject1.attributes.get('uuid')} and {subject2.attributes.get('uuid')}: {transformer_score:.4f} | (Match: {matched if (labels[i] == 1) else 'No'})")
    
            # Calculate Fellegi-Sunter probability
            probability = fs_prob(subject1, subject2, m_probs, u_probs)
            print(f"Felligi-Sunter Similarity Score between {subject1.attributes.get('uuid')} and {subject2.attributes.get('uuid')}: {probability:.4f}  | (Match: {matched if (labels[i] == 1) else 'No'})")
    
            # Print a separator for clarity
            print("-" * 80)


def module_run(modelRequested, jsonPath, doLoadModel=False, loadPath=None, doTrainModel=False, doSaveModel=False, savePath=None):
    '''
    This function is called when the module is imported.
    It runs the main function to execute the linkage and deduplication process.
    '''
    main(modelRequested=modelRequested, jsonPath=jsonPath, doLoadModel=doLoadModel, loadPath=loadPath, doTrainModel=doTrainModel, doSaveModel=doSaveModel, savePath=savePath)

if __name__ == "__main__":
    main()
