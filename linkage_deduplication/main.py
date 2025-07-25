'''
Author: Michael Kurilko
Date: 6/2/2025
Description: This script is the main entry point for the linkage and deduplication process.
'''

import os
import sys
import csv
import math

import collections
import collections.abc
collections.Container = collections.abc.Container

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkage_deduplication.evaluation_models.fellegi_sunter import fellegi_sunter_probability as fs_prob
from linkage_deduplication.evaluation_models.transformer_model import TransformerModel
from linkage_deduplication.Subject import Subject
from linkage_deduplication import ingest
from linkage_deduplication.evaluation_models.gradient_model import GradientModel

def main(modelRequested="2", jsonPath="./dataset/train.json", doLoadModel={"gradient": None, "transformer": True}, 
        loadPath={"gradient": None, "transformer": "./ercl_transfromer_weights.npz"}, doTrainModel={"gradient": None, "transformer": True}, 
        doSaveModel={"gradient": None, "transformer": None}, savePath={"gradient": None, "transformer": None}, trainParams={}):
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
    # // Used by Felligi-Sunter model
    m_probs = {
        'first_name': 0.9,
        'middle_name': 0.8,
        'last_name': 0.95,
        'dob': 0.98,
        'dod': 0.98,
        'email': 0.99,
        'phone_number': 0.99,
        'birth_city': 0.9
    }
    u_probs = {
        'first_name': 0.1,
        'middle_name': 0.2,
        'last_name': 0.005,
        'dob': 0.005,
        'dod': 0.005,
        'email': 0.3,
        'phone_number': 0.05,
        'birth_city': 0.05
    }

    # Prompt the user for the path to the dataset
    if jsonPath is not None:
        path = jsonPath 
    else:
        path = input("Enter the path to the dataset (dataset.json): ").strip()

    # Load subjects and pairs from dataset with multiprocessing support
    subjects, subject_pairs = ingest.ingest_data(path, use_multiprocessing=True)

<<<<<<< HEAD
    labels = ingest.obtain_subject_labels(subject_pairs)  # Obtain labels for the subject pairs

    n_subjects = ingest.get_len_data(path)
=======
    # Obtain labels for the subject pairs with multiprocessing support  
    labels = ingest.obtain_subject_labels(subject_pairs, use_multiprocessing=True)
>>>>>>> main
    
    # Initialize results list to store output
    results = []
    
    # Set CONSTANTS for the TransformerModel & GradientModel
<<<<<<< HEAD
    EPOCHS_CONSTANT = trainParams.get("epochs", 300)  # Number of epochs for training the TransformerModel
    LEARNING_RATE_CONSTANT = trainParams.get("lr", 1e-4)  # Learning rate for training the TransformerModel
=======
    EPOCHS_CONSTANT = trainParams.get("epochs", 10)  # Number of epochs for training the TransformerModel
    LEARNING_RATE_CONSTANT = trainParams.get("lr", 1e-6)  # Learning rate for training the TransformerModel
>>>>>>> main
    RUN_ON = "cuda"  # Device to run the model on, change to "cuda" if you have a GPU available

    # Ask the user if they want to use the TransformerModel
    if modelRequested is not None:
        model_requested = modelRequested 
    else:
        model_requested = int(input("Do you want to use 1) Gradient Model, 2) Transformer Model, " \
        "3) Fellegi-Sunter model, or 4) All? (Enter 1, 2, 3, or 4): "))


    # Run the requested model
    if model_requested == 1:
        model = GradientModel()  # Initialize GradientModel for gradient-boosted scoring

        # Prompt for training data or loading a model
        if doLoadModel.get("gradient") is not None:
            load_model = ('y' if doLoadModel.get("gradient") == True else 'n')
        else:
            load_model = input("Do you want to load a pre-trained model " \
                        "for the GradientModel? (Y/N): ").strip().lower()

        if load_model == 'y':
            if loadPath.get("gradient") is not None:
                path = loadPath.get("gradient")
            else:
                path = input("Enter the path to the pre-trained GradientModel: ").strip()

            model.load(path)
            print("Loading pre-trained GradientModel...")

        if doTrainModel.get("gradient") is not None:
            train_model = ('y' if doTrainModel.get("gradient") == True else 'n') 
        else:
            train_model = input("Do you want to train the GradientModel? (Y/N): ").strip().lower()

        if train_model == 'y':
            # Train the model with the subject pairs and labels
            model.train_gbt(subject_pairs, labels, EPOCHS_CONSTANT, LEARNING_RATE_CONSTANT, RUN_ON)

            # Save the trained model (WHEN READY)
            if doSaveModel.get("gradient") is not None:
                save_model = ('y' if doSaveModel.get("gradient") == True else 'n') 
            else:
                save_model = input("Do you want to save the trained GradientModel? (Y/N): ").strip().lower()

            if save_model == 'y':
                if savePath.get("gradient") is not None:
                    path = savePath.get("gradient")
                else:
                    path = input("Enter the path to save the GradientModel: ").strip()

                model.save(path)
                print(f"GradientModel saved to {path}")

        # Print loading message
        print("Running Gradient-Boosted Model...")

        # Calculate gradient-boosted scores for all subject pairs
        gb_scores = model.batch_gradient_boosted_score(subject_pairs)

        # Run through each pair of subjects in the dataset
        for i, (subject1, subject2) in enumerate(subject_pairs):
            # Calculate the gradient-boosted score
            gb_score = math.floor((gb_scores[i])*10000) / 10000.0  # Round to 4 decimal places
            
            is_match = labels[i] == 1
            base_id = subject2.attributes.get('base_id') if is_match else ""

            # Append results to the list
            results.append({
                "Row": i + 1,
                "Subject1_UUID": subject1.attributes.get('uuid'),
                "Subject2_UUID": subject2.attributes.get('uuid'),
                "Gradient_Boosted_Score": gb_score,
                "Transformer_Similarity_Score": "",
                "Felligi_Sunter_Similarity_Score": "",
                "Match": "Yes" if is_match else "No",
                "Base_ID": base_id
            })

        to_csv(results)  # Save results to CSV

    elif model_requested == 2:
        model = TransformerModel()
        model.to("cuda")
        print("Model on:", next(model.parameters()).device)

        if doLoadModel.get("transformer") is not None:
            load_model = ('y' if doLoadModel.get("transformer") == True else 'n')
        else:
            load_model = input("Do you want to load a pre-trained model " \
                        "for the TransformerModel? (Y/N): ").strip().lower()

        if load_model == 'y':
            if loadPath.get("transformer") is not None:
                path = loadPath.get("transformer")
            else:
                path = input("Enter the path to the pre-trained TransformerModel: ").strip()

            model.load(path)
            print("Loading pre-trained TransformerModel...")

        if doTrainModel.get("transformer") is not None:
            train_model = ('y' if doTrainModel.get("transformer") == True else 'n') 
        else:
            train_model = input("Do you want to train the TransformerModel? (Y/N): ").strip().lower()

        if train_model == 'y':
<<<<<<< HEAD
            model.train_transformer(subject_pairs, labels, epochs=EPOCHS_CONSTANT, 
                                    lr=LEARNING_RATE_CONSTANT, device=RUN_ON, max_samples=(n_subjects * (n_subjects - 1) // 2))
=======
            # Add debug mode option for transformer training
            debug_mode = input("Enable debug mode to see timing info? (Y/N): ").strip().lower() == 'y'
            use_preencoded = input("Use pre-encoded dataset for maximum efficiency? (Y/N): ").strip().lower() == 'y'
            use_old_style = input("Use old-style training approach (fallback for CUDA issues)? (Y/N): ").strip().lower() == 'y'
            
            if use_old_style:
                model.train_transformer_old_style(subject_pairs, labels, epochs=EPOCHS_CONSTANT, 
                                                lr=LEARNING_RATE_CONSTANT, device=RUN_ON)
            else:
                model.train_transformer(subject_pairs, labels, epochs=EPOCHS_CONSTANT, 
                                        lr=LEARNING_RATE_CONSTANT, device=RUN_ON, debug_mode=debug_mode, use_preencoded=use_preencoded)
>>>>>>> main

            if doSaveModel.get("transformer") is not None:
                save_model = ('y' if doSaveModel.get("transformer") == True else 'n')
            else:
                save_model = ("y").strip().lower()

            if save_model == 'y':
                if savePath.get("transformer") is not None:
                    path = savePath.get("transformer")
                else:
                    path = "ercl_transformer_weights.npz"

                model.save(path)
                print(f"TransformerModel saved to {path}")

    elif model_requested == 3:
        # Print loading message
        print("Running Fellegi-Sunter model...")

        # Run through each pair of subjects in the dataset
        for i, (subject1, subject2) in enumerate(subject_pairs):
            # Calculate the probability of a match
            probability = math.floor(
                fs_prob(subject1, subject2, m_probs, u_probs)*10000
                ) / 10000.0  # Round to 4 decimal places
            
            is_match = labels[i] == 1
            base_id = subject2.attributes.get('base_id') if is_match else ""

            # Append results to the list
            results.append({
                "Row": i + 1,
                "Subject1_UUID": subject1.attributes.get('uuid'),
                "Subject2_UUID": subject2.attributes.get('uuid'),
                "Gradient_Boosted_Score": "",
                "Transformer_Similarity_Score": "",
                "Felligi_Sunter_Similarity_Score": probability,
                "Match": "Yes" if is_match else "No",
                "Base_ID": base_id
            })

        to_csv(results)  # Save results to CSV

    elif model_requested == 4:
        # Initialize the TransformerModel
        transformer_model = TransformerModel()
        gradient_model = GradientModel()  # Initialize GradientModel for gradient-boosted scoring

        # Prompt for training data or loading a model (TransformerModel & GraidientModel)
        if doLoadModel.get("gradient") is not None:
            load_model_gradient = ('y' if doLoadModel.get("gradient") == True else 'n') 
        else:
            load_model_gradient = input("Do you want to load a pre-trained model " \
            "for the GradientModel? (Y/N): ").strip().lower()

        if doLoadModel.get("transformer") is not None:
            load_model_transformer = ('y' if doLoadModel.get("transformer") == True else 'n') 
        else:
            load_model_transformer = input("Do you want to load a pre-trained " \
                        "model for the TransformerModel? (Y/N): ").strip().lower()
            
        if load_model_gradient == 'y':
            if loadPath.get("gradient") is not None:
                path = loadPath.get("gradient") 
            else:
                path = input("Enter the path to the pre-trained GradientModel: ").strip()

            gradient_model.load(path)
            print("Loading pre-trained GradientModel...")
        if load_model_transformer == 'y':
            if loadPath.get("transformer") is not None:
                path = loadPath.get("transformer")
            else:
                path = input("Enter the path to the pre-trained TransformerModel: ").strip()

            transformer_model.load(path)
            print("Loading pre-trained TransformerModel...")

        if doTrainModel.get("gradient") is not None:
            train_model_gradient = ('y' if doTrainModel.get("gradient") == True else 'n')
        else:
            train_model_gradient = input("Do you want to train the GradientModel? (Y/N): ").strip().lower()

        if doTrainModel.get("transformer") is not None:
            train_model_transformer = ('y' if doTrainModel.get("transformer") == True else 'n') 
        else:
            train_model_transformer = input("Do you want to train the TransformerModel? (Y/N): ").strip().lower()
        
        if train_model_gradient == 'y':
            # Subject pair arew tuples of (subject1, subject2) and already created

            #Train the model with the subject pairs and labels
            gradient_model.train_gbt(subject_pairs, labels, EPOCHS_CONSTANT, 
                                     LEARNING_RATE_CONSTANT, RUN_ON)

            # Save the trained model (WHEN READY)
            if doSaveModel.get("gradient") is not None:
                save_model = ('y' if doSaveModel.get("gradient") == True else 'n')
            else:
                save_model = input("Do you want to save the trained GradientModel? (Y/N): ").strip().lower()

            if save_model == 'y':
                if savePath.get("gradient") is not None:
                    path = savePath.get("gradient") 
                else:
                    path = input("Enter the path to save the GradientModel: ").strip()

                gradient_model.save(path)
                print(f"GradientModel saved to {path}")

        if train_model_transformer == 'y':
            # Subject pair arew tuples of (subject1, subject2) and already created
            
            # Add debug mode option for transformer training
            debug_mode = input("Enable debug mode to see timing info? (Y/N): ").strip().lower() == 'y'
            use_preencoded = input("Use pre-encoded dataset for maximum efficiency? (Y/N): ").strip().lower() == 'y'
            use_old_style = input("Use old-style training approach (fallback for CUDA issues)? (Y/N): ").strip().lower() == 'y'
            
            if use_old_style:
                transformer_model.train_transformer_old_style(subject_pairs, labels, epochs=EPOCHS_CONSTANT, 
                                                            lr=LEARNING_RATE_CONSTANT, device=RUN_ON)
            else:
                #Train the model with the subject pairs and labels
                transformer_model.train_transformer(subject_pairs, labels, epochs=EPOCHS_CONSTANT, 
                                                    lr=LEARNING_RATE_CONSTANT, device=RUN_ON, debug_mode=debug_mode, use_preencoded=use_preencoded)

            # Save the trained model (WHEN READY)
            if doSaveModel.get("transformer") is not None:
                save_model = ('y' if doSaveModel.get("transformer") == True else 'n')
            else:
                save_model = input("Do you want to save the trained TransformerModel? (Y/N): ").strip().lower()

            if save_model == 'y':
                if savePath.get("transformer") is not None:
                    path = savePath.get("transformer") 
                else:
                    path = input("Enter the path to save the TransformerModel: ").strip()

                transformer_model.save(path)
                print(f"TransformerModel saved to {path}")

        # Print loading message
        print("Running all models...")

        # Calculate gradient-boosted scores for all subject pairs
        gb_scores = gradient_model.batch_gradient_boosted_score(subject_pairs)

        # Run through each pair of subjects in the dataset
        for i, (subject1, subject2) in enumerate(subject_pairs):
            # Calculate the gradient-boosted score
            gb_score = math.floor(gb_scores[i] * 10000) / 10000.0  # Round to 4 decimal places

            # Calculate the transformer similarity score
            transformer_score = math.floor(
                transformer_model.transformer_similarity(subject1, subject2)*10000
                ) / 10000.0  # Round to 4 decimal places

            # Calculate Fellegi-Sunter probability
            probability = math.floor(
                fs_prob(subject1, subject2, m_probs, u_probs)*10000
                ) / 10000.0  # Round to 4 decimal places

            is_match = labels[i] == 1
            base_id = subject2.attributes.get('base_id') if is_match else ""

            # Append results to the list
            results.append({
                "Row": i + 1,
                "Subject1_UUID": subject1.attributes.get('uuid'),
                "Subject2_UUID": subject2.attributes.get('uuid'),
                "Gradient_Boosted_Score": gb_score,
                "Transformer_Similarity_Score": transformer_score,
                "Felligi_Sunter_Similarity_Score": probability,
                "Match": "Yes" if is_match else "No",
                "Base_ID": base_id
            })

        to_csv(results)  # Save results to CSV


def to_csv(results, output_path="results.csv"):
    '''
    Save the results to a CSV file.
    '''
    # Write results to CSV
    csv_path = output_path
    with open(csv_path, "w", newline='') as csvfile:
        fieldnames = [
            "Row", "Subject1_UUID", "Subject2_UUID",
            "Gradient_Boosted_Score", "Transformer_Similarity_Score",
            "Felligi_Sunter_Similarity_Score", "Match", "Base_ID"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results written to {csv_path}")


def module_run(modelRequested, jsonPath, doLoadModel={"gradient": False, "transformer": False}, 
               loadPath={"gradient": None, "transformer": None}, doTrainModel={"gradient": False, "transformer": False}, 
               doSaveModel={"gradient": False, "transformer": False}, savePath={"gradient": None, "transformer": None},
               trainParams={}):
    '''
    This function is called when the module is imported.
    It runs the main function to execute the linkage and deduplication process.

    modelRequested: int - The model to run (1 for Gradient Model, 2 for Transformer Model, 3 for Fellegi-Sunter model, 4 for All).
    jsonPath: str - The path to the dataset JSON file.
    doLoadModel: dict - Dictionary indicating whether to load pre-trained models for Gradient and Transformer models.
    loadPath: dict - Dictionary containing paths to load pre-trained models for Gradient and Transformer models.
    doTrainModel: dict - Dictionary indicating whether to train the Gradient and Transformer models.
    doSaveModel: dict - Dictionary indicating whether to save the trained Gradient and Transformer models.
    savePath: dict - Dictionary containing paths to save the trained Gradient and Transformer models.
    '''
    main(modelRequested=modelRequested, jsonPath=jsonPath, doLoadModel=doLoadModel, 
         loadPath=loadPath, doTrainModel=doTrainModel, doSaveModel=doSaveModel, 
         savePath=savePath, trainParams=trainParams)


if __name__ == "__main__":
    main()
