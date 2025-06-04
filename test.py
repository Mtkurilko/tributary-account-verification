import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkage_deduplication.main import module_run

def main():
    # Example usage of the module_run function
    model_requested = 3  # Choose the model to run (1 for Fellegi-Sunter, 2 for ComparisonModel)
    json_path = "dataset/dataset.json"  # Path to your JSON data file
    do_load_model = False  # Set to True if you want to load a pre-trained model
    load_path = None  # Path to the pre-trained model if loading
    do_train_model = False  # Set to True if you want to train the model
    do_save_model = False  # Set to True if you want to save the trained model
    save_path = None # Path to save the trained model

    module_run(model_requested, json_path, do_load_model, load_path, do_train_model, do_save_model, save_path)

if __name__ == "__main__":
    main()
    