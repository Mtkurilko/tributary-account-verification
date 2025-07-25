"""
Author: Michael Kurilko
Date: 7/25/2025
Description: This script is the one we used to train the model.
It imports the `module_run` function from the `linkage_deduplication.main`
module and sets up the parameters for running the model.
It allows you to specify whether to run, load, train, or save the model,
and provides paths for the JSON data and pre-trained model weights.
It is designed to be run as a standalone script.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linkage_deduplication.main import module_run

def main():
    # Example usage of the module_run function
    model_requested = 2  # Choose the model to run
    json_path = "dataset/dataset.json"  # Path to your JSON data file
    do_run_model = 'n'
    do_load_model = {"gradient": False, "transformer": False}  # Set to True if you want to load a pre-trained model
    load_path = {"gradient": None, "transformer": "pretrained_weights/transformer/Avant-0.0.2.npz"}  # Path to the pre-trained model if loading
    do_train_model = {"gradient": False, "transformer": True}  # Set to True if you want to train the model
    do_save_model = {"gradient": False, "transformer": True}  # Set to True if you want to save the trained model
    save_path = {"gradient": None, "transformer": "pretrained_weights/transformer/Avant-0.0.2.npz"} # Path to save the trained model

    module_run(model_requested, json_path, do_load_model, load_path, do_train_model, doRunModel=do_run_model, doSaveModel=do_save_model, savePath=save_path)

if __name__ == "__main__":
    main()
