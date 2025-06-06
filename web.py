'''
Author: Michael Kurilko
Date: 6/6/2025
Description: Streamlit dashboard for linkage & deduplication model evaluation.
'''

import streamlit as st
import pandas as pd
import os
import glob
from linkage_deduplication.main import module_run

MATCH_THRESHOLD = 0.9  # Probability above this is considered a match

st.set_page_config(layout="wide")
st.title("Linkage and Deduplication Evaluation Dashboard")
st.write("This dashboard lets you run and evaluate the linkage/deduplication models interactively.")

# --- Sidebar for Model Controls ---
st.sidebar.header("Model Controls")

model_options = {
    "Gradient Model": 1,
    "Transformer Model": 2,
    "Fellegi-Sunter Model": 3,
    "All Models": 4
}
model_choice = st.sidebar.selectbox("Select model(s) to run:", list(model_options.keys()))
model_requested = model_options[model_choice]

uploaded_file = st.sidebar.file_uploader("Upload your dataset (.json)", type=["json"])
dataset_path = None
if uploaded_file is not None:
    dataset_path = "uploaded_dataset.json"
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# --- Training and Model Options ---
st.sidebar.markdown("---")
st.sidebar.subheader("Model Training & Parameters")

# Only allow training for a specific model or "All Models"
can_train = model_requested in [1, 2, 4]

do_train = False
do_save = False
do_load = False
train_params = {}
load_path = {}
save_path = {}

if can_train:
    do_load = st.sidebar.checkbox("Load pre-trained model(s)", value=False)
    do_train = st.sidebar.checkbox("Train selected model(s)", value=False)
    do_save = st.sidebar.checkbox("Save model(s) after training", value=False)

    load_path = {"gradient": None, "transformer": None}
    if do_load:
        # --- Gradient Model ---
        if model_requested in [1, 4]:
            gradient_files = glob.glob("pretrained_weights/gradient/*.npz")
            selected_gradient = st.sidebar.selectbox(
                "Select a pre-trained Gradient Model",
                options=["Upload a different file"] + gradient_files,
                index=0
            )
            uploaded_gradient = None
            if selected_gradient == "Upload a different file":
                uploaded_gradient = st.sidebar.file_uploader(
                    "Or upload Gradient Model (.npz)", type=["npz"], key="load_gradient"
                )
            # Priority: uploaded file > selected file
            if uploaded_gradient is not None:
                gradient_model_path = "uploaded_gradient_model.npz"
                with open(gradient_model_path, "wb") as f:
                    f.write(uploaded_gradient.getbuffer())
                load_path["gradient"] = gradient_model_path
            elif selected_gradient and selected_gradient != "Upload a different file":
                load_path["gradient"] = selected_gradient

        # --- Transformer Model ---
        if model_requested in [2, 4]:
            transformer_files = glob.glob("pretrained_weights/transformer/*.npz")
            selected_transformer = st.sidebar.selectbox(
                "Select a pre-trained Transformer Model",
                options=["Upload a different file"] + transformer_files,
                index=0
            )
            uploaded_transformer = None
            if selected_transformer == "Upload a different file":
                uploaded_transformer = st.sidebar.file_uploader(
                    "Or upload Transformer Model (.npz)", type=["npz"], key="load_transformer"
                )
            # Priority: uploaded file > selected file
            if uploaded_transformer is not None:
                transformer_model_path = "uploaded_transformer_model.npz"
                with open(transformer_model_path, "wb") as f:
                    f.write(uploaded_transformer.getbuffer())
                load_path["transformer"] = transformer_model_path
            elif selected_transformer and selected_transformer != "Upload a different file":
                load_path["transformer"] = selected_transformer

    # Save path fields (let user enter filename, default to subfolder)
    save_path = {"gradient": None, "transformer": None}
    if do_save:
        if model_requested in [1, 4]:
            save_path["gradient"] = st.sidebar.text_input(
                "Filename to save Gradient Model (.npz)",
                value="pretrained_weights/gradient/gradient_model.npz"
            )
        if model_requested in [2, 4]:
            save_path["transformer"] = st.sidebar.text_input(
                "Filename to save Transformer Model (.npz)",
                value="pretrained_weights/transformer/transformer_model.npz"
            )

    # Training parameters
    if do_train:
        epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=10)
        selected_exp = st.sidebar.slider("Learning Rate Exponent (1eX)", min_value=-8, max_value=-2, value=-6, step=1)
        lr = 10 ** selected_exp
        st.sidebar.write(f"Selected learning rate: 1e{selected_exp}")
        train_params = {"epochs": epochs, "lr": lr}
else:
    do_train = False
    do_save = False
    do_load = False
    load_path = {"gradient": None, "transformer": None}
    save_path = {"gradient": None, "transformer": None}
    train_params = {}

# --- Main Area: Run Button ---
if st.button("Run Model(s)"):
    if dataset_path is None:
        st.error("Please upload a dataset first.")
    else:
        st.info("Running models, please wait...")
        # Prepare doLoadModel and doSaveModel dicts
        doLoadModel = {"gradient": False, "transformer": False}
        doSaveModel = {"gradient": False, "transformer": False}
        doTrainModel = {"gradient": False, "transformer": False}
        # Set flags for selected model(s)
        if model_requested in [1, 4]:
            doLoadModel["gradient"] = do_load
            doSaveModel["gradient"] = do_save
            doTrainModel["gradient"] = do_train
        if model_requested in [2, 4]:
            doLoadModel["transformer"] = do_load
            doSaveModel["transformer"] = do_save
            doTrainModel["transformer"] = do_train

        # Call module_run with parameters
        module_run(
            modelRequested=model_requested,
            jsonPath=dataset_path,
            doLoadModel=doLoadModel,
            loadPath=load_path,
            doTrainModel=doTrainModel,
            doSaveModel=doSaveModel,
            savePath=save_path,
            trainParams=train_params,
        )
        st.success("Model(s) finished running! See results below.")

# --- Model Accuracy Display ---
left_col, right_col = st.columns([4, 2])
with left_col:
    st.subheader("Model Accuracies")
with right_col:
    slider_col, toggle_col = st.columns([2, 2])
    with slider_col:
        MATCH_THRESHOLD = st.slider(
            "Match Threshold",
            min_value=0.8,
            max_value=1.0,
            value=0.9,
            step=0.001,
            format="%.3f",
            help="Probability above this is considered a match"
        )
    with toggle_col:
        exclude_nos = st.checkbox(
            "Exclude 'No' matches",
            value=True,  # Checked by default
            help="Only consider 'Yes' matches in accuracy"
        )

MODEL_COLUMNS = {
    "Gradient_Boosted_Score": "Gradient Model",
    "Transformer_Similarity_Score": "Transformer Model",
    "Felligi_Sunter_Similarity_Score": "Fellegi-Sunter Model"
}
MODEL_COLORS = {
    "Gradient_Boosted_Score": "#4CAF50",
    "Transformer_Similarity_Score": "#2196F3",
    "Felligi_Sunter_Similarity_Score": "#FF9800"
}

if os.path.exists("results.csv"):
    df = pd.read_csv("results.csv")
    if "Match" in df.columns:
        match_map = {"Yes": 1, "No": 0}
        match_col = df["Match"].map(match_map)
        if exclude_nos:
            mask = match_col == 1
            df = df[mask]
            match_col = match_col[mask]
        cols = st.columns(len(MODEL_COLUMNS))
        for i, (col, label) in enumerate(MODEL_COLUMNS.items()):
            if col in df.columns and df[col].notna().any():
                preds = (df[col] > MATCH_THRESHOLD).astype(int)
                correct = (preds == match_col)
                acc = correct.sum() / len(df) if len(df) > 0 else 0
                percent = f"{acc*100:.4f}%"
                cols[i].markdown(
                    f"<div style='background-color:{MODEL_COLORS[col]};padding:0.5em 0.5em 0.2em 0.5em;border-radius:8px;text-align:center;color:white;font-weight:bold;'>"
                    f"{label}<br><span style='font-size:1.5em'>{percent}</span></div>",
                    unsafe_allow_html=True
                )
            else:
                cols[i].markdown(
                    f"<div style='background-color:#eee;padding:0.5em 0.5em 0.2em 0.5em;border-radius:8px;text-align:center;color:#888;'>"
                    f"{label}<br><span style='font-size:1.5em'>N/A</span></div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("No 'Match' column found in results.")
else:
    st.info("No results yet. Run a model to see accuracy here.")


# --- Results Display ---
st.header("Results")

results_path = "results.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="results.csv")
else:
    st.info("No results yet. Run a model to see results here.")

# --- Model Download Links ---
if do_save:
    if model_requested in [1, 4] and save_path["gradient"] and os.path.exists(save_path["gradient"]):
        with open(save_path["gradient"], "rb") as f:
            st.download_button("Download Gradient Model", f, file_name=save_path["gradient"])
    if model_requested in [2, 4] and save_path["transformer"] and os.path.exists(save_path["transformer"]):
        with open(save_path["transformer"], "rb") as f:
            st.download_button("Download Transformer Model", f, file_name=save_path["transformer"])
