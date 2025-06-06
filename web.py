'''
Author: Michael Kurilko
Date: 6/6/2025
Description: Streamlit dashboard for linkage & deduplication model evaluation.
'''

import streamlit as st
import pandas as pd
import os
import glob
import streamlit.components.v1 as components
import time

from dataset.generate import generate_from_args
from linkage_deduplication.main import module_run


MATCH_THRESHOLD = 0.9  # Probability above this is considered a match

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Hide anchor link icon next to headers and prevent centering jump */
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a,
    .stHeading a {
        display: none !important;
        pointer-events: none !important;
    }
    /* Prevent header jump/centering when anchor is clicked */
    .stHeading {
        scroll-margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Make the sidebar run button smaller */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        font-size: 0.85em !important;
        padding: 0.25em 0.75em !important;
        min-height: 1.8em !important;
        height: 2em !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Linkage and Deduplication Evaluation Dashboard")
st.write("This dashboard lets you run and evaluate the linkage/deduplication models interactively.")

# --- Sidebar for Model Controls ---
header_col, button_col = st.sidebar.columns([3, 2])
with header_col:
    st.markdown("### Model Controls")
with button_col:
    run_models_clicked = st.button("Run Model(s)", key="sidebar_run")

model_options = {
    "Gradient Model": 1,
    "Transformer Model": 2,
    "Fellegi-Sunter Model": 3,
    "All Models": 4
}
model_choice = st.sidebar.selectbox("Select model(s) to run:", list(model_options.keys()))
model_requested = model_options[model_choice]

# --- Dataset Source Selection ---
st.sidebar.markdown("---")
use_generator = st.sidebar.toggle("Generate Synthetic Dataset", value=False)

dataset_path = None
generated = False

if use_generator:
    st.sidebar.markdown("#### Synthetic Dataset Options")
    gen_num_people = st.sidebar.number_input("Number of People", min_value=2, max_value=10000, value=100)
    gen_num_edges = st.sidebar.number_input("Number of Edges", min_value=1, max_value=100000, value=200)
    gen_duplicate_likelihood = st.sidebar.slider("Duplicate Likelihood", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    gen_output_path = st.sidebar.text_input("Output Path", value="dataset/dataset.json")
    gen_seq = st.sidebar.number_input("Sequence Count (optional)", min_value=0, max_value=1000, value=0)
    gen_steps = st.sidebar.number_input("Sequence Steps (if sequence)", min_value=1, max_value=1000, value=10)
    gen_btn = st.sidebar.button("Generate Dataset", key="generate_dataset_btn")
    if gen_btn:
        if gen_seq > 0:
            generate_from_args(sequence=int(gen_seq), steps=int(gen_steps), output=gen_output_path)
        else:
            generate_from_args(num_people=int(gen_num_people), num_edges=int(gen_num_edges),
                               output=gen_output_path, duplicate_likelihood=float(gen_duplicate_likelihood))
        st.sidebar.success(f"Generated and will use dataset at {gen_output_path}.")
        dataset_path = gen_output_path
        generated = True
    elif os.path.exists(gen_output_path):
        dataset_path = gen_output_path
        generated = True
else:
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (.json)", type=["json"])
    if uploaded_file is not None:
        dataset_path = "uploaded_dataset.json"
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif os.path.exists("dataset/dataset.json"):
        dataset_path = "dataset/dataset.json"

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
if 'run_models_clicked' not in locals():
    run_models_clicked = False

if run_models_clicked:
    if dataset_path is None:
        st.error("Please upload a dataset first.")
    else:
        status_placeholder = st.empty()
        status_placeholder.info("Running models, please wait...")

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
        status_placeholder.success("Model(s) finished running! See results below.")
        time.sleep(2)  # Show the success message for 2 seconds
        status_placeholder.empty()  # Remove the message

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

# --- Graph Visualization ---
st.header("Dataset Graph")
if os.path.exists("graph.html"):
    with open("graph.html", "r") as f:
        graph_html = f.read()
    # Wrap the HTML in a styled div for rounded corners and padding
    styled_html = f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #ddd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        height: 105%;
        margin-top: -0.15em;
        margin-left: -0.2em;
    ">
        {graph_html}
    </div>
    """
    components.html(styled_html, height=500, scrolling=False)
else:
    st.info("No graph visualization available. Please generate 'graph.html'.")
