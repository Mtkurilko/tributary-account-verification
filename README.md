# Tributary Account Verification

A research and benchmarking suite for evaluating and developing account linkage and deduplication models, with a focus on realistic synthetic data, graph-based identity structures, and interactive model evaluation.

**Authors:** Michael Kurilko, Thomas Bruce  
**Date:** 2025

---

## Overview

This repository provides tools for generating realistic synthetic identity datasets, running and evaluating multiple linkage/deduplication models, and visualizing results. It is designed for research and benchmarking in the context of the [Tributary](atributary.com) project, but is general enough for broader use in entity resolution and identity graph research.

---

## Features

- **Synthetic Dataset Generation:**  
  Easily create datasets with configurable duplicate likelihood, realistic family structures, and identity evolution sequences.
- **Graph Database:**  
  Lightweight, JSON-based graph database for storing and manipulating identity graphs.
- **Model Evaluation Dashboard:**  
  Streamlit app for running, training, and comparing multiple models (Gradient Boosted, Transformer, Fellegi-Sunter).
- **Interactive Visualization:**  
  Explore the generated identity graphs with a searchable, interactive HTML visualization.
- **Extensible Model Framework:**  
  Plug in new models or adjust existing ones for rapid experimentation.

---

## Repository Structure

```
tributary-account-verification/
│
├── dataset/
│   ├── generate.py         # Synthetic data generator (CLI & importable)
│   ├── data.py             # Data structures/utilities for dataset
│   ├── dataset.json        # Example generated dataset
│   ├── sequences.json      # Example generated sequences
│   └── README.md
│
├── db/
│   ├── gdb.py              # Lightweight graph database implementation
│   └── README.md
│
├── linkage_deduplication/
│   ├── main.py             # Main entry point for model evaluation
│   ├── ingest.py           # Data ingestion and preprocessing
│   ├── Subject.py          # Subject (node) data structure
│   └── evaluation_models/  # Model implementations (gradient, transformer, etc.)
│
├── pretrained_weights/
│   ├── gradient/
│   │   └── gradient_weights.npz
│   └── transformer/
│       ├── 200people20epoch.npz
│       └── 500people15epoch.npz
│
├── web.py                  # Streamlit dashboard for interactive evaluation
├── visualize.py            # Graph visualization utilities
├── graph.html              # Generated interactive graph visualization
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate a Synthetic Dataset

You can generate a dataset via CLI or from the Streamlit dashboard.

**From CLI:**
```bash
cd dataset
python generate.py 1000 350 -d 0.3
# Generates 1000 people, ~350 connections, 30% duplicate likelihood
```

**Generate identity sequences:**
```bash
python generate.py --sequence 100 --steps 5 -o sequences.json
# 100 identity sequences, 5 steps each
```

**From Streamlit Dashboard:**  
Use the "Generate Synthetic Dataset" toggle in the sidebar to create and use a new dataset interactively.

### 3. Run the Streamlit Dashboard

```bash
streamlit run web.py
```

- Upload your own dataset or generate a synthetic one.
- Select and run models, train or load pre-trained weights.
- View results, download outputs, and explore the interactive graph.

### 4. Visualize the Identity Graph

After generating a dataset, you can visualize it:

```bash
python visualize.py dataset/dataset.json
# Produces graph.html for interactive exploration
```

---

## Model Evaluation

- **Gradient Boosted Model**
- **Transformer Model**
- **Fellegi-Sunter Model**
- **All Models** (run and compare all at once)

You can train models, load pre-trained weights, and save new weights. Results are shown in the dashboard, including per-model accuracy and downloadable CSVs.

---

## Graph Database

The `db/gdb.py` module provides a simple, in-memory graph database with:

- Node/edge operations
- Metadata support
- Persistence to/from JSON
- Query and statistics utilities

See [`db/README.md`](db/README.md) for API details.

---

## Customization & Extensibility

- Add new models in `linkage_deduplication/evaluation_models/`
- Adjust dataset generation logic in `dataset/generate.py`
- Tweak visualization in `visualize.py`

---

## Example Usage in Python

```python
from dataset.generate import generate_from_args

# Generate a dataset
generate_from_args(num_people=100, num_edges=200, output="mydata.json", duplicate_likelihood=0.2)

# Generate sequences
generate_from_args(sequence=5, steps=10, output="myseq.json")
```

---

## Credits

- **Michael Kurilko** — dashboard, tranformer/felligi-sunter development, model integration, evaluation framework
- **Thomas Bruce** — synthetic data generation, graph database, family structure modeling

---

## Contact

For questions or contributions, please open an issue or contact the authors.