# Tributary Account Verification

A comprehensive research and benchmarking suite for evaluating, developing, and visualizing account linkage, deduplication, and trust models on realistic synthetic identity graphs.

**Authors:** Michael Kurilko, Thomas Bruce  
**Date:** 2025

---

## Overview

Tributary Account Verification is a modular platform for identity graph research, entity resolution, and trust modeling. It is designed for research and benchmarking in the context of the [Tributary](https://atributary.com) project, but is general enough for broader use in entity resolution and identity graph research.
It provides:

- **Synthetic data generation** with realistic family and social structures
- **Custom graph database** for scalable, flexible experimentation
- **Multiple linkage/deduplication models** (Gradient Boosted, Transformer, Fellegi-Sunter, Composite)
- **Interactive dashboards** for model evaluation and trust simulation
- **SybilRank, Limit, and Guard Implementations** for evaluation on a slow mixing social network like Tributary
- **Rich visualizations** for both graph structure and model results

This project is designed for both academic research and practical benchmarking, and is extensible for new models, features, and datasets.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dashboards & Visualizations](#dashboards--visualizations)
- [Model Evaluation](#model-evaluation)
- [Trust System](#trust-system)
- [Graph Database](#graph-database)
- [Customization & Extensibility](#customization--extensibility)
- [Credits](#credits)
- [Contact](#contact)

---

## Features

- **Synthetic Dataset Generation:**  
  Create datasets with configurable duplicate likelihood, realistic family trees, and identity evolution sequences.
- **Custom Graph Database:**  
  Lightweight, JSON-based, and supports arbitrary metadata for nodes and edges.
- **Model Evaluation Dashboard:**  
  Streamlit app for running, training, and comparing multiple models with real-time feedback.
- **Trust System:**  
  Simulate and visualize user trust dynamics and reputation in a social network.
- **Interactive Graph Visualization:**  
  Explore generated identity graphs and model results with a custom, searchable HTML interface.
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
│   ├── gdb.py              # Custom graph database implementation
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
│       ├── Avant-0.0.1.npz
│
├── trust_system/
│   ├── run.py              # Streamlit dashboard for trust simulation
│   ├── system.py           # Core trust logic and data structures
│   ├── trust_state.json    # (Generated) Exported trust state
│   └── README.md
│
├── sybil/
│   ├── sybilguard/         # SybilGuard implementation
│   ├── sybillimit/         # SybilLimit implementation
│   └── sybilrank/          # SybilRank implementation
│
├── web.py                  # Streamlit dashboard for model evaluation
├── visualize.py            # Graph visualization utilities
├── graph.html              # Generated interactive graph visualization
├── requirements.txt        # Python dependencies
├── train.py                # Example script for model training
├── cleanup.sh              # Utility script to clean up generated files
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

### 3. Run the Model Evaluation Dashboard

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

### 5. Run the Trust System Dashboard

```bash
cd trust_system
streamlit run run.py
```

- Add users, simulate trust interactions, and export/import trust states.

---

## Dashboards & Visualizations

### Model Evaluation Dashboard (`web.py`)

> _**[Insert Screenshot Here]**_  
> _A Streamlit dashboard for running, training, and comparing linkage/deduplication models. View model accuracy, download results, and interact with the graph._

### Trust System Dashboard (`trust_system/run.py`)

> _**[Insert Screenshot Here]**_  
> _Simulate user vetting, acceptance, and reporting. Visualize trust scores and detect spammy behavior in a social network._

### Graph Visualization (`graph.html`)

> _**[Insert Screenshot Here]**_  
> _Interactive HTML visualization of the identity graph. Search for nodes by ID, explore connections, and view node metadata._

---

## Model Evaluation

- **Gradient Boosted Model:**  
  Fast, interpretable, and robust for tabular features.
- **Transformer Model:**  
  Deep learning model for complex, sequence-based or high-dimensional features.
- **Fellegi-Sunter Model:**  
  Probabilistic record linkage using field-wise match probabilities.
- **Composite Model:**  
  Combines multiple model scores for improved accuracy.

You can train models, load pre-trained weights, and save new weights. Results are shown in the dashboard, including per-model accuracy and downloadable CSVs.

---

## Trust System

The trust system simulates user reputation and trust dynamics in a social or collaborative environment.

- **Add users** and simulate interactions (vet, deny, accept, report)
- **Trust scores** evolve based on actions and system rules
- **Spam detection** flags users who abuse reporting or acceptance
- **Export/import** trust states for reproducibility

See [`trust_system/README.md`](trust_system/README.md) for full details.

---

## Graph Database

The custom graph database (`db/gdb.py`) provides:

- **Node/edge operations:** Add, remove, and query nodes/edges with arbitrary metadata
- **Directed/undirected edges:** Flexible relationship modeling
- **Persistence:** Save/load graphs as JSON
- **Query/statistics utilities:** Neighbors, paths, counts, etc.

See [`db/README.md`](db/README.md) for API and usage examples.

---

## Customization & Extensibility

- **Add new models:**  
  Place your model in `linkage_deduplication/evaluation_models/` and register it in `main.py`.
- **Adjust dataset generation:**  
  Edit `dataset/generate.py` for new features, family structures, or synthetic data logic.
- **Tweak visualization:**  
  Modify `visualize.py` for custom graph layouts, node coloring, or metadata display.
- **Trust system rules:**  
  Tune trust gain/loss, spam thresholds, and scaling in `trust_system/system.py`.

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

- **Michael Kurilko** — dashboard, transformer/fellegi-sunter development, model integration, evaluation framework
- **Thomas Bruce** — synthetic data generation, graph database, family structure modeling

---

## Contact

For questions, contributions, or collaboration, please open an issue or contact the authors.

---

> _**[Leave space here for images of the trust dashboard, web.py dashboard, and graph results]**_