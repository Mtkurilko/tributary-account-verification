# Deduplication and Probabilistic Record Linkage: A Comparative Study of Transformer-Based, Gradient Boosted, and Fellegi-Sunter Models for Identity Resolution in Synthetic Data Systems

2025

By:

- Michael Kurilko <michael.kurilko1@marist.edu> Marist University
- Thomas Bruce <thomas.bruce1@marist.edu> Marist University

## Abstract

Tributary, a new initiative founded by Professor Gormanly from Marist University, aims to be the "Wikipedia of all people". A critical step in achieving this vision is ensuring each person has a unique, one-to-one record. This paper investigates probabilistic deduplication using synthetic personal data, evaluating and comparing the effectiveness of three different models: a Transformer-based model, a Gradient Boosted Tree (GBT), and the traditional Fellegi-Sunter method. A custom-built dashboard was used to visualize performance in real time across a variety of error scenarios. Our results offer a roadmap to building an optimal deduplication system for user identity resolution.

---

## 1. Introduction

- Overview of record linkage and deduplication.
- Importance of accurate deduplication for platforms like Tributary.
- Challenges: spelling errors, missing data, varying field formats.
- Objectives:
  - Compare three deduplication models.
  - Use synthetic data and variable corruption types.
  - Build a custom dashboard to monitor model performance.
  - Propose a practical solution for deployment.

---

## 2. Background and Related Work

- **Fellegi-Sunter Theory**: Probabilistic matching and thresholding.
- **Gradient Boosted Trees**: Feature-based classification strengths.
- **Transformers**: NLP strengths for unstructured or corrupted data.
- Existing deduplication tools: Dedupe.io, Splink, etc.
- The gap: Model explainability, performance under real-world corruption, and integration with visualization systems.

---

## 3. Methodology

### 3.1 Data Generation

- Description of synthetic person generator.
- Error simulations:
  - Spelling variations
  - Field reordering
  - Abbreviations and nicknames
  - Missing and swapped fields
- Dataset sizes and structure.

### 3.2 Feature Engineering

- Similarity Score

### 3.3 Model Architectures

- **Fellegi-Sunter**: Match vs non-match probability logic.
- **GBT**: Training curve, feature importances, grid search results.
- **Transformer**: Input formatting, fine-tuning strategy, architecture used.

---

## 4. Control Dashboard

- Lightweight build using Streamlit
- Graph and JSON real-time views of:
  - Detected duplicates
  - Confidence levels
  - Per-model performance
- Dashboard use in monitoring edge cases and debugging.

---

## 5. Evaluation Metrics

- Similarity Score, Accuracy
- ROC curves and AUC for each model.
- Performance on:
  - Typographical errors
  - Name formatting inconsistencies
  - Partially missing entries
- Composite Score:
  - Strategy for hybrid model scoring.
  - Weighted confidence blending.

---

## 6. Results and Analysis

- Transformers: Excellent on language variants, slower on large data.
- GBT: High performance with engineered features, interpretable.
- Fellegi-Sunter: Transparent but rigid.
- Dashboard screenshots illustrating performance trends.

---

## 7. Discussion

- Practicality and deployability of each model.
- Proposed ideal model for Tributary’s real-world implementation.
- Use cases for composite modeling.
- Dashboard utility in catching unexpected edge cases.

---

## 8. Limitations and Future Work

- Synthetic data limitations: domain mismatch vs real user data.
- Scaling issues for Transformer models.
- Need for multilingual support and fuzzy geoparsing.
- Possibility of human-in-the-loop or active learning framework.
- Future integration with mobile or web-based user reporting tools.

---

## 9. Conclusion

- Summary of core findings.
- Recommendation for best model(s) depending on priority (speed, accuracy, explainability).
- Value of a modular pipeline approach with built-in monitoring.

---

## 10. References

- Fellegi-Sunter (1969)
- Torch/Cuda documentation
- GBT paper
- Dedupe.io, Splink
- Academic articles on entity resolution and record linkage

---

## 11. Appendices (Optional)

- Synthetic record examples
- Evaluation metric tables
- Code or dashboard interface screenshots
