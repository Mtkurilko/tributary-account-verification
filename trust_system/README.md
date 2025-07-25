# Trust System

A modular, extensible trust quantification and interaction simulation framework for user-based systems.  
Developed by Michael Kurilko, July 2025.

---

## Overview

This trust system models and manages user trust in a social or collaborative environment.  
It provides a set of quantifiers and rules for how trust is gained, lost, and manipulated through user interactions such as vetting, accepting, and reporting.  
The system is designed for research, prototyping, and demonstration of trust-based mechanisms in digital identity, social networks, or reputation systems (designed for Tributary).

---

## Core Concepts

- **Trust Score:**  
  Each user has a trust score (0–100), which evolves based on their actions and others' actions toward them.
- **Interactions:**  
  Users can vet, deny vetting, accept, or report each other. Each action affects trust scores according to configurable rules.
- **Spam Detection:**  
  The system detects and penalizes users who spam reports or acceptances.
- **State Persistence:**  
  Trust states and interaction logs can be exported to and imported from JSON files for reproducibility and analysis.

---

## How to Run

1. **Install Requirements:**  
   This module uses only Python standard libraries and [Streamlit](https://streamlit.io/).  
   Install Streamlit if you haven't:
   ```bash
   pip install streamlit
   ```

2. **Start the Dashboard:**  
   From the `trust_system` directory, run:
   ```bash
   streamlit run run.py
   ```

3. **Interact via the Web UI:**  
   - Open the provided local URL in your browser.
   - Use the sidebar to add users and manage data import/export.
   - Use the main panel to simulate interactions and view trust scores.

---

## Usage Guide

### 1. **Adding Users**
- Enter a username in the sidebar and click "Add User".
- Add at least two users to enable interaction simulation.

### 2. **Simulating Interactions**
- Select an "Acting User" and a "Target User" (cannot be the same).
- Use the buttons to:
  - **Vet:** Acting user vouches for the target, increasing their trust.
  - **Deny Vet:** Acting user denies vetting, decreasing the target's trust.
  - **Accept:** Social acceptance, modest trust gain for the target.
  - **Report:** Acting user reports the target, decreasing their trust.

### 3. **Viewing Trust & History**
- Expand each user in the "User Trust Overview" to see their current trust score and a log of all interactions.

### 4. **Spam Detection**
- The system automatically flags users who spam reports or acceptances within a short time frame.
- Flagged users are listed in the "Spam Report Detection" section.

### 5. **Import/Export State**
- Use the sidebar buttons to export the current trust state to `trust_state.json` or import from it.
- This allows you to save, share, or restore trust system states.

---

## File Structure

```
trust_system/
├── run.py         # Streamlit dashboard for the trust system
├── system.py      # Core trust logic and data structures
├── trust_state.json # (Generated) Exported trust state
├── README.md      # This file
```

---

## Customization

- **Trust Rules:**  
  Adjust constants at the top of `system.py` to tune trust gain/loss, spam thresholds, and scaling.
- **Persistence:**  
  Change the export/import path in `system.py` if you want to store state elsewhere.
- **Integration:**  
  The `TrustSystem` class