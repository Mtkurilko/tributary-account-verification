# SybilGuard vs SybilLimit Implementation Guide

## Overview

This document explains how the SybilLimit implementation replicates the structure and approach used in SybilGuard while implementing the key differences of the SybilLimit algorithm.

## Key Similarities (Structure Replication)

### 1. Class Structure
Both implementations follow the same basic class structure:
- `__init__()` method for initialization
- `select_seed_nodes()` for choosing trusted starting points
- `perform_random_walk()` for walking through the graph
- `collect_walks()` for gathering walk data
- `run_detection()` as the main orchestration method
- Command-line interface with similar arguments

### 2. Random Walk Foundation
Both algorithms use random walks as their core mechanism:
- Multiple walks from seed nodes
- Configurable walk length and number of walks
- Analysis of walk patterns to detect anomalies

### 3. Database Integration
Both use the same graph database interface:
- `GraphDatabase` from `db.gdb`
- Same methods for accessing nodes and edges
- Compatible with the existing tributary system

## Key Differences (Algorithm-Specific)

### 1. Detection Philosophy

**SybilGuard:**
- Focuses on **path diversity** and **attack edge detection**
- Uses a **suspicion score** (0-1, higher = more suspicious)
- Detects nodes that are isolated or poorly connected to honest regions

**SybilLimit:**
- Focuses on **balance verification** and **mixing properties**
- Uses an **acceptance probability** (0-1, higher = more trusted)
- Detects nodes based on statistical properties of walk distributions

### 2. Core Algorithm Components

**SybilGuard:**
```python
# Main components
def analyze_path_diversity()     # How many seeds can reach each node
def detect_attack_edges()        # Edges connecting honest/sybil regions  
def calculate_sybil_scores()     # Combine multiple suspicion factors
def _calculate_clustering_coefficient()  # Local graph structure
```

**SybilLimit:**
```python
# Main components  
def analyze_mixing_properties()       # How walks converge to stationary distribution
def perform_balance_verification()    # Statistical balance of walk distributions
def calculate_acceptance_probabilities()  # Final trust assessment
def _calculate_betweenness_centrality()   # Improved seed selection
```

### 3. Scoring Mechanisms

**SybilGuard Suspicion Score:**
- Path diversity factor (40%): Lower diversity = higher suspicion
- Degree centrality factor (30%): Unusual degree = suspicion  
- Clustering coefficient factor (30%): Lower clustering = suspicion
- **Higher score = more suspicious**

**SybilLimit Acceptance Probability:**
- Balance verification score (60%): Uniform visit distribution = trusted
- Visit frequency factor (30%): More visits = more trusted
- Mixing ratio factor (10%): Good mixing = trusted  
- **Higher score = more trusted**

### 4. Mathematical Foundations

**SybilGuard:**
- Based on **graph cut properties**
- Assumes small cut between honest and Sybil regions
- Uses **local graph metrics** (clustering, degree)

**SybilLimit:**
- Based on **random walk mixing time**
- Uses **statistical analysis** of visit distributions
- Focuses on **balance verification protocol**

### 5. Output Interpretation

**SybilGuard:**
```python
suspicious_nodes = [node for node, score in scores.items() if score >= threshold]
# threshold = 0.7 (default)
```

**SybilLimit:**
```python
sybil_nodes = [node for node, prob in probabilities.items() if prob < (1.0 - threshold)]  
# threshold = 0.8 (default, inverted logic)
```

## Implementation Replication Strategy

### 1. Structural Replication
- **Same file organization**: `sybil/sybillimit/sybillimit.py` mirrors `sybil/sybilguard/sybilguard.py`
- **Same method signatures**: Compatible interfaces for easy comparison
- **Same command-line arguments**: Can run both with same parameters
- **Same output format**: Results dictionary with same keys

### 2. Algorithmic Adaptation
- **Enhanced seed selection**: Added betweenness centrality method
- **Improved random walks**: Longer walks for better mixing analysis
- **Statistical analysis**: Added mixing property analysis
- **Balance verification**: Core SybilLimit innovation

### 3. Parameter Mapping
```python
# SybilGuard parameters
walk_length = 20
num_walks = 100  
suspicious_threshold = 0.7

# SybilLimit parameters (adapted)
walk_length = 25          # Longer for better mixing
num_walks = 100           # Same number
mixing_time = 10          # New: expected mixing time
acceptance_threshold = 0.8 # Higher threshold (inverted logic)
```

## Usage Examples

### Running SybilGuard
```bash
python sybil/sybilguard/sybilguard.py dataset.json --seeds 10 --method degree --threshold 0.7
```

### Running SybilLimit  
```bash
python sybil/sybillimit/sybillimit.py dataset.json --seeds 10 --method degree --threshold 0.8
```

### Comparing Results
Both implementations provide:
- List of detected Sybil nodes
- Scoring for all nodes (suspicion vs acceptance)
- Analysis of graph properties
- Top-K most suspicious/trusted nodes

## Integration with Existing System

The SybilLimit implementation integrates seamlessly with the existing tributary system:

1. **Database compatibility**: Uses same `GraphDatabase` interface
2. **Dataset format**: Works with same JSON dataset format  
3. **Workflow integration**: Can be called from same orchestration code
4. **Result format**: Compatible output for visualization and analysis

## Key Takeaways

By replicating the SybilGuard structure, the SybilLimit implementation:

1. **Maintains consistency** with the existing codebase
2. **Enables easy comparison** between algorithms
3. **Leverages existing infrastructure** (database, datasets, visualization)
4. **Provides alternative detection approach** with different theoretical foundations
5. **Supports algorithmic experimentation** and hybrid approaches

The core insight is that while the algorithms differ in their mathematical foundations and detection strategies, they can share the same software architecture and interfaces, making them interchangeable components in a larger Sybil detection system.
