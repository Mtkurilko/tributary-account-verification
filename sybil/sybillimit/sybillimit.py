"""
sybillimit.py

SybilLimit implementation based on the paper by Yu et al.
This replicates the structure and approach used in SybilGuard but implements
the SybilLimit algorithm which uses a different approach for detecting Sybil nodes.
"""

import os
import sys
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from db.gdb import GraphDatabase


class SybilLimit:
    """
    SybilLimit implementation
    
    SybilLimit differs from SybilGuard in that it:
    1. Uses a more refined random walk approach
    2. Focuses on the mixing time and convergence properties
    3. Uses statistical analysis of walk distributions
    4. Implements a balance verification protocol
    """

    def __init__(self, graph_db: GraphDatabase, seed_nodes: Optional[List[str]] = None):
        """
        Initialize SybilLimit with database and optional seed nodes
        """
        self.graph_db = graph_db
        self.seed_nodes = seed_nodes or []
        self.walk_length = 25  # Slightly longer walks for better mixing
        self.num_walks = 100  # Number of walks per seed
        self.mixing_time = 10  # Expected mixing time
        self.acceptance_threshold = 0.8  # Threshold for accepting nodes
        
        # Results
        self.node_scores = {}  # node_id -> acceptance probability (0-1)
        self.walk_paths = {}  # seed_id -> list of walk paths
        self.mixing_analysis = {}  # analysis of mixing properties
        self.balance_verifications = {}  # balance verification results

    def select_seed_nodes(
        self, num_seeds: int = 10, method: str = "degree"
    ) -> List[str]:
        """Select trusted seed nodes using various heuristics"""
        if method == "manual" and self.seed_nodes:
            return self.seed_nodes[:num_seeds]

        all_nodes = list(self.graph_db.nodes.keys())

        if method == "degree":
            # Select nodes with highest degree (most connections)
            node_degrees = {}
            for node_id in all_nodes:
                neighbors = self.graph_db.get_neighbors(node_id)
                node_degrees[node_id] = len(neighbors)

            sorted_nodes = sorted(
                node_degrees.items(), key=lambda x: x[1], reverse=True
            )
            self.seed_nodes = [node_id for node_id, _ in sorted_nodes[:num_seeds]]

        elif method == "betweenness":
            # Select nodes with high betweenness centrality (simplified version)
            centrality_scores = self._calculate_betweenness_centrality()
            sorted_nodes = sorted(
                centrality_scores.items(), key=lambda x: x[1], reverse=True
            )
            self.seed_nodes = [node_id for node_id, _ in sorted_nodes[:num_seeds]]

        elif method == "random":
            self.seed_nodes = random.sample(all_nodes, min(num_seeds, len(all_nodes)))

        return self.seed_nodes

    def _calculate_betweenness_centrality(self) -> Dict[str, float]:
        """Calculate simplified betweenness centrality for seed selection"""
        centrality = defaultdict(float)
        all_nodes = list(self.graph_db.nodes.keys())
        
        # Simplified betweenness: count how often a node appears in random walks
        for start_node in all_nodes[:min(50, len(all_nodes))]:  # Sample for efficiency
            for _ in range(10):  # Multiple walks per node
                walk = self.perform_random_walk(start_node, 15)
                for node in walk[1:-1]:  # Exclude start and end
                    centrality[node] += 1.0
        
        # Normalize
        max_score = max(centrality.values()) if centrality else 1
        return {node: score / max_score for node, score in centrality.items()}

    def perform_random_walk(self, start_node: str, walk_length: int) -> List[str]:
        """Perform a random walk from a starting node"""
        if start_node not in self.graph_db.nodes:
            print(f"Invalid starting node: {start_node}")
            return []

        walk = [start_node]
        current_node = start_node

        for _ in range(walk_length - 1):
            neighbors = self.graph_db.get_neighbors(current_node)

            if not neighbors:
                break

            # SybilLimit uses weighted random selection based on edge weights
            # For now, we'll use uniform random selection
            next_node = random.choice(list(neighbors))
            walk.append(next_node)
            current_node = next_node

        return walk

    def collect_walks(self) -> Dict[str, List[List[str]]]:
        """Collect random walks from all seed nodes"""
        self.walk_paths = {}

        for seed in self.seed_nodes:
            walks = []
            for _ in range(self.num_walks):
                walk = self.perform_random_walk(seed, self.walk_length)
                if walk:
                    walks.append(walk)
            self.walk_paths[seed] = walks

        return self.walk_paths

    def analyze_mixing_properties(self) -> Dict[str, Any]:
        """
        Analyze mixing properties of the random walks
        This is a key component of SybilLimit
        """
        mixing_analysis = {}
        all_nodes = set(self.graph_db.nodes.keys())
        
        # Analyze convergence to stationary distribution
        for node_id in all_nodes:
            # Count visits at different time steps
            early_visits = 0  # Visits in first half of walks
            late_visits = 0   # Visits in second half of walks
            total_visits = 0
            
            for seed, walks in self.walk_paths.items():
                for walk in walks:
                    if node_id in walk:
                        total_visits += 1
                        mid_point = len(walk) // 2
                        
                        # Count early vs late visits
                        if node_id in walk[:mid_point]:
                            early_visits += 1
                        if node_id in walk[mid_point:]:
                            late_visits += 1
            
            # Calculate mixing ratio
            if total_visits > 0:
                mixing_ratio = late_visits / total_visits if total_visits > 0 else 0
                mixing_analysis[node_id] = {
                    'total_visits': total_visits,
                    'early_visits': early_visits,
                    'late_visits': late_visits,
                    'mixing_ratio': mixing_ratio,
                    'visit_frequency': total_visits / (len(self.seed_nodes) * self.num_walks)
                }
            else:
                mixing_analysis[node_id] = {
                    'total_visits': 0,
                    'early_visits': 0,
                    'late_visits': 0,
                    'mixing_ratio': 0,
                    'visit_frequency': 0
                }
        
        self.mixing_analysis = mixing_analysis
        return mixing_analysis

    def perform_balance_verification(self) -> Dict[str, float]:
        """
        Perform balance verification protocol
        This is the core innovation of SybilLimit
        """
        balance_results = {}
        all_nodes = set(self.graph_db.nodes.keys())
        
        for node_id in all_nodes:
            # Calculate balance score based on:
            # 1. How well the node mixes with honest nodes
            # 2. Distribution of incoming walks from different seeds
            # 3. Consistency of walk patterns
            
            seed_visit_distribution = {}
            total_visits = 0
            
            # Count visits from each seed
            for seed, walks in self.walk_paths.items():
                visits_from_seed = sum(1 for walk in walks if node_id in walk)
                seed_visit_distribution[seed] = visits_from_seed
                total_visits += visits_from_seed
            
            if total_visits == 0:
                balance_results[node_id] = 0.0
                continue
            
            # Calculate distribution uniformity (key insight from SybilLimit)
            expected_visits_per_seed = total_visits / len(self.seed_nodes)
            variance = sum(
                (visits - expected_visits_per_seed) ** 2 
                for visits in seed_visit_distribution.values()
            ) / len(self.seed_nodes)
            
            # Normalize variance to get balance score
            max_possible_variance = expected_visits_per_seed ** 2
            normalized_variance = variance / max_possible_variance if max_possible_variance > 0 else 1
            
            # Balance score: higher means more balanced (less suspicious)
            balance_score = 1.0 / (1.0 + normalized_variance)
            
            # Combine with mixing properties
            mixing_info = self.mixing_analysis.get(node_id, {})
            mixing_bonus = mixing_info.get('mixing_ratio', 0) * 0.3
            
            final_score = min(1.0, balance_score + mixing_bonus)
            balance_results[node_id] = final_score
        
        self.balance_verifications = balance_results
        return balance_results

    def calculate_acceptance_probabilities(self) -> Dict[str, float]:
        """
        Calculate acceptance probabilities for all nodes
        This is the final output of SybilLimit
        """
        acceptance_probs = {}
        
        # Ensure we have balance verification results
        if not self.balance_verifications:
            self.perform_balance_verification()
        
        # Ensure we have mixing analysis
        if not self.mixing_analysis:
            self.analyze_mixing_properties()
        
        all_nodes = set(self.graph_db.nodes.keys())
        
        for node_id in all_nodes:
            balance_score = self.balance_verifications.get(node_id, 0.0)
            mixing_info = self.mixing_analysis.get(node_id, {})
            
            # Factors for acceptance probability:
            # 1. Balance verification score (most important)
            # 2. Visit frequency (nodes that are visited more often)
            # 3. Mixing ratio (how well distributed the visits are)
            
            visit_frequency = mixing_info.get('visit_frequency', 0)
            mixing_ratio = mixing_info.get('mixing_ratio', 0)
            
            # Combine factors
            acceptance_prob = (
                balance_score * 0.6 +
                min(visit_frequency * 2, 1.0) * 0.3 +
                mixing_ratio * 0.1
            )
            
            acceptance_probs[node_id] = min(1.0, acceptance_prob)
        
        self.node_scores = acceptance_probs
        return acceptance_probs

    def detect_sybil_nodes(self, threshold: Optional[float] = None) -> List[str]:
        """Detect Sybil nodes based on acceptance probabilities"""
        if not self.node_scores:
            self.calculate_acceptance_probabilities()

        threshold = threshold or self.acceptance_threshold
        
        # In SybilLimit, nodes with LOW acceptance probability are suspicious
        sybil_nodes = [
            node_id for node_id, prob in self.node_scores.items() 
            if prob < (1.0 - threshold)
        ]

        return sybil_nodes

    def run_detection(
        self, num_seeds: int = 10, seed_method: str = "degree"
    ) -> Dict[str, Any]:
        """Run the complete SybilLimit detection process"""
        print(f"Running SybilLimit with {num_seeds} seed nodes")

        # 1. Select seed nodes
        seeds = self.select_seed_nodes(num_seeds, seed_method)
        print(f"Selected seed nodes: {seeds}")

        # 2. Collect random walks
        print("Collecting random walks")
        self.collect_walks()

        # 3. Analyze mixing properties
        print("Analyzing mixing properties")
        self.analyze_mixing_properties()

        # 4. Perform balance verification
        print("Performing balance verification")
        self.perform_balance_verification()

        # 5. Calculate acceptance probabilities
        print("Calculating acceptance probabilities")
        acceptance_probs = self.calculate_acceptance_probabilities()

        # 6. Detect Sybil nodes
        print("Detecting Sybil nodes")
        sybil_nodes = self.detect_sybil_nodes()

        results = {
            "seed_nodes": seeds,
            "sybil_nodes": sybil_nodes,
            "acceptance_probabilities": acceptance_probs,
            "total_nodes": len(self.graph_db.nodes),
            "total_edges": len(self.graph_db.edges),
            "sybil_ratio": (
                len(sybil_nodes) / len(self.graph_db.nodes)
                if self.graph_db.nodes
                else 0
            ),
            "mixing_analysis": self.mixing_analysis,
            "balance_verifications": self.balance_verifications,
        }

        print("Results:\n--------------------------------")
        print(f"    Total nodes:\t{results['total_nodes']}")
        print(f"    Detected Sybils:\t{len(sybil_nodes)}")
        print(f"    Sybil ratio:\t{results['sybil_ratio']:.3f}")
        print(f"    Average acceptance prob:\t{sum(acceptance_probs.values()) / len(acceptance_probs):.3f}")

        return results

    def get_most_suspicious_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the top-k most suspicious nodes (lowest acceptance probability)"""
        if not self.node_scores:
            self.calculate_acceptance_probabilities()

        # Sort by acceptance probability (ascending - lowest first)
        sorted_nodes = sorted(
            self.node_scores.items(), key=lambda x: x[1]
        )

        return sorted_nodes[:top_k]

    def get_most_trusted_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the top-k most trusted nodes (highest acceptance probability)"""
        if not self.node_scores:
            self.calculate_acceptance_probabilities()

        # Sort by acceptance probability (descending - highest first)
        sorted_nodes = sorted(
            self.node_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_nodes[:top_k]


def main():
    """Command line interface for SybilLimit"""
    import argparse

    parser = argparse.ArgumentParser(description="Run SybilLimit detection")
    parser.add_argument("dataset", help="Path to dataset JSON")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seed nodes")
    parser.add_argument(
        "--method",
        default="degree",
        choices=["degree", "betweenness", "random", "manual"],
        help="Seed selection method",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Acceptance threshold"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Show top-k suspicious nodes"
    )

    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}")
    graph_db = GraphDatabase(args.dataset)

    sybillimit = SybilLimit(graph_db)
    sybillimit.acceptance_threshold = args.threshold

    _ = sybillimit.run_detection(args.seeds, args.method)

    print(f"\nTop {args.top_k} most suspicious nodes:")
    top_suspicious = sybillimit.get_most_suspicious_nodes(args.top_k)
    for i, (node_id, acceptance_prob) in enumerate(top_suspicious, 1):
        node_data = graph_db.get_node(node_id)
        name = (
            f"{node_data.metadata.get('first_name', 'Unknown')} {node_data.metadata.get('last_name', 'Unknown')}"
            if node_data
            else "Unknown"
        )
        suspicion_score = 1.0 - acceptance_prob
        print(f"    {i:2d}. {node_id} ({name}): {suspicion_score:.3f} (acceptance: {acceptance_prob:.3f})")

    print(f"\nTop {args.top_k} most trusted nodes:")
    top_trusted = sybillimit.get_most_trusted_nodes(args.top_k)
    for i, (node_id, acceptance_prob) in enumerate(top_trusted, 1):
        node_data = graph_db.get_node(node_id)
        name = (
            f"{node_data.metadata.get('first_name', 'Unknown')} {node_data.metadata.get('last_name', 'Unknown')}"
            if node_data
            else "Unknown"
        )
        print(f"    {i:2d}. {node_id} ({name}): {acceptance_prob:.3f}")


if __name__ == "__main__":
    main()
