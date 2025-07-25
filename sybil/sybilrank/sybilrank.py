"""
sybilrank.py
"""

import os
import sys
import math
import random
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from db.gdb import GraphDatabase


class SybilRank:
    """
    sybilrank implementation using trust propagation and random walks
    """

    def __init__(self, graph_db: GraphDatabase, seed_nodes: Optional[List[str]] = None):
        """
        init with db with database and seed nodes
        """
        self.graph_db = graph_db
        self.seed_nodes = seed_nodes or []
        
        # sybilrank specific parameters
        self.trust_threshold = 0.1  # minimum trust score to be considered honest
        self.max_iterations = 100   # maximum iterations for trust propagation
        self.convergence_threshold = 1e-6  # convergence threshold
        self.damping_factor = 0.85  # damping factor for random walk
        self.restart_probability = 0.15  # probability of restarting at seed nodes
        
        self.trust_scores = {}  # node_id -> trust score (0-1, higher = more trusted)
        self.sybil_scores = {}  # node_id -> sybil score (0-1, higher = more suspicious)
        self.trust_propagation_history = []  # history of trust propagation

    def select_seed_nodes(
        self, num_seeds: int = 10, method: str = "degree"
    ) -> List[str]:
        """select trusted seed nodes heuristically"""
        if method == "manual" and self.seed_nodes:
            return self.seed_nodes[:num_seeds]

        all_nodes = list(self.graph_db.nodes.keys())

        if method == "degree":
            # select based on highest degree (most connections)
            node_degrees = {}
            for node_id in all_nodes:
                neighbors = self.graph_db.get_neighbors(node_id)
                node_degrees[node_id] = len(neighbors)

            # sort by degree and select top nodes
            sorted_nodes = sorted(
                node_degrees.items(), key=lambda x: x[1], reverse=True
            )
            self.seed_nodes = [node_id for node_id, _ in sorted_nodes[:num_seeds]]

        elif method == "random":
            self.seed_nodes = random.sample(all_nodes, min(num_seeds, len(all_nodes)))

        elif method == "clustering":
            # select nodes with high clustering coefficient (likely honest)
            node_clustering = {}
            for node_id in all_nodes:
                clustering = self._calculate_clustering_coefficient(node_id)
                node_clustering[node_id] = clustering

            sorted_nodes = sorted(
                node_clustering.items(), key=lambda x: x[1], reverse=True
            )
            self.seed_nodes = [node_id for node_id, _ in sorted_nodes[:num_seeds]]

        return self.seed_nodes

    def _calculate_clustering_coefficient(self, node_id: str) -> float:
        """calculate clustering coefficient for a node"""
        neighbors = self.graph_db.get_neighbors(node_id)

        if len(neighbors) < 2:
            return 0.0

        triangles = 0
        neighbor_list = list(neighbors)

        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if self.graph_db.get_edge(neighbor_list[i], neighbor_list[j]):
                    triangles += 1

        max_triangles = len(neighbors) * (len(neighbors) - 1) / 2
        return triangles / max_triangles if max_triangles > 0 else 0.0

    def initialize_trust_scores(self) -> Dict[str, float]:
        """initialize trust scores for all nodes"""
        all_nodes = set(self.graph_db.nodes.keys())
        trust_scores = {}
        
        for node_id in all_nodes:
            if node_id in self.seed_nodes:
                trust_scores[node_id] = 1.0  # seed nodes are fully trusted
            else:
                trust_scores[node_id] = 0.0  # all other nodes start with 0 trust
                
        self.trust_scores = trust_scores
        return trust_scores

    def propagate_trust(self) -> Dict[str, float]:
        """propagate trust scores through the network using random walk with restart"""
        if not self.trust_scores:
            self.initialize_trust_scores()
            
        all_nodes = list(self.graph_db.nodes.keys())
        n_nodes = len(all_nodes)
        node_to_index = {node: i for i, node in enumerate(all_nodes)}
        
        # init transition matrix
        transition_matrix = np.zeros((n_nodes, n_nodes))
        
        # build transition matrix
        for i, node_id in enumerate(all_nodes):
            neighbors = self.graph_db.get_neighbors(node_id)
            if neighbors:
                for neighbor in neighbors:
                    if neighbor in node_to_index:
                        j = node_to_index[neighbor]
                        transition_matrix[i][j] = 1.0 / len(neighbors)
        
        # init trust vector
        trust_vector = np.zeros(n_nodes)
        for node_id in self.seed_nodes:
            if node_id in node_to_index:
                trust_vector[node_to_index[node_id]] = 1.0
        
        # random walk with restart
        current_trust = trust_vector.copy()
        iteration = 0
        
        while iteration < self.max_iterations:
            # random walk step
            new_trust = self.damping_factor * np.dot(transition_matrix.T, current_trust)
            
            # restart step - add trust from seed nodes
            restart_vector = np.zeros(n_nodes)
            for node_id in self.seed_nodes:
                if node_id in node_to_index:
                    restart_vector[node_to_index[node_id]] = 1.0
            
            new_trust += self.restart_probability * restart_vector
            
            # check convergence
            if np.linalg.norm(new_trust - current_trust) < self.convergence_threshold:
                break
                
            current_trust = new_trust
            iteration += 1
            
            # store history for analysis
            self.trust_propagation_history.append(current_trust.copy())
        
        # update trust scores
        for i, node_id in enumerate(all_nodes):
            self.trust_scores[node_id] = float(current_trust[i])
            
        return self.trust_scores

    def calculate_sybil_scores(self) -> Dict[str, float]:
        """calculate sybil scores based on trust scores"""
        if not self.trust_scores:
            self.propagate_trust()
            
        sybil_scores = {}
        
        for node_id, trust_score in self.trust_scores.items():
            # sybil score is inverse of trust score
            sybil_score = 1.0 - trust_score
            sybil_scores[node_id] = sybil_score
            
        self.sybil_scores = sybil_scores
        return sybil_scores

    def detect_sybil_nodes(self, threshold: Optional[float] = None) -> List[str]:
        """detect Sybil nodes based on trust scores"""
        if not self.sybil_scores:
            self.calculate_sybil_scores()
            
        threshold = threshold or (1.0 - self.trust_threshold)
        sybil_nodes = [
            node_id for node_id, score in self.sybil_scores.items() 
            if score >= threshold
        ]
        
        return sybil_nodes

    def detect_honest_nodes(self, threshold: Optional[float] = None) -> List[str]:
        """detect honest nodes based on trust scores"""
        if not self.trust_scores:
            self.propagate_trust()
            
        threshold = threshold or self.trust_threshold
        honest_nodes = [
            node_id for node_id, score in self.trust_scores.items() 
            if score >= threshold
        ]
        
        return honest_nodes

    def analyze_trust_distribution(self) -> Dict[str, Any]:
        """analyze the distribution of trust scores"""
        if not self.trust_scores:
            self.propagate_trust()
            
        scores = list(self.trust_scores.values())
        
        analysis = {
            "mean_trust": np.mean(scores),
            "median_trust": np.median(scores),
            "std_trust": np.std(scores),
            "min_trust": np.min(scores),
            "max_trust": np.max(scores),
            "trust_threshold": self.trust_threshold,
            "nodes_above_threshold": len([s for s in scores if s >= self.trust_threshold]),
            "total_nodes": len(scores)
        }
        
        return analysis

    def get_trust_rankings(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """get the top-k most trusted nodes"""
        if not self.trust_scores:
            self.propagate_trust()
            
        sorted_nodes = sorted(
            self.trust_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        return sorted_nodes[:top_k]

    def get_sybil_rankings(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """get the top-k most suspicious nodes"""
        if not self.sybil_scores:
            self.calculate_sybil_scores()
            
        sorted_nodes = sorted(
            self.sybil_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        return sorted_nodes[:top_k]

    def run_detection(
        self, num_seeds: int = 10, seed_method: str = "degree"
    ) -> Dict[str, Any]:
        """run the complete sybilrank detection process"""
        print(f"running sybilrank with {num_seeds} seed nodes")
        
        seeds = self.select_seed_nodes(num_seeds, seed_method)
        print(f"selected seed nodes: {seeds}")
        
        print("propagating trust scores")
        trust_scores = self.propagate_trust()
        
        print("calculating Sybil scores")
        sybil_scores = self.calculate_sybil_scores()
        
        print("detecting Sybil nodes")
        sybil_nodes = self.detect_sybil_nodes()
        
        print("detecting honest nodes")
        honest_nodes = self.detect_honest_nodes()
        
        print("analyzing trust distribution")
        trust_analysis = self.analyze_trust_distribution()
        
        results = {
            "seed_nodes": seeds,
            "sybil_nodes": sybil_nodes,
            "honest_nodes": honest_nodes,
            "trust_scores": trust_scores,
            "sybil_scores": sybil_scores,
            "trust_analysis": trust_analysis,
            "total_nodes": len(self.graph_db.nodes),
            "total_edges": len(self.graph_db.edges),
            "sybil_ratio": len(sybil_nodes) / len(self.graph_db.nodes) if self.graph_db.nodes else 0,
            "honest_ratio": len(honest_nodes) / len(self.graph_db.nodes) if self.graph_db.nodes else 0,
        }
        
        print("results:\n--------------------------------\n")
        print(f"    total nodes:\t{results['total_nodes']}")
        print(f"    detected Sybils:\t{len(sybil_nodes)}")
        print(f"    detected Honest:\t{len(honest_nodes)}")
        print(f"    sybil ratio:\t{results['sybil_ratio']:.3f}")
        print(f"    honest ratio:\t{results['honest_ratio']:.3f}")
        print(f"    mean trust:\t{trust_analysis['mean_trust']:.3f}")
        print(f"    trust threshold:\t{self.trust_threshold}")
        
        return results


def main():
    """main function for running sybilrank detection"""
    import argparse

    parser = argparse.ArgumentParser(description="run sybilrank detection")
    parser.add_argument("dataset", help="path to dataset json")
    parser.add_argument("--seeds", type=int, default=10, help="number of seed nodes")
    parser.add_argument(
        "--method",
        default="degree",
        choices=["degree", "random", "manual", "clustering"],
        help="seed selection method",
    )
    parser.add_argument(
        "--trust-threshold", type=float, default=0.1, help="trust threshold for honest detection"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="show top-k trusted/suspicious nodes"
    )
    parser.add_argument(
        "--damping", type=float, default=0.85, help="damping factor for random walk"
    )

    args = parser.parse_args()

    print(f"loading dataset from {args.dataset}")
    graph_db = GraphDatabase(args.dataset)

    sybilrank = SybilRank(graph_db)
    sybilrank.trust_threshold = args.trust_threshold
    sybilrank.damping_factor = args.damping

    results = sybilrank.run_detection(args.seeds, args.method)

    print(f"\ntop {args.top_k} most trusted nodes:")
    top_trusted = sybilrank.get_trust_rankings(args.top_k)
    for i, (node_id, score) in enumerate(top_trusted, 1):
        node_data = graph_db.get_node(node_id)
        name = (
            f"{node_data.metadata.get('first_name', 'Unknown')} {node_data.metadata.get('last_name', 'Unknown')}"
            if node_data
            else "unknown"
        )
        print(f"    {i:2d}. {node_id} ({name}): {score:.3f}")

    print(f"\ntop {args.top_k} most suspicious nodes:")
    top_suspicious = sybilrank.get_sybil_rankings(args.top_k)
    for i, (node_id, score) in enumerate(top_suspicious, 1):
        node_data = graph_db.get_node(node_id)
        name = (
            f"{node_data.metadata.get('first_name', 'Unknown')} {node_data.metadata.get('last_name', 'Unknown')}"
            if node_data
            else "unknown"
        )
        print(f"    {i:2d}. {node_id} ({name}): {score:.3f}")


if __name__ == "__main__":
    main() 
