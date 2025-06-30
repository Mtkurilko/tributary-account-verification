"""
sybilguard.py
"""

import os
import sys
import math
import random
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from db.gdb import GraphDatabase


class SybilGuard:
    """
    sybilguard implementation
    """

    def __init__(self, graph_db: GraphDatabase, seed_nodes: Optional[List[str]] = None):
        """
        init sybilguard with db and optional seed nodes
        """
        self.graph_db = graph_db
        self.seed_nodes = seed_nodes or []
        self.walk_length = 20  # length of random walks
        self.num_walks = 100  # number of walks per seed
        self.suspicious_threshold = 0.7

        # results
        self.node_scores = {}  # node_id -> sybil score (0-1, higher = suspicious)
        self.walk_paths = {}  # seed_id -> list of walk paths
        self.attack_edges = []  # detected attack edges

    def select_seed_nodes(
        self, num_seeds: int = 10, method: str = "degree"
    ) -> List[str]:
        """select trusted seed nodes heuristically"""
        if method == "manual" and self.seed_nodes:
            return self.seed_nodes[:num_seeds]

        all_nodes = list(self.graph_db.nodes.keys())

        if method == "degree":
            # select based off of having the highest degree (most connections)
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

        return self.seed_nodes

    def perform_random_walk(self, start_node: str, walk_length: int) -> List[str]:
        """perform a random walk from a starting node"""
        if start_node not in self.graph_db.nodes:
            print(f"invalid starting node: {start_node}")
            return []

        walk = [start_node]
        current_node = start_node

        for _ in range(walk_length - 1):
            neighbors = self.graph_db.get_neighbors(current_node)

            if not neighbors:
                break

            # randomly select next node
            next_node = random.choice(list(neighbors))
            walk.append(next_node)
            current_node = next_node

        return walk

    def collect_walks(self) -> Dict[str, List[List[str]]]:
        """collect walks from all seed nodes"""
        self.walk_paths = {}

        for seed in self.seed_nodes:
            walks = []
            for _ in range(self.num_walks):
                walk = self.perform_random_walk(seed, self.walk_length)
                if walk:
                    walks.append(walk)
            self.walk_paths[seed] = walks

        return self.walk_paths

    def analyze_path_diversity(self) -> Dict[str, float]:
        """analyze path diversity for each node"""
        node_diversity = {}
        all_nodes = set(self.graph_db.nodes.keys())

        for node_id in all_nodes:
            reachable_from_seeds = set()

            for seed, walks in self.walk_paths.items():
                for walk in walks:
                    if node_id in walk:
                        reachable_from_seeds.add(seed)
                        break

            if reachable_from_seeds:
                diversity = len(reachable_from_seeds) / len(self.seed_nodes)
                node_diversity[node_id] = diversity
            else:
                node_diversity[node_id] = 0.0

        return node_diversity

    def detect_attack_edges(self) -> List[Tuple[str, str]]:
        """detect potential attack edges connecting sybil/honest regions"""
        attack_edges = []

        # analyze edges which appear in walks
        edge_frequencies = defaultdict(int)
        total_walks = sum(len(walks) for walks in self.walk_paths.values())

        for seed, walks in self.walk_paths.items():
            for walk in walks:
                for i in range(len(walk) - 1):
                    edge = (walk[i], walk[i + 1])
                    edge_frequencies[edge] += 1

        # identify edges with suspicious frequencies
        for edge, frequency in edge_frequencies.items():
            normalized_freq = frequency / total_walks

            if normalized_freq > 0.1:
                attack_edges.append(edge)

        self.attack_edges = attack_edges
        return attack_edges

    def calculate_sybil_scores(self) -> Dict[str, float]:
        """calculate sybil scores for all nodes"""
        diversity_scores = self.analyze_path_diversity()

        node_scores = {}
        all_nodes = set(self.graph_db.nodes.keys())

        for node_id in all_nodes:
            score = 0.0

            # factor 1: path diversity (lower diversity, higher suspicion)
            diversity = diversity_scores.get(node_id, 0.0)
            score += (1.0 - diversity) * 0.4

            # factor 2: degree centrality (very high/low degree = suspicion)
            neighbors = self.graph_db.get_neighbors(node_id)
            degree = len(neighbors)
            avg_degree = sum(
                len(self.graph_db.get_neighbors(n)) for n in all_nodes
            ) / len(all_nodes)

            if degree > 0:
                degree_ratio = degree / avg_degree
                if degree_ratio > 3 or degree_ratio < 0.3:
                    score += 0.3
                else:
                    score += (1.0 - min(degree_ratio / 3.0, 1.0)) * 0.3

            # factor 3: clustering coefficient
            clustering = self._calculate_clustering_coefficient(node_id)
            score += (1.0 - clustering) * 0.3

            node_scores[node_id] = min(score, 1.0)

        self.node_scores = node_scores
        return node_scores

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

    def detect_sybil_nodes(self, threshold: Optional[float] = None) -> List[str]:
        """detect sybil nodes based on scores"""
        if not self.node_scores:
            self.calculate_sybil_scores()

        threshold = threshold or self.suspicious_threshold
        sybil_nodes = [
            node_id for node_id, score in self.node_scores.items() if score >= threshold
        ]

        return sybil_nodes

    def run_detection(
        self, num_seeds: int = 10, seed_method: str = "degree"
    ) -> Dict[str, Any]:
        """run the complete sybilguard process"""
        print(f"running sybilguard with {num_seeds} seed nodes")

        # 1: select seed nodes
        seeds = self.select_seed_nodes(num_seeds, seed_method)
        print(f"selected seed nodes: {seeds}")

        # 2: collect random walks
        print("collecting random walks")
        self.collect_walks()

        # 3: calculate sybil scores
        print("calculating sybil scores")
        scores = self.calculate_sybil_scores()

        # 4: detect sybil nodes
        print("detecting sybil nodes")
        sybil_nodes = self.detect_sybil_nodes()

        # 5: detect attack edges
        print("detecting attack edges")
        attack_edges = self.detect_attack_edges()

        results = {
            "seed_nodes": seeds,
            "sybil_nodes": sybil_nodes,
            "attack_edges": attack_edges,
            "node_scores": scores,
            "total_nodes": len(self.graph_db.nodes),
            "total_edges": len(self.graph_db.edges),
            "sybil_ratio": (
                len(sybil_nodes) / len(self.graph_db.nodes)
                if self.graph_db.nodes
                else 0
            ),
        }

        print("results:\n--------------------------------\n")
        print(f"    total nodes:\t{results['total_nodes']}")
        print(f"    detected sybils:\t{len(sybil_nodes)}")
        print(f"    sybil ratio:\t{results['sybil_ratio']:.3f}")
        print(f"    attack edges:\t{len(attack_edges)}")

        return results

    def get_top_suspicious_nodes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """get the top-k most suspicious nodes"""
        if not self.node_scores:
            self.calculate_sybil_scores()

        sorted_nodes = sorted(
            self.node_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_nodes[:top_k]


def main():
    """using it"""
    import argparse

    parser = argparse.ArgumentParser(description="run sybilguard detection")
    parser.add_argument("dataset", help="path to dataset json")
    parser.add_argument("--seeds", type=int, default=10, help="number of seed nodes")
    parser.add_argument(
        "--method",
        default="degree",
        choices=["degree", "random", "manual"],
        help="seed selection method",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="sybil detection threshold"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="show top-k suspicious nodes"
    )

    args = parser.parse_args()

    print(f"loaing dataset from {args.dataset}")
    graph_db = GraphDatabase(args.dataset)

    sybilguard = SybilGuard(graph_db)
    sybilguard.suspicious_threshold = args.threshold

    results = sybilguard.run_detection(args.seeds, args.method)

    print(f"\ntop {args.top_k} most suspicious nodes:")
    top_suspicious = sybilguard.get_top_suspicious_nodes(args.top_k)
    for i, (node_id, score) in enumerate(top_suspicious, 1):
        node_data = graph_db.get_node(node_id)
        name = (
            f"{node_data.metadata.get('first_name', 'Unknown')} {node_data.metadata.get('last_name', 'Unknown')}"
            if node_data
            else "Unknown"
        )
        print(f"    {i:2d}. {node_id} ({name}): {score:.3f}")

    if results["attack_edges"]:
        print("\ndetected attack edges:")
        for source, target in results["attack_edges"][:10]:
            source_data = graph_db.get_node(source)
            target_data = graph_db.get_node(target)
            source_name = (
                f"{source_data.metadata.get('first_name', 'Unknown')} {source_data.metadata.get('last_name', 'Unknown')}"
                if source_data
                else "Unknown"
            )
            target_name = (
                f"{source_data.metadata.get('first_name', 'Unknown')} {source_data.metadata.get('last_name', 'Unknown')}"
                if target_data
                else "Unknown"
            )
            print(f"    {source} ({source_name}) -> {target} ({target_name})")


if __name__ == "__main__":
    main()
