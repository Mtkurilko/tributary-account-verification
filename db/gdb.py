import os
import json

from typing import Dict, List, Set, Any, Optional, Tuple, Union
from collections import defaultdict, deque


class Node:
    """
    represents a node in the graph with arbitrary metadata

    attrib:
        node_id: unique identifier for the node
        metadata: dictionary of key-val pairs
    """

    def __init__(self, node_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Node(id='{self.node_id}', metadata={self.metadata})"

    def to_dict(self) -> Dict[str, Any]:
        """convert node to dictionary for serialization"""
        return {"node_id": self.node_id, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """create node from dictionary"""
        return cls(data["node_id"], data["metadata"])


class Edge:
    """
    represents an edge in the graph with arbitrary metadata

    attrib:
        source: source node id
        target: target node id
        directed: is the edge directed?
        metadata: dict of key-val pairs
    """

    def __init__(
        self,
        source: str,
        target: str,
        directed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.target = target
        self.directed = directed
        self.metadata = metadata or {}

    def __repr__(self):
        arrow = "->" if self.directed else "<->"
        return f"Edge({self.source} {arrow} {self.target}, metadata={self.metadata})"

    def to_dict(self) -> Dict[str, Any]:
        """convert edge to dictionary for serialization"""
        return {
            "source": self.source,
            "target": self.target,
            "directed": self.directed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """create edge from dictionary"""
        return cls(data["source"], data["target"], data["directed"], data["metadata"])


class GraphDatabase:
    """
    graph db for benchmarking sybil detection algorithms

    - add/remove nodes and edges (directed and undirected)
    - arbitrary metadata on nodes and edges
    - simple queries (neighbours, paths)
    - persistence to json files
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        init db

        args:
            db_path: path to persist the database (optional)
        """
        self.db_path = db_path
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}

        # adjacency lists for neighbour searches
        self.outgoing: Dict[str, Set[str]] = defaultdict(set)
        self.incoming: Dict[str, Set[str]] = defaultdict(set)
        self.undirected: Dict[str, Set[str]] = defaultdict(set)

        # load from file
        if self.db_path and os.path.exists(self.db_path):
            self.load()

    def add_node(self, node_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        add node

        args:
            node_id: unique identifier for the node
            metadata: metadata dictionary

        returns:
            true if node was added, false if it already exists
        """
        if node_id in self.nodes:
            return False

        self.nodes[node_id] = Node(node_id, metadata)
        return True

    def remove_node(self, node_id: str) -> bool:
        """
        remove a node and all its edges from the graph.

        args:
            node_id: id of the node to remove

        returns:
            true if the node was removed, false if it didn't exist
        """
        if node_id not in self.nodes:
            return False

        # remove all edges connected to this node
        edges_to_remove = []
        for (source, target), edge in self.edges.items():
            if source == node_id or target == node_id:
                edges_to_remove.append((source, target))

        for source, target in edges_to_remove:
            self.remove_edge(source, target)

        # remove the node
        del self.nodes[node_id]

        # clean up adjacency lists
        if node_id in self.outgoing:
            del self.outgoing[node_id]
        if node_id in self.incoming:
            del self.incoming[node_id]
        if node_id in self.undirected:
            del self.undirected[node_id]

        return True

    def add_edge(
        self,
        source: str,
        target: str,
        directed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        add an edge to the graph

        args:
            source: source node id
            target: target node id
            directed: whether the edge is directed
            metadata: metadata dictionary

        Returns:
            true if edge was added, false if it already exists
        """
        # ensure both nodes exist
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(
                f"both nodes must exist before adding edge: {source} -> {target}"
            )

        edge_key = (source, target)
        if edge_key in self.edges:
            return False

        # create and store the edge
        edge = Edge(source, target, directed, metadata)
        self.edges[edge_key] = edge

        # update adjacency lists
        if directed:
            self.outgoing[source].add(target)
            self.incoming[target].add(source)
        else:
            self.undirected[source].add(target)
            self.undirected[target].add(source)

        return True

    def remove_edge(self, source: str, target: str) -> bool:
        """
        remove an edge from the graph

        args:
            source: source node id
            target: target node id

        returns:
            true if edge was removed, false if it didn't exist
        """
        edge_key = (source, target)
        if edge_key not in self.edges:
            return False

        edge = self.edges[edge_key]

        # update adjacency lists
        if edge.directed:
            self.outgoing[source].discard(target)
            self.incoming[target].discard(source)
        else:
            self.undirected[source].discard(target)
            self.undirected[target].discard(source)

        # remove the edge
        del self.edges[edge_key]

        return True

    def get_neighbors(self, node_id: str, direction: str = "all") -> Set[str]:
        """
        get all neighbours of a node

        args:
            node_id: id of the node
            direction: 'all', 'outgoing', 'incoming', or 'undirected'

        returns:
            set of neighbour ids
        """
        if node_id not in self.nodes:
            return set()

        neighbors = set()

        if direction in ("all", "outgoing"):
            neighbors.update(self.outgoing[node_id])

        if direction in ("all", "incoming"):
            neighbors.update(self.incoming[node_id])

        if direction in ("all", "undirected"):
            neighbors.update(self.undirected[node_id])

        return neighbors

    def find_paths(
        self, source: str, target: str, max_steps: int = 5
    ) -> List[List[str]]:
        """
        find all paths between two nodes up to n steps.

        args:
            source: source node id
            target: target node id
            max_steps: maximum number of steps in the path

        returns:
            list of paths, where each path is a list of node ids
        """
        if source not in self.nodes or target not in self.nodes:
            return []

        if source == target:
            return [[source]]

        paths = []
        queue = deque([(source, [source])])

        while queue:
            current_node, path = queue.popleft()

            if len(path) > max_steps:
                continue

            neighbors = self.get_neighbors(current_node)

            for neighbor in neighbors:
                if neighbor in path:  # Avoid cycles
                    continue

                new_path = path + [neighbor]

                if neighbor == target:
                    paths.append(new_path)
                elif len(new_path) < max_steps:
                    queue.append((neighbor, new_path))

        return paths

    def get_node(self, node_id: str) -> Optional[Node]:
        """get a node by its id"""
        return self.nodes.get(node_id)

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """get an edge by source and target node ids"""
        return self.edges.get((source, target))

    def node_count(self) -> int:
        """get the number of nodes in the graph"""
        return len(self.nodes)

    def edge_count(self) -> int:
        """get the number of edges in the graph"""
        return len(self.edges)

    def save(self, path: Optional[str] = None) -> None:
        """
        args:
            path: path to save to
        """
        save_path = path or self.db_path
        if not save_path:
            raise ValueError("no save path specified")

        data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """
        args:
            path: path to load from
        """
        load_path = path or self.db_path
        if not load_path or not os.path.exists(load_path):
            raise ValueError(f"load path does not exist: {load_path}")

        with open(load_path, "r") as f:
            data = json.load(f)

        # clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.outgoing.clear()
        self.incoming.clear()
        self.undirected.clear()

        # load nodes
        for node_data in data["nodes"]:
            node = Node.from_dict(node_data)
            self.nodes[node.node_id] = node

        # load edges
        for edge_data in data["edges"]:
            edge = Edge.from_dict(edge_data)
            edge_key = (edge.source, edge.target)
            self.edges[edge_key] = edge

            # rebuild adjacency lists
            if edge.directed:
                self.outgoing[edge.source].add(edge.target)
                self.incoming[edge.target].add(edge.source)
            else:
                self.undirected[edge.source].add(edge.target)
                self.undirected[edge.target].add(edge.source)

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.outgoing.clear()
        self.incoming.clear()
        self.undirected.clear()

    def __repr__(self):
        return f"GraphDatabase(nodes={len(self.nodes)}, edges={len(self.edges)})"
