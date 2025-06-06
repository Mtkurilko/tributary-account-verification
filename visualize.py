from pyvis.network import Network
import networkx as nx
import sys
import argparse
import colorsys
from typing import Dict, List, Tuple, Optional
from db.gdb import GraphDatabase


def convert_to_networkx(db: GraphDatabase) -> nx.Graph:
    """convert GraphDatabase to networkx"""
    G = nx.Graph()

    for node_id, node in db.nodes.items():
        G.add_node(node_id, **node.metadata)

    for (source, target), edge in db.edges.items():
        edge_attrs = dict(edge.metadata)
        edge_attrs["directed"] = edge.directed
        G.add_edge(source, target, **edge_attrs)

    return G


def get_node_color(connections: int, max_connections: int) -> str:
    """generate color based on connection count"""
    if max_connections == 0:
        return "#888888"

    hue = 0.7 * (1 - connections / max_connections)  # 0.7 = blue, 0 = red
    saturation = 0.8
    value = 0.9

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def visualize_graph(
    db: GraphDatabase,
    title: str = "database visualization",
    height: str = "750px",
    width: str = "100%",
    physics: str = "barnes_hut",
) -> Network:
    """
    create a pyvis network visualization of the graph

    args:
        db: GraphDatabase instance to visualize
        title: title for the visualization
        height: height of the visualization
        width: width of the visualization
        physics: physics simulation type ('barnes_hut', 'force_atlas_2based', 'hierarchical_repulsion')

    returns:
        pyvis network object
    """

    if db.node_count() == 0:
        print("warn: empty graph database")
        return None

    G = convert_to_networkx(db)

    net = Network(
        height=height,
        width=width,
        bgcolor="#121212",
        font_color="white",
        notebook=False,
    )

    if physics == "barnes_hut":
        net.barnes_hut()
    elif physics == "force_atlas_2based":
        net.force_atlas_2based()
    elif physics == "hierarchical_repulsion":
        net.hrepulsion()
    elif physics == "disabled":
        net.toggle_physics(False)

    net.set_options(
        """
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 150,
          "updateInterval": 50
        },
        "barnesHut": {
          "gravitationalConstant": -4000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.2,
          "avoidOverlap": 0.5
        }
      },
      "edges": {
        "smooth": false,
        "width": 2
      },
      "nodes": {
        "physics": true
      }
    }
    """
    )

    connection_counts = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    max_connections = max(connection_counts.values()) if connection_counts else 0

    for node_id in G.nodes():
        node = db.get_node(node_id)
        connections = connection_counts[node_id]

        # create hover title with metadata
        if node.metadata:
            metadata_lines = [f"{k}: {v}" for k, v in node.metadata.items()]
            metadata_str = "\n".join(metadata_lines)
        else:
            metadata_str = "no metadata"

        title = (
            f"{node_id}\nconnections: {connections}\n--- metadata ---\n{metadata_str}"
        )

        color = get_node_color(connections, max_connections)

        size = 20 + (connections / max_connections * 30) if max_connections > 0 else 20

        net.add_node(
            node_id,
            label=str(node_id),
            title=title,
            color=color,
            size=size,
            font={"size": 16, "color": "white", "face": "arial"},
        )

    for (source, target), edge in db.edges.items():
        weight = edge.metadata.get("weight", 1.0)

        edge_title = f"{'directed' if edge.directed else 'undirected'} edge\nfrom: {source}\nto: {target}\nweight: {weight:.2f}"

        if edge.directed:
            net.add_edge(
                source,
                target,
                value=weight,
                title=edge_title,
                arrows="to",
                color={"color": "#aaaaaa", "highlight": "#ffffff"},
                smooth=False,
            )
        else:
            net.add_edge(
                source,
                target,
                value=weight,
                title=edge_title,
                color={"color": "#888888", "highlight": "#ffffff"},
                dashes=True,
                smooth=False,
            )

    return net


def save_visualization(
    db: GraphDatabase, filename: str = "graph.html", **kwargs
) -> None:
    net = visualize_graph(db, **kwargs)
    if net:
        net.show(filename)
        print(f"graph visualization saved to {filename}")
    else:
        print("failed to create visualization - empty graph")


def main():
    parser = argparse.ArgumentParser(description="visualize graph database")
    parser.add_argument("database_file", help="path to graph database")
    parser.add_argument(
        "--output",
        "-o",
        default="graph.html",
        help="output html filename",
    )
    parser.add_argument(
        "--title",
        "-t",
        default="graph database visualization",
        help="title for the visualization",
    )
    parser.add_argument(
        "--physics",
        "-p",
        default="barnes_hut",
        choices=[
            "barnes_hut",
            "force_atlas_2based",
            "hierarchical_repulsion",
            "disabled",
        ],
        help="physics simulation type",
    )
    parser.add_argument(
        "--height",
        default="750px",
        help="height of the visualization",
    )
    parser.add_argument(
        "--width",
        default="100%",
        help="width of the visualization",
    )

    args = parser.parse_args()

    try:
        print(f"loading database from {args.database_file}")
        db = GraphDatabase(args.database_file)

        if db.node_count() == 0:
            print("warn: loaded database is empty")
            return

        print(f"graph loaded: {db.node_count()} nodes, {db.edge_count()} edges")

        save_visualization(
            db,
            filename=args.output,
            title=args.title,
            physics=args.physics,
            height=args.height,
            width=args.width,
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
