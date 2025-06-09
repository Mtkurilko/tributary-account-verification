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
    height: str = "600px",
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

    node_count = db.node_count()

    # scale physics parameters based on graph size
    if node_count <= 50:
        gravitational_constant = -2000
        spring_length = 150
        spring_constant = 0.04
    elif node_count <= 200:
        gravitational_constant = -4000
        spring_length = 200
        spring_constant = 0.02
    else:
        gravitational_constant = -6000
        spring_length = 250
        spring_constant = 0.01

    net.set_options(
        f"""
    var options = {{
      "physics": {{
        "enabled": true,
        "stabilization": {{
          "enabled": true,
          "iterations": {min(300, max(100, node_count * 2))},
          "updateInterval": 50
        }},
        "barnesHut": {{
          "gravitationalConstant": {gravitational_constant},
          "centralGravity": 0.1,
          "springLength": {spring_length},
          "springConstant": {spring_constant},
          "damping": 0.2,
          "avoidOverlap": 0.5
        }}
      }},
      "edges": {{
        "smooth": false,
        "width": 2,
        "arrows": {{
          "to": {{
            "enabled": true,
            "scaleFactor": 0.8
          }}
        }}
      }},
      "nodes": {{
        "physics": true,
        "chosen": {{
          "node": true,
          "label": false
        }}
      }},
      "interaction": {{
        "hover": true,
        "hoverConnectedEdges": true,
        "selectConnectedEdges": false
      }}
    }}
    """
    )

    connection_counts = {node: len(list(G.neighbors(node))) for node in G.nodes()}
    max_connections = max(connection_counts.values()) if connection_counts else 0

    # scalable node sizing
    base_size = max(15, min(30, 100 / max(1, node_count / 20)))
    max_size_bonus = max(20, min(50, 200 / max(1, node_count / 10)))

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

        size = (
            base_size + (connections / max_connections * max_size_bonus)
            if max_connections > 0
            else base_size
        )

        # scale font size based on graph size
        font_size = max(12, min(18, 50 / max(1, node_count / 30)))

        net.add_node(
            node_id,
            label=str(node_id),
            title=title,
            color=color,
            size=size,
            font={"size": font_size, "color": "white", "face": "Arial"},
        )

    for (source, target), edge in db.edges.items():
        weight = edge.metadata.get("weight", 1.0)

        edge_title = f"{'directed' if edge.directed else 'undirected'} edge\nfrom: {source}\nto: {target}\nweight: {weight:.2f}"

        # scale edge thickness based on graph size
        edge_width = max(1, min(4, 10 / max(1, node_count / 50)))

        if edge.directed:
            net.add_edge(
                source,
                target,
                value=weight,
                title=edge_title,
                arrows="to",
                color={"color": "#aaaaaa", "highlight": "#ffffff"},
                width=edge_width,
                smooth=False,
            )
        else:
            net.add_edge(
                source,
                target,
                value=weight,
                title=edge_title,
                color={"color": "#888888", "highlight": "#ffffff"},
                width=edge_width,
                dashes=True,
                smooth=False,
            )

    return net


def save_visualization(
    db: GraphDatabase, filename: str = "graph.html", **kwargs
) -> None:
    net = visualize_graph(db, **kwargs)
    if net:
        # get the width parameter to set card width
        width = kwargs.get("width", "100%")

        # generate the html
        net.show(filename)

        with open(filename, "r") as f:
            html_content = f.read()

        # create search bar and centering functionality
        search_functionality = f"""
        <style>
        body, html {{
            margin: 0 !important;
            padding: 0 !important;
        }}
        .card {{
            width: {width} !important;
            max-width: {width} !important;
            margin: 0 auto;
            padding: 0 !important;
            box-sizing: border-box;
            position: relative;
        }}
        #search-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #444;
        }}
        #search-input {{
            background: #333;
            color: white;
            border: 1px solid #666;
            padding: 8px;
            border-radius: 3px;
            width: 200px;
            font-size: 14px;
        }}
        #search-button {{
            background: #007acc;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 3px;
            margin-left: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        #search-button:hover {{
            background: #005a9e;
        }}
        #search-status {{
            color: #ccc;
            font-size: 12px;
            margin-top: 5px;
        }}
        </style>
        
        <script>
        function centerOnNode(nodeId) {{
            if (network && network.body && network.body.nodes[nodeId]) {{
                const nodePosition = network.getPositions([nodeId])[nodeId];
                
                network.moveTo({{
                    position: nodePosition,
                    scale: 1.0,
                    animation: {{
                        duration: 1000,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
                
                network.selectNodes([nodeId]);
                setTimeout(() => {{
                    network.unselectAll();
                }}, 2000);
                
                return true;
            }}
            return false;
        }}
        
        function searchNode() {{
            const input = document.getElementById('search-input');
            const status = document.getElementById('search-status');
            const nodeId = input.value.trim();
            
            if (!nodeId) {{
                status.textContent = 'please enter a node id';
                status.style.color = '#ffb6b6';
                return;
            }}
            
            if (centerOnNode(nodeId)) {{
                status.textContent = `found and centered on node: ${{nodeId}}`;
                status.style.color = '#51cf66';
            }} else {{
                status.textContent = `node not found: ${{nodeId}}`;
                status.style.color = '#ff6b6b';
            }}
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            const searchInput = document.getElementById('search-input');
            if (searchInput) {{
                searchInput.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') {{
                        searchNode()
                    }}
                }});
            }}
        }});
        </script>
        """

        # inject search bar HTML into the body
        search_bar_html = """
        <div id="search-overlay">
            <input type="text" id="search-input" placeholder="Enter node ID to center view">
            <button id="search-button" onclick="searchNode()">Find</button>
            <div id="search-status"></div>
        </div>
        """

        html_content = html_content.replace("</head>", search_functionality + "</head>")
        html_content = html_content.replace(
            '<div class="card"', search_bar_html + '<div class="card"'
        )

        with open(filename, "w") as f:
            f.write(html_content)

        print(f"graph visualization saved to {filename}")
    else:
        print("failed to create visualization - empty graph")

def visualize_graph_from_file(database_file: str):
    db = GraphDatabase(database_file)
    save_visualization(db, filename="graph.html")

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
        default="100vh",
        help="height of the visualization",
    )
    parser.add_argument(
        "--width",
        default="100vw",
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
