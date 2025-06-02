# gdb

small graph db implementation for benchmarking trust models

## features

- add/remove nodes and edges
- directed/undirected edges
- arbitrary metadata on nodes and edges
- basic queries (neighbours, paths)
- persistence via json

## reference

### node operations

```python
# add a node
db.add_node(node_id: str, metadata: Dict[str, Any] = None) -> bool

# remove a node (and all its connected edges)
db.remove_node(node_id: str) -> bool

# get a node
node = db.get_node(node_id: str)
```

### edge operations

```python
# add an edge
db.add_edge(source: str, 
            target: str, 
            directed: bool = True, 
            metadata: Dict[str, Any] = None) -> bool

# remove an edge
db.remove_edge(source: str, target: str) -> bool

# get an edge
edge = db.get_edge(source: str, target: str)
```

### queries

```python
# get neighbors
neighbors = db.get_neighbors(node_id: str, direction: str = "all")
# direction can be: "all", "outgoing", "incoming", or "undirected"

# find paths between nodes
paths = db.find_paths(source: str, target: str, max_steps: int = 5)
```

### statistics

```python
# get node count
count = db.node_count()

# get edge count
count = db.edge_count()
```

### persistence

```python
# save database to file
db.save()  # saves to path specified during initialization
db.save("new_path.json")  # save to specific path

# load database from file
db.load()  # loads from path specified during initialization
db.load("other_graph.json")  # load from specific path

# clear database
db.clear()  # removes all nodes and edges
```

## notes

- node ids must be unique strings
- both nodes must exist before adding an edge between
- the database supports arbitrary metadata as dictionaries
- when removing a node, all connected edges are automatically removed
