# my_brep_to_graph/converter/brep_converter.py
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Union, Optional, Dict, Any, Tuple

# OCCWL imports
from occwl.entity_mapper import EntityMapper
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell


from bachelor_thesis.mappings import SURFACE_TYPE_MAPPING, CURVE_TYPE_MAPPING

# Type definition for shape types
ShapeType = Union[Shell, Solid, Compound]

def brepgat_face_adjacency(
    shape: ShapeType,
    feature_extractor,
    add_parallel: bool = False,
    self_loops: bool = False
) -> nx.DiGraph:
    """
    Build a face-level adjacency graph from the given B-Rep shape, 
    storing BRepGAT-like node & edge descriptors:
    
    1) Node = face index
       node attributes:
         - "face": the original OCCWL face
         - "face_descriptor": a dict from feature_extractor.get_face_descriptor(face)
    2) Edge = adjacency among faces
       edge attributes:
         - "edge_descriptor": a dict from feature_extractor.get_edge_descriptor(...)
           or a special 'parallel' descriptor if faces are parallel

    Args:
        shape (Shell, Solid, or Compound): The B-Rep shape
        feature_extractor: Object/class that provides:
           - get_face_descriptor(face) -> dict
           - get_edge_descriptor(edge, face1, face2) -> dict
           - check_parallel(face1, face2) -> bool
           - compute_parallel_distance(face1, face2) -> float
        add_parallel (bool): Whether to also add edges for parallel faces
        self_loops (bool): Whether to add loops for seam edges

    Returns:
        nx.DiGraph: A directed graph where each node is a face, 
                    and edges represent either shared-edge adjacency or parallel adjacency.
    """
    # Validate input
    assert isinstance(shape, (Shell, Solid, Compound)), \
        "shape must be a Shell, Solid, or Compound"

    # Create a graph & entity mapper
    graph = nx.DiGraph()
    mapper = EntityMapper(shape)

    # 1) Add nodes (faces) with descriptors
    all_faces = _add_face_nodes_to_graph(graph, shape, mapper, feature_extractor)

    # 2) Add topological adjacency edges from shared edges
    _add_topological_edges_to_graph(graph, shape, mapper, feature_extractor, all_faces, self_loops)

    # 3) Optionally add parallel face edges
    if add_parallel:
        _add_parallel_edges_to_graph(graph, all_faces, feature_extractor)

    return graph


def _add_face_nodes_to_graph(
    graph: nx.DiGraph,
    shape: ShapeType,
    mapper: EntityMapper,
    feature_extractor
) -> list:
    """
    Add one node per face in the shape to the graph, storing face descriptors.
    
    Returns:
        all_faces (list): The list of Face objects in the shape
    """
    all_faces = list(shape.faces())
    for face in all_faces:
        face_idx = mapper.face_index(face)
        face_desc = feature_extractor.get_face_descriptor(face)
        graph.add_node(face_idx,
                       face=face,
                       face_descriptor=face_desc)
    return all_faces


def _add_topological_edges_to_graph(
    graph: nx.DiGraph,
    shape: ShapeType,
    mapper: EntityMapper,
    feature_extractor,
    all_faces: list,
    self_loops: bool
):
    """
    Traverse all edges in the shape to find two-face adjacencies
    and add them as directed edges in the graph, storing an "edge_descriptor".
    If self_loops=True, also handle seam edges.
    
    This version uses a set to track added edges to prevent duplicates.
    """
    # Track added edges to prevent duplicates
    added_edges = set()
    
    for edge in shape.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(shape.faces_from_edge(edge))
        if not connected_faces:
            continue

        # Case 1: Seam edge (single face)
        if len(connected_faces) == 1 and self_loops and edge.seam(connected_faces[0]):
            face_idx = mapper.face_index(connected_faces[0])
            seam_desc = feature_extractor.get_edge_descriptor(edge, connected_faces[0], connected_faces[0])
            edge_key = (face_idx, face_idx, edge.__hash__())
            if edge_key not in added_edges:
                graph.add_edge(face_idx, face_idx, edge_descriptor=seam_desc, edge_obj=edge)
                added_edges.add(edge_key)

        # Case 2: Standard manifold adjacency (two faces)
        elif len(connected_faces) == 2:
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue

            left_idx = mapper.face_index(left_face)
            right_idx = mapper.face_index(right_face)

            # Add left->right edge if not already added
            edge_key_lr = (left_idx, right_idx, edge.__hash__())
            if edge_key_lr not in added_edges:
                edesc_lr = feature_extractor.get_edge_descriptor(edge, left_face, right_face)
                graph.add_edge(left_idx, right_idx, edge_descriptor=edesc_lr, edge_obj=edge)
                added_edges.add(edge_key_lr)

            # Add right->left edge if not already added
            edge_rev = edge.reversed_edge()
            edge_key_rl = (right_idx, left_idx, edge_rev.__hash__())
            if edge_key_rl not in added_edges:
                edesc_rl = feature_extractor.get_edge_descriptor(edge_rev, right_face, left_face)
                graph.add_edge(right_idx, left_idx, edge_descriptor=edesc_rl, edge_obj=edge_rev)
                added_edges.add(edge_key_rl)

        # Case 3: Non-manifold (more than 2 faces)
        elif len(connected_faces) > 2:
            # In case of non-manifold edges, we need a more complex handling
            # For now we'll raise a warning and continue
            print(f"Warning: Non-manifold edge detected with {len(connected_faces)} faces")
            continue


def _add_parallel_edges_to_graph(
    graph: nx.DiGraph,
    all_faces: list,
    feature_extractor
):
    """
    Find parallel faces and add them as special edges to the graph.
    
    This is an optional enhancement feature for the BRepGAT model.
    """
    # Skip implementation for now - can be added later
    # This would compare all face pairs to find parallel ones
    pass


def visualize_graph(graph: nx.DiGraph, output_path: str = "graph_visualization.png"):
    """
    Visualize the BRepGAT graph using matplotlib.
    
    Args:
        graph (nx.DiGraph): The graph to visualize
        output_path (str): Where to save the output image
    """
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for visualization
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    
    # Draw edges - use different styles for different edge types
    normal_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                    if 'edge_descriptor' in d]
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, 
                           width=1.5, alpha=0.7, arrows=True)
    
    # Label nodes with their face indices
    nx.draw_networkx_labels(graph, pos, font_size=14)
    
    # Add title and basic stats
    plt.title(f"BRepGAT Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    plt.axis('off')
    
    # Save to file
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {output_path}")
    plt.close()


def analyze_graph(graph: nx.DiGraph):
    """
    Analyze the graph structure and print statistics.
    """
    print("\n===== Graph Analysis =====")
    print(f"Number of nodes (faces): {graph.number_of_nodes()}")
    print(f"Number of edges (adjacencies): {graph.number_of_edges()}")
    
    # Count the number of edges by type
    edge_types = defaultdict(int)
    for _, _, data in graph.edges(data=True):
        if 'edge_descriptor' in data:
            edge_desc = data['edge_descriptor']
            if hasattr(edge_desc, 'curve_type'):
                curve_type = edge_desc.curve_type
                edge_types[curve_type] += 1
    
    # Print edge type distribution
    if edge_types:
        print("\nEdge types distribution:")
        for edge_type, count in sorted(edge_types.items()):
            curve_type_name = [k for k, v in CURVE_TYPE_MAPPING.items() if v == edge_type]
            name = curve_type_name[0] if curve_type_name else f"Type {edge_type}"
            print(f"  {name}: {count} edges")
    
    # Print node degree statistics
    degrees = [d for _, d in graph.degree()]
    if degrees:
        print(f"\nNode degree statistics:")
        print(f"  Min degree: {min(degrees)}")
        print(f"  Max degree: {max(degrees)}")
        print(f"  Avg degree: {sum(degrees)/len(degrees):.2f}")
    
    # Check for duplicated edges
    duplicate_check = {}
    for u, v, data in graph.edges(data=True):
        key = (u, v)
        if key not in duplicate_check:
            duplicate_check[key] = 1
        else:
            duplicate_check[key] += 1
    
    duplicates = {k: v for k, v in duplicate_check.items() if v > 1}
    if duplicates:
        print(f"\nWARNING: Found {len(duplicates)} duplicated edges!")
        for (u, v), count in duplicates.items():
            print(f"  Edge {u}->{v} appears {count} times")
    else:
        print("\nNo duplicated edges found!")