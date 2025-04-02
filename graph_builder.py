import networkx as nx
from pathlib import Path
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple
from occwl.entity_mapper import EntityMapper
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell
from mappings import CURVE_TYPE_MAPPING

ShapeType = Union[Shell, Solid, Compound]

def brepgat_face_adjacency(
    shape: ShapeType,
    feature_extractor,
    add_parallel: bool = False,
    self_loops: bool = False,
    face_descriptors: Optional[List[Dict]] = None,
    edge_descriptors: Optional[Dict[Tuple[int, int], Dict]] = None
) -> nx.DiGraph:
    """
    Build a face-level graph storing node and edge descriptors as in BRepGAT.

    Args:
        shape: OCCWL B-Rep shape (Solid, Shell, or Compound)
        feature_extractor: FeatureExtractor object
        add_parallel: Whether to add parallel face edges
        self_loops: Include seam (loop) edges
        face_descriptors: Optional list of precomputed face descriptors
        edge_descriptors: Optional dict {(face_i, face_j): edge_desc_dict}

    Returns:
        nx.DiGraph with descriptors as node/edge attributes
    """
    assert isinstance(shape, (Shell, Solid, Compound))
    graph = nx.DiGraph()
    mapper = EntityMapper(shape)
    all_faces = list(shape.faces())

    for i, face in enumerate(all_faces):
        face_idx = mapper.face_index(face)
        descriptor = face_descriptors[i] if face_descriptors else feature_extractor.get_face_descriptor(face).to_dict()
        graph.add_node(face_idx, face_descriptor=descriptor)

    _add_topological_edges_to_graph(graph, shape, mapper, feature_extractor, all_faces, self_loops, edge_descriptors)

    if add_parallel:
        _add_parallel_edges_to_graph(graph, all_faces, feature_extractor)

    return graph


def _add_topological_edges_to_graph(
    graph: nx.DiGraph,
    shape: ShapeType,
    mapper: EntityMapper,
    feature_extractor,
    all_faces: list,
    self_loops: bool,
    edge_descriptors: Optional[Dict[Tuple[int, int], Dict]]
):
    added_edges = set()
    for edge in shape.edges():
        if not edge.has_curve():
            continue

        connected_faces = list(shape.faces_from_edge(edge))
        if not connected_faces:
            continue

        # Self-loop
        if len(connected_faces) == 1 and self_loops and edge.seam(connected_faces[0]):
            f = connected_faces[0]
            idx = mapper.face_index(f)
            key = (idx, idx)
            edesc = edge_descriptors.get(key) if edge_descriptors and key in edge_descriptors else \
                feature_extractor.get_edge_descriptor(edge, f, f).to_dict()
            edge_key = (idx, idx, edge.__hash__())
            if edge_key not in added_edges:
                graph.add_edge(idx, idx, edge_descriptor=edesc)
                added_edges.add(edge_key)

        # Standard edge between two faces
        elif len(connected_faces) == 2:
            f1, f2 = edge.find_left_and_right_faces(connected_faces)
            if f1 is None or f2 is None:
                continue
            i1, i2 = mapper.face_index(f1), mapper.face_index(f2)

            # Forward
            k12 = (i1, i2)
            key12 = (i1, i2, edge.__hash__())
            if key12 not in added_edges:
                edesc12 = edge_descriptors.get(k12) if edge_descriptors and k12 in edge_descriptors else \
                    feature_extractor.get_edge_descriptor(edge, f1, f2).to_dict()
                graph.add_edge(i1, i2, edge_descriptor=edesc12)
                added_edges.add(key12)

            # Reverse
            edge_rev = edge.reversed_edge()
            k21 = (i2, i1)
            key21 = (i2, i1, edge_rev.__hash__())
            if key21 not in added_edges:
                edesc21 = edge_descriptors.get(k21) if edge_descriptors and k21 in edge_descriptors else \
                    feature_extractor.get_edge_descriptor(edge_rev, f2, f1).to_dict()
                graph.add_edge(i2, i1, edge_descriptor=edesc21)
                added_edges.add(key21)

        elif len(connected_faces) > 2:
            print(f"Warning: Non-manifold edge with {len(connected_faces)} faces")


def _add_parallel_edges_to_graph(graph: nx.DiGraph, all_faces: list, feature_extractor):
    pass  # Optional extension


def analyze_graph(graph: nx.DiGraph):
    print("\n===== Graph Analysis =====")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    from collections import defaultdict
    edge_types = defaultdict(int)
    for _, _, data in graph.edges(data=True):
        if 'edge_descriptor' in data:
            etype = data['edge_descriptor'].get("curve_type", -1)
            edge_types[etype] += 1

    if edge_types:
        print("\nEdge type distribution:")
        for t, c in edge_types.items():
            name = [k for k, v in CURVE_TYPE_MAPPING.items() if v == t]
            print(f"  {name[0] if name else f'Type {t}'}: {c}")


def visualize_graph(graph: nx.DiGraph, graph_name: str = "graph"):
    """
    Visualize the BRepGAT graph and save it to the visualizations/ folder.

    Args:
        graph (nx.DiGraph): The graph to visualize
        graph_name (str): A name used in the filename (e.g. part ID)
    """
    vis_dir = Path(__file__).resolve().parent.parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    output_path = vis_dir / f"{graph_name}.png"

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title(f"Graph: {graph_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📷 Saved: {output_path}")
    plt.close()