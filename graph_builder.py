import networkx as nx
from pathlib import Path
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple

import numpy as np
from occwl.entity_mapper import EntityMapper
from occwl.compound import Compound
from occwl.solid import Solid
from occwl.shell import Shell

from mappings import CURVE_TYPE_MAPPING

ShapeType = Union[Shell, Solid, Compound]


def brepgat_face_adjacency(
    shape: ShapeType,
    feature_extractor,
    add_parallel: bool = True,
    self_loops: bool = True,
    face_attributes: Optional[List[Dict]] = None,
    edge_attributes: Optional[Dict[Tuple[int, int], Dict]] = None,
) -> nx.DiGraph:
    """Build a face‑level graph storing node and edge descriptors as in BRepGAT."""

    assert isinstance(shape, (Shell, Solid, Compound))
    graph = nx.DiGraph()
    mapper = EntityMapper(shape)
    all_faces = list(shape.faces())

    # --- Nodes ---
    for i, face in enumerate(all_faces):
        face_idx = mapper.face_index(face)
        attrs = (
            face_attributes[i]
            if face_attributes is not None
            else feature_extractor.get_face_descriptor(face).to_dict()
        )
        graph.add_node(face_idx, face_attributes=attrs)

    # --- Topological edges ---
    _add_topological_edges_to_graph(
        graph,
        shape,
        mapper,
        feature_extractor,
        all_faces,
        self_loops,
        edge_attributes,
    )

    # --- Parallel‑face synthetic edges ---
    if add_parallel:
        _add_parallel_edges_to_graph(graph, all_faces, mapper)

    return graph


def _add_topological_edges_to_graph(
    graph: nx.DiGraph,
    shape: ShapeType,
    mapper: EntityMapper,
    feature_extractor,
    all_faces: list,
    self_loops: bool,
    edge_attributes: Optional[Dict[Tuple[int, int], Dict]],
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
            eattrs = (
                edge_attributes.get(key)
                if edge_attributes and key in edge_attributes
                else feature_extractor.get_edge_descriptor(edge, f, f).to_dict()
            )
            edge_key = (idx, idx, edge.__hash__())
            if edge_key not in added_edges:
                graph.add_edge(idx, idx, edge_attributes=eattrs)
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
                eattrs12 = (
                    edge_attributes.get(k12)
                    if edge_attributes and k12 in edge_attributes
                    else feature_extractor.get_edge_descriptor(edge, f1, f2).to_dict()
                )
                graph.add_edge(i1, i2, edge_attributes=eattrs12)
                added_edges.add(key12)

            # Reverse
            edge_rev = edge.reversed_edge()
            k21 = (i2, i1)
            key21 = (i2, i1, edge_rev.__hash__())
            if key21 not in added_edges:
                eattrs21 = (
                    edge_attributes.get(k21)
                    if edge_attributes and k21 in edge_attributes
                    else feature_extractor.get_edge_descriptor(edge_rev, f2, f1).to_dict()
                )
                graph.add_edge(i2, i1, edge_attributes=eattrs21)
                added_edges.add(key21)

        elif len(connected_faces) > 2:
            print(f"Warning: Non‑manifold edge with {len(connected_faces)} faces")



def _add_parallel_edges_to_graph(
    graph: nx.DiGraph,
    all_faces: list,
    mapper: EntityMapper,
    tol: float = 1e-3,
):
    """
    Examine every existing edge (u→v).  If the two faces are parallel
    (|n₁·n₂| ≈ 1) set
        edge_attributes['parallel'] = True
        edge_attributes['distance'] = |(p₂ – p₁)·n₁|
    Otherwise guarantee those keys exist (False, 0.0).
    Nothing else in the dict is modified; no new edges are added.
    """
    if graph.number_of_edges() == 0:
        return

    # cache face normals & centroids once
    normals, centers = {}, {}
    for f in all_faces:
        idx = mapper.face_index(f)
        c_uv = f.uv_bounds().center()
        normals[idx] = np.array(f.normal(c_uv))
        centers[idx] = np.array(f.point(c_uv))

    for u, v, data in graph.edges(data=True):
        eattrs = data["edge_attributes"]
        n1, n2 = normals[u], normals[v]
        cos = abs(np.dot(n1, n2))

        if 1.0 - cos < tol:                         # parallel
            dist = float(abs(np.dot(centers[v] - centers[u], n1)))
            eattrs["parallel"] = True
            eattrs["distance"] = dist
        else:
            # ensure keys exist but leave other info untouched
            eattrs.setdefault("parallel", False)
            eattrs.setdefault("distance", 0.0)



# -----------------------------------------------------------------------------
# Diagnostics / visualisation helpers (unchanged)
# -----------------------------------------------------------------------------

def analyze_graph(graph: nx.DiGraph):
    print("\n===== Graph Analysis =====")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    edge_types = defaultdict(int)
    for _, _, data in graph.edges(data=True):
        if "edge_attributes" in data:
            etype = data["edge_attributes"].get("curve_type", -1)
            edge_types[etype] += 1

    if edge_types:
        print("\nEdge type distribution:")
        for t, c in edge_types.items():
            name = [k for k, v in CURVE_TYPE_MAPPING.items() if v == t]
            print(f"  {name[0] if name else f'Type {t}'}: {c}")


def visualize_graph(graph: nx.DiGraph, graph_name: str = "graph"):
    vis_dir = Path(__file__).resolve().parent.parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    output_path = vis_dir / f"{graph_name}.png"

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.title(f"Graph: {graph_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📷 Saved: {output_path}")
    plt.close()
