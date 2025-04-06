import re
from pathlib import Path
import pickle
import logging
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional

import torch
from torch_geometric.data import Data
from occwl.compound import Compound
from occwl.entity_mapper import EntityMapper
from feature_extractor import FeatureExtractor
from graph_builder import brepgat_face_adjacency 
from torch_geometric.utils import from_networkx
from descriptors.face_attributes import FaceAttributes
from descriptors.edge_attributes import EdgeAttributes

# === Config ===
DATASET_DIR = Path("original_datasets/MFCAD++_dataset/step/train")
VAL_LIST = Path("original_datasets/MFCAD++_dataset/train.txt")
GRAPH_DIR = Path("graph_data")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging: log both to a file and the console
LOG_FILE = GRAPH_DIR / "error_log.txt"
logger = logging.getLogger("dataset_preprocesser")
logger.setLevel(logging.INFO)

# Create file handler which logs messages to file
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.INFO)
fh_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

# Create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
ch.setFormatter(ch_formatter)
logger.addHandler(ch)

# Optionally capture warnings from the warnings module
logging.captureWarnings(True)

# Feature selection configuration
FACE_FEATURES = {
    "surface_type": True,
    "surface_area": True,
    "surface_normal": True,
    "bounding_box_ratio": True,
    "outer_loop_adj_faces": True,
    "outer_loop_c0_continuity": True,
    "outer_loop_perpendicular": True,
    "inner_loop": True
}

EDGE_FEATURES = {
    "curve_type": True,
    "curve_length": True,
    "convexity": True,
    "perpendicular": True,
    "parallel": True,
    "distance": True
}


def extract_feature_labels_from_text(step_content):
    feature_labels = []
    for line in step_content.splitlines():
        match = re.match(r"#\d+ = ADVANCED_FACE\('(\d+)'", line)
        if match:
            try:
                label = int(match.group(1))
            except ValueError:
                label = 0
            feature_labels.append(label)
    return feature_labels


def get_feature_indices(attrs_class, feature_config: Dict[str, bool]) -> List[str]:
    """
    Get the list of feature names that are enabled in the config.
    """
    return [name for name, enabled in feature_config.items() if enabled]


def attributes_to_tensor(attrs: Dict[str, Any], 
                         feature_config: Dict[str, bool],
                         attrs_class) -> torch.Tensor:
    """
    Convert attributes dictionary to a tensor of numerical features based on config.
    
    Args:
        attrs: Dictionary of attributes
        feature_config: Dictionary mapping feature names to boolean flags
        attrs_class: The attribute class (FaceAttributes or EdgeAttributes)
    
    Returns:
        Tensor of selected numerical features
    """
    features = []
    for name, enabled in feature_config.items():
        if not enabled:
            continue
            
        value = attrs[name]
        if isinstance(value, (list, tuple)):
            features.extend(value)  # Unpack lists (e.g., surface_normal)
        else:
            features.append(float(value))  # Convert bools and other types to float
            
    return torch.tensor(features, dtype=torch.float)


# === Load STEP filenames ===
with open(VAL_LIST, "r") as f:
    step_files = [line.strip() for line in f if line.strip()]

logger.info("🚀 Starting BRepGAT graph generation...")
logger.info("\nSelected face features: " + str(get_feature_indices(FaceAttributes, FACE_FEATURES)))
logger.info("Selected edge features: " + str(get_feature_indices(EdgeAttributes, EDGE_FEATURES)))

# Process each STEP file with error logging and resumability
for file_name in tqdm(step_files, desc="STEP files"):
    step_path = DATASET_DIR / f"{file_name}.step"

    if not step_path.exists():
        logger.warning(f"File not found: {step_path}")
        continue

    # Check if graph already exists (resume support)
    out_path = GRAPH_DIR / f"{file_name}.pkl"
    if out_path.exists():
        logger.info(f"Graph already exists, skipping: {file_name}")
        continue

    try:
        # Load solid from STEP
        compound = Compound.load_from_step(step_path)
        solids = list(compound.solids())
        if not solids:
            logger.warning(f"No solids found in {file_name}. Skipping...")
            continue
        solid = solids[0]

        # === Feature + Label Extraction ===
        feature_extractor = FeatureExtractor(solid)
        face_attributes = []
        step_text = step_path.read_text()
        labels = extract_feature_labels_from_text(step_text)

        for i, face in enumerate(solid.faces()):
            attrs = feature_extractor.get_face_descriptor(face)
            attrs.label = labels[i] if i < len(labels) else 0
            face_attributes.append(attrs.to_dict())

        # === Edge Attributes ===
        edge_attributes = {}
        mapper = EntityMapper(solid)

        for edge in solid.edges():
            connected_faces = list(solid.faces_from_edge(edge))
            if len(connected_faces) == 2:
                f1, f2 = edge.find_left_and_right_faces(connected_faces)
                if f1 is None or f2 is None:
                    continue
                i1, i2 = mapper.face_index(f1), mapper.face_index(f2)
                attrs12 = feature_extractor.get_edge_descriptor(edge, f1, f2).to_dict()
                edge_attributes[(i1, i2)] = attrs12

                edge_rev = edge.reversed_edge()
                attrs21 = feature_extractor.get_edge_descriptor(edge_rev, f2, f1).to_dict()
                edge_attributes[(i2, i1)] = attrs21

        # === Build Graph ===
        graph_nx = brepgat_face_adjacency(
            solid,
            feature_extractor,
            face_attributes=face_attributes,
            edge_attributes=edge_attributes
        )

        # === Convert to PyG Data ===
        pyg_data = from_networkx(graph_nx)

        # Prepare node features (x) and labels (y)
        pyg_data.x = torch.stack([
            attributes_to_tensor(d["face_attributes"], FACE_FEATURES, FaceAttributes)
            for _, d in graph_nx.nodes(data=True)
        ])
        pyg_data.y = torch.tensor(
            [d["face_attributes"]["label"] for _, d in graph_nx.nodes(data=True)],
            dtype=torch.long
        )

        # Prepare edge features
        edge_attrs = torch.stack([
            attributes_to_tensor(d["edge_attributes"], EDGE_FEATURES, EdgeAttributes)
            for _, _, d in graph_nx.edges(data=True)
        ])
        pyg_data.edge_attr = edge_attrs

        # === Save to graph_data/ ===
        with open(out_path, "wb") as f:
            pickle.dump(pyg_data, f)

        logger.info(f"Successfully processed {file_name}")
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")

logger.info("✅ All STEP files processed and converted to graph data.")
