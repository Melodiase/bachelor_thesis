# my_brep_to_graph/converter/brep_converter.py

import sys
from occwl.io import load_step
from occwl.shape import Shape
from occwl.face import Face

from .feature_extractor import FeatureExtractor

class BRepConverter:
    """
    Loads a STEP file, converts it into B-rep topological data, 
    and uses FeatureExtractor to get descriptors for faces & edges.
    """

    def __init__(self, step_file_path: str):
        self.step_file_path = step_file_path
        self.shape: Shape = None

        # These lists will hold the final descriptors
        self.face_descriptors = []  # List[FaceDescriptor]
        self.edge_descriptors = []  # List[EdgeDescriptor]

        # For adjacency, map face_index -> [neighbor_face_indices]
        self.face_adjacency = {}

        # Our separate FeatureExtractor
        self.feature_extractor = FeatureExtractor()


    def load_shape(self) -> None:
        """
        Loads the STEP file into an OCCWL Shape object.
        """
        self.shape = load_step(self.step_file_path)
        if not self.shape:
            raise RuntimeError(f"Could not load shape from {self.step_file_path}")
        print(f"Loaded shape from {self.step_file_path}")


    def convert_to_graph(self) -> None:
        """
        Performs the B-rep to graph conversion:
         1) Extract face descriptors
         2) Extract edge descriptors
         3) Build adjacency among faces
        """
        self._compute_face_descriptors()
        self._compute_edge_descriptors_and_adjacency()


    def _compute_face_descriptors(self) -> None:
        """
        Creates a FaceDescriptor for each face in the shape.
        """
        faces = list(self.shape.faces())
        self.face_adjacency = {i: [] for i in range(len(faces))}

        for i, face in enumerate(faces):
            f_desc = self.feature_extractor.get_face_descriptor(face)
            self.face_descriptors.append(f_desc)
            

    def _compute_edge_descriptors_and_adjacency(self) -> None:
        """
        Builds edge descriptors for all unique edges 
        and collects adjacency info between faces.
        """
        faces = list(self.shape.faces())

        for i, face in enumerate(faces):
            for edge in face.edges():
                # Get the faces connected by this edge
                connected_faces = list(edge.connected_faces())

                # If there are exactly 2 connected faces, we can build adjacency
                if len(connected_faces) == 2:
                    f_idx_1 = faces.index(connected_faces[0])
                    f_idx_2 = faces.index(connected_faces[1])

                    # Add adjacency in both directions, if not already
                    if f_idx_2 not in self.face_adjacency[f_idx_1]:
                        self.face_adjacency[f_idx_1].append(f_idx_2)
                    if f_idx_1 not in self.face_adjacency[f_idx_2]:
                        self.face_adjacency[f_idx_2].append(f_idx_1)

                    # Build an EdgeDescriptor
                    edge_desc = self.feature_extractor.get_edge_descriptor(
                        edge,
                        face1=connected_faces[0],
                        face2=connected_faces[1]
                    )
                    self.edge_descriptors.append(edge_desc)

                else:
                    # This edge may belong to only one face (open edge) 
                    # or more than 2 (uncommon in solids).
                    # You could still build an EdgeDescriptor if desired,
                    # but we skip adjacency in that case.
                    edge_desc = self.feature_extractor.get_edge_descriptor(
                        edge,
                        face1=connected_faces[0] if connected_faces else None,
                        face2=connected_faces[1] if len(connected_faces) > 1 else None
                    )
                    self.edge_descriptors.append(edge_desc)
