# my_brep_to_graph/feature_extractor/feature_extractor.py

from typing import List
import math
import numpy as np

from occwl.face import Face
from occwl.edge import Edge
from descriptors.face_descriptor import FaceDescriptor
from descriptors.edge_descriptor import EdgeDescriptor




class FeatureExtractor:
    """
    Extracts feature descriptors for faces and edges.
    """

    def get_face_descriptor(self, face: Face) -> FaceDescriptor:
        """
        Create a FaceDescriptor from an OCCWL Face.
        """
        surface_type_id = self._classify_surface_type(face)
        area = face.area()
        normal = face.normal()
        normal_list = [normal.X(), normal.Y(), normal.Z()]

        # For bounding box ratio in parametric domain, placeholder
        bounding_box_ratio = self._compute_uv_bounding_box_ratio(face)

        # Outer/inner loops - placeholders for demonstration
        outer_loop_adj = [0.0]*33
        outer_loop_c0  = [0.0]*11
        outer_loop_perp= [0.0]*11
        inner_loop_info= [0.0, 0.0]

        return FaceDescriptor(
            surface_type=surface_type_id,
            surface_area=area,
            surface_normal=normal_list,
            bounding_box_ratio=bounding_box_ratio,
            outer_loop_adj_faces=outer_loop_adj,
            outer_loop_c0_continuity=outer_loop_c0,
            outer_loop_perpendicular=outer_loop_perp,
            inner_loop=inner_loop_info
        )


    def get_edge_descriptor(self, edge: Edge, 
                            face1: Face = None, 
                            face2: Face = None) -> EdgeDescriptor:
        """
        Create an EdgeDescriptor from an OCCWL Edge.
        If `face1` and `face2` are provided, we can compute convexity, parallel, etc.
        """
        curve_type_id = self._classify_curve_type(edge)
        length = edge.curve_length() or 0.0

        # For convexity, we need face1, face2 normals
        is_convex = False
        if face1 and face2:
            is_convex = self._compute_convexity(face1, face2, edge)

        # For perpendicular, parallel, and distance:
        is_perp = False
        is_parallel = False
        distance = 0.0
        if face1 and face2:
            is_perp      = self._check_perpendicular(face1, face2)
            is_parallel  = self._check_parallel(face1, face2)
            if is_parallel:
                distance = self._compute_parallel_distance(face1, face2)

        return EdgeDescriptor(
            curve_type=curve_type_id,
            curve_length=length,
            convexity=is_convex,
            perpendicular=is_perp,
            parallel=is_parallel,
            distance=distance
        )


    # ------------------------
    # Internal helper methods
    # ------------------------

    ################
    # FACES
    #####################

    def classify_surface_type(self, face: Face) -> int:
        """
        Classify the Face's surface into an integer ID based on Table 1.

        Mapped as follows (0-based):
            0 = Unknown
            1 = Bezier surface
            2 = B-spline surface
            3 = Rectangular trimmed surface (no implementation)
            4 = Conical surface
            5 = Cylindrical surface
            6 = Plane
            7 = Spherical surface
            8 = Toroidal surface
            9 = Surface of linear extrusion
            10 = Surface of revolution
            11 = Any (generic fallback)

        Args:
            face (occwl.face.Face): The occwl Face object to classify.
        Returns:
            int: An integer identifier (0..11).
        """

        return face.surface_type()

    '''
        surf = face.surface()

        if surf.is_plane(): 
            print ("Yo")
            return 6  # Plane ✅ (Checked in OCCWL)
        if surf.is_cylinder(): return 5  # Cylinder ✅ (Checked in OCCWL)
        if surf.is_cone(): return 4  # Cone ✅ (Checked in OCCWL)
        if surf.is_sphere(): return 7  # Sphere ✅ (Checked in OCCWL)
        if surf.is_torus(): return 8  # Torus ✅ (Checked in OCCWL)
        if surf.is_bezier(): return 1  # Bezier Surface ✅ (Checked in OCCWL)
        if surf.is_bspline(): return 2  # B-spline Surface ✅ (Checked in OCCWL)

        # TODO: Implement checks for Rectangular Trimmed Surface, Extrusion, and Revolution

        return 11  # Fallback: "Any" or "Unknown"'''
    



    def _compute_uv_bounding_box_ratio(self, face: Face) -> float:
        """
        Compute the bounding box aspect ratio in the UV parameter space.
        In practice, use face.uv_bounds() if available, or approximate in 3D.
        """
        try:
            u_min, u_max, v_min, v_max = face.uv_bounds()
            u_length = abs(u_max - u_min)
            v_length = abs(v_max - v_min)
            return min(u_length, v_length) / max(u_length, v_length) if max(u_length, v_length) > 0 else 1.0
        except:
            return 1.0  # Default ratio

    def _compute_outer_loop_adj_faces(self, face: Face) -> List[float]:
        """
        Computes the adjacency ratio for different surface types × convexity.
        """
        # TODO: Implement adjacency queries
        return [0.0] * 33  # Placeholder

    def _compute_outer_loop_c0_continuity(self, face: Face) -> List[float]:
        """
        Computes the ratio of adjacent faces with C0 continuity.
        """
        # TODO: Implement continuity check
        return [0.0] * 11  # Placeholder

    def _compute_outer_loop_perpendicular(self, face: Face) -> List[float]:
        """
        Computes the ratio of adjacent faces that are perpendicular.
        """
        # TODO: Implement perpendicularity check
        return [0.0] * 11  # Placeholder

    def _compute_inner_loop(self, face: Face) -> List[float]:
        """
        Extracts the location and convexity of the inner loop (if any).
        """
        # TODO: Implement inner loop detection
        return [0.0, 0.0]  # Placeholder


    ################
    # EDGES
    #####################
    

    def _classify_curve_type(self, edge: Edge) -> int:
        """
        Return an integer representing the curve type.
            0 = Unknown
            1 = B-spline
            5 = Circle
            9 = Line
            etc.
        """
        crv = edge.curve()
        if crv is None:
            return 0
        if crv.is_line():
            return 9
        elif crv.is_circle():
            return 5
        elif crv.is_bspline():
            return 1
        # etc.
        return 0


    def _compute_convexity(self, face1: Face, face2: Face, edge: Edge) -> bool:
        """
        Example approach to compute convex/concave using face normals
        and the edge direction, as per the paper.
        """
        # Simplified logic just for illustration:
        normal1 = face1.normal()
        normal2 = face2.normal()
        # Paper’s approach involves cross/dot products with the edge direction
        # We'll do something minimal:
        dot_value = normal1.Dot(normal2)
        # If dot_value < 0 => concave, else convex (very rough heuristic)
        return dot_value >= 0
    

    def _check_perpendicular(self, face1: Face, face2: Face) -> bool:
        n1 = face1.normal()
        n2 = face2.normal()
        dot_val = abs(n1.Dot(n2))
        # If nearly 0 => perpendicular
        return dot_val < 1e-4


    def _check_parallel(self, face1: Face, face2: Face) -> bool:
        n1 = face1.normal()
        n2 = face2.normal()
        # Norm dot close to ±1 => parallel
        dot_val = abs(n1.Dot(n2))
        return abs(dot_val - 1.0) < 1e-4
    

    def _compute_parallel_distance(self, face1: Face, face2: Face) -> float:
        """
        If two faces are parallel, compute distance between them.
        In practice, you'd pick a point on face1 and face2 and measure 
        the distance along the normal.  This is a placeholder.
        """
        # E.g., find an average point on face1, project it onto face2, measure distance, etc.
        return 20.0  # example value

import pathlib
import json
import re
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.vertex import Vertex
from occwl.viewer import Viewer
from occwl.compound import Compound


def _compute_surface_normal(self, face: Face) -> List[float]:
    """
    Compute the normal of the surface at the center of its UV domain.
    """
    uv_center = face.uv_bounds().center()
    normal = face.normal(uv_center)
    return normal.tolist()

if __name__ == "__main__":
    STEP_FILE = "C:/Users/synkh/BachelorThesis/original_datasets/MFCAD++_dataset/step/val/1154.step"
    compound = Compound.load_from_step(pathlib.Path(__file__).resolve().parent.joinpath(STEP_FILE))
    solid = next(compound.solids())
    faces = list(solid.faces())
    print(faces)

    feature_extractor = FeatureExtractor()
    for i in range(0, len(faces)):
        feature_type = feature_extractor.classify_surface_type(faces[i])
        print(i, " Surface type:", feature_type)

    face = faces[6]
    print(" Surface type:", feature_extractor.classify_surface_type(face), " with area = ", face.area())
    face = faces[8]
    print(8, " Surface type:", feature_extractor.classify_surface_type(faces[8]), " with area = ", faces[8].area())
    face = faces[34]
    print(34, " Surface type:", feature_extractor.classify_surface_type(faces[34]), " with area = ", faces[34].area())
    