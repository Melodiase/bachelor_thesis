from typing import List, Tuple
import numpy as np

from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid

from descriptors.edge_attributes import EdgeAttributes, EdgeConvexity
from mappings import CURVE_TYPE_MAPPING
from extractors.face_extractor import FaceExtractor

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods_Edge
from OCC.Core.Geom import Geom_TrimmedCurve, Geom_BoundedCurve, Geom_Conic
from occwl.edge_data_extractor import EdgeDataExtractor

# Helper function to compute the angle between two normalized vectors.
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)

class EdgeExtractor:
    """
    Extracts edge features and creates edge descriptors.
    """

    def __init__(self, solid: Solid):
        """
        Initialize with a parent solid for adjacency queries.
        
        Args:
            solid: The OCCWL Solid to extract features from
        """
        self.solid = solid

    def get_edge_descriptor(self, edge: Edge, 
                              face1: Face = None, 
                              face2: Face = None,
                              angle_tol_rads: float = 0.1) -> EdgeAttributes:
        """
        Create an EdgeDescriptor from an OCCWL Edge.
        
        Args:
            edge: The OCCWL Edge to extract features from
            face1: First adjacent face (optional)
            face2: Second adjacent face (optional)
            angle_tol_rads: Tolerance for angle calculations
            
        Returns:
            A complete EdgeDescriptor object
        """
        # Get basic properties
        curve_type_id = self.classify_curve_type(edge)
        length = edge.length() 

        # Default values for properties that require faces
        is_convex = False
        is_perp = False
        is_parallel = False
        distance = 0.0
        
        # Check if both faces are provided
        if face1 is not None and face2 is not None:
            # Determine convexity using enhanced method
            is_convex = self.check_edge_convexity(edge, face1, face2, angle_tol_rads)
            
            # Check for perpendicular and parallel faces
            # Get normal vectors using FaceExtractor
            face_extractor = FaceExtractor(self.solid)
            normal1 = np.array(face_extractor.compute_surface_normal(face1))
            normal2 = np.array(face_extractor.compute_surface_normal(face2))
            
            # Calculate dot product for angle between normals
            dot_product = abs(np.dot(normal1, normal2))
            
            # Perpendicular if dot product is close to 0
            is_perp = abs(dot_product) < 1e-3
            
            # Parallel if dot product is close to 1 (allowing for some tolerance)
            is_parallel = abs(dot_product - 1.0) < 1e-4
            
            # If faces are parallel, compute distance properly.
            if is_parallel:
                distance = self.compute_parallel_distance(face1, face2)
    
        # Create and return the EdgeDescriptor
        return EdgeAttributes(
            curve_type=curve_type_id,
            curve_length=length,
            convexity=is_convex,
            perpendicular=is_perp,
            parallel=is_parallel,
            distance=distance
        )

    def classify_curve_type(self, edge: Edge) -> int:
        """
        Classify the Edge's curve into an integer ID.
        See mappings.py -> CURVE_TYPE_MAPPING
        
        Args:
            edge: The OCCWL edge to classify
            
        Returns:
            int: An integer identifier from CURVE_TYPE_MAPPING
        """
        curve_type = edge.curve_type()

        if curve_type == "unknown":
            curve_type = self._identify_additional_curve_types(edge)
        
        return CURVE_TYPE_MAPPING.get(curve_type, 0)

    def _identify_additional_curve_types(self, edge: Edge) -> str:
        """
        Identify additional curve types not covered by the basic classification.
        
        Args:
            edge: The OCCWL edge to analyze
            
        Returns:
            str: The identified curve type string
        """
        curve, _, _ = BRep_Tool.Curve(topods_Edge(edge.topods_shape()))
        
        # Use DownCast to check specific curve types
        if Geom_TrimmedCurve.DownCast(curve) is not None:
            return "trimmed"
        elif Geom_BoundedCurve.DownCast(curve) is not None:
            return "bounded"
        elif Geom_Conic.DownCast(curve) is not None:
            return "conic"

        return "unknown"

    def check_edge_convexity(self, edge: Edge, face1: Face, face2: Face, angle_tol_rads: float) -> bool:
        """
        Determines convexity between two adjacent faces.
        
        Enhancement:
        - If the face-center normals (computed at UV-center) are nearly parallel,
          then recompute normals at the centroid of the shared edge.
        - Otherwise, use the occwl EdgeDataExtractor as is.
        
        Args:
            edge: The edge to check
            face1: First face
            face2: Second face
            angle_tol_rads: Angle tolerance in radians
            
        Returns:
            bool: True if convex, False if concave or smooth.
        """
        # Get face-center normals via face methods
        uv_center1 = face1.uv_bounds().center()
        uv_center2 = face2.uv_bounds().center()
        normal1 = np.array(face1.normal(uv_center1))
        normal2 = np.array(face2.normal(uv_center2))
        
        # If the normals are nearly parallel, use edge centroid approach
        if angle_between(normal1, normal2) < 0.1:
            # Attempt to use ordered_vertices; fall back to start/end vertices if unavailable.
            try:
                vertices = list(edge.ordered_vertices())
            except AttributeError:
                vertices = [edge.start_vertex(), edge.end_vertex()]
            if vertices:
                pts = np.array([v.point() for v in vertices])
                centroid = np.mean(pts, axis=0)
                # Reproject centroid to each face
                uv1 = face1.point_to_parameter(centroid)
                uv2 = face2.point_to_parameter(centroid)
                normal1 = np.array(face1.normal(uv1))
                normal2 = np.array(face2.normal(uv2))
            # Else, fallback to the face center normals (unlikely case)
        
            # Compute a representative tangent (using midpoint of parameter domain)
            interval = edge.u_bounds()
            mid_u = (interval.a + interval.b) / 2.0
            tangent = np.array(edge.tangent(mid_u))
            # Determine convexity manually
            cross_prod = np.cross(normal1, normal2)
            dot_val = np.dot(cross_prod, tangent)
            return dot_val > 0
        else:
            # Use occwl's EdgeDataExtractor if normals are not nearly parallel.
            edge_extractor = EdgeDataExtractor(edge, [face1, face2])
            convexity = edge_extractor.edge_convexity(angle_tol_rads)
            return convexity == EdgeConvexity.CONVEX

    def compute_parallel_distance(self, face1: Face, face2: Face) -> float:
        """
        If two faces are parallel, compute the distance between them.
        
        Enhancement:
        - Compute an approximate distance by projecting a representative point from face1 onto face2
          along face1's normal. This works best for nearly planar regions.
        
        Args:
            face1: First face.
            face2: Second face.
            
        Returns:
            float: Approximate distance between the parallel faces.
        """
        # Get a representative point from face1 (use its UV center)
        uv_center1 = face1.uv_bounds().center()
        point1 = face1.point(uv_center1)
        normal1 = np.array(face1.normal(uv_center1))
        
        # Project point1 onto face2 by obtaining the UV coordinates from face2.
        uv_proj = face2.point_to_parameter(point1)
        point2 = face2.point(uv_proj)
        
        # The distance along the normal direction is the dot product of the difference and normal.
        diff = np.array(point2) - np.array(point1)
        distance = abs(np.dot(diff, normal1))
        return distance