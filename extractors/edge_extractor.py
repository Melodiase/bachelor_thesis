from typing import List
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
            # Determine convexity using EdgeDataExtractor
            is_convex = self.check_edge_convexity(edge, face1, face2, angle_tol_rads)
            
            # Check for perpendicular and parallel faces
            # Get normal vectors
            face_extractor = FaceExtractor(self.solid)
            normal1 = np.array(face_extractor.compute_surface_normal(face1))
            normal2 = np.array(face_extractor.compute_surface_normal(face2))
            
            # Calculate dot product for angle between normals
            dot_product = abs(np.dot(normal1, normal2))
            
            # Perpendicular if dot product is close to 0
            is_perp = abs(dot_product) < 1e-3
            
            # Parallel if dot product is close to 1
            is_parallel = abs(dot_product - 1.0) < 1e-4
            
            # If faces are parallel, compute distance
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
        Creates an EdgeDataExtractor, checks convexity, and returns a boolean.
        Returns True if the edge is convex, False if concave or smooth.
        
        Args:
            edge: The edge to check
            face1: First face
            face2: Second face
            angle_tol_rads: Angle tolerance in radians
            
        Returns:
            bool: True if convex, False if concave or smooth
        """
        # Create EdgeDataExtractor within this function
        edge_extractor = EdgeDataExtractor(edge, [face1, face2])
        
        # Determine convexity
        convexity = edge_extractor.edge_convexity(angle_tol_rads)

        # Return True only if convex, False otherwise
        return convexity == EdgeConvexity.CONVEX

    def compute_parallel_distance(self, face1: Face, face2: Face) -> float:
        """
        If two faces are parallel, compute distance between them.
        
        Args:
            face1: First face
            face2: Second face
            
        Returns:
            float: Distance between the parallel faces
        """
        # TODO: Implement a proper calculation
        # In practice, you'd pick a point on face1 and face2 and measure 
        # the distance along the normal.
        
        # Sample implementation: 
        # 1. Get centroid of face1
        # 2. Get normal at that point
        # 3. Project a ray along that normal
        # 4. Find intersection with face2
        # 5. Measure distance
        
        # This is a placeholder - implement proper calculation
        return 20.0 