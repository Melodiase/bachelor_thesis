from collections import defaultdict
from typing import List
import numpy as np

from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid

from descriptors.face_attributes import FaceAttributes
from mappings import SURFACE_TYPE_MAPPING

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods_Face
from OCC.Core.Geom import Geom_RectangularTrimmedSurface
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.TopAbs import TopAbs_REVERSED


class FaceExtractor:
    """
    Extracts face features and creates face descriptors.
    """

    def __init__(self, solid: Solid):
        """
        Initialize with a parent solid for adjacency queries.
        
        Args:
            solid: The OCCWL Solid to extract features from
        """
        self.solid = solid

    def get_face_descriptor(self, face: Face) -> FaceAttributes:
        """
        Create a FaceDescriptor from an OCCWL Face.
        
        Args:
            face: The OCCWL Face to extract features from
            
        Returns:
            A complete FaceDescriptor object
        """
        surface_type_id = self.classify_surface_type(face)
        area = face.area()
        normal_list = self.compute_surface_normal(face)
        bounding_box_ratio = self.compute_uv_bounding_box_ratio(face)
        outer_loop_c0 = self.compute_outer_loop_c0_continuity(face)
        outer_loop_perp = self.compute_outer_loop_perpendicular(face)
        
        # TODO: Implement these properly if needed
        outer_loop_adj = [0.0] * 3 * len(SURFACE_TYPE_MAPPING) 
        inner_loop_info = [0.0, 0.0]
        
        return FaceAttributes(
            surface_type=surface_type_id,
            surface_area=area,
            surface_normal=normal_list,
            bounding_box_ratio=bounding_box_ratio,
            outer_loop_adj_faces=outer_loop_adj,
            outer_loop_c0_continuity=outer_loop_c0,
            outer_loop_perpendicular=outer_loop_perp,
            inner_loop=inner_loop_info
        )

    def classify_surface_type(self, face: Face) -> int:
        """
        Classify the Face's surface into an integer ID based on Table 1.
        See mappings.py -> SURFACE_TYPE_MAPPING

        Args:
            face: The OCCWL face to classify
            
        Returns:
            int: An integer identifier (0..11)
        """
        surface_type = face.surface_type()

        if surface_type == "unknown":
            # Additional check for rectangular trimmed surface
            if self._is_rectangular_trimmed_surface(face):
                surface_type = "rectangular_trimmed"
        
        return SURFACE_TYPE_MAPPING.get(surface_type, 0)
    
    def _is_rectangular_trimmed_surface(self, face: Face) -> bool:
        """
        Check if the given face is a rectangular trimmed surface.
        
        Args:
            face: The OCCWL face to check
            
        Returns:
            bool: True if it's a rectangular trimmed surface
        """
        surface = BRep_Tool.Surface(topods_Face(face.topods_shape()))
        return Geom_RectangularTrimmedSurface.DownCast(surface) is not None

    def compute_surface_normal(self, face: Face) -> List[float]:
        """
        Compute the normal of the surface at the center of its UV domain.
        
        Args:
            face: The OCCWL face to compute normal for
            
        Returns:
            List[float]: The normal vector as a list [x, y, z]
        """
        uv_center = face.uv_bounds().center()
        normal = face.normal(uv_center)
        return normal.tolist()

    def compute_uv_bounding_box_ratio(self, face: Face) -> float:
        """
        Compute the bounding box ratio (shorter side / longer side) in the UV parameter space.

        Returns:
            float: The bounding box aspect ratio between U and V directions (always ≤ 1).
        """
        try:
            umin, umax, vmin, vmax = breptools_UVBounds(face.topods_shape())
            width = abs(umax - umin)
            height = abs(vmax - vmin)

            epsilon = 1e-8  # Avoid division by zero
            max_side = max(width, height)
            min_side = min(width, height)

            if max_side < epsilon:
                return 1.0  # Fallback ratio

            return min_side / max_side  # Always ≤ 1

        except Exception as e:
            print(f"Warning: Unable to compute UV bounding box ratio. Error: {e}")
            return 1.0
        
    def compute_outer_loop_c0_continuity(self, face: Face) -> List[float]:
        """
        Find adjacent faces and compute C0 continuity ratio by surface type.
        
        Args:
            face: The OCCWL face to compute C0 continuity for
            
        Returns:
            List[float]: Array of C0 continuity ratios by surface type
        """
        edges = list(face.edges())
        total_adj_faces = 0
        c0_continuity_count = defaultdict(int)

        for edge in edges:
            # Use the SOLID to find all faces that share 'edge'
            adjacent_faces = [f for f in self.solid.faces_from_edge(edge) if f != face]

            for adj_face in adjacent_faces:
                total_adj_faces += 1
                if edge.continuity(face, adj_face) == GeomAbs_C0:
                    st_id = self.classify_surface_type(adj_face)
                    c0_continuity_count[st_id] += 1

        # Suppose your SURFACE_TYPE_MAPPING has size 'N'
        num_surface_types = max(SURFACE_TYPE_MAPPING.values()) + 1
        continuity_array = [0.0] * num_surface_types

        if total_adj_faces > 0:
            for surface_idx, count in c0_continuity_count.items():
                continuity_array[surface_idx] = count / total_adj_faces

        return continuity_array

    def compute_outer_loop_perpendicular(self, face: Face) -> List[float]:
        """
        Identify adjacent faces that share the OUTER loop edges,
        determine if they're perpendicular, and store the ratio by surface type.
        
        Args:
            face: The OCCWL face to compute perpendicularity for
            
        Returns:
            List[float]: Array of perpendicularity ratios by surface type
        """
        # 1) Identify the wire(s) that is/are the outer loop
        outer_wires = []
        for wire in face.wires():
            if wire.topods_shape().Orientation() == TopAbs_REVERSED:
                outer_wires.append(wire)
                
        # 2) Collect edges from these outer wires only
        outer_edges = []
        for wire in outer_wires:
            for edge in wire.ordered_edges():
                outer_edges.append(edge)

        total_adj_faces = 0
        perpendicular_count = defaultdict(int)

        # Reference normal for main face
        ref_normal = np.array(self.compute_surface_normal(face))

        # 3) Loop over each "outer" edge to find adjacent faces
        for edge in outer_edges:
            adjacent_faces = [adj_face for adj_face in self.solid.faces_from_edge(edge) if adj_face != face]
            
            for adj_face in adjacent_faces:
                total_adj_faces += 1
                # Compute normal of adjacent face
                adj_normal = np.array(self.compute_surface_normal(adj_face))
                # Check if perpendicular
                if self._are_normals_perpendicular(ref_normal, adj_normal):
                    stype_id = self.classify_surface_type(adj_face)
                    perpendicular_count[stype_id] += 1

        # 4) Build ratio array
        num_surface_types = max(SURFACE_TYPE_MAPPING.values()) + 1
        perp_array = [0.0] * num_surface_types

        if total_adj_faces > 0:
            for stype_idx, count in perpendicular_count.items():
                perp_array[stype_idx] = count / total_adj_faces

        return perp_array

    def _are_normals_perpendicular(self, n1: np.ndarray, n2: np.ndarray, tol=1e-3) -> bool:
        """
        Consider two normalized vectors perpendicular if their dot product is near zero.
        
        Args:
            n1: First normal vector
            n2: Second normal vector
            tol: Tolerance for dot product comparison
            
        Returns:
            bool: True if the normals are perpendicular
        """
        dot_val = np.dot(n1, n2)
        return abs(dot_val) < tol

    def compute_face_adj_type_convexity(self, face: Face, angle_tol_rads=0.1) -> List[float]:
        """
        For a given face, compute a 33-length array describing adjacency counts 
        per (surface_type, convexity) pair across ALL edges.
        
        Args:
            face: The OCCWL face to compute convexity for
            angle_tol_rads: Tolerance for convexity classification
            
        Returns:
            List[float]: Array of adjacency ratios by (surface_type, convexity) pairs
        """
        # 11 surface types × 3 convexity states => 33 bins
        NUM_SURFACE_TYPES = len(SURFACE_TYPE_MAPPING)
        NUM_CONVEXITY_STATES = 3  # e.g. 0=SMOOTH/UNKNOWN, 1=CONVEX, 2=CONCAVE
        adjacency_bins = [0.0] * (NUM_SURFACE_TYPES * NUM_CONVEXITY_STATES)
        
        total_adjacencies = 0

        # 1) Get ALL edges from 'face'
        edges = list(face.edges())

        for edge in edges:
            # 2) Find all faces that share this edge
            connected_faces = [adj_face for adj_face in self.solid.faces_from_edge(edge) if adj_face != face]
            if len(connected_faces) < 1:
                continue

            # 3) For each adjacent face that isn't 'face', gather surface type + convexity
            for adj_face in connected_faces:
                if adj_face == face:
                    continue  # skip if it's the same face
                total_adjacencies += 1

                # 3.1) Surface Type
                st_idx = self.classify_surface_type(adj_face)  # 0..10
                if st_idx < 0 or st_idx >= NUM_SURFACE_TYPES:
                    st_idx = 0  # fallback

                # 3.2) Convexity (0=SMOOTH, 1=CONVEX, 2=CONCAVE)
                from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
                
                extractor = EdgeDataExtractor(edge, [face, adj_face])
                ctype = extractor.edge_convexity(angle_tol_rads)
                
                conv_idx = 0  # Default SMOOTH/UNKNOWN
                if ctype == EdgeConvexity.CONVEX:
                    conv_idx = 1
                elif ctype == EdgeConvexity.CONCAVE:
                    conv_idx = 2

                # 3.3) increment the bin
                bin_index = st_idx * NUM_CONVEXITY_STATES + conv_idx
                adjacency_bins[bin_index] += 1.0

        # 4) Convert to ratios
        if total_adjacencies > 0:
            adjacency_bins = [count / total_adjacencies for count in adjacency_bins]

        return adjacency_bins 