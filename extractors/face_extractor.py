from collections import defaultdict
from typing import List, Tuple
import numpy as np

from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid
from occwl.wire import Wire

from descriptors.face_attributes import FaceAttributes
from mappings import SURFACE_TYPE_MAPPING

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods_Face, topods_Wire
from OCC.Core.Geom import Geom_RectangularTrimmedSurface
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.GeomAbs import GeomAbs_C0
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_FORWARD, TopAbs_INTERNAL, TopAbs_EXTERNAL
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.gp import gp_Pnt2d, gp_Pln

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
        
        # Get outer and inner loops
        outer_wire, inner_wires = self.identify_outer_and_inner_wires(face)
        
        # Process outer loop properties as per paper:
        #   (i) Ratio of adjacent faces per (surface type, convexity) pair (33 bins)
        #   (ii) Ratio of adjacent faces with C0 continuity per surface type
        #   (iii) Ratio of adjacent faces that are perpendicular per surface type
        outer_loop_adj = self.compute_face_adj_type_convexity(face)
        outer_loop_c0 = self.compute_outer_loop_c0_continuity(face, outer_wire)
        outer_loop_perp = self.compute_outer_loop_perpendicular(face, outer_wire)
        
        # Process inner loop properties as per paper:
        # Store the "location type" and "convexity" of the inner loop.
        inner_loop_info = self.compute_inner_loop_info(face, inner_wires)
        
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

    def identify_outer_and_inner_wires(self, face: Face) -> Tuple[Wire, List[Wire]]:
        """
        Correctly identify the outer wire (boundary) and inner wires (holes) of a face.
        
        Args:
            face: The OCCWL face to analyze
            
        Returns:
            Tuple containing (outer_wire, list_of_inner_wires)
        """
        wires = list(face.wires())
        
        if len(wires) == 1:
            return wires[0], []
        
        outer_candidates = []
        inner_candidates = []
        
        for wire in wires:
            orientation = wire.topods_shape().Orientation()
            if orientation == TopAbs_FORWARD:
                outer_candidates.append(wire)
            else:
                inner_candidates.append(wire)
        
        if len(outer_candidates) == 1:
            return outer_candidates[0], inner_candidates
        
        if not outer_candidates and len(wires) > 0:
            wire_areas = [(wire, self._compute_wire_area(wire, face)) for wire in wires]
            wire_areas.sort(key=lambda x: x[1], reverse=True)
            outer_wire = wire_areas[0][0]
            inner_wires = [w for w, _ in wire_areas[1:]]
            return outer_wire, inner_wires
        
        if len(outer_candidates) > 1:
            outer_wire = None
            for wire in outer_candidates:
                for edge in wire.ordered_edges():
                    adjacent_faces = [f for f in self.solid.faces_from_edge(edge) if f != face]
                    if adjacent_faces:
                        outer_wire = wire
                        break
                if outer_wire:
                    break
            if outer_wire:
                inner_wires = [w for w in wires if w != outer_wire]
                return outer_wire, inner_wires
        
        return wires[0], wires[1:]

    def _compute_wire_area(self, wire: Wire, face: Face) -> float:
        """
        Computes the approximate area enclosed by a wire on a face.
        This heuristic is used to help identify the outer wire.
        
        For planar faces, a temporary face is constructed using the plane of the face
        and its area is computed. For non-planar faces, a perimeter-based area estimate
        is used as a fallback.
        
        Args:
            wire: The wire to compute the area for
            face: The face containing the wire
            
        Returns:
            Approximate area enclosed by the wire.
        """
        try:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.GeomAbs import GeomAbs_Plane
            
            adaptor = BRepAdaptor_Surface(face.topods_shape())
            if adaptor.GetType() == GeomAbs_Plane:
                # Create a temporary face bounded by the wire on the plane of the face
                plane_face = BRepBuilderAPI_MakeFace(adaptor.Plane(), wire.topods_shape()).Face()
                props = GProp_GProps()
                brepgprop_SurfaceProperties(plane_face, props)
                return props.Mass()
            
            # Fallback for non-planar surfaces: estimate area from perimeter
            perimeter = sum(edge.length() for edge in wire.ordered_edges())
            from math import pi
            return (perimeter * perimeter) / (4 * pi)
                
        except Exception as e:
            print(f"Warning: Wire area calculation failed: {e}. Using perimeter approximation.")
            perimeter = sum(edge.length() for edge in wire.ordered_edges())
            from math import pi
            return (perimeter * perimeter) / (4 * pi)

    def compute_inner_loop_info(self, face: Face, inner_wires: List[Wire]) -> List[float]:
        """
        Compute inner loop information as specified by the paper:
        store the location type and convexity of the inner loop.
        
        - Location type: computed as the normalized distance between the inner loop's centroid
          and the face's centroid.
        - Convexity: 1.0 if the projected inner loop (in UV space) is convex; 0.0 otherwise.
        
        If multiple inner wires exist, the average values are returned.
        If no inner wire exists, returns [0.0, 0.0].
        
        Returns:
            List[float]: [location_type, convexity]
        """
        if not inner_wires:
            return [0.0, 0.0]
        
        # Compute face center in 3D from the UV center of the face
        face_uv_center = face.uv_bounds().center()
        face_center = face.point(face_uv_center)
        
        # Get face bounding box diagonal (in 3D)
        from occwl.geometry.box import Box
        bbox = face.box()
        diag = np.linalg.norm(bbox.diagonal())
        if diag == 0:
            diag = 1.0
        
        location_types = []
        convexities = []
        
        # Helper to test convexity of a polygon in 2D (UV space)
        def is_convex(points: List[np.ndarray]) -> float:
            if len(points) < 4:
                return 1.0
            sign = 0
            n = len(points)
            for i in range(n):
                dx1 = points[(i+1)%n][0] - points[i][0]
                dy1 = points[(i+1)%n][1] - points[i][1]
                dx2 = points[(i+2)%n][0] - points[(i+1)%n][0]
                dy2 = points[(i+2)%n][1] - points[(i+1)%n][1]
                cross = dx1 * dy2 - dy1 * dx2
                current_sign = 1 if cross > 0 else -1 if cross < 0 else 0
                if current_sign != 0:
                    if sign == 0:
                        sign = current_sign
                    elif sign != current_sign:
                        return 0.0
            return 1.0
        
        for wire in inner_wires:
            # Compute inner loop centroid in 3D from its vertices
            vertices = list(wire.ordered_vertices())
            if not vertices:
                continue
            pts = np.array([v.point() for v in vertices])
            inner_centroid = np.mean(pts, axis=0)
            loc_dist = np.linalg.norm(inner_centroid - face_center)
            location_types.append(loc_dist / diag)
            
            # Project vertices to UV space using face.point_to_parameter
            uv_points = [face.point_to_parameter(v.point()) for v in vertices]
            convexities.append(is_convex(uv_points))
        
        avg_location = float(np.mean(location_types)) if location_types else 0.0
        avg_convexity = float(np.mean(convexities)) if convexities else 0.0
        return [avg_location, avg_convexity]

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
            if self._is_rectangular_trimmed_surface(face):
                surface_type = "rectangular_trimmed"
        return SURFACE_TYPE_MAPPING.get(surface_type, 0)
    
    # CHECK
    def _is_rectangular_trimmed_surface(self, face: Face) -> bool:
        """
        Check if the given face is a rectangular trimmed surface.
        
        Args:
            face: The OCCWL face to check
            
        Returns:
            bool: True if it's a rectangular trimmed surface
        """
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.Geom import Geom_RectangularTrimmedSurface
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
            epsilon = 1e-8
            max_side = max(width, height)
            min_side = min(width, height)
            if max_side < epsilon:
                return 1.0
            return min_side / max_side
        except Exception as e:
            print(f"Warning: Unable to compute UV bounding box ratio. Error: {e}")
            return 1.0
        
    def compute_outer_loop_c0_continuity(self, face: Face, outer_wire: Wire = None) -> List[float]:
        """
        Find adjacent faces and compute C0 continuity ratio by surface type.
        
        Returns:
            List[float]: Array of C0 continuity ratios by surface type
        """
        if outer_wire is None:
            outer_wire, _ = self.identify_outer_and_inner_wires(face)
        outer_edges = list(outer_wire.ordered_edges())
        total_adj_faces = 0
        c0_continuity_count = defaultdict(int)
        for edge in outer_edges:
            adjacent_faces = [f for f in self.solid.faces_from_edge(edge) if f != face]
            for adj_face in adjacent_faces:
                total_adj_faces += 1
                if edge.continuity(face, adj_face) == GeomAbs_C0:
                    st_id = self.classify_surface_type(adj_face)
                    c0_continuity_count[st_id] += 1
        num_surface_types = max(SURFACE_TYPE_MAPPING.values()) + 1
        continuity_array = [0.0] * num_surface_types
        if total_adj_faces > 0:
            for surface_idx, count in c0_continuity_count.items():
                continuity_array[surface_idx] = count / total_adj_faces
        return continuity_array

    def compute_outer_loop_perpendicular(self, face: Face, outer_wire: Wire = None) -> List[float]:
        """
        Identify adjacent faces that share the outer loop edges, determine if they're perpendicular,
        and store the ratio by surface type.
        
        Returns:
            List[float]: Array of perpendicularity ratios by surface type
        """
        if outer_wire is None:
            outer_wire, _ = self.identify_outer_and_inner_wires(face)
        outer_edges = list(outer_wire.ordered_edges())
        total_adj_faces = 0
        perpendicular_count = defaultdict(int)
        ref_normal = np.array(self.compute_surface_normal(face))
        for edge in outer_edges:
            adjacent_faces = [adj_face for adj_face in self.solid.faces_from_edge(edge) if adj_face != face]
            for adj_face in adjacent_faces:
                total_adj_faces += 1
                adj_normal = np.array(self.compute_surface_normal(adj_face))
                if self._are_normals_perpendicular(ref_normal, adj_normal):
                    stype_id = self.classify_surface_type(adj_face)
                    perpendicular_count[stype_id] += 1
        num_surface_types = max(SURFACE_TYPE_MAPPING.values()) + 1
        perp_array = [0.0] * num_surface_types
        if total_adj_faces > 0:
            for stype_idx, count in perpendicular_count.items():
                perp_array[stype_idx] = count / total_adj_faces
        return perp_array

    def _are_normals_perpendicular(self, n1: np.ndarray, n2: np.ndarray, tol=1e-3) -> bool:
        """
        Consider two normalized vectors perpendicular if their dot product is near zero.
        
        Returns:
            bool: True if the normals are perpendicular
        """
        dot_val = np.dot(n1, n2)
        return abs(dot_val) < tol

    def compute_face_adj_type_convexity(self, face: Face, angle_tol_rads=0.1) -> List[float]:
        """
        For a given face, compute a 33-length array describing adjacency counts per 
        (surface_type, convexity) pair across all edges.
        
        Returns:
            List[float]: Array of adjacency ratios by (surface_type, convexity) pairs.
        """
        NUM_SURFACE_TYPES = len(SURFACE_TYPE_MAPPING)
        NUM_CONVEXITY_STATES = 3  # 0: SMOOTH/UNKNOWN, 1: CONVEX, 2: CONCAVE
        adjacency_bins = [0.0] * (NUM_SURFACE_TYPES * NUM_CONVEXITY_STATES)
        total_adjacencies = 0
        outer_wire, _ = self.identify_outer_and_inner_wires(face)
        edges = list(outer_wire.ordered_edges())
        for edge in edges:
            connected_faces = [adj_face for adj_face in self.solid.faces_from_edge(edge) if adj_face != face]
            if len(connected_faces) < 1:
                continue
            for adj_face in connected_faces:
                total_adjacencies += 1
                st_idx = self.classify_surface_type(adj_face)
                if st_idx < 0 or st_idx >= NUM_SURFACE_TYPES:
                    st_idx = 0
                from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
                extractor = EdgeDataExtractor(edge, [face, adj_face])
                conv_idx = 0  # Default SMOOTH/UNKNOWN
                if extractor.good:
                    ctype = extractor.edge_convexity(angle_tol_rads)
                    if ctype == EdgeConvexity.CONVEX:
                        conv_idx = 1
                    elif ctype == EdgeConvexity.CONCAVE:
                        conv_idx = 2
                # If extractor.good is False, we keep the default conv_idx = 0
                bin_index = st_idx * NUM_CONVEXITY_STATES + conv_idx
                adjacency_bins[bin_index] += 1.0
        if total_adjacencies > 0:
            adjacency_bins = [count / total_adjacencies for count in adjacency_bins]
        return adjacency_bins
