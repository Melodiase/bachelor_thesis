from collections import defaultdict
from typing import List, Tuple
import numpy as np

from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid
from occwl.wire import Wire
from occwl.uvgrid import uvgrid

from descriptors.face_attributes import FaceAttributes
from mappings import SURFACE_TYPE_MAPPING

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopoDS import topods_Face, topods_Wire
from OCC.Core.Geom import Geom_RectangularTrimmedSurface
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.GeomAbs import GeomAbs_C0
from math import acos, degrees
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_FORWARD, TopAbs_INTERNAL, TopAbs_EXTERNAL
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.gp import gp_Pnt2d, gp_Pln

# Helper function to compute the angle between two normalized vectors.
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle between two normalized vectors.
    
    Args:
        v1: First normalized vector
        v2: Second normalized vector
        
    Returns:
        float: Angle between vectors in radians
    """
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)

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
        # flag so we only compute the bbox once
        self._bbox_cached = False

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
        outer_loop_adj = self.compute_face_adj_type_convexity(face)
        outer_loop_c0 = self.compute_outer_loop_c0_continuity(face, outer_wire)
        outer_loop_perp = self.compute_outer_loop_perpendicular(face, outer_wire)
        
        # Process inner loop properties as per paper:
        inner_loop_info = self.compute_inner_loop_info(face, inner_wires)

        signK, magK = self.compute_gaussian_curvature(face)

        dr = self.compute_depth_ratio(face)
        mda = self.compute_mean_dihedral(face)
        cham= self.compute_chamfer_angle_norm(face)
        
        return FaceAttributes(
            surface_type=surface_type_id,
            surface_area=area,
            surface_normal=normal_list,
            bounding_box_ratio=bounding_box_ratio,
            outer_loop_adj_faces=outer_loop_adj,
            outer_loop_c0_continuity=outer_loop_c0,
            outer_loop_perpendicular=outer_loop_perp,
            inner_loop=inner_loop_info,
            sign_gaussian_curvature=signK,
            mag_gaussian_curvature=magK,
            depth_ratio=dr,
            mean_dihedral_angle=mda,
            chamfer_angle_norm = cham
        )

    def identify_outer_and_inner_wires(self, face: Face) -> Tuple[Wire, List[Wire]]:
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
        In addition, if the normals computed at the face centers of the two adjacent faces are nearly parallel,
        re-compute the normals at the centroid of the shared edge.
        
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
                # Check if face centers provide nearly parallel normals.
                center_uv_face = face.uv_bounds().center()
                center_uv_adj = adj_face.uv_bounds().center()
                normal_face_center = np.array(face.normal(center_uv_face))
                normal_adj_center = np.array(adj_face.normal(center_uv_adj))
                if angle_between(normal_face_center, normal_adj_center) < 0.1:
                    # Compute centroid of the shared edge.
                    try:
                        vertices = list(edge.ordered_vertices())
                    except AttributeError:
                        vertices = [edge.start_vertex(), edge.end_vertex()]
                    if vertices:
                        pts = np.array([v.point() for v in vertices])
                        centroid = np.mean(pts, axis=0)
                        # Reproject centroid onto each face to get UV parameters.
                        uv_face = face.point_to_parameter(centroid)
                        uv_adj = adj_face.point_to_parameter(centroid)
                        n_face = np.array(face.normal(uv_face))
                        n_adj = np.array(adj_face.normal(uv_adj))
                    else:
                        n_face = normal_face_center
                        n_adj = normal_adj_center
                else:
                    # Use the edge_data_extractor from occwl.
                    from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
                    extractor = EdgeDataExtractor(edge, [face, adj_face])
                    if extractor.good:
                        ctype = extractor.edge_convexity(angle_tol_rads)
                        if ctype == EdgeConvexity.CONVEX:
                            conv_idx = 1
                        elif ctype == EdgeConvexity.CONCAVE:
                            conv_idx = 2
                        else:
                            conv_idx = 0
                        bin_index = st_idx * NUM_CONVEXITY_STATES + conv_idx
                        adjacency_bins[bin_index] += 1.0
                        continue  # Process next adjacent face
                    else:
                        n_face = normal_face_center
                        n_adj = normal_adj_center
                
                # When using re-computed normals at the edge centroid:
                # Determine convexity manually.
                # Use the tangent at the midpoint of the edge parameter domain.
                mid_u = (edge.u_bounds().a + edge.u_bounds().b) / 2.0
                tangent = np.array(edge.tangent(mid_u))
                cross_prod = np.cross(n_face, n_adj)
                dot_val = np.dot(cross_prod, tangent)
                conv_idx = 1 if dot_val > 0 else 2  # 1: Convex, 2: Concave
                bin_index = st_idx * NUM_CONVEXITY_STATES + conv_idx
                adjacency_bins[bin_index] += 1.0
        
        if total_adjacencies > 0:
            adjacency_bins = [count / total_adjacencies for count in adjacency_bins]
        return adjacency_bins

    def _ensure_bbox(self):
        if getattr(self, "_bbox_cached", False):
            return

        bbox_geom   = self.solid.box()            # occwl.geometry.box.Box
        self._bbox_min = bbox_geom.min_point()    # np.array([x,y,z])
        self._bbox_max = bbox_geom.max_point()
        self._bbox_diag_sq  = float(np.sum((self._bbox_max - self._bbox_min) ** 2))
        self._bbox_height   = float(abs(self._bbox_max[2] - self._bbox_min[2]))

        self._bbox_cached = True

    # ----------------------------------------------------------------------
    #  Gaussian curvature signature  (sign ∈ {-1,0,+1},  mag ∈ (0,1] )
    # ----------------------------------------------------------------------
    def compute_gaussian_curvature(self, face: Face) -> Tuple[float, float]:
        """
        Return two scale-free scalars per face:
            signK  :  -1, 0, +1          (convex / developable / saddle)
            magK   :  tanh(|K̄| · d²)    (0…1, clipped smoothly)

        * Fast analytic shortcut for planes / cylinders / spheres.
        * 3x3 UV sampling (with progressive inset) for everything else.
        * Median aggregation is used for robustness.
        """
        # ---------- analytic surfaces first (cheap) ----------
        stype = face.surface_type()
        if stype in ("plane", "cylinder", "cone", "extrusion"):
            return 0.0, 0.0
        if stype == "sphere":
            self._ensure_bbox()
            r     = face.specific_surface().Radius()
            k_val = 1.0 / (r * r)
            mag   = np.tanh(k_val * self._bbox_diag_sq)
            return 1.0, mag  # sphere curvature is always positive

        # ---------- numeric sampling for the rest ----------
        from OCC.Core.BRepTools import breptools_UVBounds
        umin, umax, vmin, vmax = breptools_UVBounds(face.topods_shape())
        MIN_SAMPLES = 5

        for inset_frac in (0.10, 0.05, 0.0):
            du = (umax - umin) * inset_frac
            dv = (vmax - vmin) * inset_frac
            us = np.linspace(umin + du, umax - du, 3)
            vs = np.linspace(vmin + dv, vmax - dv, 3)
            uu, vv = np.meshgrid(us, vs)
            k_vals = []

            for u, v in np.column_stack([uu.ravel(), vv.ravel()]):
                try:
                    k_vals.append(face.gaussian_curvature((u, v)))
                except RuntimeError as e:
                    if "LProp_NotDefined" in str(e):
                        continue
                    raise

            if len(k_vals) >= MIN_SAMPLES:
                break

        # robust aggregate
        K_avg = float(np.median(k_vals)) if k_vals else 0.0
        signK = float(np.sign(K_avg))

        self._ensure_bbox()
        magK  = np.tanh(abs(K_avg) * self._bbox_diag_sq)  # 0…1

        return signK, magK

    # ----------------------------------------------------------------------
    #  Depth ratio  (0 at highest exposed faces  →  1 at deepest)
    # ----------------------------------------------------------------------
    def compute_depth_ratio(self, face: Face) -> float:
        """
        Generic depth measure independent of part size and orientation.

        If a custom reference axis has been set in self._depth_axis
        (e.g. PCA direction), use it; otherwise fall back to global +Z.
        """
        self._ensure_bbox()

        # -- centroid via OCC surface properties (handles trims robustly)
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

        props = GProp_GProps()
        brepgprop_SurfaceProperties(face.topods_shape(), props)
        centroid = np.array(props.CentreOfMass().Coord(), dtype=float)

        # -- choose axis
        axis_dir = getattr(self, "_depth_axis", np.array([0.0, 0.0, 1.0]))
        axis_dir = axis_dir / np.linalg.norm(axis_dir)

        # -- project centroid and extremes onto axis
        d_top    = np.dot(self._bbox_max, axis_dir)
        d_bottom = np.dot(self._bbox_min, axis_dir)
        d_face   = np.dot(centroid,      axis_dir)

        denom = d_top - d_bottom
        if denom <= 1e-9:
            return 0.0

        depth_ratio = (d_top - d_face) / denom
        return float(np.clip(depth_ratio, 0.0, 1.0))

    def compute_mean_dihedral(self, face: Face) -> float:
        """
        0 → coplanar; ~0.5 → 90°; 1 → 180° folds.

        For the given face, gather its adjacent faces via shared boundary edges,
        compute the unsigned dihedral angle at each junction, and return
        the mean angle normalized to [0,1], with special handling so that
        cylindrical side faces (which only have planar-cap neighbours) yield 0.
        """
        import math
        import numpy as np
        from mappings import SURFACE_TYPE_MAPPING

        # 1) Compute this face's normal at a robust internal point
        n0 = np.array(self.compute_surface_normal(face), dtype=float)
        norm0 = np.linalg.norm(n0)
        if norm0 < 1e-8:
            return 0.0
        n0 /= norm0

        # 2) Gather unique neighbours via each edge of the outer wire
        root_shape = face.topods_shape()
        outer_wire, _ = self.identify_outer_and_inner_wires(face)

        neighbours: List[Face] = []
        for edge in outer_wire.ordered_edges():
            for nbr in self.solid.faces_from_edge(edge):
                # skip the face itself
                if nbr.topods_shape().IsSame(root_shape):
                    continue
                # skip duplicates
                if any(nbr.topods_shape().IsSame(existing.topods_shape())
                        for existing in neighbours):
                    continue
                neighbours.append(nbr)

        # 3) For cylindrical side faces, ignore planar-cap neighbours
        stype_id = self.classify_surface_type(face)
        if stype_id == SURFACE_TYPE_MAPPING["cylinder"]:
            neighbours = [
                nbr for nbr in neighbours
                if self.classify_surface_type(nbr) == stype_id
            ]

        # 4) Compute unsigned dihedral angles ∈ [0, π/2]
        thetas: List[float] = []
        for nbr in neighbours:
            n1 = np.array(self.compute_surface_normal(nbr), dtype=float)
            norm1 = np.linalg.norm(n1)
            if norm1 < 1e-8:
                continue
            n1 /= norm1

            dot = abs(float(np.dot(n0, n1)))
            dot = float(np.clip(dot, 0.0, 1.0))
            thetas.append(math.acos(dot))

        # 5) If no valid neighbours, return 0
        if not thetas:
            return 0.0

        # 6) Return mean angle normalized by (π/2)
        mean_theta = float(sum(thetas) / len(thetas))
        return min(1.0, mean_theta / (math.pi / 2.0))


    def compute_chamfer_angle_norm(
        self,
        face: Face,
        stock_area_threshold: float = 1e6,
        consistency_tol_deg: float = 3.0
    ) -> float:
        """
        Return a scalar in [0, 1] that encodes the chamfer angle of a
        planar face, or 0 if the face is not a consistent chamfer.

        * 0          → pocket floor / stock plane
        * ≈0.50      → 45° chamfer
        * 1          → vertical wall (90°)

        """
        # -------------------------------------------------- 1. Planarity
        if face.surface_type() != "plane":
            return 0.0

        # -------------------------------------------------- 2. Skip stock
        if face.area() > stock_area_threshold:
            return 0.0

        # -------------------------------------------------- 3. Find sharp C0 edges
        angles = []
        outer_wire, _ = self.identify_outer_and_inner_wires(face)

        for edge in outer_wire.ordered_edges():
            # adjacent faces
            nbrs = list(self.solid.faces_from_edge(edge))
            if len(nbrs) != 2:
                continue

            # keep only truly sharp (C0) edges
            if edge.continuity(nbrs[0], nbrs[1]) != GeomAbs_C0:
                continue

            # determine "other" face
            other = nbrs[1] if nbrs[0].topods_shape().IsSame(face.topods_shape()) else nbrs[0]

            # Get 3D midpoint of the edge using OCCWL API
            u_interval = edge.u_bounds()
            if u_interval.invalid():
                continue
            mid_u = u_interval.middle()
            p3d = edge.point(mid_u)

            # Get UV coordinates for the 3D point on each face
            try:
                uv_face = face.point_to_parameter(p3d)
                uv_other = other.point_to_parameter(p3d)
            except:
                # If projection fails, skip this edge
                continue

            # Get outward normals using correct OCCWL API
            try:
                n0 = np.asarray(face.normal(uv_face), dtype=float)
                n1 = np.asarray(other.normal(uv_other), dtype=float)
            except:
                # If normal computation fails, skip this edge
                continue

            # Ensure normals are valid
            if np.linalg.norm(n0) < 1e-8 or np.linalg.norm(n1) < 1e-8:
                continue

            # Normalize normals
            n0 = n0 / np.linalg.norm(n0)
            n1 = n1 / np.linalg.norm(n1)

            # Compute unsigned dihedral angle (0–90 deg)
            cos_theta = abs(np.dot(n0, n1))
            cos_theta = max(-1.0, min(1.0, cos_theta))       # robust clamp
            theta_deg = degrees(acos(cos_theta))
            if theta_deg > 90.0:
                theta_deg = 180.0 - theta_deg
            angles.append(theta_deg)

        # -------------------------------------------------- 4. No sharp edges
        if not angles:
            return 0.0

        # -------------------------------------------------- 5. Consistency check
        if max(angles) - min(angles) > consistency_tol_deg:
            return 0.0

        # -------------------------------------------------- 6. Normalize mean angle
        mean_angle = float(sum(angles) / len(angles))        # degrees
        return min(1.0, mean_angle / 90.0)
