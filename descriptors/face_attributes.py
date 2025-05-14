from dataclasses import dataclass, field
from typing import List, Optional
from mappings import SURFACE_TYPE_MAPPING


@dataclass # node in graph
class FaceAttributes:
    """
    Stores node (face) feature information, as in Table 1 of the paper.

    `label`: Integer (0-17)  -- Optional for training

    `surface_type`: Integer (0-11)   <-- as we are using other datasets for testing not only MFCAD
        ^--- see mappings.py 
    
    `surface_area`: Float (total face area).
    `surface_normal`: Float[3] ([x, y, z] unit vector).
    `bounding_box_ratio`: Float (width/height ratio).
    `outer_loop_adj_faces`: Float[33] (surface type x convexity).
    `outer_loop_c0_continuity`: Float[11] (C0 continuity by surface type).
    `outer_loop_perpendicular`: Float[11] (perpendicularity by surface type).
    `inner_loop`: Float[2] ([location, convexity]).
    `sign_gaussian_curvature`: Float (sign of Gaussian curvature).
    `mag_gaussian_curvature`: Float (magnitude of Gaussian curvature).
    `depth_ratio`: Float (depth ratio).
    """
    surface_type_size = len (SURFACE_TYPE_MAPPING)

    surface_type: int
    surface_area: float
    surface_normal: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounding_box_ratio: float = 0.0
    outer_loop_adj_faces: List[float] = field(default_factory=lambda size=surface_type_size: [0.0] * size * 3)
    outer_loop_c0_continuity: List[float] = field(default_factory=lambda size=surface_type_size: [0.0] * size)
    outer_loop_perpendicular: List[float] = field(default_factory=lambda size=surface_type_size: [0.0] * size)
    inner_loop: List[float] = field(default_factory=lambda: [0.0, 0.0])
    # Gaussian curvature signature
    sign_gaussian_curvature: float = 0.0
    mag_gaussian_curvature: float = 0.0
    depth_ratio: float = 0.0
    label: Optional[int] = None


    def to_dict(self) -> dict:
        """
        Convert descriptor to a dictionary (e.g. for JSON export).
        """
        return {
            "label": self.label,
            "surface_type": self.surface_type,
            "surface_area": self.surface_area,  
            "surface_normal": self.surface_normal,
            "bounding_box_ratio": self.bounding_box_ratio,
            "outer_loop_adj_faces": self.outer_loop_adj_faces,
            "outer_loop_c0_continuity": self.outer_loop_c0_continuity,
            "outer_loop_perpendicular": self.outer_loop_perpendicular,
            "inner_loop": self.inner_loop,
            "sign_gaussian_curvature": self.sign_gaussian_curvature,
            "mag_gaussian_curvature": self.mag_gaussian_curvature,
            "depth_ratio": self.depth_ratio
        }
