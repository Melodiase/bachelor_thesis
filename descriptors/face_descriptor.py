from dataclasses import dataclass, field
from typing import List, Optional

@dataclass # node in graph
class FaceDescriptor:
    """
    Stores node (face) feature information, as in Table 1 of the paper.

    `label`: Integer (0-17)  -- Optional for training

    `surface_type`: Integer (0-11)   <-- as we are using other datasets for testing
        Mapped as follows (0-based):
            0 = Unknown
            1 = Bezier surface
            2 = B-spline surface
            3 = Rectangular trimmed surface
            4 = Conical surface
            5 = Cylindrical surface
            6 = Plane
            7 = Spherical surface
            8 = Toroidal surface
            9 = Surface of linear extrusion
            10 = Surface of revolution
            11 = Any (generic fallback)
    `surface_area`: Float (total face area).
    `surface_normal`: Float[3] ([x, y, z] unit vector).
    `bounding_box_ratio`: Float (width/height ratio).
    `outer_loop_adj_faces`: Float[33] (surface type × convexity).
    `outer_loop_c0_continuity`: Float[11] (C0 continuity by surface type).
    `outer_loop_perpendicular`: Float[11] (perpendicularity by surface type).
    `inner_loop`: Float[2] ([location, convexity]).

    """

    label: Optional[int] = None
    surface_type: int
    surface_area: float
    surface_normal: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounding_box_ratio: float = 0.0
    outer_loop_adj_faces: List[float] = field(default_factory=lambda: [0.0] * 33)
    outer_loop_c0_continuity: List[float] = field(default_factory=lambda: [0.0] * 11)
    outer_loop_perpendicular: List[float] = field(default_factory=lambda: [0.0] * 11)
    inner_loop: List[float] = field(default_factory=lambda: [0.0, 0.0])


    def to_dict(self) -> dict:
        """
        Convert descriptor to a dictionary (e.g. for JSON export).
        """
        return {
            "label": self.label,
            "surface_type": self.surface_type,
            "surface_area": self.surface_area,  # +
            "surface_normal": self.surface_normal,
            "bounding_box_ratio": self.bounding_box_ratio,
            "outer_loop_adj_faces": self.outer_loop_adj_faces,
            "outer_loop_c0_continuity": self.outer_loop_c0_continuity,
            "outer_loop_perpendicular": self.outer_loop_perpendicular,
            "inner_loop": self.inner_loop
        }
