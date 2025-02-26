from dataclasses import dataclass
from typing import List

@dataclass
class EdgeDescriptor:
    """
    Stores edge (b_edge) feature information, as in Table 2 of the paper.

    curve_type : Unknown, B-spline, Bezier curve, trimmed curve, bounded curve, circle, ellipse,
      hyperbola, parabola, conic, line, offset curve, extended_complex curve, any 
    
    distance: If parallel is true, this attribute can have value.   

    """
    curve_type: int
    curve_length: float
    convexity: bool
    perpendicular: bool
    parallel: bool
    distance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "curve_type": self.curve_type,
            "curve_length": self.curve_length,
            "convexity": self.convexity,
            "perpendicular": self.perpendicular,
            "parallel": self.parallel,
            "distance": self.distance
        }
