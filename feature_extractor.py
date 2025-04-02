# my_brep_to_graph/feature_extractor/feature_extractor.py

from collections import defaultdict
from typing import List
import math
import numpy as np
import warnings

from occwl.geometry.geom_utils import to_numpy
from occwl.edge_data_extractor import EdgeDataExtractor

from occwl.face import Face
from occwl.edge import Edge
from occwl.solid import Solid
from occwl.shape import Shape
from occwl.compound import Compound

from descriptors.face_attributes import FaceAttributes
from descriptors.edge_attributes import EdgeConvexity, EdgeAttributes

from OCC.Core.BRep import BRep_Tool, BRep_Tool_Surface
from OCC.Core.TopoDS import topods_Face, topods_Edge
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.Geom import Geom_RectangularTrimmedSurface
from OCC.Core.BRepTools import breptools_UVBounds

from mappings import SURFACE_TYPE_MAPPING, CURVE_TYPE_MAPPING

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.GeomAbs import (GeomAbs_BezierSurface, GeomAbs_BSplineSurface,
                              GeomAbs_C0, GeomAbs_C1, GeomAbs_C2, GeomAbs_C3,
                              GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_G1,
                              GeomAbs_G2, GeomAbs_OffsetSurface,
                              GeomAbs_OtherSurface, GeomAbs_Plane,
                              GeomAbs_Sphere, GeomAbs_SurfaceOfExtrusion,
                              GeomAbs_SurfaceOfRevolution, GeomAbs_Torus)
from OCC.Core.Geom import (
    Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface,
    Geom_SphericalSurface, Geom_ToroidalSurface, Geom_BSplineSurface
)
from OCC.Core.Geom import (
    Geom_TrimmedCurve, Geom_BoundedCurve, Geom_Conic
)
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Dir, gp_Pnt, gp_Pnt2d, gp_TrsfForm
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_REVERSED, TopAbs_FORWARD
from OCC.Core.TopAbs import TopAbs_INTERNAL, TopAbs_EXTERNAL
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Face

from occwl.base import BoundingBoxMixin, TriangulatorMixin, WireContainerMixin, \
    EdgeContainerMixin, VertexContainerMixin, SurfacePropertiesMixin

import occwl.geometry.geom_utils as geom_utils
import occwl.geometry.interval as Interval
from occwl.geometry.box import Box

from extractors.face_extractor import FaceExtractor
from extractors.edge_extractor import EdgeExtractor

class FeatureExtractor:
    """
    Main extractor class that combines face and edge feature extraction.
    
    This class acts as a facade to the specialized extractors, providing
    a simple interface to get face and edge descriptors.
    """

    def __init__(self, solid: Solid):
        self.solid = solid
        
        # Initialize specialized extractors
        self.face_extractor = FaceExtractor(solid)
        self.edge_extractor = EdgeExtractor(solid)


    def get_face_descriptor(self, face: Face) -> FaceAttributes:
        """
        Create a FaceDescriptor from an OCCWL Face.
        
        Args:
            face: The OCCWL Face to extract features from
            
        Returns:
            A complete FaceDescriptor object
        """
        return self.face_extractor.get_face_descriptor(face)


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
        return self.edge_extractor.get_edge_descriptor(edge, face1, face2, angle_tol_rads)