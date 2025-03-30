# missing 'any' from original BrepGAT paper
SURFACE_TYPE_MAPPING = {
    "unknown": 0,              # Fallback (default unknown)
    "plane": 1,                # GeomAbs_Plane
    "cylinder": 2,             # GeomAbs_Cylinder
    "cone": 3,                 # GeomAbs_Cone
    "sphere": 4,               # GeomAbs_Sphere
    "torus": 5,                # GeomAbs_Torus
    "bezier": 6,               # GeomAbs_BezierSurface
    "bspline": 7,              # GeomAbs_BSplineSurface
    "revolution": 8,           # GeomAbs_SurfaceOfRevolution
    "extrusion": 9,            # GeomAbs_SurfaceOfExtrusion
    "offset": 10,              # GeomAbs_OffsetSurface
    "rectangular_trimmed": 11, # Not explicitly in OpenCASCADE, but useful for BRep models
    "other": 12                # GeomAbs_OtherSurface (generic fallback)
}

CURVE_TYPE_MAPPING = {
    "unknown": 0,              # Fallback (default unknown)
    "line": 1,                 
    "circle": 2,               
    "ellipse": 3,             
    "hyperbola": 4,            
    "parabola": 5,             
    "bezier": 6,              
    "bspline": 7,              
    "offset": 8,               
    "conic": 9,                
    "extended_complex": 10,    
    "trimmed": 11,     
    "bounded": 12,        
    "other": 13                
}

INITIAL_MACHINING_FEATUTES_DICT = {
    "0": "Chamfer",
    "1": "Through hole",
    "2": "Triangular passage",
    "3": "Rectangular passage",
    "4": "Six-sided passage",
    "5": "Triangular through slot",
    "6": "Rectangular through slot",
    "7": "Circular through slot",
    "8": "Rectangular through step",
    "9": "Two-sided through step",
    "10": "Slanted through step",
    "11": "O Ring",
    "12": "Blind hole",
    "13": "Triangular pocket",
    "14": "Rectangular pocket",
    "15": "Six-sided pocket",
    "16": "Circular end pocket",
    "17": "Rectangular blind slot",
    "18": "Vertical circular end blind slot",
    "19": "Horizontal circular end blind slot",
    "20": "Triangular blind step",
    "21": "Circular blind step",
    "22": "Rectangular blind step",
    "23": "Round",
    "24": "Stock"
}