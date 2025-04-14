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

FINAL_COMBINED_FEATURES = {
    0: "Chamfer",
    1: "Through hole",
    2: "Triangular passage",
    3: "Rectangular passage",
    4: "Six-sided passage",
    5: "Triangular through slot",
    6: "General slot",
    7: "Circular through slot",
    8: "General step",
    9: "O Ring",
    10: "Blind hole",
    11: "Triangular pocket",
    12: "Rectangular pocket",
    13: "Six-sided pocket",
    14: "Circular end pocket",
    15: "Horizontal circular end blind slot",
    16: "Round",
    17: "Stock"
}

# Maps the old label ID -> new label ID
# This yields 18 unique labels total
LABEL_MERGE_MAP = {
    0: 0,   # chamfer
    1: 1,   # through hole
    2: 2,   # triangular passage
    3: 3,   # rectangular passage
    4: 4,   # six-sided passage
    5: 5,   # triangular through slot
    6: 6,   # (6,17,18) => general slot
    17: 6,
    18: 6,
    7: 7,   # circular through slot
    8: 8,   # (8,9,10,20,21,22) => general step
    9: 8,
    10: 8,
    20: 8,
    21: 8,
    22: 8,
    11: 9,   # O ring
    12: 10,  # blind hole
    13: 11,  # triangular pocket
    14: 12,  # rectangular pocket
    15: 13,  # six-sided pocket
    16: 14,  # circular end pocket
    19: 15,  # horizontal circular end blind slot
    23: 16,  # round
    24: 17   # stock
}
