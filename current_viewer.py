import pathlib
import re
from occwl.edge import Edge
from occwl.face import Face
from occwl.vertex import Vertex
from occwl.viewer import Viewer
from occwl.compound import Compound

from mappings import INITIAL_MACHINING_FEATUTES_DICT

STEP_FILE = "C:/Users/synkh/BachelorThesis/original_datasets/MFCAD++_dataset/step/val/1154.step"

v = Viewer(backend="wx")

def select_vertex(event=None):
    v.selection_mode_none()
    v.selection_mode_vertex()

def select_edge(event=None):
    v.selection_mode_none()
    v.selection_mode_edge()

def select_face(event=None):
    v.selection_mode_none()
    v.selection_mode_face()

def dump(event=None):
    filename = pathlib.Path(__file__).resolve().parent.joinpath("screenshot.png")
    v.save_image(filename)

# Add menu for selection mode
v.add_menu("select")
v.add_submenu("select", select_vertex)
v.add_submenu("select", select_edge)
v.add_submenu("select", select_face)

# Add menu for screencapture
v.add_menu("screenshot")
v.add_submenu("screenshot", dump)

# Load the STEP file and extract the solid
compound = Compound.load_from_step(pathlib.Path(__file__).resolve().parent.joinpath(STEP_FILE))
solid = next(compound.solids())
v.display(solid, transparency=0.5)
v.selection_mode_face()

# ✅ Step 1: Extract Feature Labels from the STEP File
def extract_feature_labels(step_file):
    """
    Parses the STEP file to extract feature labels associated with faces.
    Returns a list of feature labels in the order they appear.
    """
    feature_labels = []
    with open(step_file, "r") as file:
        for line in file:
            # Look for lines describing an ADVANCED_FACE('some_number',...
            match = re.match(r"#\d+ = ADVANCED_FACE\('(\d+)',", line)
            if match:
                feature_labels.append(match.group(1))  # Store feature label

    return feature_labels

# Extract feature labels in order
feature_labels = extract_feature_labels(STEP_FILE)

# ✅ Step 2: Load Machining Feature Names
machining_features = INITIAL_MACHINING_FEATUTES_DICT

# ✅ Step 3: Match Faces in OCC Model with Feature Labels
faces = list(solid.faces())

if len(faces) != len(feature_labels):
    print("Warning: Mismatch between faces in the STEP file and OCC model!")

# Create a mapping between faces and their feature labels
face_label_map = {
    face: feature_labels[i] if i < len(feature_labels) else "?"
    for i, face in enumerate(faces)
}

# Create an index map so we can quickly retrieve the face index
face_index_map = {face: i for i, face in enumerate(faces)}

# ✅ Step 4: Display Face Information on Selection
tooltip = None

def callback(selected_shapes, x, y):
    global tooltip

    # If nothing is selected, return
    if len(selected_shapes) == 0:
        return

    # Remove old tooltip
    if tooltip is not None:
        tooltip.Erase()

    # Assume one entity is selected
    entity = selected_shapes[0]
    
    if isinstance(entity, Face):
        uv = entity.uv_bounds().center()
        pt = entity.point(uv)

        # Get face type (e.g., Plane, Cylinder)
        face_type = entity.surface_type()

        # Get the feature label from our mapping
        feature_label = face_label_map.get(entity, "?")  # Default if not found

        # Match feature label with machining feature name
        machining_feature_name = machining_features.get(feature_label, "Unknown")

        # Retrieve face index
        face_index = face_index_map.get(entity, -1)

        # Display the tooltip with surface type, face index, feature label, and machining feature
        tooltip_text = (
            f"Face {face_index}: {face_type} {feature_label} - {machining_feature_name}"
        )
        tooltip = v.display_text(pt, tooltip_text, height=30)

    elif isinstance(entity, Edge):
        u = entity.u_bounds().middle()
        pt = entity.point(u)
        tooltip = v.display_text(pt, entity.curve_type(), height=30)

    elif isinstance(entity, Vertex):
        pt = entity.point()
        tooltip = v.display_text(pt, f"{pt[0]:2.3f}, {pt[1]:2.3f}, {pt[2]:2.3f}", height=30)

# Register callback for selection
v.on_select(callback)

# Show viewer
v.fit()
v.show()
