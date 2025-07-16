# BRepGAT for Machining Feature Segmentation

## 📋 Overview

This package implements **BRepGAT (Boundary Representation Graph Attention Networks)** for automatic segmentation of machining features in 3D CAD models. The implementation explores different geometric feature representations to classify 18 distinct machining feature categories.

## 🎯 Core Features

- **Graph Neural Network Architecture** - 5-layer GAT with multi-head attention
- **Multiple Feature Extraction Strategies** - Various geometric descriptors
- **CAD Model Processing** - From STEP files to graph representations
- **Robust Training Pipeline** - Complete model training and evaluation

## 🎨 Machining Feature Classes

The model classifies 18 distinct machining features:

```json
{
    "0": "Chamfer",
    "1": "Through hole", 
    "2": "Triangular passage",
    "3": "Rectangular passage",
    "4": "Six-sided passage",
    "5": "Triangular through slot",
    "6": "Circular through slot",
    "7": "O Ring",
    "8": "Blind hole",
    "9": "Triangular pocket",
    "10": "Rectangular pocket", 
    "11": "Six-sided pocket",
    "12": "Circular end pocket",
    "13": "Horizontal circular end blind slot",
    "14": "Round",
    "15": "General step",
    "16": "General slot",
    "17": "Stock"
}
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install PythonOCC wrapper
pip install occwl

# Install package
pip install -e .
```

## 📁 Package Structure

```
bachelor_thesis/
├── model/                          # Neural network implementation
│   ├── brepgat_architecture.py     # Main BRepGAT model
│   └── train.py                   # Training pipeline
├── extractors/                    # Feature extraction modules
│   ├── face_extractor.py          # Face-level geometric features
│   └── edge_extractor.py          # Edge-level geometric features
├── descriptors/                   # Geometric descriptors
│   ├── face_attributes.py         # Face geometric properties
│   └── edge_attributes.py         # Edge geometric properties
├── feature_extractor.py           # Main feature extraction pipeline
├── graph_builder.py               # Graph construction from CAD models
├── dataset_preprocessor.py        # Data preprocessing utilities
├── current_viewer.py              # CAD model visualization
├── mappings.py                    # Feature type mappings
└── requirements.txt               # Package dependencies
```

## 🏗️ Architecture Details

### BRepGAT Network
- **5-layer Graph Attention Network** with edge features
- **Multi-head attention** (configurable heads: 4-10)
- **Hidden dimension**: 64
- **MLP decoder** for classification
- **Dropout regularization**: 0.5

### Feature Configurations
- **Node features**: 10-78 dimensions (varies by feature set)
- **Edge features**: 6-7 dimensions  
- **Graph representation**: Face adjacency with geometric attributes

## 🔧 Usage

### 1. Data Preprocessing
```python
from bachelor_thesis.dataset_preprocessor import DatasetPreprocessor

preprocessor = DatasetPreprocessor()
preprocessor.process_dataset("path/to/step/files", "output/directory")
```

### 2. Feature Extraction
```python
from bachelor_thesis.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
graph_data = extractor.extract_features("model.step")
```

### 3. Graph Construction
```python
from bachelor_thesis.graph_builder import GraphBuilder

builder = GraphBuilder()
graph = builder.build_graph(solid_object, labels)
```

### 4. Model Training
```python
from bachelor_thesis.model.brepgat_architecture import BRepGAT
from bachelor_thesis.model.train import train_model

# Initialize model
model = BRepGAT(
    node_in_dim=74,
    edge_in_dim=6,
    hidden_dim=64,
    num_classes=18,
    heads=10
)

# Train model
train_model(model, train_data, val_data, config)
```

### 5. CAD Model Visualization
```python
from bachelor_thesis.current_viewer import display_model

display_model("model.step")
```

## 💾 Graph Data Storage

The package processes CAD models into graph representations stored as PyTorch Geometric `Data` objects:

- **Node features**: Face-level geometric descriptors
- **Edge features**: Edge-level connectivity and geometric properties  
- **Edge indices**: Face adjacency relationships
- **Labels**: Ground truth machining feature classifications

Graph data is typically saved as `.pkl` files for efficient loading during training.

## 🔬 Feature Extraction Strategies

The package supports multiple geometric feature extraction approaches:

1. **Combined Features** - Comprehensive geometric descriptor set
2. **Chamfer Distance** - Distance-based geometric measures
3. **Edge Continuity** - Topological connectivity features
4. **Depth Ratio** - Surface depth and geometric ratios
5. **Dihedral Angles** - Angular relationships between surfaces
6. **Gaussian Curvature** - Surface curvature descriptors
7. **Paper Baseline** - Original BRepGAT implementation

## 🎛️ Configuration

### Model Parameters
```python
# Standard configuration
node_in_dim: 74          # Node feature dimensions
edge_in_dim: 6           # Edge feature dimensions
hidden_dim: 64           # Hidden layer size
num_classes: 18          # Machining feature classes
dropout: 0.5             # Regularization
heads: 10                # Attention heads
```

### Training Parameters
```python
learning_rate: 0.001     # Adam optimizer learning rate
batch_size: 64           # Training batch size
epochs: 100              # Training epochs
scheduler: ReduceLROnPlateau  # Learning rate scheduling
```

## 📚 Core Components

### Feature Extractors
- **FaceExtractor**: Extracts geometric properties from CAD face entities
- **EdgeExtractor**: Computes edge-level geometric and topological features

### Descriptors
- **FaceAttributes**: Area, surface type, curvature, normal vectors
- **EdgeAttributes**: Convexity, length, curve type, continuity measures

### Model Architecture
- **BRepGAT**: Main graph attention network implementation
- **Training Pipeline**: Complete training loop with validation and checkpointing

## 🔄 Workflow

1. **CAD Import** → Load STEP files using PythonOCC
2. **Feature Extraction** → Compute geometric descriptors for faces and edges
3. **Graph Construction** → Build face adjacency graph with features
4. **Model Training** → Train BRepGAT on graph data
5. **Evaluation** → Assess performance on test set

## 📄 References

Implementation based on BRepGAT methodology for machining feature segmentation using graph neural networks on 3D CAD boundary representations.

---

**Framework**: PyTorch, PyTorch Geometric, PythonOCC  
**Focus**: Geometric deep learning for CAD analysis