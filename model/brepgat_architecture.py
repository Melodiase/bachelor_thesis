import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class BRepGAT(nn.Module):
    """
    Basic BRepGAT model:
    - 1st~4th layers: GATConv -> ReLU
    - 5th layer: Dropout
    - 6th layer: GATConv -> (no ReLU)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 640,
        out_channels: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels (int):  Dimension of node input features.
            hidden_channels (int): Hidden dimension for GATConv layers.
            out_channels (int):   Dimension of the output (e.g., number of segmentation classes).
            dropout (float):      Dropout probability (5th layer).
        """
        super(BRepGAT, self).__init__()
        
        # 1st~4th: GATConv + ReLU
        self.gat1 = GATConv(in_channels, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)
        self.gat3 = GATConv(hidden_channels, hidden_channels)
        self.gat4 = GATConv(hidden_channels, hidden_channels)
        
        # 5th: dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # 6th: GATConv (final layer, outputs segmentation logits)
        self.gat5 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of BRepGAT.

        Args:
            x (Tensor):         Node feature matrix [num_nodes, in_channels].
            edge_index (Tensor):Graph connectivity [2, num_edges].
        
        Returns:
            Tensor: Segmentation logits [num_nodes, out_channels].
        """
        # 1st to 4th layers: GATConv + ReLU
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))
        x = F.relu(self.gat4(x, edge_index))
        
        # 5th layer: dropout
        x = self.dropout(x)
        
        # 6th layer: GATConv without ReLU -> output segmentation logits
        x = self.gat5(x, edge_index)
        
        return x
