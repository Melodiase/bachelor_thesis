import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class BRepGAT(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_classes, dropout=0.5, heads=4):  # HEADS = 4
        super(BRepGAT, self).__init__()

        self.dropout = dropout

        # 5 GATConv layers with edge features
        self.gat1 = GATConv(node_in_dim, hidden_dim, heads=heads, edge_dim=edge_in_dim)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_in_dim)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_in_dim)
        self.gat4 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_in_dim)
        #self.gat5 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_in_dim)  # New extra layer
        #self.gat6 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_in_dim)  # New extra layer
        self.gat5 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=edge_in_dim)  # Final layer (no ReLU)

        # MLP Decoder
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = F.relu(self.gat3(x, edge_index, edge_attr))
        x = F.relu(self.gat4(x, edge_index, edge_attr))
        #x = F.relu(self.gat5(x, edge_index, edge_attr))  # New extra layer with ReLU   
        #x = F.relu(self.gat6(x, edge_index, edge_attr))  # New extra layer with ReLU  

        x = self.gat5(x, edge_index, edge_attr)  # No activation after final GAT
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)
