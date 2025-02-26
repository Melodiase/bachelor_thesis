import torch
from torch_geometric.loader import DataLoader
import numpy as np

def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=20, device='cuda'):
    """
    Example training routine with early stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- TRAINING ----
        model.train()
        train_loss = 0.0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            out = model(batch_data.x, batch_data.edge_index)
            
            # Ensure shape for cross-entropy: [num_nodes, num_classes], [num_nodes]
            loss = criterion(out, batch_data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_data.num_nodes

        # Average training loss per node (optional metric)
        train_loss /= sum([d.num_nodes for d in train_loader.dataset])

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                out = model(batch_data.x, batch_data.edge_index)
                loss = criterion(out, batch_data.y)
                val_loss += loss.item() * batch_data.num_nodes
        
        val_loss /= sum([d.num_nodes for d in val_loader.dataset])

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- EARLY STOPPING CHECK ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Optionally save a model checkpoint
            torch.save(model.state_dict(), "best_brepgat_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best validation loss:", best_val_loss)

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    # Suppose you have train_dataset and val_dataset as PyG Dataset objects
    # train_dataset = ...
    # val_dataset = ...
    
    # Wrap them in DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Instantiate the BRepGAT model
    model = BRepGAT(
        in_channels=64,       # e.g., if you have 64-dimensional node features
        hidden_channels=640,
        out_channels=3,       # e.g., 3 segmentation classes
        dropout=0.5
    )
    
    # Train
    train_model(model, train_loader, val_loader, epochs=200, lr=0.001, patience=20, device='cuda')
