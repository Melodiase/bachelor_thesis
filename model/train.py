import os
import glob
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from model import BRepGAT
import pickle
from sklearn.model_selection import train_test_split

# === Configuration ===
data_dir = "graph_data"
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
LR = 0.001

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load graph data ===
def load_graphs():
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    data_list = []
    for f in files:
        with open(f, "rb") as infile:
            data = pickle.load(infile)
            data_list.append(data)
    return data_list

print("🔍 Loading graph data...")
data_list = load_graphs()
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = PyGDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = PyGDataLoader(val_data, batch_size=BATCH_SIZE)

# === Initialize model ===
print("🚀 Initializing model...")
sample = data_list[0]
model = BRepGAT(
    node_in_dim=sample.x.shape[1],
    edge_in_dim=sample.edge_attr.shape[1],
    hidden_dim=64,
    num_classes=int(sample.y.max().item()) + 1,
    dropout=0.5
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training ===
def train():
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate()
        print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("✅ Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping.")
                break

# === Validation ===
def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            total_loss += loss.item()
    return total_loss

# === Inference ===
def predict(batch):
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)  # <== Inference: softmax applied
        preds = probs.argmax(dim=1)
    return preds

if __name__ == "__main__":
    train()
