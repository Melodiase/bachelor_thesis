import os
import glob
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from brepgat_architecture import BRepGAT
import pickle
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
import shutil
import sys

# === Command line arguments ===
parser = argparse.ArgumentParser(
    description="Train BRepGAT with checkpointing, logging, and plotting in an experiment folder."
)
parser.add_argument(
    "--exp_name",
    type=str,
    required=True,
    help="Name for the experiment. Outputs will be saved in experiments/<exp_name>/",
)
parser.add_argument(
    "--new",
    action="store_true",
    help="Flag to start a new experiment. If set, the experiment folder must not already exist.",
)
args = parser.parse_args()

# === Define experiment output directory and file paths ===
output_dir = os.path.join("experiments", args.exp_name)
checkpoint_file = os.path.join(output_dir, "checkpoint.pt")
best_model_file = os.path.join(output_dir, "best_model.pt")
loss_plot_file = os.path.join(output_dir, "loss_plot.png")
log_file = os.path.join(output_dir, "training.log")

# Determine mode based on --new flag
if args.new:
    if os.path.exists(output_dir):
        print(f"Error: Experiment folder '{output_dir}' already exists. Choose a different name or omit --new to resume.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting new experiment: '{args.exp_name}'. Folder created at {output_dir}")
else:
    if not os.path.exists(output_dir):
        print(f"Error: Experiment folder '{output_dir}' does not exist. Use --new to create a new experiment.")
        sys.exit(1)
    print(f"Resuming experiment: '{args.exp_name}' from folder {output_dir}")

# === Setup logging ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("Starting training script...")

# === Configuration ===
data_dir = "/home/ms23911/BachelorThesis/graph_data"
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
LR = 0.001

# === Device configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# === Load graph data ===
def load_graphs():
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    logger.info(f"Found {len(files)} graph files")
    data_list = []
    for f in files:
        with open(f, "rb") as infile:
            data = pickle.load(infile)
            data_list.append(data)
    logger.info(f"Successfully loaded {len(data_list)} graphs")
    return data_list

logger.info("🔍 Loading graph data...")
data_list = load_graphs()
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = PyGDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = PyGDataLoader(val_data, batch_size=BATCH_SIZE)

# === Initialize model ===
logger.info("🚀 Initializing model...")
sample = data_list[0]
model = BRepGAT(
    node_in_dim=sample.x.shape[1],
    edge_in_dim=sample.edge_attr.shape[1],
    hidden_dim=64,
    num_classes=int(sample.y.max().item()) + 1,
    dropout=0.5
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Checkpoint functions ---
def save_checkpoint(epoch, best_loss, patience_counter, filename=checkpoint_file):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter
    }
    torch.save(checkpoint, filename)
    logger.info(f"🔖 Checkpoint saved at epoch {epoch} to {filename}")

def load_checkpoint(filename=checkpoint_file):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"🔄 Loaded checkpoint from epoch {checkpoint['epoch']} from {filename}")
        return checkpoint["epoch"], checkpoint["best_loss"], checkpoint["patience_counter"]
    return 0, float('inf'), 0

# --- Load checkpoint if exists ---
start_epoch, best_loss, patience_counter = load_checkpoint()

# Lists for tracking losses
train_loss_history = []
val_loss_history = []

# === Validation function ===
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

# === Training function ===
def train():
    global best_loss, patience_counter, start_epoch
    for epoch in range(start_epoch + 1, EPOCHS + 1):
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
        logger.info(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f}")
        train_loss_history.append(total_loss)

        val_loss = evaluate()
        logger.info(f"Epoch {epoch:03d} | Validation Loss: {val_loss:.4f}")
        val_loss_history.append(val_loss)

        # Save best model if improvement is seen
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_file)
            logger.info(f"✅ Saved best model to {best_model_file}.")
        else:
            patience_counter += 1
            logger.info(f"No improvement, patience counter: {patience_counter}")
            if patience_counter >= PATIENCE:
                logger.info("⏹️ Early stopping triggered.")
                break

        # Save a checkpoint at the end of each epoch
        save_checkpoint(epoch, best_loss, patience_counter)

    # --- Plot loss curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_file)
    logger.info(f"📈 Loss plot saved as {loss_plot_file}")
    plt.show()

# === Inference function ===
def predict(batch):
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
    return preds

if __name__ == "__main__":
    train()
