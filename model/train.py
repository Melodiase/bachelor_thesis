import os
import glob
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader
from brepgat_architecture import BRepGAT
import pickle
import logging
import matplotlib.pyplot as plt
import shutil
import sys
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
acc_plot_file = os.path.join(output_dir, "accuracy_plot.png")
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
# Instead of one data_dir, we now use separate directories for official splits.
train_data_dir = "/home/ms23911/BachelorThesis/graph_data_new_features/all_new/train_18"
val_data_dir   = "/home/ms23911/BachelorThesis/graph_data_new_features/all_new/val_18"
BATCH_SIZE = 64
EPOCHS = 220
PATIENCE = 35
LR = 0.001

# === Device configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# === Load graph data functions (Modified to use official split folders) ===
def load_graphs_from_dir(directory: str):
    files = glob.glob(os.path.join(directory, "*.pkl"))
    logger.info(f"Found {len(files)} graph files in {directory}")
    data_list = []
    for f in files:
        try:
            with open(f, "rb") as infile:
                data = pickle.load(infile)
                data_list.append(data)
        except Exception as e:
            logger.error(f"Error loading graph from {f}: {e}")
    logger.info(f"Successfully loaded {len(data_list)} graphs from {directory}")
    return data_list

logger.info("🔍 Loading training graph data...")
train_data = load_graphs_from_dir(train_data_dir)
logger.info("🔍 Loading validation graph data...")
val_data   = load_graphs_from_dir(val_data_dir)

# Create DataLoaders using the loaded official splits.
train_loader = PyGDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = PyGDataLoader(val_data, batch_size=BATCH_SIZE)

# === Initialize model ===
logger.info("🚀 Initializing model...")
sample = train_data[0]  # using a sample from training data

model = BRepGAT(
    node_in_dim=sample.x.shape[1],
    edge_in_dim=sample.edge_attr.shape[1],
    hidden_dim=64,
    num_classes=int(sample.y.max().item()) + 1,
    dropout=0.5,
    heads=10
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
logger.info(
    f"📐 Input feature sizes → node_in_dim={sample.x.shape[1]}, "
    f"edge_in_dim={sample.edge_attr.shape[1]}, "
    f"num_classes={int(sample.y.max().item()) + 1}"
)

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

# Lists for tracking losses and metrics
train_loss_history = []
val_loss_history = []
val_acc_history = []
val_f1_history = []

# === Evaluation function (No changes to the core logic) ===
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return total_loss, acc, f1, cm

# === Training function (Retaining core logic and adding metric tracking) ===
def train():
    global best_loss, patience_counter, start_epoch
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        logger.info(f"Epoch {epoch:03d} | Train Loss: {total_train_loss:.4f}")
        train_loss_history.append(total_train_loss)

        # Evaluate on validation set
        val_loss, val_acc, val_f1, val_cm = evaluate(val_loader)
        logger.info(f"Epoch {epoch:03d} | Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
        logger.debug(f"Confusion Matrix at Epoch {epoch}:\n{val_cm}")
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_f1_history.append(val_f1)

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

    # --- Plot loss and accuracy curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_file)
    logger.info(f"📈 Loss plot saved as {loss_plot_file}")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_plot_file)
    logger.info(f"📈 Accuracy plot saved as {acc_plot_file}")
    plt.close()

# === Inference function (No changes) ===
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
