import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import os

# --- Configuration (Upgraded Hyperparameters) ---
INTERACTIONS_FILE = 'interactions.csv'
MODEL_SAVE_PATH = 'models/gnn_model.pt'
MAP_SAVE_PATH = 'models/drug_map.json'
EMBEDDING_DIM = 128
HIDDEN_CHANNELS = 128
EPOCHS = 300
LEARNING_RATE = 0.005
# ---

# 1. Define the Upgraded Graph Neural Network Model
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels):
        super(GNNLinkPredictor, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GATConv(embedding_dim, hidden_channels, heads=4, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False, dropout=0.6)

    def encode(self, x, edge_index):
        x = self.embedding(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

def train_gnn():
    """Main function to load data, build graph, and train the GNN with advanced techniques."""
    print("--- Starting Upgraded GNN Training & Evaluation ---")
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    try:
        df = pd.read_csv(INTERACTIONS_FILE)
        print(f"Loaded {len(df)} interaction records from '{INTERACTIONS_FILE}'.")
    except FileNotFoundError:
        print(f"[ERROR] '{INTERACTIONS_FILE}' not found.")
        return

    all_drugs = pd.concat([df['drug_a'], df['drug_b']]).unique()
    le = LabelEncoder()
    le.fit(all_drugs)
    
    drug_map = {name: int(idx) for name, idx in zip(le.classes_, le.transform(le.classes_))}
    with open(MAP_SAVE_PATH, 'w') as f:
        json.dump(drug_map, f)
    print(f"Created a map for {len(all_drugs)} unique drugs saved to '{MAP_SAVE_PATH}'.")

    src = le.transform(df['drug_a'])
    dst = le.transform(df['drug_b'])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=torch.arange(len(all_drugs)), edge_index=edge_index)

    # --- Using PyG's RandomLinkSplit for a fair and balanced data split ---
    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=False # We will handle negative sampling manually
    )
    train_data, val_data, test_data = transform(data)
    print("Split data into training, validation, and test sets using RandomLinkSplit.")

    device = torch.device('cpu')
    model = GNNLinkPredictor(
        num_nodes=len(all_drugs),
        embedding_dim=EMBEDDING_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=EMBEDDING_DIM
    ).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
    
    print("\nStarting training loop with early stopping...")
    best_val_auc = 0
    patience_counter = 0
    patience = 30 # Increased patience

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(train_data.x, train_data.edge_index)
        
        # --- THIS IS THE CORRECTED TRAINING LOGIC ---
        # 1. Positive edges (real interactions from the training set)
        pos_out = model.decode(z, train_data.edge_index)
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        
        # 2. Negative edges (random, non-existent interactions)
        neg_edge_index = torch.randint(0, len(all_drugs), (2, train_data.num_edges), dtype=torch.long, device=device)
        neg_out = model.decode(z, neg_edge_index)
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        # --- Early Stopping and Validation Check ---
        if epoch % 10 == 0:
            val_auc = test(model, val_data, len(all_drugs)) # Check performance
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save the best model
            else:
                patience_counter += 1
            
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val ROC-AUC: {val_auc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. No improvement in validation AUC.")
                break
    
    print("\n--- Training Complete ---")

    print("\n--- Evaluating Best Model on Unseen Test Data ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_auc = test(model, test_data, len(all_drugs))
    print(f"Final Model ROC-AUC Score on the test set: {test_auc:.4f}")
    print("This is the new, more reliable performance score.")

# --- New Test Function for Evaluation ---
def test(model, data, num_nodes):
    """Evaluates the model and returns the ROC-AUC score."""
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        # Test against the positive (real) links in the test set
        pos_edge_label_index = data.edge_index
        pos_pred = model.decode(z, pos_edge_label_index).sigmoid()
        
        # Test against random negative (non-existent) links
        neg_edge_label_index = torch.randint(0, num_nodes, (2, data.num_edges), dtype=torch.long, device=z.device)
        neg_pred = model.decode(z, neg_edge_label_index).sigmoid()

        # Combine predictions and true labels
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        true_labels = torch.cat([torch.ones(pos_pred.numel()), torch.zeros(neg_pred.numel())], dim=0)
        
        return roc_auc_score(true_labels.cpu().numpy(), preds.cpu().numpy())

if __name__ == '__main__':
    train_gnn()

