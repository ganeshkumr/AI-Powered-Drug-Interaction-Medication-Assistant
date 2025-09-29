import pandas as pd
import json
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuration ---
INTERACTIONS_FILE = 'interactions.csv'
MODEL_SAVE_PATH = 'models/gnn_model.pt'
MAP_SAVE_PATH = 'models/drug_map.json'
EMBEDDING_DIM = 64
EPOCHS = 200 
LEARNING_RATE = 0.01
# ---

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim=64):
        super(GNNLinkPredictor, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, embedding_dim * 2)
        self.conv2 = SAGEConv(embedding_dim * 2, embedding_dim)

    def encode(self, x, edge_index):
        x = self.embedding(x)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

def train_gnn():
    """Main function to load data, build graph, train, and evaluate the GNN."""
    print("--- Starting GNN Training & Evaluation ---")
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    try:
        df = pd.read_csv(INTERACTIONS_FILE)
        print(f"Loaded {len(df)} interaction records from '{INTERACTIONS_FILE}'.")
    except FileNotFoundError:
        print(f"[ERROR] '{INTERACTIONS_FILE}' not found. Please ensure it is in the project directory.")
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

    transform = RandomLinkSplit(is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(data)
    print("Split data into training, validation, and test sets for proper evaluation.")

    device = torch.device('cpu')
    model = GNNLinkPredictor(num_nodes=len(all_drugs), embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_data, test_data = train_data.to(device), test_data.to(device)
    
    print("\nStarting training loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        out = model.decode(z, train_data.edge_label_index)
        loss = criterion(out, train_data.edge_label)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
    
    print("\n--- Training Complete ---")

    print("\n--- Evaluating Model on Unseen Data ---")
    model.eval()
    with torch.no_grad():
        z = model.encode(test_data.x, test_data.edge_index)
        out = model.decode(z, test_data.edge_label_index)
        pred = (out > 0).float()
        correct = (pred == test_data.edge_label).sum()
        accuracy = int(correct) / int(test_data.edge_label.numel())
        print(f"Model Accuracy on the test set: {accuracy:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to '{MODEL_SAVE_PATH}'.")

if __name__ == '__main__':
    train_gnn()

