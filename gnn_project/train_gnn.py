import torch
import os
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load graph data
def load_data(graph_dir="graphs2"):
    graphs = []
    for file in os.listdir(graph_dir):
        if file.endswith(".pt"):
            graph = torch.load(os.path.join(graph_dir, file))
            graphs.append(graph)
    if not graphs:
        print(f"Error: No graphs found in {graph_dir}. Ensure .pt files are present.")
    return DataLoader(graphs, batch_size=32, shuffle=True)

# Train the model
def train(model, data_loader, epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if len(data_loader) == 0:
        print("Error: DataLoader is empty. Check your graph data.")
        return

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data in data_loader:
            optimizer.zero_grad()
            out = model(data)  # Output shape: [num_nodes, num_classes]

            if out.size(0) != data.y.size(0):
                print(f"Error: Output size {out.size(0)} and label size {data.y.size(0)} mismatch.")
                return
            
            loss = criterion(out, data.y)  # Node-level labels
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, pred = torch.max(out, dim=1)  # Node-level predictions
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Main function to execute training
def main():
    model = GNNModel(input_dim=1, hidden_dim=64, output_dim=2)
    data_loader = load_data()
    print(f"Number of graphs in DataLoader: {len(data_loader)}")
    train(model, data_loader)
    torch.save(model.state_dict(), "gnn_model.pth")
    print("Model trained and saved as 'gnn_model.pth'.")

if __name__ == "__main__":
    main()








