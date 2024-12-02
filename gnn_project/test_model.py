import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
import os

# Define the GNN model (GCN)
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

# Load the trained model
def load_trained_model(model_path):
    model = GNNModel(input_dim=1, hidden_dim=64, output_dim=2)  # Define the model with the same architecture
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Load saved graph data from 'graphs/' directory
def load_data():
    graphs = []
    for graph_file in os.listdir('graphs'):
        if graph_file.endswith('.pt'):
            graph = torch.load(os.path.join('graphs', graph_file))
            graphs.append(graph)
    
    return DataLoader(graphs, batch_size=1, shuffle=False)  # Load data in non-shuffling mode for testing

# Test the model on the test data
def test(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data in data_loader:
            out = model(data)  # Model's predictions
            # Assuming we're doing node classification and out contains logits
            _, predicted = torch.max(out, dim=1)  # Get the predicted class
            correct += (predicted == data.y).float().sum().item()  
  # Compare prediction with ground truth (labels)
            total += data.y.size(0)  # Add the number of nodes in this batch
    
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Main function to run testing
def main():
    # Load the trained model
    model = GNNModel(input_dim=1, hidden_dim=64, output_dim=2)  # Ensure model parameters match the training setup
    model.load_state_dict(torch.load('gnn_model.pth'))
    model.eval()

    # Load the test data
    data_loader = load_data()  # Make sure load_data() is properly implemented

    # Test the model
    test(model, data_loader)

if __name__ == "__main__":
    main()