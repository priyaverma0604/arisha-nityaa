import networkx as nx
import json
import torch
from torch_geometric.data import Data
import sys

def create_graph(transaction_data):
    # Create an empty graph
    G = nx.Graph()

    # Add edges with weights (transaction values or frequency)
    for tx in transaction_data:
        sender = tx['sender']
        recipient = tx['recipient']
        amount = float(tx['amount'])

        # Add edges with weight (transaction amount)
        if G.has_edge(sender, recipient):
            G[sender][recipient]['weight'] += amount  # If edge exists, accumulate weight
        else:
            G.add_edge(sender, recipient, weight=amount)

    return G

def convert_to_pyg_graph(G):
    # Map nodes (wallet addresses) to unique integers
    node_mapping = {node: idx for idx, node in enumerate(G.nodes)}

    # Convert edges and weights
    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])  # Map string nodes to indices
        edge_attr.append(data['weight'])  # Edge weights

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create node features (e.g., degree of each node)
    node_features = torch.tensor(
        [G.degree[node] for node in G.nodes], dtype=torch.float
    ).view(-1, 1)

    # Create the Data object for PyTorch Geometric
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    return data

def main():
    # Read the formatted transaction data passed from JavaScript (node)
    transaction_data = json.loads(sys.argv[1])

    # Step 1: Create the graph from transaction data
    G = create_graph(transaction_data)

    # Step 2: Convert the graph to PyTorch Geometric format
    pyg_data = convert_to_pyg_graph(G)

    # Print the PyTorch Geometric Data object
    print(pyg_data)

if __name__ == "__main__":
    main()
