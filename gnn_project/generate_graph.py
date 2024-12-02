
# import json
# import os
# import random
# import torch
# import networkx as nx
# from torch_geometric.data import Data

# def create_transaction_graph(file_path, graph_range, save_graphs=True):
#     """
#     Create a transaction graph from a JSON file and save it as PyTorch Geometric Data.
#     """
#     try:
#         with open(file_path, "r") as file:
#             transactions = json.load(file)
#     except FileNotFoundError:
#         print(f"Error: File not found at {file_path}")
#         return None
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON format in {file_path}")
#         return None

#     # Slice transactions for the given range
#     transactions = transactions[graph_range[0]:graph_range[1]]

#     # Create directed graph
#     G = nx.DiGraph()
#     for tx in transactions:
#         sender = tx.get('sender', 'Unknown Sender')
#         recipient = tx.get('recipient', 'Unknown Recipient')
#         try:
#             amount = float(tx.get('amount', 0))
#         except ValueError:
#             amount = 0
#         G.add_edge(sender, recipient, weight=amount)

#     if len(G.nodes) == 0:
#         print(f"Warning: No valid nodes found in the specified transaction range {graph_range}.")
#         return None

#     # Map node strings to indices
#     node_map = {node: idx for idx, node in enumerate(G.nodes)}
#     edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges], dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)
#     node_features = torch.ones((len(G.nodes), 1))  # Dummy feature: one feature per node

#     # Add dummy labels (0 or 1) for each node
#     node_labels = torch.tensor([random.randint(0, 1) for _ in range(len(G.nodes))], dtype=torch.long)

#     data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_labels)

#     # Save the graph
#     if save_graphs:
#         os.makedirs('graphs', exist_ok=True)
#         filename = f"graphs/graph_{graph_range[0]}_{graph_range[1]}.pt"
#         torch.save(data, filename)
#         print(f"Graph saved at {filename}")

#     return data

# def main():
#     file_path = "C:/Users/HP/OneDrive/Documents/NISHITAAAAAA/project-part2/gnn_project/transactions.json"  # Path to your transactions.json
#     if not os.path.exists(file_path):
#         print(f"Error: {file_path} does not exist.")
#         return

#     # Load transactions to get total count
#     try:
#         with open(file_path, "r") as file:
#             transactions = json.load(file)
#     except Exception as e:
#         print(f"Error loading transactions: {e}")
#         return

#     total_transactions = len(transactions)
#     print(f"Total transactions: {total_transactions}")
    
#     batch_size = 50  # Change this to 10 or 50
#     for i in range(0, total_transactions, batch_size):
#         graph_range = (i, min(i + batch_size, total_transactions))
#         print(f"Processing transactions {graph_range[0]} to {graph_range[1]}")
#         create_transaction_graph(file_path, graph_range)


# if __name__ == "__main__":
#     main()
import json
import os
import random
import torch
import networkx as nx
from torch_geometric.data import Data
from networkx.algorithms.community import greedy_modularity_communities  # Louvain Clustering

def create_transaction_graph(file_path, graph_range, save_graphs=True):
    """
    Create a transaction graph from a JSON file, cluster it, and save as PyTorch Geometric Data.
    """
    try:
        with open(file_path, "r") as file:
            transactions = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

    # Slice transactions for the given range
    transactions = transactions[graph_range[0]:graph_range[1]]

    # Create directed graph
    G = nx.DiGraph()
    for tx in transactions:
        sender = tx.get('sender', 'Unknown Sender')
        recipient = tx.get('recipient', 'Unknown Recipient')
        try:
            amount = float(tx.get('amount', 0))
        except ValueError:
            amount = 0
        G.add_edge(sender, recipient, weight=amount)

    if len(G.nodes) == 0:
        print(f"Warning: No valid nodes found in the specified transaction range {graph_range}.")
        return None

    # Clustering: Find communities
    communities = list(greedy_modularity_communities(G.to_undirected()))  # Louvain clustering
    node_cluster_map = {node: idx for idx, community in enumerate(communities) for node in community}

    # Map node strings to indices
    node_map = {node: idx for idx, node in enumerate(G.nodes)}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)

    # Node features: Add cluster ID as a feature
    node_features = torch.tensor([[node_cluster_map.get(node, -1)] for node in G.nodes], dtype=torch.float)

    # Add dummy labels (or use clustering as labels for graph classification)
    node_labels = torch.tensor([random.randint(0, 1) for _ in range(len(G.nodes))], dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_labels)

    # Save the graph
    if save_graphs:
        os.makedirs('graphs2', exist_ok=True)
        filename = f"graphs2/graph_{graph_range[0]}_{graph_range[1]}.pt"
        torch.save(data, filename)
        print(f"Graph saved at {filename}")

    return data

def main():
    file_path = "C:/Users/HP/OneDrive/Documents/NISHITAAAAAA/project-part2/gnn_project/transactions.json"  # Path to your transactions.json
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return

    # Load transactions to get total count
    try:
        with open(file_path, "r") as file:
            transactions = json.load(file)
    except Exception as e:
        print(f"Error loading transactions: {e}")
        return

    total_transactions = len(transactions)
    print(f"Total transactions: {total_transactions}")
    
    batch_size = 50  # Adjust as per your dataset
    for i in range(0, total_transactions, batch_size):
        graph_range = (i, min(i + batch_size, total_transactions))
        print(f"Processing transactions {graph_range[0]} to {graph_range[1]}")
        create_transaction_graph(file_path, graph_range)

if __name__ == "__main__":
    main()
