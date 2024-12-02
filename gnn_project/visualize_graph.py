import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import os

# Specify the directory where .pt files are stored
graph_dir = "C:/Users/HP/OneDrive/Documents/NISHITAAAAAA/project-part2/graphs2"

# List all .pt files
pt_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]

# Visualize graphs
for pt_file in pt_files[:5]:  # Visualize the first 5 graphs
    file_path = os.path.join(graph_dir, pt_file)
    data = torch.load(file_path)  # Load the graph

    # Convert to a NetworkX graph (undirected for visualization)
    nx_graph = to_networkx(data, to_undirected=True)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, with_labels=True, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title(f"Graph from {pt_file}")
    plt.show()

