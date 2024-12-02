# import json

# # Path to your JSON file
# file_path = 'C:/Users/HP/OneDrive/Documents/NISHITAAAAAA/project-part2/gnn_project/transactions.json'

# # Load the data from the JSON file
# with open(file_path, 'r') as file:
#     transactions = json.load(file)

# # Print the first few transactions to ensure it's loaded correctly
# print(f"First few transactions: {transactions[:5]}")

# # Get the total number of transactions
# print(f"Total number of transactions: {len(transactions)}")

import torch

# Path to your .pt file
file_path = 'C:/Users/HP\OneDrive/Documents/NISHITAAAAAA/project-part2/graphs2/graph_0_50.pt'

# Load the .pt file
data = torch.load(file_path)

# Check the type of data
print(f"Type of data: {type(data)}")

# Inspect the content
if isinstance(data, dict):
    print("Keys in the file:", data.keys())
elif isinstance(data, torch.nn.Module):
    print("This is a PyTorch model. Model architecture:")
    print(data)
else:
    print("File content:", data)

