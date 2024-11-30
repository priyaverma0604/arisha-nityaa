import json
import networkx as nx
import matplotlib.pyplot as plt

def create_transaction_graph(file_path, max_transactions=10):
    """
    Create a network graph from transaction data, limited to max_transactions
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
    
    # Limit transactions to max_transactions
    transactions = transactions[:max_transactions]
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track unique edges to prevent duplicates
    processed_edges = set()
    
    # Add edges to the graph
    for transaction in transactions:
        # Flexible key handling
        sender = transaction.get('sender', transaction.get('from', 'Unknown Sender'))
        recipient = transaction.get('recipient', transaction.get('to', 'Unknown Recipient'))
        amount = transaction.get('amount', transaction.get('value', 0))
        
        # Ensure amount is numeric
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            amount = 0
        
        # Create unique edge
        edge = (sender, recipient)
        if edge not in processed_edges:
            G.add_edge(sender, recipient, weight=amount)
            processed_edges.add(edge)
    
    return G

def visualize_graph(G, layout_type='spring', node_size=1000, node_color='skyblue'):
    """
    Visualize the transaction network graph
    """
    if not G or len(G.edges()) == 0:
        print("No graph to visualize")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Select layout
    pos = nx.spring_layout(G, k=0.9)  # Increased k for more spread-out graph
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                            node_color=node_color, 
                            node_size=node_size, 
                            alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                            edge_color='gray', 
                            arrows=True, 
                            arrowsize=20, 
                            width=1)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                             font_size=8, 
                             font_weight='bold')
    
    # Draw edge weights 
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, 
                                      edge_labels={k: f'{v:.2f}' for k, v in edge_labels.items()}, 
                                      font_size=7)
    
    plt.title(f"Transaction Network Graph (Limited to {len(G.edges())} Transactions)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Use the exact file path you provided
    file_path = r"D:\SIH final\fetch-data\project-part2\Python\transactions.json"
    
    # Create graph with maximum 10 transactions
    transaction_graph = create_transaction_graph(file_path, max_transactions=10)
    
    if transaction_graph:
        # Visualize graph
        visualize_graph(transaction_graph)

if __name__ == "__main__":
    main()