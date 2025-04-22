
"""
Subgraph Construction and Graph Feature Integration

This script defines utilities to construct network graphs and transform them into subgraph representations
suitable for PyTorch Geometric processing. It includes handling of node and edge features and
conversion to the `torch_geometric.data.Data` format for GNN training.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, global_mean_pool, LayerNorm

# --------------------------
# DEPRECATED UTILITY FUNCTIONS
# --------------------------
def node_features_to_dataframe(G):
    """Convert node attributes from a NetworkX graph to a pandas DataFrame."""
    node_features = dict(G.nodes(data=True))
    df = pd.DataFrame.from_dict(node_features, orient='index')
    return df

def edge_features_to_dataframe(G):
    """Convert edge attributes from a NetworkX graph to a pandas DataFrame."""
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({'from': u, 'to': v, **d})
    df = pd.DataFrame(edge_data)
    return df

def extract_1hop_subgraph(G, node):
    """Extract a 1-hop subgraph around a given node with associated features."""
    neighbors = list(nx.all_neighbors(G, node))
    subgraph = G.subgraph([node] + neighbors)
    nx_graph = nx.from_pandas_edgelist(edge_features_to_dataframe(subgraph), "from", "to")
    data = from_networkx(nx_graph)
    node_features = np.asarray(node_features_to_dataframe(subgraph), dtype="float32")
    data.X = torch.tensor(node_features, dtype=torch.float)
    edge_features = edge_features_to_dataframe(subgraph)
    data.Y = torch.tensor(np.asarray(edge_features[["connection", "sum_spikes", "distance"]], dtype="float32"), dtype=torch.float)
    return data

def CreateNetwork(network_pram):
    """
    Create a NetworkX graph with annotated nodes and edges from parameter dictionaries.

    Args:
        network_pram (dict): Dictionary containing node features, edge features, and node/edge definitions.

    Returns:
        G (networkx.Graph): Fully constructed NetworkX graph.
    """
    edges = network_pram["edges"].copy()
    edge_features = network_pram["edges_feature"].copy()
    node_features = network_pram["node_feature"].copy()
    node_features["Node"] = network_pram["nodes"].copy()
    node_features.index = node_features["Node"]
    G = nx.Graph()

    for index, row in node_features.iterrows():
        node_id = row['Node']
        G.add_node(node_id,
                   closeness=row['closeness'],
                   betweenness=row['betweenness'],
                   degree=row['degree'],
                   centrality=row['centrality'],
                   page_rank=row['page_rank'],
                   fire_rate=row['fire_rate'])

    for index, row in edges.iterrows():
        from_node = row['from']
        to_node = row['to']
        G.add_edge(from_node, to_node,
                   connection=edge_features.loc[index, 'connection'],
                   sum_spikes=edge_features.loc[index, 'sum_spikes'],
                   distance=edge_features.loc[index, 'distance'])

    return G

# --------------------------
# SUBGRAPH CREATION FUNCTION
# --------------------------
def BuildSubgraph(list_graph):
    """
    Build torch_geometric data subgraphs from a list of dictionaries with edge and expression data.

    Args:
        list_graph (list): List of graph data dictionaries, each containing 'edges' and 'expression'.

    Returns:
        subgraphs (list): List of PyTorch Geometric Data objects.
    """
    subgraphs = []
    for input in tqdm(list_graph):
        nx_graph = nx.from_pandas_edgelist(input["edges"], "from", "to")
        data = from_networkx(nx_graph)
        node_features = np.asarray(input["expression"], dtype="float32")
        data.X = torch.tensor(node_features, dtype=torch.float)
        subgraphs.append(data)
    return subgraphs





