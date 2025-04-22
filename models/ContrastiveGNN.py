"""
Graph Contrastive Learning with Graph Attention Network (GAT)

This script implements a GAT-based Graph Neural Network for graph-level representation learning using contrastive learning.
It includes graph augmentations, contrastive loss functions (InfoNCE and NT-Xent), early stopping, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import from_networkx

# --------------------------
# GNN MODEL DEFINITION
# --------------------------
class GNN(nn.Module):
    """
    Graph Neural Network using GATConv layers.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layers.
        output_dim (int): Dimension of output embedding.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.X, data.edge_index
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return x

# --------------------------
# DATA AUGMENTATION FUNCTIONS
# --------------------------
def create_negative_sample(data):
    """Create a negative sample by shuffling node features."""
    negative_data = data.clone()
    node_indices = torch.randperm(data.X.size(0))
    negative_data.X = data.X[node_indices]
    return negative_data

def create_positive_sample(data, mask_ratio=0.1):
    """Create a positive sample by masking a fraction of node features."""
    positive_data = data.clone()
    num_nodes_to_mask = int(mask_ratio * data.X.size(0))
    mask_indices = torch.randperm(data.X.size(0))[:num_nodes_to_mask]
    positive_data.X[mask_indices] = 0
    return positive_data

# --------------------------
# CONTRASTIVE LOSS FUNCTIONS
# --------------------------
def contrastive_loss(z1, z_pos, z_neg, temperature=0.5):
    """InfoNCE loss implementation."""
    z1 = F.normalize(z1, dim=1)
    z_pos = F.normalize(z_pos, dim=1)
    z_neg = F.normalize(z_neg, dim=1)
    pos_sim = torch.sum(z1 * z_pos, dim=1)
    neg_sim = torch.einsum('ik,jk->ij', z1, z_neg)
    numerator = torch.exp(pos_sim / temperature)
    denominator = torch.exp(neg_sim / temperature).sum(dim=1)
    loss = -torch.log(numerator / (denominator + numerator)).mean()
    return loss

class NTXentLoss(nn.Module):
    """NT-Xent loss implementation."""
    def __init__(self, temperature=1.0):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        sim = self.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature
        logits = torch.cat([sim, sim.T], dim=0)
        labels = torch.arange(z_i.size(0), device=sim.device)
        labels = torch.cat([labels, labels], dim=0)
        loss = F.cross_entropy(logits, labels)
        return loss

# --------------------------
# EARLY STOPPING
# --------------------------
class EarlyStopping:
    """Basic early stopping based on validation loss."""
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# --------------------------
# SIMILARITY & EVALUATION
# --------------------------
def compute_cosine_similarity(embedding1, embedding2):
    embedding1 = F.normalize(embedding1, dim=1)
    embedding2 = F.normalize(embedding2, dim=1)
    return torch.mm(embedding1, embedding2.t())

def evaluate_contrastive_learning(latent_space, positive_pairs, negative_pairs, threshold=0.5):
    positive_similarities = []
    negative_similarities = []
    for orig_idx, aug_idx in positive_pairs:
        cos_sim = cosine_similarity([latent_space[orig_idx]], [latent_space[aug_idx]])[0][0]
        positive_similarities.append(cos_sim)
    for orig_idx, neg_idx in negative_pairs:
        cos_sim = cosine_similarity([latent_space[orig_idx]], [latent_space[neg_idx]])[0][0]
        negative_similarities.append(cos_sim)
    pos_acc = np.mean(np.array(positive_similarities) >= threshold)
    neg_acc = np.mean(np.array(negative_similarities) < threshold)
    total_acc = (pos_acc + neg_acc) / 2
    print(f"Positive Accuracy: {pos_acc*100:.2f}%, Negative Accuracy: {neg_acc*100:.2f}%, Total: {total_acc*100:.2f}%")
    return positive_similarities, negative_similarities, total_acc

def get_pairs(original_embeddings, augmented_embeddings):
    return [(i, i) for i in range(original_embeddings.shape[0])]

# --------------------------
# RANDOM GRAPH GENERATION
# --------------------------
def generate_random_graph(num_nodes, feature_dim, edge_dim):
    random_graph_nx = nx.gnp_random_graph(num_nodes, 0.5)
    random_graph = from_networkx(random_graph_nx)
    random_graph.X = torch.randn(num_nodes, feature_dim)
    return random_graph

def shuffle_edges(graph):
    negative_graph = graph.clone()
    edge_indices = negative_graph.edge_index
    perm = torch.randperm(edge_indices.size(1))
    negative_graph.edge_index = edge_indices[:, perm]
    return negative_graph

def generate_negative_sample(graph):
    negative_graph = graph.clone()
    idx = torch.randperm(graph.X.size(0))
    negative_graph.X = graph.X[idx]
    return negative_graph
