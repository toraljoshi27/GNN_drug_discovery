
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# class GCN(nn.Module):
#     def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, out_dim=1, task="binary"):
#         super().__init__()
#         self.convs = nn.ModuleList()
#         self.convs.append(GCNConv(in_dim, hidden_dim))
#         for _ in range(num_layers-1):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
#         self.dropout = dropout
#         self.head = nn.Linear(hidden_dim, out_dim)
#         self.task = task
#
#     def forward(self, x, edge_index, batch, edge_attr=None):
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_mean_pool(x, batch)
#         return self.head(x)

# src/models/gcn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool  # or your pooling

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, out_dim, task):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.task = task
        self.head = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # >>> CRITICAL LINE: force float for GCNConv <<<
        if not torch.is_floating_point(x):
            x = x.float()

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.head(x)
        return out
