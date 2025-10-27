
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=5, dropout=0.2, out_dim=1, task="binary"):
        super().__init__()
        self.layers = nn.ModuleList()
        def mlp(in_c, out_c):
            return nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))
        self.layers.append(GINConv(mlp(in_dim, hidden_dim)))
        for _ in range(num_layers-1):
            self.layers.append(GINConv(mlp(hidden_dim, hidden_dim)))
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, out_dim)
        self.task = task

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)
